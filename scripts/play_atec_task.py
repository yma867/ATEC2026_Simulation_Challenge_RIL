# Created by skywoodsz on 2026/02/07.

import argparse
import os
import sys
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play Atec Tasks (ENV only, no RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints for per-step reward/time metrics.",
)
parser.add_argument(
    "--gt_nav",
    action="store_true",
    default=False,
    help="Enable GT navigation mode (directly read object positions from simulation).",
)
parser.add_argument(
    "--wbc",
    action="store_true",
    default=False,
    help="Enable WBC mode (use WBC to control the robot).",
)
parser.add_argument(
    "--show_score",
    action="store_true",
    default=False,
    help="Show real-time score and elapsed time during play.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# If recording video, need cameras enabled in IsaacLab/Kit
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec


def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    # -------------------------------------------------------------------------
    # Initialize solution (dynamic import based on --gt_nav flag)
    # -------------------------------------------------------------------------
    global solution
    if args_cli.gt_nav:
        print("[INFO] Enabling GT Navigation mode")
        from demo.solution_gt import AlgSolution
        solution = AlgSolution(env=None)
    else:
        from demo.solution import AlgSolution
        solution = AlgSolution(env=None)

    # TODO: simulate getting action spec from jason string (e.g. from a file or network)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    action_spec_json = json.dumps(action_spec) if action_spec else None

    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # New Feature: apply safe action spec to env config (e.g. for scaling/clipping actions from your solution)
    if action_spec_json:
        env_cfg = apply_safe_action_spec(env_cfg, action_spec_json)

    def make_env(current_env_cfg):
        env_inst = gym.make(args_cli.task, cfg=current_env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env_inst.unwrapped, DirectMARLEnv):
            env_inst = multi_agent_to_single_agent(env_inst)
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.abspath(os.path.join("logs", "videos", args_cli.task, "play")),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env_inst = gym.wrappers.RecordVideo(env_inst, **video_kwargs)
        return env_inst

    env = make_env(env_cfg)
    solution.env = env

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    last_reward = 0.0  # Track last reward for delta display
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # ===== Your controller goes here =====
            resp = solution.predicts(obs, total_episode_reward)
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            # 视角跟随：尊重 solution 中的 camera_follow_enabled 开关
            #  在 demo/solution_gt.py 顶部修改 ATEC_CAMERA_FOLLOW_ROBOT 或设置
            #   ATEC_TASKB_CAMERA_FOLLOW=1 来启用视角跟随
            if not is_task_e and getattr(solution, "camera_follow_enabled", True):
                camera_follow(env)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            if args_cli.debug:
                print(f"total_episode_reward:{total_episode_reward: .2f}")
                print(f"total_elapsed_time:{total_elapsed_time: .2f}")
            
            # Show score only when reward increases
            if args_cli.show_score:
                reward_delta = total_episode_reward - last_reward
                if reward_delta > 0.001:  # Only show when there's a meaningful increase
                    print(f"[+{reward_delta:.2f}] Score: {total_episode_reward:.2f} | Time: {total_elapsed_time:.2f}s")
                last_reward = total_episode_reward

            done = (terminated.item() or truncated.item())
            if done:
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()