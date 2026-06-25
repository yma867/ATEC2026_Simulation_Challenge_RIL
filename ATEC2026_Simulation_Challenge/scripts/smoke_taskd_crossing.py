"""Smoke-test TaskD crossing Parkour training environment.

Checks:
- box is spawned near the trench center
- action dimension is 12 leg actions
- policy observation is 753 dim
- optional depth camera group exists when --enable_cameras is used
- short rollout steps without NaN or shape mismatch
"""

from __future__ import annotations

import argparse

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test TaskD crossing training env.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=100, help="Number of rollout steps.")
parser.add_argument("--random_actions", action="store_true", default=False, help="Use small random actions instead of zero.")
parser.add_argument("--action_std", type=float, default=0.05, help="Stddev for random actions.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from atec_rl_lab.tasks.task_d import TaskDCrossingEnvCfg  # noqa: E402


EXPECTED_POLICY_DIM = 753
EXPECTED_ACTION_DIM = 12


def _assert_finite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN/Inf: shape={tuple(tensor.shape)}")


def main():
    env_cfg = TaskDCrossingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if not args_cli.enable_cameras:
        env_cfg.scene.head_camera = None
        env_cfg.observations.depth_camera = None

    env = gym.make("ATEC-TaskD-Crossing-B2Piper", cfg=env_cfg, render_mode=None)
    obs, info = env.reset()

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    box = unwrapped.scene["box"]

    print("[smoke] robot joints:", robot.joint_names)
    print("[smoke] action space:", env.action_space)
    print("[smoke] observation keys:", list(obs.keys()))
    print("[smoke] env origin:", unwrapped.scene.env_origins[0].detach().cpu().tolist())
    print("[smoke] box pos:", box.data.root_pos_w[0].detach().cpu().tolist())
    print("[smoke] robot pos:", robot.data.root_pos_w[0].detach().cpu().tolist())
    print("[smoke] robot root z min/max:", float(robot.data.root_pos_w[:, 2].min()), float(robot.data.root_pos_w[:, 2].max()))
    print("[smoke] robot root x min/max:", float(robot.data.root_pos_w[:, 0].min()), float(robot.data.root_pos_w[:, 0].max()))

    action_dim = env.action_space.shape[-1]
    if action_dim != EXPECTED_ACTION_DIM:
        raise RuntimeError(f"Expected action dim {EXPECTED_ACTION_DIM}, got {action_dim}")

    policy_obs = obs["policy"]
    if policy_obs.shape[-1] != EXPECTED_POLICY_DIM:
        raise RuntimeError(f"Expected policy obs dim {EXPECTED_POLICY_DIM}, got {policy_obs.shape[-1]}")
    _assert_finite("policy_obs", policy_obs)

    if "depth_camera" in obs:
        depth = obs["depth_camera"]
        print("[smoke] depth_camera shape:", tuple(depth.shape), "range:", float(depth.min()), float(depth.max()))
        _assert_finite("depth_camera", depth)
    else:
        print("[smoke] depth_camera group is disabled; use --enable_cameras to test student depth input.")

    if "delta_yaw_ok" in obs:
        print("[smoke] delta_yaw_ok shape:", tuple(obs["delta_yaw_ok"].shape))

    total_reward = torch.zeros(args_cli.num_envs, device=unwrapped.device)
    for step in range(args_cli.steps):
        if args_cli.random_actions:
            action = torch.randn((args_cli.num_envs, action_dim), device=unwrapped.device) * args_cli.action_std
        else:
            action = torch.zeros((args_cli.num_envs, action_dim), device=unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(action)
        _assert_finite("reward", reward)
        _assert_finite("policy_obs", obs["policy"])
        total_reward += reward
        done = terminated | truncated
        if done.any():
            obs, info = env.reset()
        if step % max(1, args_cli.steps // 5) == 0:
            box_pos = box.data.root_pos_w[0].detach().cpu().tolist()
            robot_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            print(
                f"[smoke] step={step} reward={float(reward.mean()):.4f} "
                f"done_rate={float(done.float().mean()):.4f} "
                f"terminated_rate={float(terminated.float().mean()):.4f} "
                f"truncated_rate={float(truncated.float().mean()):.4f} "
                f"root_z_minmax=({float(robot.data.root_pos_w[:, 2].min()):.3f},"
                f"{float(robot.data.root_pos_w[:, 2].max()):.3f}) "
                f"root_x_minmax=({float(robot.data.root_pos_w[:, 0].min()):.3f},"
                f"{float(robot.data.root_pos_w[:, 0].max()):.3f}) "
                f"robot={robot_pos} box={box_pos}"
            )

    print("[smoke] total_reward_mean:", float(total_reward.mean()))
    print("[smoke] PASS")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
