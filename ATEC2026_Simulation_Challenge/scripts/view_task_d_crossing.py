# Created by Codex on 2026-06-22.
"""Visualize the TaskD crossing training environment."""

from __future__ import annotations

import argparse
import itertools

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View TaskD crossing terrain.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--task",
    type=str,
    default="ATEC-TaskD-Crossing-B2Piper",
    choices=[
        "ATEC-TaskD-Crossing-B2Piper",
        "ATEC-TaskD-Crossing-B2Piper-Easy",
        "ATEC-TaskD-Crossing-B2Piper-Mid",
        "ATEC-TaskD-Crossing-B2Piper-Hard",
    ],
    help="Which crossing terrain stage to visualize.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from atec_rl_lab.tasks.task_d import (  # noqa: E402
    TaskDCrossingEasyEnvCfg,
    TaskDCrossingEnvCfg,
    TaskDCrossingHardEnvCfg,
    TaskDCrossingMidEnvCfg,
)


TASK_CFGS = {
    "ATEC-TaskD-Crossing-B2Piper": TaskDCrossingEnvCfg,
    "ATEC-TaskD-Crossing-B2Piper-Easy": TaskDCrossingEasyEnvCfg,
    "ATEC-TaskD-Crossing-B2Piper-Mid": TaskDCrossingMidEnvCfg,
    "ATEC-TaskD-Crossing-B2Piper-Hard": TaskDCrossingHardEnvCfg,
}


def main():
    env_cfg = TASK_CFGS[args_cli.task]()
    env_cfg.scene.num_envs = args_cli.num_envs
    if not args_cli.enable_cameras:
        env_cfg.scene.head_camera = None
        env_cfg.observations.depth_camera = None

    env = ManagerBasedRLEnv(env_cfg)
    obs, info = env.reset()

    print("Robot joints:", env.scene["robot"].joint_names)
    print("Action dim:", env.action_space.shape)
    print("Obs keys:", obs.keys())
    print("Task:", args_cli.task)
    print("Env origin:", env.scene.env_origins[0].detach().cpu().tolist())
    if "box" in env.scene.rigid_objects:
        print("Box initial pos:", env.scene["box"].data.root_pos_w[0].detach().cpu().tolist())
    else:
        print("WARNING: box rigid object is not present in the scene.")

    for _ in itertools.count():
        if not simulation_app.is_running():
            break
        action = torch.zeros(env.action_space.shape, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        if done.any():
            env.reset(env_ids=done.nonzero(as_tuple=False).squeeze(-1))


if __name__ == "__main__":
    main()
    simulation_app.close()
