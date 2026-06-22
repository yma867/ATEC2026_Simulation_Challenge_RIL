#!/usr/bin/env python3
"""
在有仿真的机器上跑 Task B，只测感知层并打印跟踪结果。

  cd ATEC2026_Simulation_Challenge
  python taskb_perception/scripts/run_sim_debug.py --task ATEC-TaskB-B2Piper --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--debug", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from taskb_perception import PerceptionModule, PerceptionConfig  # noqa: E402


def main():
    perception = PerceptionModule(PerceptionConfig(
        yolo_weights="taskb_perception/weights/taskb_yolo.pt",
        use_color_fallback=True,
    ))
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    perception.reset()

    action_dim = (int(obs["proprio"].shape[-1]) - 12) // 3

    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break

        out = perception.update(obs)
        if args_cli.debug and step % 10 == 0:
            nt = out.next_target
            nt_str = f"id={nt.track_id} dist={nt.distance_xy():.2f}" if nt else "None"
            print(f"step={step} active={out.num_active} next={nt_str}")

        action = torch.zeros((1, action_dim), device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated.item() or truncated.item():
            obs, _ = env.reset()
            perception.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
