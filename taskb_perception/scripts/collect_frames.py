#!/usr/bin/env python3
"""
在有 Isaac Sim 的机器上采集 head RGB-D 帧，供本地离线调试。

必须在 ATEC 仓库根目录、已激活 isaaclab 环境下运行:

  conda activate isaaclab
  pip install "numpy<2.0.0" scipy ultralytics opencv-python

  cd ATEC2026_Simulation_Challenge
  python taskb_perception/scripts/collect_frames.py --task ATEC-TaskB-B2Piper --headless --enable_cameras --max_steps 500 --out taskb_perception/saved_frames

输出:
  saved_frames/head_rgb_000000.png
  saved_frames/head_depth_000000.npy
  saved_frames/meta_000000.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Isaac 必须先启动 AppLauncher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Task B perception frames.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--interval", type=int, default=5, help="每隔 N 个 control step 存一帧")
parser.add_argument("--out", type=str, default="taskb_perception/saved_frames")
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
from taskb_perception.obs_utils import parse_depth, parse_proprio, parse_rgb  # noqa: E402


def main():
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    saved = 0
    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break

        action_dim = (int(obs["proprio"].shape[-1]) - 12) // 3
        action = torch.zeros((1, action_dim), device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(action)

        if step % args_cli.interval != 0:
            continue

        image_obs = obs.get("image", {})
        rgb = parse_rgb(image_obs, "head_rgb")
        depth = parse_depth(image_obs, "head_depth")
        if rgb is None:
            continue

        idx = saved
        cv2.imwrite(str(out_dir / f"head_rgb_{idx:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if depth is not None:
            np.save(out_dir / f"head_depth_{idx:06d}.npy", depth)
        meta = {"step": step, "proprio": parse_proprio(obs["proprio"])["joint_pos"].tolist()}
        (out_dir / f"meta_{idx:06d}.json").write_text(json.dumps(meta), encoding="utf-8")
        saved += 1
        print(f"saved frame {idx} @ sim_step {step}")

        if terminated.item() or truncated.item():
            obs, _ = env.reset()

    env.close()
    simulation_app.close()
    print(f"完成，共保存 {saved} 帧 -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
