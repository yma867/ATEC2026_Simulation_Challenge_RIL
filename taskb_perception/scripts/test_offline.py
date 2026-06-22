#!/usr/bin/env python3
"""
离线测试感知模块（不需要 Isaac Sim / 渲染）。

用法:
  conda activate perception   # 或 isaaclab
  pip install "numpy<2.0.0" scipy ultralytics opencv-python
  cd taskb_perception
  python scripts/test_offline.py
  python scripts/test_offline.py --data_dir ./saved_frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from taskb_perception import PerceptionModule, PerceptionConfig


def make_synthetic_frame(seed: int = 0) -> tuple[dict, np.ndarray]:
    """生成一张带彩色块的假 RGB + 深度，用于验证 pipeline 能跑通。"""
    rng = np.random.default_rng(seed)
    h, w = 480, 640
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)

    # 黄块模拟 mustard
    cx, cy = int(rng.integers(200, 440)), int(rng.integers(120, 360))
    cv2.rectangle(rgb, (cx - 30, cy - 40), (cx + 30, cy + 40), (240, 210, 40), -1)

    depth = np.zeros((h, w), dtype=np.float32)
    depth[:, :] = 3.5
    depth[cy - 40 : cy + 40, cx - 30 : cx + 30] = 1.2

    obs = {
        "proprio": np.zeros(72, dtype=np.float32),  # B2 piper 维度
        "image": {
            "head_rgb": rgb,
            "head_depth": depth,
            "ee_rgb": None,
            "ee_depth": None,
        },
    }
    vis = rgb.copy()
    return obs, vis


def load_saved_frame(data_dir: Path, index: int) -> tuple[dict, np.ndarray | None]:
    rgb_path = data_dir / f"head_rgb_{index:06d}.png"
    depth_path = data_dir / f"head_depth_{index:06d}.npy"
    meta_path = data_dir / f"meta_{index:06d}.json"
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)

    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path) if depth_path.exists() else None
    proprio = np.zeros(72, dtype=np.float32)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "proprio" in meta:
            proprio = np.array(meta["proprio"], dtype=np.float32)

    obs = {
        "proprio": proprio,
        "image": {"head_rgb": rgb, "head_depth": depth},
    }
    return obs, rgb


def draw_tracks(img: np.ndarray, output) -> np.ndarray:
    vis = img.copy()
    for t in output.tracks:
        # 简单把 base x,y 投影回屏幕中心附近仅作 debug 占位
        u = int(320 + t.pos_b[1] * 80)
        v = int(240 - t.pos_b[0] * 80)
        color = (0, 255, 0) if t.status.value == "active" else (0, 128, 255)
        cv2.circle(vis, (u, v), 8, color, -1)
        cv2.putText(
            vis,
            f"id={t.track_id} {t.obj_class.value} c={t.confidence:.2f}",
            (u + 10, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if output.next_target:
        t = output.next_target
        cv2.putText(
            vis,
            f"next: id={t.track_id} dist={t.distance_xy():.2f}m",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="师姐采集的帧目录")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--show", action="store_true", help="弹窗显示（有显示器时）")
    args = parser.parse_args()

    cfg = PerceptionConfig(use_color_fallback=True)
    module = PerceptionModule(cfg)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        indices = sorted(int(p.stem.split("_")[-1]) for p in data_dir.glob("head_rgb_*.png"))
        if not indices:
            print(f"目录里没有 head_rgb_*.png: {data_dir}")
            return
        for i in indices[: args.frames]:
            obs, rgb = load_saved_frame(data_dir, i)
            out = module.update(obs)
            print(f"[frame {i}] active={out.num_active} tracks={[ (t.track_id, t.obj_class.value, round(t.confidence,2)) for t in out.tracks ]}")
            if rgb is not None and args.show:
                cv2.imshow("perception", cv2.cvtColor(draw_tracks(rgb, out), cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
    else:
        print("使用合成数据测试（无仿真）...")
        for i in range(args.frames):
            obs, rgb = make_synthetic_frame(seed=i)
            out = module.update(obs)
            print(f"[syn {i}] tracks={len(out.tracks)} next={out.next_target}")
            if args.show:
                cv2.imshow("perception", cv2.cvtColor(draw_tracks(rgb, out), cv2.COLOR_RGB2BGR))
                cv2.waitKey(200)
        if args.show:
            cv2.destroyAllWindows()

    print("离线测试完成。")


if __name__ == "__main__":
    main()
