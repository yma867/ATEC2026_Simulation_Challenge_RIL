#!/usr/bin/env python3
"""
训练 YOLOv8-Seg（EE 俯视分割 → 几何中心）。

  python taskb_perception/scripts/train_yolo_seg.py \\
    --data taskb_perception/dataset_ee_seg/dataset.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="taskb_perception/dataset_ee_seg/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out", type=str, default="taskb_perception/weights/taskb_ee_seg.pt")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到 {data_path}，请先运行 manual_collect_ee.py 并手动标注 seg txt")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="taskb_perception/runs",
        name="seg",
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"Seg 权重: {out.resolve()}")


if __name__ == "__main__":
    main()
