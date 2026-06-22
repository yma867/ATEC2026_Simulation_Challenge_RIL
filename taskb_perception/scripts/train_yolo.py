#!/usr/bin/env python3
"""
训练 YOLOv8（纯 2D 检测，不需要仿真）。

  conda activate perception   # 或 isaaclab
  pip install "numpy<2.0.0" scipy ultralytics opencv-python

  python taskb_perception/scripts/train_yolo.py
  python taskb_perception/scripts/train_yolo.py --data taskb_perception/dataset/dataset.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="taskb_perception/dataset_head/dataset.yaml",
        help="head 检测数据集 yaml",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练 backbone")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--out",
        type=str,
        default="taskb_perception/weights/taskb_head_yolo.pt",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"找不到 {data_path}。请先:\n"
            "  python taskb_perception/scripts/init_yolo_datasets.py --which head\n"
            "  采集 + 标注 dataset_head，见 taskb_perception/LABELING_TRAINING.md"
        )

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="taskb_perception/runs",
        name="detect",
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"训练完成，权重已复制到: {out.resolve()}")


if __name__ == "__main__":
    main()
