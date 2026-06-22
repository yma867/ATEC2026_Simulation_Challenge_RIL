#!/usr/bin/env python3
"""创建 head 检测 / ee 分割 两套 YOLO 数据集目录 + dataset.yaml。"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "head": {
        "dir": "dataset_head",
        "yaml_name": "dataset.yaml",
        "desc": "head_rgb → YOLO detect (矩形框)",
    },
    "ee": {
        "dir": "dataset_ee_seg",
        "yaml_name": "dataset.yaml",
        "desc": "ee_rgb 俯视 → YOLO-seg (多边形)",
    },
}


def _write_yaml(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    yaml = out_root / "dataset.yaml"
    yaml.write_text(
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: sugar\n"
        "  1: mustard\n"
        "  2: banana\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        choices=("head", "ee", "both"),
        default="both",
        help="head=检测数据集  ee=分割数据集  both=两个都建",
    )
    args = parser.parse_args()

    keys = ("head", "ee") if args.which == "both" else (args.which,)
    for key in keys:
        info = DATASETS[key]
        out = ROOT / info["dir"]
        _write_yaml(out)
        print(f"[OK] {info['desc']}")
        print(f"     {out.resolve()}")
        print(f"     图片 → {out / 'images/train'}")
        print(f"     标签 → {out / 'labels/train'}")
    print("\n标注完成后:")
    print("  python taskb_perception/scripts/check_dataset.py --task head  taskb_perception/dataset_head")
    print("  python taskb_perception/scripts/check_dataset.py --task ee    taskb_perception/dataset_ee_seg")
    print("  python taskb_perception/scripts/train_yolo.py     --data taskb_perception/dataset_head/dataset.yaml")
    print("  python taskb_perception/scripts/train_yolo_seg.py --data taskb_perception/dataset_ee_seg/dataset.yaml")


if __name__ == "__main__":
    main()
