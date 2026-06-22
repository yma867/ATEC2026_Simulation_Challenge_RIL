#!/usr/bin/env python3
"""检查 YOLO 数据集：head=矩形框 detect，ee=多边形 seg。"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_detect_line(parts: list[str]) -> bool:
    return len(parts) == 5


def _parse_seg_line(parts: list[str]) -> bool:
    return len(parts) >= 7 and (len(parts) - 1) % 2 == 0


def check(root: Path, task: str) -> int:
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    if not img_dir.is_dir():
        print(f"[ERROR] 缺少 {img_dir}")
        return 1

    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not imgs:
        print(f"[WARN] {img_dir} 无图片")
        return 1

    missing, empty, bad_fmt, ok = [], [], [], 0
    for img in imgs:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.is_file():
            missing.append(img.name)
            continue
        text = lbl.read_text(encoding="utf-8").strip()
        if not text:
            empty.append(img.name)
            continue
        valid = True
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            if task == "head" and not _parse_detect_line(parts):
                valid = False
                break
            if task == "ee" and not _parse_seg_line(parts):
                valid = False
                break
        if valid:
            ok += 1
        else:
            bad_fmt.append(img.name)

    print(f"任务类型 : {'YOLO detect (head)' if task == 'head' else 'YOLO-seg (ee)'}")
    print(f"根目录   : {root.resolve()}")
    print(f"图片     : {len(imgs)}")
    print(f"有效标签 : {ok}")
    print(f"缺 txt   : {len(missing)}")
    print(f"空 txt   : {len(empty)}")
    print(f"格式错   : {len(bad_fmt)}")
    if missing[:3]:
        print("  缺标签:", missing[:3])
    if bad_fmt[:3]:
        print("  格式错:", bad_fmt[:3])
        if task == "head":
            print("  head 每行应为: class cx cy w h  (5 个数，YOLO 矩形)")
        else:
            print("  ee 每行应为: class x1 y1 x2 y2 ... (≥3 点，YOLO-seg 多边形)")

    if ok == 0:
        print("\n[结论] 还不能训练，请先完成标注。")
        return 1
    if ok < len(imgs) * 0.9:
        print("\n[结论] 大部分图还没标完，可以继续标或删掉无标签图。")
        return 0
    print("\n[结论] 可以开始训练。")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        nargs="?",
        default="taskb_perception/dataset_head",
        help="数据集根目录",
    )
    parser.add_argument(
        "--task",
        choices=("head", "ee"),
        required=True,
        help="head=检测框  ee=分割多边形",
    )
    args = parser.parse_args()
    raise SystemExit(check(Path(args.root), args.task))


if __name__ == "__main__":
    main()
