#!/usr/bin/env python3
"""离线测试 EE seg 几何中心：对单张 ee_rgb 画质心与对准十字。

用法（在仓库根目录 ATEC2026_Simulation_Challenge-main/ 下）:

  # 自动找 dataset_ee_seg 里第一张图
  python taskb_perception/scripts/test_ee_seg_center.py

  # 指定图片
  python taskb_perception/scripts/test_ee_seg_center.py \\
    --image taskb_perception/dataset_ee_seg/images/train/000001.jpg \\
    --out debug_center.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "taskb_perception"
sys.path.insert(0, str(PKG_ROOT))

from taskb_perception.config import PerceptionConfig, WEIGHT_EE_SEG  # noqa: E402
from taskb_perception.ee_seg_align import EESegAligner  # noqa: E402


def resolve_repo_path(p: str | Path) -> Path:
    """相对仓库根目录解析路径；已是绝对路径则直接用。"""
    path = Path(p)
    if path.is_file():
        return path.resolve()
    cand = (REPO_ROOT / path).resolve()
    if cand.is_file():
        return cand
    cand2 = (PKG_ROOT / path).resolve()
    if cand2.is_file():
        return cand2
    return path


def find_sample_ee_image() -> Path | None:
    """在常见 EE 数据集目录里找一张测试图。"""
    search_dirs = [
        PKG_ROOT / "dataset_ee_seg" / "images" / "train",
        PKG_ROOT / "dataset_ee_seg" / "images" / "val",
        PKG_ROOT / "dataset" / "images" / "train",
        PKG_ROOT / "dataset" / "images",
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
            files = sorted(d.glob(ext))
            if files:
                return files[0]
    return None


def print_usage_hints() -> None:
    print("\n[提示] --image 需要真实文件路径，不要用文档占位符 path/to/ee_frame.jpg")
    print("  1) 若已采集 EE 数据，例如:")
    print("     python taskb_perception/scripts/test_ee_seg_center.py \\")
    print("       --image taskb_perception/dataset_ee_seg/images/train/000001.jpg")
    print("  2) 或省略 --image，脚本会自动在 dataset_ee_seg/images/train 找第一张 jpg")
    print("  3) 还没有图时，先在仿真里采集:")
    print("     python taskb_perception/scripts/manual_collect_ee.py \\")
    print("       --task ATEC-TaskB-B2Piper --enable_cameras --resume")
    print("     按 P 存 ee_rgb 后再跑本脚本")
    print(f"\n  仓库根目录: {REPO_ROOT}")
    print(f"  seg 权重默认: {REPO_ROOT / WEIGHT_EE_SEG}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EE seg mask 几何中心测试")
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="ee_rgb 图片；省略则自动找 dataset_ee_seg/images/train 下第一张",
    )
    parser.add_argument("--weights", type=str, default=None, help="seg 权重，默认 config")
    parser.add_argument("--out", type=str, default="", help="可视化输出路径")
    parser.add_argument("--target-u", type=float, default=None)
    parser.add_argument("--target-v", type=float, default=None)
    args = parser.parse_args()

    if args.image:
        img_path = resolve_repo_path(args.image)
        if not img_path.is_file():
            print(f"[ERROR] 找不到图片: {args.image}")
            print(f"        解析为: {img_path}")
            print_usage_hints()
            sys.exit(1)
    else:
        img_path = find_sample_ee_image()
        if img_path is None:
            print("[ERROR] 未指定 --image，且 dataset_ee_seg 下没有 jpg/png")
            print_usage_hints()
            sys.exit(1)
        print(f"[INFO] 自动使用测试图: {img_path}")

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"[ERROR] OpenCV 无法读取: {img_path}")
        sys.exit(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cfg = PerceptionConfig()
    if args.weights:
        cfg.yolo_ee_seg_weights = str(resolve_repo_path(args.weights))
    else:
        cfg.yolo_ee_seg_weights = str(resolve_repo_path(cfg.yolo_ee_seg_weights or WEIGHT_EE_SEG))
    if args.target_u is not None:
        cfg.ee_align_target_u = args.target_u
    if args.target_v is not None:
        cfg.ee_align_target_v = args.target_v

    weight_path = Path(cfg.yolo_ee_seg_weights)
    if not weight_path.is_file():
        print(f"[ERROR] seg 权重不存在: {weight_path}")
        print("  请确认已解压 ee_seg.zip，或指定 --weights")
        sys.exit(1)

    aligner = EESegAligner(cfg)
    if not aligner.ready:
        print("[ERROR] seg 模型未加载")
        sys.exit(1)

    centers = aligner.detect_centers(rgb)
    if not centers:
        print("[WARN] 未检测到分割实例（可换一张有物体的 ee 图，或降低 conf_threshold）")
    for i, c in enumerate(centers):
        print(
            f"[{i}] {c.obj_class.value} conf={c.confidence:.3f} "
            f"center=({c.u:.1f},{c.v:.1f}) err=({c.err_u:+.1f},{c.err_v:+.1f}) "
            f"aligned={c.aligned} area={c.area_px:.0f}px"
        )
        dx, dy = aligner.arm_xy_nudge_from_error(c, cfg.ee_align_gain_m_per_px)
        print(f"     nudge base xy: dx={dx:+.4f} dy={dy:+.4f} m")

    vis = aligner.draw_debug(rgb, centers)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    out = args.out or str(img_path.with_name(img_path.stem + "_center.jpg"))
    cv2.imwrite(out, vis_bgr)
    print(f"[OK] 可视化已保存: {out}")


if __name__ == "__main__":
    main()
