"""
Headless 合成训练数据生成器
无需仿真器渲染，用投影几何生成 YCB 物体检测数据集

原理:
    已知物体的世界坐标 + 相机模型 → 投影到像素 → 生成 bbox 标签
    在空白背景上绘制彩色矩形模拟物体 → 生成 RGB 图
    
输出 (YOLO 格式):
    datasets/
    ├── images/train/   # RGB 图像 (640×480 PNG)
    ├── images/val/     # 验证集
    ├── labels/train/   # YOLO 标注 (class_id cx cy w h 归一化)
    └── labels/val/

用法:
    conda activate perception
    cd taskb_perception
    python generate_synthetic_dataset.py --num_train 2000 --num_val 200
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    HEAD_CAM_MATRIX, IMG_W, IMG_H, HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX, HEAD_CAM_ROT_MATRIX_INV,  # cam2robot / robot2cam
    ROBOT_INIT_POS, BIN_CENTER, BIN_RADIUS,
    OBJECT_SIZES, CLASS_NAMES,
)

# =============================================================================
#  Class ID 映射 (YOLO 格式)
# =============================================================================
CLASS_TO_ID = {
    "sugar_box": 0,
    "mustard_bottle": 1,
    "banana": 2,
}

# =============================================================================
#  物体颜色 (用于绘制合成图)
# =============================================================================
CLASS_COLORS = {
    "sugar_box": (0, 255, 255),       # 黄色 BGR
    "mustard_bottle": (255, 0, 255),  # 紫色 BGR
    "banana": (0, 255, 0),            # 绿色 BGR
}

# =============================================================================
#  场景参数范围
# =============================================================================
# 物体可能出现的世界坐标范围 (相对初始机器人位置)
OBJ_X_RANGE = (-12.0, -1.0)    # x: -12 到 -1 (机器人在 -10)
OBJ_Y_RANGE = (-13.0, -1.0)    # y: -13 到 -1 (机器人在 -10)
OBJ_Z_BOTTOM = 0.02            # 物体底面的 z 坐标 (桌面高度)

# 机器人起始位置变化范围 (±1m)
ROBOT_X_JITTER = 1.0
ROBOT_Y_JITTER = 1.0
ROBOT_YAW_RANGE = (-0.5, 0.5)  # ±0.5 弧度

# 每帧物体数量范围
NUM_OBJ_PER_FRAME = (3, 12)

# 物体深度范围 (相机光轴方向 z_cam, 即 OpenCV cam_Z)
# 太远: bbox太小无法训练; 太近: 出 FOV
CAM_Z_MIN = 0.8    # 最小深度 (m)
CAM_Z_MAX = 8.0    # 最大深度 (m)

# 物体 bbox 有效像素范围
BBOX_MIN_W = 12    # 最小 bbox 宽度 (像素)
BBOX_MIN_H = 12
BBOX_MAX_RATIO = 0.7  # 单物体最多占图像比例


def world_to_camera(point_world, robot_pos, robot_yaw, cam_matrix, cam_pos_robot,
                    robot2cam: np.ndarray = None):
    """世界坐标 → 像素坐标 (含 30° 俯仰角)
    
    变换链:
      1. world → robot (平移 + 绕 Z 旋转)
      2. robot → OpenCV camera (robot2cam 矩阵, 含轴交换 + 俯仰角)
      3. pin-hole 投影
    
    robot2cam: 从 config.HEAD_CAM_ROT_MATRIX_INV 传入，若为 None 回退到默认
    """
    if robot2cam is None:
        robot2cam = HEAD_CAM_ROT_MATRIX_INV

    # Step 1: world → robot
    dx = point_world[0] - robot_pos[0]
    dy = point_world[1] - robot_pos[1]
    dz = point_world[2] - robot_pos[2]

    cos_yaw = np.cos(-robot_yaw)
    sin_yaw = np.sin(-robot_yaw)
    x_robot = cos_yaw * dx - sin_yaw * dy
    y_robot = sin_yaw * dx + cos_yaw * dy
    z_robot = dz

    # Step 2: robot → camera (含俯仰角)
    # robot frame: X=前, Y=右, Z=上
    p_robot_offset = np.array([
        x_robot - cam_pos_robot[0],
        y_robot - cam_pos_robot[1],
        z_robot - cam_pos_robot[2],
    ], dtype=np.float32)

    # robot → OpenCV cam: 轴交换 + 俯仰逆旋转
    p_cam = robot2cam @ p_robot_offset
    cam_x, cam_y, cam_z = p_cam[0], p_cam[1], p_cam[2]

    if cam_z <= 0.05:
        return None

    # Step 3: 针孔投影
    u_homo = cam_matrix @ np.array([cam_x, cam_y, cam_z])
    u = u_homo[0] / u_homo[2]
    v = u_homo[1] / u_homo[2]

    u_int = int(round(u))
    v_int = int(round(v))

    if 0 <= u_int < IMG_W and 0 <= v_int < IMG_H:
        return (u_int, v_int)
    return None


def compute_bbox_3d(point_world, size_3d, robot_pos, robot_yaw, cam_matrix, cam_pos_robot,
                    robot2cam: np.ndarray = None):
    """
    根据物体 3D 尺寸和位姿，计算投影后的 2D bbox (含俯仰角)
    
    在物体周围采样 3D 边界框的 8 个角点 + 中心点，投影后取外接矩形。
    
    Args:
        point_world: (3,) 物体中心世界坐标
        size_3d: (3,) 物体长宽高 (lx, ly, lz)
        robot2cam: robot->camera 旋转矩阵 (含俯仰角), None 则用默认
        
    Returns:
        (x1, y1, x2, y2) 像素坐标，若不可见返回 None
    """
    if robot2cam is None:
        robot2cam = HEAD_CAM_ROT_MATRIX_INV

    lx, ly, lz = size_3d
    half_lx, half_ly, half_lz = lx / 2, ly / 2, lz / 2

    # 物体坐标系的 8 个角点 (世界坐标，假设无旋转)
    corners_local = np.array([
        [-half_lx, -half_lx,  half_lx,  half_lx, -half_lx, -half_lx,  half_lx,  half_lx],
        [-half_ly,  half_ly,  half_ly, -half_ly, -half_ly,  half_ly,  half_ly, -half_ly],
        [-half_lz, -half_lz, -half_lz, -half_lz,  half_lz,  half_lz,  half_lz,  half_lz],
    ])  # (3, 8)

    corners_world = corners_local + point_world.reshape(3, 1)

    pixels = []
    for i in range(8):
        corner = corners_world[:, i]
        pix = world_to_camera(corner, robot_pos, robot_yaw, cam_matrix, cam_pos_robot,
                              robot2cam)
        if pix is not None:
            pixels.append(pix)

    if len(pixels) < 2:
        return None

    pixels = np.array(pixels)
    x1 = max(0, int(pixels[:, 0].min()))
    y1 = max(0, int(pixels[:, 1].min()))
    x2 = min(IMG_W - 1, int(pixels[:, 0].max()))
    y2 = min(IMG_H - 1, int(pixels[:, 1].max()))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def bbox_to_yolo(bbox, img_w, img_h):
    """
    将像素 bbox 转为 YOLO 归一化格式
    
    Args:
        bbox: (x1, y1, x2, y2) 像素坐标
        img_w, img_h: 图像尺寸
        
    Returns:
        (cx, cy, w, h) 归一化到 [0,1]
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def generate_random_scene():
    """
    生成一帧随机场景 (优化版: 先定像素位置→反算世界坐标, 零拒绝率)
    
    Returns:
        rgb: (H, W, 3) uint8 合成 RGB 图
        labels: list of (class_id, cx, cy, w, h) YOLO 格式标注
        metadata: dict 场景元数据 (调试用)
    """
    # ---- 随机机器人位姿 ----
    robot_pos = np.array([
        ROBOT_INIT_POS[0] + np.random.uniform(-ROBOT_X_JITTER, ROBOT_X_JITTER),
        ROBOT_INIT_POS[1] + np.random.uniform(-ROBOT_Y_JITTER, ROBOT_Y_JITTER),
        ROBOT_INIT_POS[2],
    ], dtype=np.float32)
    robot_yaw = np.random.uniform(*ROBOT_YAW_RANGE)
    robot_yaw_f32 = np.float32(robot_yaw)

    # ---- 预计算: 世界→机器人旋转矩阵 ----
    cos_y = np.cos(robot_yaw_f32)
    sin_y = np.sin(robot_yaw_f32)
    world2robot_rot = np.array([
        [cos_y,  sin_y, 0],
        [-sin_y, cos_y, 0],
        [0,      0,     1],
    ], dtype=np.float32)

    # ---- 随机背景 ----
    bg_choices = np.array([
        [180, 190, 200], [200, 195, 185], [170, 180, 190],
        [190, 185, 175], [160, 170, 180],
    ], dtype=np.uint8)
    bg_color = tuple(bg_choices[np.random.randint(len(bg_choices))].tolist())

    rgb = np.full((IMG_H, IMG_W, 3), bg_color, dtype=np.uint8)

    # 画地面 (下半部深色)
    ground_color = (
        np.random.randint(80, 120),
        np.random.randint(80, 120),
        np.random.randint(80, 120),
    )
    horizon = IMG_H // 2 + np.random.randint(-15, 15)
    cv2.rectangle(rgb, (0, horizon), (IMG_W, IMG_H), ground_color, -1)

    # 随机纹理 (简单的条纹/斑点模拟)
    if np.random.random() < 0.4:
        for _ in range(np.random.randint(3, 10)):
            px = np.random.randint(0, IMG_W)
            py = np.random.randint(horizon, IMG_H)
            size = np.random.randint(5, 30)
            intensity = np.random.randint(0, 40)
            color = tuple(np.clip(np.array(ground_color) + intensity, 0, 255).tolist())
            cv2.circle(rgb, (px, py), size, color, -1)

    # ---- 随机物体 (像素→世界 逆投影, 保证在视野内) ----
    num_objs = np.random.randint(*NUM_OBJ_PER_FRAME)
    bin_center = np.array(BIN_CENTER)

    labels = []
    objects_placed = []  # [(cls_name, bbox)]

    # 重试上限 (避免死循环, 正常情况不会触发)
    max_retries_per_obj = 8

    for _ in range(num_objs):
        for retry in range(max_retries_per_obj):
            # ---- 随机选类别 ----
            class_id = np.random.choice(list(CLASS_NAMES.keys()))
            cls_name = CLASS_NAMES[class_id]
            size_dict = OBJECT_SIZES.get(cls_name, {"lx": 0.15, "ly": 0.10, "lz": 0.05})
            size_3d = (size_dict["lx"], size_dict["ly"], size_dict["lz"])

            # ---- Step 1: 随机像素中心 + 深度 ----
            u = (np.random.random() * 0.90 + 0.05) * IMG_W  # 5%~95% 宽度
            v = (np.random.random() * 0.80 + 0.10) * IMG_H  # 10%~90% 高度
            z_cam = np.random.uniform(CAM_Z_MIN, CAM_Z_MAX)  # 深度

            # ---- Step 2: 像素 → cam (OpenCV 针孔反投影) ----
            cam_x = (u - HEAD_CAM_MATRIX[0, 2]) / HEAD_CAM_MATRIX[0, 0] * z_cam
            cam_y = (v - HEAD_CAM_MATRIX[1, 2]) / HEAD_CAM_MATRIX[1, 1] * z_cam
            p_cam = np.array([cam_x, cam_y, z_cam], dtype=np.float32)

            # ---- Step 3: cam → robot (cam2robot = 轴交换 + 俯仰逆) ----
            p_robot_cam = HEAD_CAM_ROT_MATRIX @ p_cam
            p_robot = HEAD_CAM_POS_ROBOT + p_robot_cam

            # ---- Step 4: robot → world ----
            # robot 坐标系下物体在地面, robot_z 方向应修正到地面
            # 物体底面在地面 (world z ≈ 0)
            obj_center_z_world = size_3d[2] / 2 + OBJ_Z_BOTTOM
            # 反算正确的 world 坐标: 先用 xy + yaw 算 world, 再强制设 z
            p_world_xy = (robot_pos[:2]
                          + np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
                          @ p_robot[:2])

            p_world = np.array([p_world_xy[0], p_world_xy[1], obj_center_z_world], dtype=np.float32)

            # ---- Step 5: 正向投影验证 bbox (确保尺寸合理) ----
            bbox = compute_bbox_3d(p_world, size_3d, robot_pos, robot_yaw_f32,
                                   HEAD_CAM_MATRIX, HEAD_CAM_POS_ROBOT,
                                   HEAD_CAM_ROT_MATRIX_INV)

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if (bbox_w < BBOX_MIN_W or bbox_h < BBOX_MIN_H
                    or bbox_w > IMG_W * BBOX_MAX_RATIO
                    or bbox_h > IMG_H * BBOX_MAX_RATIO):
                continue

            if x1 < 0 or y1 < 0 or x2 >= IMG_W or y2 >= IMG_H:
                continue

            # 避免与已放置物体重叠太多
            overlap = False
            for _, prev_bbox in objects_placed:
                if compute_iou(bbox, prev_bbox) > 0.25:
                    overlap = True
                    break
            if overlap:
                continue

            # ---- 成功! 保留此物体 ----
            # 垃圾桶内判定 (世界坐标 vs bin_center)
            dist_to_bin = float(np.linalg.norm(p_world[:2] - bin_center[:2]))
            in_bin = dist_to_bin < BIN_RADIUS * 0.7

            # YOLO 标注
            cx, cy, w, h = bbox_to_yolo(bbox, IMG_W, IMG_H)
            labels.append((class_id, cx, cy, w, h))
            objects_placed.append((cls_name, bbox))

            # ---- 画物体 ----
            color = list(CLASS_COLORS[cls_name])

            # 随机颜色变化 (模拟光照)
            if np.random.random() < 0.5:
                for c in range(3):
                    color[c] = np.clip(color[c] + np.random.randint(-30, 30), 0, 255)
            # 转 Python 原生 int (OpenCV 不接受 numpy int)
            color_tup = tuple(int(c) for c in color)

            if in_bin:
                overlay = rgb.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), -1)
                cv2.addWeighted(overlay, 0.3, rgb, 0.7, 0, rgb)
            else:
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color_tup, -1)
                if np.random.random() < 0.3:
                    hl = tuple(np.clip(np.array(color_tup) + 40, 0, 255).tolist())
                    margin = max(2, (x2 - x1) // 5)
                    cv2.rectangle(rgb, (x1 + margin, y1 + margin),
                                  (x2 - margin, y2 - margin), hl, -1)

            border_color = tuple(max(0, c - 80) for c in color_tup)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), border_color, 1)

            # 噪声斑点
            if np.random.random() < 0.3:
                nx = np.random.randint(x1 + 2, x2 - 2)
                ny = np.random.randint(y1 + 2, y2 - 2)
                ns = np.random.randint(2, max(3, (x2 - x1) // 4))
                cv2.circle(rgb, (nx, ny), ns,
                           (np.random.randint(0, 100),
                            np.random.randint(0, 100),
                            np.random.randint(0, 100)), -1)

            break  # 物体生成成功, 退出重试循环

    # ---- 全局噪声 ----
    noise = np.random.randint(-5, 6, rgb.shape, dtype=np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if np.random.random() < 0.3:
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0.5)

    return rgb, labels, {
        'robot_pos': robot_pos.tolist(),
        'robot_yaw': robot_yaw_f32.item(),
        'n_objects': len(labels),
    }


def compute_iou(bbox1, bbox2):
    """计算两个 bbox 的 IoU"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def create_directories():
    """创建数据集目录结构"""
    dirs = [
        'datasets/images/train',
        'datasets/images/val',
        'datasets/labels/train',
        'datasets/labels/val',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_dataset(num_train: int, num_val: int, seed: int = 42):
    """生成完整训练/验证集"""
    np.random.seed(seed)

    create_directories()

    total = num_train + num_val
    generated = 0

    # 先生成训练集
    for split, num in [('train', num_train), ('val', num_val)]:
        img_dir = Path(f'datasets/images/{split}')
        lbl_dir = Path(f'datasets/labels/{split}')

        print(f"\nGenerating {split} set: {num} images...")
        count = 0
        attempts = 0
        max_attempts = num * 5  # 防止无限循环

        while count < num and attempts < max_attempts:
            attempts += 1

            rgb, labels, meta = generate_random_scene()

            # 至少有 2 个可见物体才保留
            if len(labels) < 2:
                continue

            frame_name = f"synth_{count:06d}"

            # 保存图像
            img_path = img_dir / f"{frame_name}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # 保存标注
            lbl_path = lbl_dir / f"{frame_name}.txt"
            with open(lbl_path, 'w') as f:
                for class_id, cx, cy, w, h in labels:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            count += 1
            generated += 1

            if count % 100 == 0:
                print(f"  {split}: {count}/{num} (total: {generated}/{total})")

        print(f"  {split}: {count}/{num} complete (max attempts: {max_attempts})")

    print(f"\nTotal generated: {generated}/{total}")
    print(f"Dataset saved to: {os.path.abspath('datasets')}")

    # ---- 统计 ----
    print("\nDataset statistics:")
    for split in ['train', 'val']:
        lbl_dir = Path(f'datasets/labels/{split}')
        lbl_files = list(lbl_dir.glob('*.txt'))
        total_instances = 0
        class_counts = {0: 0, 1: 0, 2: 0}
        for lf in lbl_files:
            with open(lf) as f:
                lines = f.readlines()
            total_instances += len(lines)
            for line in lines:
                cid = int(line.strip().split()[0])
                class_counts[cid] = class_counts.get(cid, 0) + 1

        print(f"  {split}: {len(lbl_files)} images, {total_instances} instances")
        for cid, cname in {0: 'sugar_box', 1: 'mustard_bottle', 2: 'banana'}.items():
            print(f"    {cname}: {class_counts[cid]}")


def generate_sample_visualization(num_samples: int = 5):
    """生成示例可视化图片 (用于检查数据质量)"""
    os.makedirs('datasets/viz', exist_ok=True)

    for i in range(num_samples):
        rgb, labels, meta = generate_random_scene()

        # 在图上画 bbox
        viz = rgb.copy()
        for class_id, cx_norm, cy_norm, w_norm, h_norm in labels:
            cx = int(cx_norm * IMG_W)
            cy = int(cy_norm * IMG_H)
            w = int(w_norm * IMG_W)
            h = int(h_norm * IMG_H)
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = cx + w // 2
            y2 = cy + h // 2

            cls_name = list(CLASS_TO_ID.keys())[list(CLASS_TO_ID.values()).index(class_id)]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))

            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz, f"{cls_name} (#{class_id})",
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        cv2.imwrite(f'datasets/viz/sample_{i:02d}.png',
                    cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic YOLO dataset for Task B objects"
    )
    parser.add_argument('--num_train', type=int, default=2000,
                        help='Number of training images')
    parser.add_argument('--num_val', type=int, default=200,
                        help='Number of validation images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--viz', type=int, default=5,
                        help='Number of sample visualizations (0 to skip)')
    args = parser.parse_args()

    print("=" * 70)
    print("  ATEC 2026 Task B — Synthetic Dataset Generator")
    print("  Headless mode — no simulator required")
    print("=" * 70)
    print(f"  Train images: {args.num_train}")
    print(f"  Val images:   {args.num_val}")
    print(f"  Image size:   {IMG_W}×{IMG_H}")
    print(f"  Classes:      sugar_box(0), mustard_bottle(1), banana(2)")
    print(f"  Output dir:   {os.path.abspath('datasets')}")
    print("=" * 70)

    # 生成数据集
    generate_dataset(args.num_train, args.num_val, args.seed)

    # 生成可视化样本
    if args.viz > 0:
        print(f"\nGenerating {args.viz} visualization samples...")
        generate_sample_visualization(args.viz)
        print(f"  Saved to datasets/viz/")

    print("\n" + "=" * 70)
    print("  DATASET GENERATION COMPLETE")
    print("=" * 70)
    print("""
Next steps:
  1. Inspect samples:
     Check datasets/viz/sample_*.png to verify bbox quality

  2. Train YOLO:
     python train_yolo.py --epochs 50 --batch 16

  3. Deploy:
     cp runs/train/taskb_ycb/weights/best.pt ./taskb_ycb.pt
     Update perception_pipeline.py to use taskb_ycb.pt

Note: Synthetic data may not perfectly match simulation appearance.
      Consider collecting real simulation screenshots later for fine-tuning.
""")


if __name__ == "__main__":
    main()