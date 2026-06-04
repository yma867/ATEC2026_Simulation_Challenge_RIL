"""
离线可视化调试工具
输入: RGB图 + 深度图 (从仿真器保存的 numpy)
输出: 标注后的图像（bbox、类别、追踪ID、3D位置、距离等）
"""

import argparse
import cv2
import numpy as np
import torch
from perception_pipeline import PerceptionPipeline, CLASS_NAMES


def draw_results(rgb, perception_output):
    """
    在 RGB 图上绘制感知结果

    Args:
        rgb: (H, W, 3) uint8 numpy
        perception_output: dict, from pipeline.process()

    Returns:
        annotated: (H, W, 3) uint8 numpy
    """
    annotated = rgb.copy()
    H, W = annotated.shape[:2]

    # 颜色表（BGR）
    colors = {
        "sugar_box": (0, 255, 255),       # 黄
        "mustard_bottle": (255, 0, 255),  # 紫
        "banana": (0, 255, 0),            # 绿
    }

    for obj in perception_output.get('objects_detailed', []):
        bbox = obj['bbox']
        cls_name = obj['class']
        track_id = obj['id']
        conf = obj['conf']
        dist = obj['dist_to_robot']
        in_bin = obj['in_bin']
        pos_w = obj['pos_world']
        color = colors.get(cls_name, (255, 255, 255))

        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x1 = min(x1, W - 1)
        x2 = min(x2, W - 1)
        y1 = min(y1, H - 1)
        y2 = min(y2, H - 1)

        # 画 bbox
        thickness = 2
        if in_bin:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), thickness)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # 标签
        label = f"ID{track_id} {cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 3D 位置信息
        info = f"dist={dist:.2f}m world=({pos_w[0]:.1f},{pos_w[1]:.1f},{pos_w[2]:.2f})"
        if in_bin:
            info += " [BIN]"
        cv2.putText(annotated, info, (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # ---- 画垃圾桶方向 (顶部信息条) ----
    bin_info = perception_output.get('bin', {})
    robot_info = perception_output.get('robot', {})
    progress = perception_output.get('progress', {})
    gripper = perception_output.get('gripper', {})

    info_lines = [
        f"Robot: pos=({robot_info.get('pos_world', [0, 0])}) yaw={robot_info.get('yaw', 0):.2f}",
        f"Bin: dist={bin_info.get('dist_to_robot', 0):.2f}m yaw_rel={bin_info.get('yaw_rel', 0):.2f}",
        f"Progress: {progress.get('inside_bin', 0)}/{progress.get('total', 18)} in bin ({progress.get('remaining', 0)} left)",
        f"Gripper: holding={gripper.get('is_holding', False)} width={gripper.get('width', 0):.3f}m",
    ]

    y_offset = 15
    for line in info_lines:
        cv2.putText(annotated, line, (5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        y_offset += 18

    # ---- 目标高亮 ----
    target = perception_output.get('target')
    if target is not None:
        cv2.putText(annotated,
                    f">> TARGET: ID{target['id']} {target['class']} dist={target['dist_to_robot']:.2f}m",
                    (5, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Perception Visualization Tool")
    parser.add_argument('--rgb', type=str, required=True, help='Path to RGB image (npy or png/jpg)')
    parser.add_argument('--depth', type=str, required=True, help='Path to depth image (npy)')
    parser.add_argument('--proprio', type=str, default=None, help='Path to proprio array (npy, 72,)')
    parser.add_argument('--output', type=str, default='annotated.png', help='Output image path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for YOLO')
    args = parser.parse_args()

    # ---- 加载数据 ----
    if args.rgb.endswith('.npy'):
        rgb = np.load(args.rgb)
    else:
        rgb = cv2.imread(args.rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if args.depth.endswith('.npy'):
        depth = np.load(args.depth)
    else:
        raise ValueError("Depth must be .npy file")

    if args.proprio:
        proprio = np.load(args.proprio)
    else:
        proprio = np.zeros(72, dtype=np.float32)

    # 确保形状正确
    if rgb.ndim == 4:
        rgb = rgb.squeeze(0)
    if depth.ndim == 4:
        depth = depth.squeeze(0)

    H, W = rgb.shape[:2]

    # ---- 运行 Pipeline ----
    pipeline = PerceptionPipeline(device=args.device)

    # 构造 obs dict
    obs = {
        'image': {
            'head_rgb': torch.from_numpy(rgb).unsqueeze(0).to(torch.uint8),
            'head_depth': torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1).to(torch.float32),
        },
        'proprio': torch.from_numpy(proprio).unsqueeze(0).to(torch.float32),
    }

    perception_output = pipeline.process(obs, dt=0.02)

    # ---- 绘制 ----
    annotated = draw_results(rgb, perception_output)

    # 保存
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, annotated_bgr)
    print(f"Annotated image saved to: {args.output}")

    # ---- 打印文本输出 ----
    print("\n" + "=" * 60)
    print("PERCEPTION OUTPUT")
    print("=" * 60)
    target = perception_output['target']
    if target:
        print(f"Target: ID={target['id']} class={target['class']} conf={target['conf']:.2f} "
              f"dist={target['dist_to_robot']:.2f}m")
        print(f"  pos_world=({target['pos_world'][0]:.2f}, {target['pos_world'][1]:.2f}, "
              f"{target['pos_world'][2]:.3f})")
        print(f"  grasp_quat=({target['grasp_quat_world'][0]:.3f}, {target['grasp_quat_world'][1]:.3f}, "
              f"{target['grasp_quat_world'][2]:.3f}, {target['grasp_quat_world'][3]:.3f})")
    else:
        print("Target: None")

    print(f"\nObjects remaining: {len(perception_output['objects_remaining'])}")
    for obj in perception_output['objects_remaining']:
        flag = "[BIN]" if obj['in_bin'] else ""
        print(f"  ID{obj['id']:3d} {obj['class']:20s} dist={obj['dist']:5.2f}m {flag}")

    print(f"\nBin: center={perception_output['bin']['center_world']} "
          f"dist={perception_output['bin']['dist_to_robot']:.2f}m "
          f"yaw_rel={perception_output['bin']['yaw_rel']:.2f}")
    print(f"Progress: {perception_output['progress']['inside_bin']}/{perception_output['progress']['total']} "
          f"in bin")
    print(f"Robot: pos={perception_output['robot']['pos_world']} yaw={perception_output['robot']['yaw']:.2f}")
    print(f"Gripper: holding={perception_output['gripper']['is_holding']} "
          f"width={perception_output['gripper']['width']:.4f}m")


if __name__ == "__main__":
    main()