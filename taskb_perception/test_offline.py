"""
离线测试脚本 — 无需仿真器，用合成数据验证完整感知 Pipeline
适用于 headless 服务器 (AutoDL 4090)

用法:
    python test_offline.py --gt      # GT模式: 验证坐标变换链
    python test_offline.py           # YOLO模式: 端到端测试
"""

import numpy as np
import torch
import cv2
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception_pipeline import PerceptionPipeline, CLASS_NAMES
from config import (
    HEAD_CAM_MATRIX, IMG_W, IMG_H,
    HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX, HEAD_CAM_ROT_MATRIX_INV,
    ROBOT_INIT_POS, BIN_CENTER, BIN_RADIUS,
    OBJECT_SIZES,
)

ROBOT_POS = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
ROBOT_YAW = 0.12
BIN = np.array(BIN_CENTER, dtype=np.float32)
NUM_GT_OBJECTS = 8
CAM_Z_MIN, CAM_Z_MAX = 1.0, 7.0
SEEDS = [42, 43, 44, 45, 46]


def pixel_to_world(u, v, z_cam, robot_pos, robot_yaw):
    """像素+深度 → 世界坐标 (与 world_to_pixel 严格互逆)"""
    cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
    cam_x = (u - HEAD_CAM_MATRIX[0, 2]) / HEAD_CAM_MATRIX[0, 0] * z_cam
    cam_y = (v - HEAD_CAM_MATRIX[1, 2]) / HEAD_CAM_MATRIX[1, 1] * z_cam
    p_cam = np.array([cam_x, cam_y, z_cam], dtype=np.float32)
    p_robot = HEAD_CAM_POS_ROBOT + HEAD_CAM_ROT_MATRIX @ p_cam
    return np.array([
        robot_pos[0] + cy * p_robot[0] - sy * p_robot[1],
        robot_pos[1] + sy * p_robot[0] + cy * p_robot[1],
        robot_pos[2] + p_robot[2],
    ], dtype=np.float32)


def world_to_pixel(point_world, robot_pos, robot_yaw, K):
    """世界坐标 → 像素坐标 (含俯仰角, 与 pipeline _cam_to_robot 互逆)"""
    dx, dy, dz = point_world - robot_pos
    cy, sy = np.cos(-robot_yaw), np.sin(-robot_yaw)
    xr = cy * dx - sy * dy
    yr = sy * dx + cy * dy
    zr = dz
    p_off = np.array([xr - HEAD_CAM_POS_ROBOT[0],
                       yr - HEAD_CAM_POS_ROBOT[1],
                       zr - HEAD_CAM_POS_ROBOT[2]], dtype=np.float32)
    p_cam = HEAD_CAM_ROT_MATRIX_INV @ p_off
    if p_cam[2] <= 0.05:
        return None
    ui = int(round(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]))
    vi = int(round(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))
    if 0 <= ui < IMG_W and 0 <= vi < IMG_H:
        return (ui, vi)
    return None


def generate_synthetic_frame(frame_idx):
    np.random.seed(SEEDS[frame_idx % len(SEEDS)])
    robot_pos = ROBOT_POS.copy() if frame_idx < 3 else \
        ROBOT_POS + np.array([(frame_idx - 3) * 0.05 * np.cos(ROBOT_YAW),
                              (frame_idx - 3) * 0.05 * np.sin(ROBOT_YAW), 0.0], dtype=np.float32)
    robot_yaw = ROBOT_YAW

    rgb = np.full((IMG_H, IMG_W, 3), 180, dtype=np.uint8)
    cv2.rectangle(rgb, (0, IMG_H // 2), (IMG_W, IMG_H), (100, 100, 100), -1)
    depth = np.ones((IMG_H, IMG_W), dtype=np.float32) * 10.0

    COLORS_BGR = {"sugar_box": (0, 255, 255), "mustard_bottle": (255, 0, 255), "banana": (0, 255, 0)}
    ALL_CLASSES = list(CLASS_NAMES.values())

    gt_objects, placed_bboxes = [], []

    for i in range(NUM_GT_OBJECTS):
        cls_name = ALL_CLASSES[i % 3]
        for retry in range(30):
            u = (np.random.random() * 0.85 + 0.075) * IMG_W
            v = (np.random.random() * 0.75 + 0.15) * IMG_H
            z_cam = np.random.uniform(CAM_Z_MIN, CAM_Z_MAX)

            world_pos = pixel_to_world(u, v, z_cam, robot_pos, robot_yaw)  # FIXED: no size_3d
            pixel = world_to_pixel(world_pos, robot_pos, robot_yaw, HEAD_CAM_MATRIX)
            if pixel is None:
                continue

            bbox_size = max(10, int(60 / max(z_cam, 0.5)))
            half = bbox_size // 2
            x1, y1 = max(0, pixel[0] - half), max(0, pixel[1] - half)
            x2, y2 = min(IMG_W - 1, pixel[0] + half), min(IMG_H - 1, pixel[1] + half)
            if x2 - x1 < 10 or y2 - y1 < 10 or x1 < 0 or y1 < 0 or x2 >= IMG_W or y2 >= IMG_H:
                continue

            bbox = [x1, y1, x2, y2]
            if any(compute_iou(bbox, pb) > 0.2 for pb in placed_bboxes):
                continue

            in_bin = bool(np.linalg.norm(world_pos[:2] - BIN[:2]) < BIN_RADIUS * 0.7)
            placed_bboxes.append(bbox)

            gt_objects.append({
                "class": cls_name, "pos_world": world_pos.tolist(),
                "in_bin": in_bin, "visible": True,
                "pixel": pixel, "cam_z": float(z_cam), "bbox_pixel": bbox,
            })

            color = list(COLORS_BGR[cls_name])
            if np.random.random() < 0.5:
                color = [np.clip(c + np.random.randint(-30, 30), 0, 255) for c in color]
            ct = tuple(int(c) for c in color)

            if in_bin:
                ov = rgb.copy()
                cv2.rectangle(ov, (x1, y1), (x2, y2), (128, 128, 128), -1)
                cv2.addWeighted(ov, 0.3, rgb, 0.7, 0, rgb)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (150, 150, 150), 2)
            else:
                cv2.rectangle(rgb, (x1, y1), (x2, y2), ct, -1)
                if np.random.random() < 0.3:
                    hl = tuple(min(255, c + 40) for c in ct)
                    m = max(2, (x2 - x1) // 5)
                    cv2.rectangle(rgb, (x1 + m, y1 + m), (x2 - m, y2 - m), hl, -1)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), tuple(max(0, c - 80) for c in ct), 1)
            cv2.putText(rgb, cls_name[:6], (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.rectangle(depth, (x1, y1), (x2, y2), float(z_cam), -1)
            break

    # 垃圾桶标记
    bp = world_to_pixel(BIN, robot_pos, robot_yaw, HEAD_CAM_MATRIX)
    if bp:
        cv2.circle(rgb, bp, 8, (0, 0, 255), -1)
        cv2.putText(rgb, "BIN", (bp[0] + 10, bp[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.line(rgb, (0, IMG_H // 2 - 20), (IMG_W, IMG_H // 2 - 20), (60, 60, 60), 1)
    cv2.putText(rgb, f"Frame {frame_idx} | {len(gt_objects)} objects", (5, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    proprio = np.zeros(72, dtype=np.float32)
    proprio[9], proprio[10], proprio[11] = -np.sin(ROBOT_YAW), -np.cos(ROBOT_YAW), -1.0
    for j in range(12):
        proprio[12 + j] = [-0.3, 0.6, -1.2][j % 3]
    for j in range(8):
        proprio[24 + j] = [-0.5, 0.8, -1.5, 0.3, -0.2, 0.1, 0.04, 0.04][j]

    obs = {
        'proprio': torch.from_numpy(proprio).unsqueeze(0).to(torch.float32),
        'image': {
            'head_rgb': torch.from_numpy(rgb).unsqueeze(0).to(torch.uint8),
            'head_depth': torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1).to(torch.float32),
        },
    }
    return obs, gt_objects, rgb, depth, robot_pos


def compute_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    denom = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / denom if denom > 0 else 0.0


def run_pipeline_with_gt_bboxes(pipeline, obs, gt_objects):
    hd = obs['image']['head_depth'].squeeze(0).squeeze(-1)
    hd_np = (hd.cpu() if hd.device.type == 'cuda' else hd).numpy().astype(np.float32)
    proprio_np = obs['proprio'].squeeze(0).cpu().numpy().astype(np.float32)
    pipeline._update_robot_pose(proprio_np, dt=0.02)

    from config import OBJECT_SIZES as _OS, DEFAULT_OBJECT_SIZE as _DOS, \
        GRASP_DEPTH_OFFSET, BIN_CENTER as _BC, BIN_RADIUS as _BR

    objects_list, target, best_dist, tid = [], None, float('inf'), 0
    for gt in gt_objects:
        if not gt.get('visible', False): continue
        tid += 1
        p_cam = pipeline._depth_to_cam(hd_np, gt['bbox_pixel'])
        if p_cam is None: continue
        p_world = pipeline._robot_to_world(pipeline._cam_to_robot(p_cam))
        dist = float(np.linalg.norm(p_world[:2] - pipeline.robot_pos[:2]))
        in_bin = pipeline._check_in_bin(p_world)
        size = _OS.get(gt['class'], _DOS)
        o = {'id': tid, 'class': gt['class'], 'conf': 1.0,
             'pos_world': p_world.tolist(),
             'pos_robot': pipeline._world_to_robot(p_world).tolist(),
             'grasp_pos_world': (p_world - np.array([0, 0, GRASP_DEPTH_OFFSET], dtype=np.float32)).tolist(),
             'grasp_quat_world': pipeline._compute_grasp_quat(gt['class'], p_world).tolist(),
             'dist_to_robot': dist, 'yaw_rel': 0.0, 'in_bin': in_bin,
             'size_world': [size['lx'], size['ly'], size['lz']], 'bbox': gt['bbox_pixel']}
        objects_list.append(o)
        if not in_bin and dist < best_dist:
            best_dist, target = dist, o

    objs = sorted(objects_list, key=lambda o: o['dist_to_robot'])
    return {
        'target': target, 'objects_detailed': objs,
        'objects_remaining': [{"id": o['id'], "class": o['class'], "dist": o['dist_to_robot'],
                                "in_bin": o['in_bin']} for o in objs],
        'bin': {'center_world': _BC.tolist(), 'radius': _BR, 'drop_height': 0.50,
                'dist_to_robot': float(np.linalg.norm(_BC[:2] - pipeline.robot_pos[:2])),
                'yaw_rel': float(np.arctan2(_BC[1] - pipeline.robot_pos[1],
                                            _BC[0] - pipeline.robot_pos[0]) - pipeline.robot_yaw)},
        'progress': {'total': len(gt_objects), 'inside_bin': 0, 'remaining': len(gt_objects)},
        'robot': {'pos_world': pipeline.robot_pos.tolist(), 'yaw': float(pipeline.robot_yaw)},
    }


def evaluate(pout, gt_objects, frame_idx, robot_pos):
    print(f"\n{'='*70}\n  FRAME {frame_idx} — ACCURACY EVALUATION\n{'='*70}")
    t = pout.get('target')
    if t:
        print(f"\n  🎯 Target: {t['class']} dist={t['dist_to_robot']:.2f}m in_bin={t['in_bin']}")

    ps = {o['class'] for o in pout.get('objects_detailed', [])}
    gs = {o['class'] for o in gt_objects if o['visible']}
    print(f"\n  GT visible: {gs}  ({len(gs)} objects)")
    print(f"  Pipe detected: {ps}  ({len(ps)} objects)")
    missed, extra = gs - ps, ps - gs
    if missed: print(f"  ⚠️  MISSED: {missed}")
    if extra: print(f"  ⚠️  EXTRA (FP): {extra}")
    if not missed and not extra: print(f"  ✅ All visible objects detected")

    print(f"\n  {'─'*60}\n  📐 3D POSITION ACCURACY\n  {'─'*60}")
    print(f"  {'Class':<16s} {'GT (x,y,z)':>22s}  {'Pipe (x,y,z)':>22s}  {'Δxy':>7s} {'Δz':>7s} {'Δ3D':>7s}\n  {'─'*60}")

    pbc, t3, txy, tz, nm = {}, 0.0, 0.0, 0.0, 0
    for o in pout.get('objects_detailed', []):
        pbc.setdefault(o['class'], []).append(o)
    for gt in gt_objects:
        if not gt['visible']: continue
        cls, gp = gt['class'], np.array(gt['pos_world'])
        if cls not in pbc:
            print(f"  {cls:<16s} (undetected)")
            continue
        best = min(pbc[cls], key=lambda x: np.linalg.norm(np.array(x['pos_world']) - gp))
        pp = np.array(best['pos_world'])
        e3, exy, ez = float(np.linalg.norm(pp - gp)), float(np.linalg.norm(pp[:2] - gp[:2])), float(abs(pp[2] - gp[2]))
        f = "✅" if e3 < 0.15 else ("⚠️" if e3 < 0.30 else "❌")
        print(f"  {cls:<16s} ({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f})  ({pp[0]:.3f},{pp[1]:.3f},{pp[2]:.3f})  {exy:>6.3f} {ez:>6.3f} {e3:>6.3f} {f}")
        t3 += e3; txy += exy; tz += ez; nm += 1

    a3 = float('inf')
    if nm > 0:
        a3 = t3 / nm
        print(f"  {'─'*60}\n  AVERAGE: Δxy={txy/nm:.3f}m Δz={tz/nm:.3f}m Δ3D={a3:.3f}m")
        g = "🏆 EXCELLENT (<5cm)" if a3 < 0.05 else ("✅ GOOD (<10cm)" if a3 < 0.10 else "⚠️ ACCEPTABLE")
        print(f"  Grade: {g}")

    bi = pout.get('bin', {})
    print(f"\n  🗑️  Bin dist: pipe={bi.get('dist_to_robot', 0):.2f}m GT={np.linalg.norm(BIN[:2]-robot_pos[:2]):.2f}m")
    ri = pout.get('robot', {})
    ye = abs(ri.get('yaw', 0) - ROBOT_YAW)
    print(f"  🤖 Robot yaw: pipe={np.rad2deg(ri.get('yaw', 0)):.1f}° GT={np.rad2deg(ROBOT_YAW):.1f}° Δ={np.rad2deg(min(ye, 2*np.pi-ye)):.1f}°")
    return nm, a3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", action="store_true")
    p.add_argument("--frames", type=int, default=5)
    a = p.parse_args()
    print("=" * 70 + f"\n  ATEC Task B — {'GT' if a.gt else 'YOLO'} MODE — 30° camera pitch\n" + "=" * 70)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {dev}  PyTorch: {torch.__version__}")
    pl = PerceptionPipeline(device=dev)
    print(f"Pipeline OK.\n")
    anns, errs = [], []
    for fi in range(a.frames):
        print(f"\n{'▬'*70}\n  Frame {fi+1}/{a.frames}\n{'▬'*70}")
        obs, gto, rgb, _, rp = generate_synthetic_frame(fi)
        if a.gt:
            pl.in_bin_ids, pl.frame_count = set(), fi
            po = run_pipeline_with_gt_bboxes(pl, obs, gto)
        else:
            po = pl.process(obs, dt=0.02)
        n, e = evaluate(po, gto, fi, rp)
        if n > 0: errs.append(e)
        from visualize import draw_results
        an = draw_results(rgb, po)
        cv2.putText(an, f"GT: {len(gto)} obj", (5, an.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        anns.append(an)
        cv2.imwrite(f"test_frame_{fi:03d}.png", cv2.cvtColor(an, cv2.COLOR_RGB2BGR))
        print(f"  Saved: test_frame_{fi:03d}.png")
    if anns:
        th, tw = 160, 213
        cv2.imwrite(f"test_all_frames_{'gt' if a.gt else 'yolo'}.png",
                    cv2.cvtColor(np.hstack([cv2.resize(f, (tw, th)) for f in anns]), cv2.COLOR_RGB2BGR))
    print("\n" + "=" * 70 + "\n  TEST COMPLETE")
    if errs:
        oa = np.mean(errs)
        print(f"  🎯 Overall avg 3D error: {oa:.4f}m ({oa*100:.1f}cm)")
        print(f"  {'🏆 EXCELLENT' if oa < 0.05 else '✅ GOOD' if oa < 0.10 else '⚠️ NEEDS TUNING'}")
    print("=" * 70)


if __name__ == "__main__":
    main()