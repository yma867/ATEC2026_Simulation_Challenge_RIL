#!/usr/bin/env python3
"""
Task B YOLO 自动采集 + 巡逻移动（走向每个物体，比站原地抖腿快很多）。

逻辑:
  reset → 18 个物体随机位置 → 按距离排序依次走过去 → 路上自动存图+自动标注

用法（isaaclab + 已下载 B2 baseline policy）:

  conda activate isaaclab
  pip install "numpy<2.0.0" scipy ultralytics opencv-python

  cd ATEC2026_Simulation_Challenge
  python taskb_perception/scripts/collect_yolo_dataset.py \\
    --task ATEC-TaskB-B2Piper --headless --enable_cameras \\
    --num_episodes 30 --interval 2
    # policy 默认 demo/policy.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto-label YOLO dataset with patrol locomotion.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--num_episodes", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=600, help="单 episode 最多步数，走完所有物体或达到上限")
parser.add_argument("--interval", type=int, default=2, help="每隔 N step 存一帧")
parser.add_argument("--out", type=str, default="taskb_perception/dataset")
parser.add_argument("--val_ratio", type=float, default=0.15)
parser.add_argument(
    "--policy",
    type=str,
    default="demo/policy.pt",
    help="B2 locomotion policy，默认 demo/policy.pt",
)
parser.add_argument("--arrive_dist", type=float, default=1.3, help="距物体多少米算到达")
parser.add_argument(
    "--no_patrol",
    action="store_true",
    help="关闭巡逻（回退到旧版随机抖腿，不推荐）",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from taskb_perception.collector_nav import (  # noqa: E402
    B2LocomotionDriver,
    ObjectPatrolController,
    resolve_policy_path,
)
from taskb_perception.obs_utils import parse_rgb  # noqa: E402
from taskb_perception.yolo_labels import CLASS_NAMES, NUM_OBJECTS, OBJ_HALF_EXTENTS_3D, object_index_to_class  # noqa: E402

CAM_W, CAM_H = 640, 480
FX = 24.0 / 20.955 * CAM_W
FY = FX
CX, CY = CAM_W / 2.0, CAM_H / 2.0


def _aabb_corners_local(half: tuple[float, float, float]) -> np.ndarray:
    hx, hy, hz = half
    return np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [-hx, hy, hz], [hx, hy, hz],
        ],
        dtype=np.float32,
    )


def project_world_to_pixel(points_w, cam_pos_w, cam_quat_w):
    q_inv = quat_inv(torch.tensor(cam_quat_w, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    pos = torch.tensor(cam_pos_w, dtype=torch.float32)
    pts = torch.tensor(points_w, dtype=torch.float32)
    rel = pts - pos
    p_cam = quat_apply(q_inv.unsqueeze(0).expand(len(pts), -1), rel)
    x, y, z = p_cam[:, 0].numpy(), p_cam[:, 1].numpy(), p_cam[:, 2].numpy()
    valid = z > 0.05
    u = FX * x / np.maximum(z, 1e-6) + CX
    v = FY * y / np.maximum(z, 1e-6) + CY
    return np.stack([u, v], axis=1), valid


def object_to_yolo_line(obj_pos_w, obj_quat_w, cls_id, cam_pos_w, cam_quat_w) -> str | None:
    half = OBJ_HALF_EXTENTS_3D[cls_id]
    corners_local = _aabb_corners_local(half)
    q = torch.tensor(obj_quat_w, dtype=torch.float32).unsqueeze(0)
    corners_w = quat_apply(q.expand(len(corners_local), -1), torch.tensor(corners_local)) + torch.tensor(obj_pos_w)
    uv, valid = project_world_to_pixel(corners_w.numpy(), cam_pos_w, cam_quat_w)
    if valid.sum() < 4:
        return None
    uv = uv[valid]
    x1, y1 = uv[:, 0].min(), uv[:, 1].min()
    x2, y2 = uv[:, 0].max(), uv[:, 1].max()
    x1, x2 = np.clip(x1, 0, CAM_W - 1), np.clip(x2, 0, CAM_W - 1)
    y1, y2 = np.clip(y1, 0, CAM_H - 1), np.clip(y2, 0, CAM_H - 1)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    return f"{cls_id} {(x1+x2)/2/CAM_W:.6f} {(y1+y2)/2/CAM_H:.6f} {bw/CAM_W:.6f} {bh/CAM_H:.6f}"


def random_leg_action(action_dim: int, device) -> torch.Tensor:
    action = torch.zeros((1, action_dim), device=device)
    if action_dim >= 12:
        action[0, :12] = torch.randn(12, device=device) * 0.15
    return action


def save_frame(out_root, split, stem, rgb, label_lines):
    img_path = out_root / "images" / split / f"{stem}.jpg"
    lbl_path = out_root / "labels" / split / f"{stem}.txt"
    cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def main():
    out_root = Path(args_cli.out)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    use_patrol = not args_cli.no_patrol
    driver = None
    if use_patrol:
        policy_path = resolve_policy_path(args_cli.policy, REPO_ROOT)
        if policy_path is None:
            print(
                "[ERROR] 找不到 B2 locomotion policy，无法巡逻。\n"
                f"  请确认: {REPO_ROOT / 'demo' / 'policy.pt'}\n"
                "  临时可 --no_patrol 使用旧版抖腿（慢且效果差）"
            )
            return
        print(f"[INFO] 使用 locomotion policy: {policy_path}")
        driver = B2LocomotionDriver(device=args_cli.device, policy_path=policy_path)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    cam = unwrapped.scene.sensors["head_camera"]
    patrol = ObjectPatrolController(num_objects=NUM_OBJECTS, arrive_dist=args_cli.arrive_dist)

    global_idx = 0
    for ep in range(args_cli.num_episodes):
        obs, _ = env.reset()
        patrol.reset(unwrapped)
        action_dim = (int(obs["proprio"].shape[-1]) - 12) // 3

        for step in range(args_cli.max_steps):
            if not simulation_app.is_running():
                break

            if use_patrol and driver is not None:
                vel_cmd = patrol.get_velocity_command(unwrapped)
                action = driver.compute_action(obs, vel_cmd)
            else:
                action = random_leg_action(action_dim, unwrapped.device)

            obs, _, terminated, truncated, _ = env.step(action)

            if step % args_cli.interval == 0:
                rgb = parse_rgb(obs.get("image", {}), "head_rgb")
                if rgb is not None:
                    cam_pos_w = cam.data.pos_w[0].detach().cpu().numpy()
                    cam_quat_w = cam.data.quat_w_world[0].detach().cpu().numpy()
                    label_lines = []
                    for obj_idx in range(1, NUM_OBJECTS + 1):
                        obj = unwrapped.scene[f"object_{obj_idx}"]
                        pos_w = obj.data.root_pos_w[0, :3].detach().cpu().numpy()
                        quat_w = obj.data.root_quat_w[0].detach().cpu().numpy()
                        line = object_to_yolo_line(pos_w, quat_w, object_index_to_class(obj_idx), cam_pos_w, cam_quat_w)
                        if line:
                            label_lines.append(line)
                    if label_lines:
                        split = "val" if random.random() < args_cli.val_ratio else "train"
                        save_frame(out_root, split, f"{global_idx:06d}", rgb, label_lines)
                        global_idx += 1

            if global_idx > 0 and global_idx % 50 == 0:
                print(f"ep {ep+1}/{args_cli.num_episodes}  visits={patrol.visits}/18  saved={global_idx}")

            if patrol.done:
                print(f"ep {ep+1}: 已访问全部 {patrol.visits} 个物体目标，saved={global_idx}")
                break
            if terminated.item() or truncated.item():
                break

    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "path": str(out_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {i: n for i, n in enumerate(CLASS_NAMES)},
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    env.close()
    simulation_app.close()
    print(f"完成: {global_idx} 帧 -> {out_root.resolve()}")
    print("下一步: python taskb_perception/scripts/train_yolo.py")


if __name__ == "__main__":
    main()
