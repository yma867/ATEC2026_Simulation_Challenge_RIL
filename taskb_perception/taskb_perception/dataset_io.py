"""手动采集时复用：3D 真值 → YOLO 标签 + 存图。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from isaaclab.utils.math import quat_apply, quat_inv, subtract_frame_transforms

from .yolo_labels import CLASS_NAMES, NUM_OBJECTS, OBJ_HALF_EXTENTS_3D, object_index_to_class

CAM_W, CAM_H = 640, 480
FX = 24.0 / 20.955 * CAM_W
FY = FX
CX, CY = CAM_W / 2.0, CAM_H / 2.0

# 中心点投影失败时的默认框大小（归一化 w,h，按类别）
DEFAULT_NORM_BOX: dict[int, tuple[float, float]] = {
    0: (0.085, 0.095),  # sugar
    1: (0.055, 0.130),  # mustard
    2: (0.140, 0.075),  # banana
}


def _read_cam_pose(cam) -> tuple[np.ndarray, np.ndarray]:
    pos = cam.data.pos_w[0].detach().cpu().numpy()
    for attr in ("quat_w_world", "quat_w_ros", "quat_w"):
        if hasattr(cam.data, attr):
            quat = getattr(cam.data, attr)[0].detach().cpu().numpy()
            return pos, quat
    raise AttributeError("camera sensor has no quat_w_world / quat_w_ros")


def _read_intrinsics(cam) -> tuple[float, float, float, float]:
    if hasattr(cam.data, "intrinsic_matrices"):
        k = cam.data.intrinsic_matrices[0].detach().cpu().numpy()
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    return FX, FY, CX, CY


def _aabb_corners_local(half: tuple[float, float, float]) -> np.ndarray:
    hx, hy, hz = half
    return np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [-hx, hy, hz], [hx, hy, hz],
        ],
        dtype=np.float32,
    )


def _world_to_cam(points_w: np.ndarray, cam_pos_w: np.ndarray, cam_quat_w: np.ndarray) -> np.ndarray:
    """世界系 Nx3 -> 相机系 Nx3（Isaac subtract_frame_transforms）。"""
    device = torch.device("cpu")
    pos = torch.tensor(cam_pos_w, dtype=torch.float32, device=device)
    quat = torch.tensor(cam_quat_w, dtype=torch.float32, device=device).unsqueeze(0)
    out = []
    for p in points_w:
        pw = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
        pc, _ = subtract_frame_transforms(pos.unsqueeze(0), quat, pw, None)
        out.append(pc[0].numpy())
    return np.stack(out, axis=0)


def _cam_to_pixel(p_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """尝试多种相机光学系，返回 (uv, valid)。"""
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

    for depth, px, py in (
        (z, x, y),           # z 朝前 (ROS 光学常见)
        (-z, x, -y),         # -z 朝前 (OpenGL 常见)
        (x, y, z),           # x 朝前 (备选)
    ):
        d = np.where(np.abs(depth) > 1e-4, depth, 1e-4)
        valid = d > 0.05
        u = fx * px / d + cx
        v = fy * py / d + cy
        in_img = valid & (u >= 0) & (u < CAM_W) & (v >= 0) & (v < CAM_H)
        candidates.append((np.stack([u, v], axis=1), in_img))

    # 选「在画面内点数最多」的投影
    best_uv, best_valid = candidates[0]
    best_count = int(best_valid.sum())
    for uv, valid in candidates[1:]:
        c = int(valid.sum())
        if c > best_count:
            best_uv, best_valid = uv, valid
            best_count = c
    return best_uv, best_valid


def _project_world_to_pixel(
    points_w: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    p_cam = _world_to_cam(points_w, cam_pos_w, cam_quat_w)
    return _cam_to_pixel(p_cam, fx, fy, cx, cy)


def _yolo_line_from_uv(
    uv: np.ndarray,
    valid: np.ndarray,
    cls_id: int,
    min_side: float = 6.0,
) -> str | None:
    if valid.sum() < 1:
        return None
    pts = uv[valid]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    x1, x2 = np.clip(x1, 0, CAM_W - 1), np.clip(x2, 0, CAM_W - 1)
    y1, y2 = np.clip(y1, 0, CAM_H - 1), np.clip(y2, 0, CAM_H - 1)
    bw, bh = x2 - x1, y2 - y1
    if bw < min_side and bh < min_side:
        # 单点 + 默认框
        if pts.shape[0] >= 1:
            cx_px, cy_px = float(pts[0, 0]), float(pts[0, 1])
            nw, nh = DEFAULT_NORM_BOX.get(cls_id, (0.08, 0.08))
            cx_n, cy_n = cx_px / CAM_W, cy_px / CAM_H
            if 0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0:
                return f"{cls_id} {cx_n:.6f} {cy_n:.6f} {nw:.6f} {nh:.6f}"
        return None
    if bw < min_side or bh < min_side:
        cx_px, cy_px = (x1 + x2) / 2, (y1 + y2) / 2
        nw, nh = max(bw / CAM_W, DEFAULT_NORM_BOX.get(cls_id, (0.08, 0.08))[0]), max(
            bh / CAM_H, DEFAULT_NORM_BOX.get(cls_id, (0.08, 0.08))[1]
        )
        return f"{cls_id} {cx_px/CAM_W:.6f} {cy_px/CAM_H:.6f} {nw:.6f} {nh:.6f}"
    return f"{cls_id} {(x1+x2)/2/CAM_W:.6f} {(y1+y2)/2/CAM_H:.6f} {bw/CAM_W:.6f} {bh/CAM_H:.6f}"


def object_to_yolo_line(
    obj_pos_w: np.ndarray,
    obj_quat_w: np.ndarray,
    cls_id: int,
    cam_pos_w: np.ndarray,
    cam_quat_w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> str | None:
    half = OBJ_HALF_EXTENTS_3D[cls_id]
    corners_local = _aabb_corners_local(half)
    q = torch.tensor(obj_quat_w, dtype=torch.float32).unsqueeze(0)
    corners_w = quat_apply(q.expand(len(corners_local), -1), torch.tensor(corners_local)) + torch.tensor(obj_pos_w)
    uv, valid = _project_world_to_pixel(corners_w.numpy(), cam_pos_w, cam_quat_w, fx, fy, cx, cy)
    line = _yolo_line_from_uv(uv, valid, cls_id)
    if line is not None:
        return line
    # 中心点 fallback
    uv_c, valid_c = _project_world_to_pixel(
        obj_pos_w.reshape(1, 3), cam_pos_w, cam_quat_w, fx, fy, cx, cy
    )
    return _yolo_line_from_uv(uv_c, valid_c, cls_id, min_side=0.0)


def build_labels_from_env(env, cam, debug: bool = False) -> list[str]:
    cam_pos_w, cam_quat_w = _read_cam_pose(cam)
    fx, fy, cx, cy = _read_intrinsics(cam)
    lines: list[str] = []
    n_try = 0
    for obj_idx in range(1, NUM_OBJECTS + 1):
        obj = env.scene[f"object_{obj_idx}"]
        pos_w = obj.data.root_pos_w[0, :3].detach().cpu().numpy()
        quat_w = obj.data.root_quat_w[0].detach().cpu().numpy()
        cls_id = object_index_to_class(obj_idx)
        line = object_to_yolo_line(pos_w, quat_w, cls_id, cam_pos_w, cam_quat_w, fx, fy, cx, cy)
        n_try += 1
        if line:
            lines.append(line)
    if debug:
        print(
            f"[label debug] fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
            f"objects_in_view={len(lines)}/{n_try} cam_z={cam_pos_w[2]:.2f}"
        )
    return lines


def save_snapshot(
    out_root: Path,
    stem: str,
    rgb: np.ndarray,
    label_lines: list[str],
    split: str = "train",
    allow_empty: bool = False,
) -> tuple[Path, Path] | None:
    if not label_lines and not allow_empty:
        return None
    out_root = ensure_dataset_dirs(out_root)
    img_path = out_root / "images" / split / f"{stem}.jpg"
    lbl_path = out_root / "labels" / split / f"{stem}.txt"
    cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    lbl_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return img_path, lbl_path


def resolve_output_dir(out: str | Path, repo_root: Path, package_root: Path) -> Path:
    p = Path(out)
    if p.is_absolute():
        return p.resolve()
    if p.parts and p.parts[0] == "taskb_perception":
        return (repo_root / p).resolve()
    return (package_root / p).resolve()


def ensure_dataset_dirs(out_root: Path) -> Path:
    out_root = out_root.resolve()
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"无法创建输出目录: {out_root}\n{exc}") from exc
    return out_root


def write_dataset_yaml(out_root: Path) -> None:
    out_root = ensure_dataset_dirs(out_root)
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
