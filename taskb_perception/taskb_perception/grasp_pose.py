"""抓取姿态估计（移植自 scripts/act/task_e/state_machine.py 思路）。"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_TOPDOWN_QUAT_W, GRASP_Z_OFFSET, PRE_GRASP_CLEARANCE
from .math3d import matrix_to_quat, quat_to_matrix
from .types import ObjectClass


def _build_grasp_matrix(long_axis: np.ndarray, grip_z: np.ndarray) -> np.ndarray:
    jaw_dir = np.cross(long_axis, grip_z)
    n = np.linalg.norm(jaw_dir)
    if n < 1e-6:
        jaw_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        jaw_dir = jaw_dir / n
    align_dir = np.cross(jaw_dir, grip_z)
    align_dir = align_dir / max(np.linalg.norm(align_dir), 1e-6)
    return np.stack([align_dir, jaw_dir, grip_z], axis=1)


def compute_grasp_quat_for_object(
    obj_class: ObjectClass,
    obj_quat_b: np.ndarray | None = None,
) -> np.ndarray:
    """根据物体类别 / 朝向计算 base 系抓取四元数 (wxyz)。"""
    default = DEFAULT_TOPDOWN_QUAT_W.copy()
    if obj_class == ObjectClass.SUGAR:
        return default
    if obj_quat_b is None:
        return default

    R_obj = quat_to_matrix(obj_quat_b)
    grip_z = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    norms, axes_xy = [], []
    for col in range(3):
        ax = np.array([R_obj[0, col], R_obj[1, col], 0.0], dtype=np.float32)
        norms.append(float(np.linalg.norm(ax)))
        axes_xy.append(ax)

    best_norm = max(norms)
    candidates = [axes_xy[c] / max(norms[c], 1e-6) for c in range(3) if norms[c] >= best_norm - 1e-3]
    if not candidates:
        return default

    best_cos = -2.0
    long_axis = candidates[0]
    for cand in candidates:
        R_grip = _build_grasp_matrix(cand, grip_z)
        q_cand = matrix_to_quat(R_grip)
        cos_sim = abs(float(np.dot(q_cand, default)))
        if cos_sim > best_cos:
            best_cos = cos_sim
            long_axis = cand

    return matrix_to_quat(_build_grasp_matrix(long_axis, grip_z))


def compute_grasp_waypoints(pos_b: np.ndarray, grasp_quat_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (pre_grasp_pos_b, grasp_pos_b)。"""
    pre = pos_b.copy()
    pre[2] = max(pre[2], 0.05) + PRE_GRASP_CLEARANCE
    grasp = pos_b.copy()
    grasp[2] = max(grasp[2], 0.02) + GRASP_Z_OFFSET
    return pre.astype(np.float32), grasp.astype(np.float32)
