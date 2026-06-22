"""从 obs 解析图像 / proprio。"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return np.squeeze(x)


def parse_rgb(obs_image: dict, key: str = "head_rgb") -> np.ndarray | None:
    """返回 uint8 HWC RGB，shape (H, W, 3)。"""
    if key not in obs_image or obs_image[key] is None:
        return None
    img = _to_numpy(obs_image[key])
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def parse_depth(obs_image: dict, key: str = "head_depth") -> np.ndarray | None:
    """返回 float32 深度图 (H, W)，单位 m。"""
    if key not in obs_image or obs_image[key] is None:
        return None
    depth = _to_numpy(obs_image[key]).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def parse_proprio(proprio: Any) -> dict[str, np.ndarray]:
    """按 readme 顺序解析 proprio 向量。"""
    p = _to_numpy(proprio).astype(np.float32)
    if p.ndim == 2:
        p = p[0]
    action_dim = (len(p) - 12) // 3
    idx = 0
    out = {}
    out["base_lin_vel"] = p[idx : idx + 3]
    idx += 3
    out["base_ang_vel"] = p[idx : idx + 3]
    idx += 3
    out["velocity_commands"] = p[idx : idx + 3]
    idx += 3
    out["projected_gravity"] = p[idx : idx + 3]
    idx += 3
    out["joint_pos"] = p[idx : idx + action_dim]
    idx += action_dim
    out["joint_vel"] = p[idx : idx + action_dim]
    idx += action_dim
    out["actions"] = p[idx : idx + action_dim]
    out["action_dim"] = np.array([action_dim], dtype=np.int32)
    return out


def pixel_to_cam(u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z], dtype=np.float32)


def sample_depth_median(depth: np.ndarray, u: int, v: int, radius: int = 3) -> float:
    h, w = depth.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[(patch > 0.01) & np.isfinite(patch)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))
