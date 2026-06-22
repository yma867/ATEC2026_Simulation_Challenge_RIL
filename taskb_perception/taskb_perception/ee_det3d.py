"""EE 远距离导航：YOLO detect + ee_depth → 每个框的 base 系 3D。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PerceptionConfig
from .detector import Detection2D, ObjectDetector
from .ee_collect_utils import EE_CAM_H, EE_CAM_W, EE_CX, EE_CY, EE_FX, EE_FY
from .math3d import transform_point_cam_to_base
from .obs_utils import pixel_to_cam, sample_depth_median

_CLASS_BGR = {
    "sugar": (220, 220, 220),
    "mustard": (0, 200, 255),
    "banana": (0, 220, 255),
    "unknown": (128, 255, 0),
}


def draw_ee_detections(
    rgb: np.ndarray,
    detections: list[EEDetection3D],
    *,
    copy: bool = True,
    show_pos_b: bool = False,
) -> np.ndarray:
    """在 RGB 上画 detect 框、类别、置信度、ee_depth 真实深度（米）。"""
    import cv2

    vis = rgb.copy() if copy else rgb
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = _CLASS_BGR.get(det.obj_class, (0, 255, 128))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        u, v = int(det.u), int(det.v)
        cv2.drawMarker(vis, (u, v), color, cv2.MARKER_CROSS, 12, 2)
        label = f"{det.obj_class} {det.confidence:.2f} D={det.depth_m:.2f}m"
        if show_pos_b:
            label += f" b=({det.pos_b[0]:.1f},{det.pos_b[1]:.1f})"
        ty = max(y1 - 8, 16)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        f"det={len(detections)}",
        (8, vis.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


@dataclass
class EEDetection3D:
    """EE detect 单目标：2D 框 + 深度 + base 系 3D。"""

    u: float
    v: float
    depth_m: float
    bbox: tuple[int, int, int, int]
    obj_class: str
    confidence: float
    p_cam: np.ndarray
    pos_b: np.ndarray

    def distance_xy(self) -> float:
        return float(np.linalg.norm(self.pos_b[:2]))


def ee_pixel_depth_to_base(
    u: float,
    v: float,
    depth_m: float,
    cfg: PerceptionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """EE 像素 (u,v) + 深度(m) → (p_cam, pos_b)。

    base 系：x 前、y 左、z 上，单位米。
    """
    cfg = cfg or PerceptionConfig()
    p_cam = pixel_to_cam(u, v, depth_m, EE_FX, EE_FY, EE_CX, EE_CY)
    if cfg.robot.ee_cam is None:
        raise RuntimeError("robot.ee_cam 未配置")
    ext = cfg.robot.ee_cam
    pos_b = transform_point_cam_to_base(p_cam, ext.pos_b, ext.quat_b)
    return p_cam.astype(np.float32), pos_b.astype(np.float32)


def ee_bbox_to_base(
    bbox: tuple[int, int, int, int],
    depth: np.ndarray,
    cfg: PerceptionConfig | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray] | None:
    """检测框 (x1,y1,x2,y2) + ee_depth 图 → 框中心与 base 3D。

    Returns:
        (u, v, depth_m, p_cam, pos_b) 或 None（无有效深度）
    """
    x1, y1, x2, y2 = bbox
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    d = EEDetector.depth_in_bbox(depth, x1, y1, x2, y2, u, v)
    if d <= 0:
        return None
    p_cam, pos_b = ee_pixel_depth_to_base(u, v, d, cfg)
    return float(u), float(v), d, p_cam, pos_b


class EEDetector:
    """EE 相机 detect 权重 + EE 内参 + ee_depth → base 坐标。"""

    def __init__(self, cfg: PerceptionConfig | None = None):
        self.cfg = cfg or PerceptionConfig()
        ee_cfg = PerceptionConfig(
            conf_threshold=self.cfg.conf_threshold,
            use_color_fallback=False,
            class_names=self.cfg.class_names,
        )
        ee_weights = self.cfg.yolo_ee_det_weights
        self._detector = ObjectDetector(ee_cfg, weights=ee_weights)
        self.fx, self.fy, self.cx, self.cy = EE_FX, EE_FY, EE_CX, EE_CY

    @property
    def ready(self) -> bool:
        return self._detector._yolo is not None

    @staticmethod
    def depth_in_bbox(
        depth: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        fallback_u: float,
        fallback_v: float,
    ) -> float:
        """框内有效深度中值；失败则回退框中心。"""
        h, w = depth.shape
        x1c = max(0, min(w - 1, x1))
        x2c = max(0, min(w, x2))
        y1c = max(0, min(h - 1, y1))
        y2c = max(0, min(h, y2))
        patch = depth[y1c:y2c, x1c:x2c]
        valid = patch[(patch > 0.05) & (patch < 8.0) & np.isfinite(patch)]
        if valid.size >= 5:
            return float(np.median(valid))
        return sample_depth_median(depth, int(fallback_u), int(fallback_v), radius=4)

    def detection_to_base(self, det: Detection2D) -> tuple[np.ndarray, np.ndarray]:
        return ee_pixel_depth_to_base(det.u, det.v, det.depth, self.cfg)

    def detect_all(
        self,
        ee_rgb: np.ndarray,
        ee_depth: np.ndarray | None,
    ) -> list[EEDetection3D]:
        """视角内所有 detect 目标 → 带 depth 与 pos_b。"""
        if ee_rgb is None or ee_depth is None:
            return []

        raw = self._detector.detect(ee_rgb, ee_depth, source="ee")
        out: list[EEDetection3D] = []

        for det in raw:
            x1, y1, x2, y2 = det.bbox
            d = self.depth_in_bbox(ee_depth, x1, y1, x2, y2, det.u, det.v)
            if d <= 0:
                continue
            det_refined = Detection2D(
                u=det.u,
                v=det.v,
                depth=d,
                obj_class=det.obj_class,
                confidence=det.confidence,
                bbox=det.bbox,
            )
            p_cam, pos_b = self.detection_to_base(det_refined)
            out.append(
                EEDetection3D(
                    u=det.u,
                    v=det.v,
                    depth_m=d,
                    bbox=det.bbox,
                    obj_class=det.obj_class.value,
                    confidence=det.confidence,
                    p_cam=p_cam,
                    pos_b=pos_b,
                )
            )
        out.sort(key=lambda x: x.distance_xy())
        return out

    def detect_2d(
        self,
        ee_rgb: np.ndarray,
        ee_depth: np.ndarray | None = None,
    ) -> list[Detection2D]:
        """仅 2D 框中心，用于光轴中线导航；depth 可选（仅停车距离）。"""
        if ee_rgb is None:
            return []
        raw = self._detector.detect(ee_rgb, ee_depth, source="ee", require_depth=False)
        out: list[Detection2D] = []
        for det in raw:
            d = 0.0
            if ee_depth is not None:
                x1, y1, x2, y2 = det.bbox
                d = self.depth_in_bbox(ee_depth, x1, y1, x2, y2, det.u, det.v)
            out.append(
                Detection2D(
                    u=det.u,
                    v=det.v,
                    depth=float(d) if d > 0 else 0.0,
                    obj_class=det.obj_class,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )
        out.sort(key=lambda d: -d.confidence)
        return out

    def detect_nearest(
        self,
        ee_rgb: np.ndarray,
        ee_depth: np.ndarray | None,
    ) -> EEDetection3D | None:
        dets = self.detect_all(ee_rgb, ee_depth)
        return dets[0] if dets else None
