"""EE 垂直俯视：YOLO-seg mask → 几何中心 (u,v)，用于图像中心对准（无需 depth）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CAM_HEIGHT, CAM_WIDTH, PerceptionConfig
from .types import ObjectClass

# EE 相机分辨率与 head 相同（640×480）
EE_CAM_W, EE_CAM_H = CAM_WIDTH, CAM_HEIGHT
EE_ALIGN_U0 = EE_CAM_W / 2.0
EE_ALIGN_V0 = EE_CAM_H / 2.0


@dataclass
class EESegCenter:
    """单个 EE 分割实例的几何中心（像素系，左上角为原点）。"""

    u: float
    v: float
    obj_class: ObjectClass
    confidence: float
    bbox: tuple[int, int, int, int]
    area_px: float
    target_u: float
    target_v: float
    err_u: float
    err_v: float
    aligned: bool
    polygon: np.ndarray | None = None

    @property
    def pixel_error(self) -> tuple[float, float]:
        return self.err_u, self.err_v

    @property
    def pixel_error_norm(self) -> float:
        return float(np.hypot(self.err_u, self.err_v))


def mask_centroid_from_polygon(
    polygon_xy: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[float, float, float]:
    """多边形 → 面积加权几何中心 (u,v) 与 mask 像素面积。

    polygon_xy: (N,2) 像素坐标 [u,v] = [x,y]
    """
    poly = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        u = float(poly[:, 0].mean()) if poly.size else 0.0
        v = float(poly[:, 1].mean()) if poly.size else 0.0
        return u, v, 0.0

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    M = cv2.moments(mask, binaryImage=True)
    area = float(M["m00"])
    if area < 1e-3:
        return float(poly[:, 0].mean()), float(poly[:, 1].mean()), 0.0
    u = float(M["m10"] / M["m00"])
    v = float(M["m01"] / M["m00"])
    return u, v, area


def mask_centroid_from_binary(
    mask: np.ndarray,
) -> tuple[float, float, float]:
    """二值 mask (H,W) → (u,v,area)。"""
    mask_u8 = (mask > 0).astype(np.uint8)
    M = cv2.moments(mask_u8, binaryImage=True)
    area = float(M["m00"])
    if area < 1e-3:
        return 0.0, 0.0, 0.0
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]), area


def mask_centroid_from_yolo(
    result,
    index: int,
    img_h: int = EE_CAM_H,
    img_w: int = EE_CAM_W,
) -> tuple[float, float, float, np.ndarray | None]:
    """Ultralytics seg 单实例 → (u, v, area, polygon)。"""
    polygon = None
    if result.masks is not None and hasattr(result.masks, "xy"):
        xy = result.masks.xy[index]
        if xy is not None and len(xy) >= 3:
            polygon = np.asarray(xy, dtype=np.float32)
            u, v, area = mask_centroid_from_polygon(polygon, img_h, img_w)
            return u, v, area, polygon

    if result.boxes is not None:
        x1, y1, x2, y2 = result.boxes.xyxy[index].tolist()
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        return u, v, float((x2 - x1) * (y2 - y1)), polygon

    return 0.0, 0.0, 0.0, polygon


def compute_align_error(
    u: float,
    v: float,
    target_u: float = EE_ALIGN_U0,
    target_v: float = EE_ALIGN_V0,
    tol_px: float = 8.0,
) -> tuple[float, float, bool]:
    """相对标定对准点的像素误差与是否已对准。"""
    err_u = float(u - target_u)
    err_v = float(v - target_v)
    aligned = abs(err_u) <= tol_px and abs(err_v) <= tol_px
    return err_u, err_v, aligned


class EESegAligner:
    """EE 垂直状态下：seg 推理 + mask 几何中心 + 对准误差。"""

    def __init__(self, cfg: PerceptionConfig | None = None):
        self.cfg = cfg or PerceptionConfig()
        self.target_u = float(getattr(self.cfg, "ee_align_target_u", EE_ALIGN_U0))
        self.target_v = float(getattr(self.cfg, "ee_align_target_v", EE_ALIGN_V0))
        self.align_tol_px = float(getattr(self.cfg, "ee_align_tol_px", 8.0))
        self._yolo = None
        weights = self.cfg.yolo_ee_seg_weights or self.cfg.yolo_weights
        if weights:
            self._load_weights(weights)

    def _load_weights(self, weights: str | Path) -> None:
        path = Path(weights)
        if not path.is_file():
            repo_root = Path(__file__).resolve().parents[2]
            alt = (repo_root / path).resolve()
            if alt.is_file():
                path = alt
        if not path.is_file():
            print(f"[EESegAligner] 权重不存在: {weights}")
            return
        try:
            from ultralytics import YOLO

            self._yolo = YOLO(str(path))
            print(f"[EESegAligner] 已加载 seg 权重: {path}")
        except Exception as exc:
            print(f"[EESegAligner] 加载失败: {exc}")

    @property
    def ready(self) -> bool:
        return self._yolo is not None

    def detect_centers(self, ee_rgb: np.ndarray) -> list[EESegCenter]:
        """对 ee_rgb 做 seg，返回所有实例的几何中心。"""
        if self._yolo is None or ee_rgb is None:
            return []

        h, w = ee_rgb.shape[:2]
        bgr = ee_rgb[..., ::-1]
        results = self._yolo.predict(bgr, verbose=False, conf=self.cfg.conf_threshold)
        centers: list[EESegCenter] = []

        for r in results:
            if r.boxes is None:
                continue
            n = len(r.boxes)
            for i in range(n):
                conf = float(r.boxes.conf[i].item())
                if conf < self.cfg.conf_threshold:
                    continue
                cls_id = int(r.boxes.cls[i].item())
                cls_name = self.cfg.class_names.get(cls_id, "unknown")
                obj_class = (
                    ObjectClass(cls_name)
                    if cls_name in ObjectClass._value2member_map_
                    else ObjectClass.UNKNOWN
                )
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                u, v, area, poly = mask_centroid_from_yolo(r, i, h, w)
                err_u, err_v, aligned = compute_align_error(
                    u, v, self.target_u, self.target_v, self.align_tol_px
                )
                centers.append(
                    EESegCenter(
                        u=u,
                        v=v,
                        obj_class=obj_class,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        area_px=area,
                        target_u=self.target_u,
                        target_v=self.target_v,
                        err_u=err_u,
                        err_v=err_v,
                        aligned=aligned,
                        polygon=poly,
                    )
                )
        centers.sort(key=lambda c: c.confidence, reverse=True)
        return centers

    def best_center(
        self,
        ee_rgb: np.ndarray,
        target_class: ObjectClass | None = None,
    ) -> EESegCenter | None:
        """置信度最高（或指定类别）的一个中心。"""
        centers = self.detect_centers(ee_rgb)
        if not centers:
            return None
        if target_class is not None:
            for c in centers:
                if c.obj_class == target_class:
                    return c
            return None
        return centers[0]

    @staticmethod
    def arm_xy_nudge_from_error(
        center: EESegCenter,
        gain_m_per_px: float = 0.0004,
    ) -> tuple[float, float]:
        """像素误差 → base 系 XY 微调量 (dx, dy)，供 IK 对准（符号需在仿真标定）。"""
        dx_b = -gain_m_per_px * center.err_u
        dy_b = -gain_m_per_px * center.err_v
        return dx_b, dy_b

    @staticmethod
    def draw_debug(
        ee_rgb: np.ndarray,
        centers: list[EESegCenter],
        copy: bool = True,
    ) -> np.ndarray:
        """可视化：mask 轮廓、质心、对准十字。"""
        vis = ee_rgb.copy() if copy else ee_rgb
        tu = int(centers[0].target_u) if centers else int(EE_ALIGN_U0)
        tv = int(centers[0].target_v) if centers else int(EE_ALIGN_V0)
        cv2.drawMarker(vis, (tu, tv), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        for c in centers:
            if c.polygon is not None:
                pts = c.polygon.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], True, (255, 128, 0), 2)
            color = (0, 255, 0) if c.aligned else (0, 0, 255)
            cv2.circle(vis, (int(c.u), int(c.v)), 6, color, -1)
            cv2.putText(
                vis,
                f"{c.obj_class.value} {c.confidence:.2f}",
                (int(c.u) + 8, int(c.v) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return vis