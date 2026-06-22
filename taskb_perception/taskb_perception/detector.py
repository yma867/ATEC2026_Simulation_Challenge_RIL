"""2D 检测：优先 YOLO，无权重时回退到颜色+深度 blob。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CAM_CX, CAM_CY, CAM_FX, CAM_FY, DEPTH_MAX, DEPTH_MIN, PerceptionConfig
from .obs_utils import pixel_to_cam, sample_depth_median
from .types import ObjectClass


@dataclass
class Detection2D:
    u: float
    v: float
    depth: float
    obj_class: ObjectClass
    confidence: float
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2


class ObjectDetector:
    def __init__(self, cfg: PerceptionConfig, weights: str | Path | None = None):
        self.cfg = cfg
        self._yolo = None
        w = weights or cfg.yolo_head_weights or cfg.yolo_weights
        if w:
            weights_path = self._resolve_weights(w)
            if not weights_path.is_file():
                print(f"[ObjectDetector] 权重不存在: {weights_path}，将使用颜色回退")
            else:
                try:
                    from ultralytics import YOLO

                    self._yolo = YOLO(str(weights_path))
                except Exception as exc:
                    print(f"[ObjectDetector] YOLO 加载失败，将使用颜色回退: {exc}")

    @staticmethod
    def _resolve_weights(weights: str | Path) -> Path:
        path = Path(weights)
        if path.is_file():
            return path.resolve()
        repo_root = Path(__file__).resolve().parents[2]
        alt = (repo_root / path).resolve()
        return alt if alt.is_file() else path

    def detect(self, rgb: np.ndarray, depth: np.ndarray | None, source: str = "head", *, require_depth: bool = True) -> list[Detection2D]:
        if self._yolo is not None:
            dets = self._detect_yolo(rgb, depth, require_depth=require_depth)
            if dets:
                return dets
        if self.cfg.use_color_fallback and require_depth:
            return self._detect_color_blob(rgb, depth)
        return []

    def _detect_yolo(self, rgb: np.ndarray, depth: np.ndarray | None, *, require_depth: bool = True) -> list[Detection2D]:
        results = self._yolo.predict(rgb, verbose=False, conf=self.cfg.conf_threshold)
        dets: list[Detection2D] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if conf < self.cfg.conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)
                d = self._depth_at(depth, int(u), int(v)) if depth is not None else 0.0
                if require_depth and d <= 0:
                    continue
                if not require_depth and d <= 0:
                    d = 1.0
                cls_name = self.cfg.class_names.get(cls_id, "unknown")
                obj_class = ObjectClass(cls_name) if cls_name in ObjectClass._value2member_map_ else ObjectClass.UNKNOWN
                dets.append(
                    Detection2D(
                        u=u, v=v, depth=d,
                        obj_class=obj_class,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )
        return dets

    def _detect_color_blob(self, rgb: np.ndarray, depth: np.ndarray | None) -> list[Detection2D]:
        """简易回退：HSV 颜色 + 连通域。仿真里足够联调，正式比赛请训 YOLO。"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        dets: list[Detection2D] = []

        # 粗略颜色区间（可在真机/仿真上再标定）
        color_rules = [
            (ObjectClass.SUGAR, np.array([0, 0, 160]), np.array([180, 40, 255])),      # 白盒
            (ObjectClass.MUSTARD, np.array([18, 80, 80]), np.array([38, 255, 255])),   # 黄瓶
            (ObjectClass.BANANA, np.array([20, 60, 80]), np.array([35, 255, 255])),    # 黄香蕉
        ]

        for obj_class, lo, hi in color_rules:
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 120:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                u, v = x + w // 2, y + h // 2
                d = self._depth_at(depth, u, v)
                if d <= 0:
                    continue
                conf = min(0.9, 0.35 + area / 5000.0)
                dets.append(
                    Detection2D(
                        u=float(u), v=float(v), depth=d,
                        obj_class=obj_class,
                        confidence=conf,
                        bbox=(x, y, x + w, y + h),
                    )
                )
        return dets

    @staticmethod
    def _depth_at(depth: np.ndarray | None, u: int, v: int) -> float:
        if depth is None:
            return 0.0
        d = sample_depth_median(depth, u, v, radius=4)
        if d < DEPTH_MIN or d > DEPTH_MAX:
            return 0.0
        return d

    @staticmethod
    def detection_to_cam(det: Detection2D) -> np.ndarray:
        return pixel_to_cam(det.u, det.v, det.depth, CAM_FX, CAM_FY, CAM_CX, CAM_CY)

    @staticmethod
    def detection_to_cam_ee(det: Detection2D) -> np.ndarray:
        from .ee_collect_utils import EE_CX, EE_CY, EE_FX, EE_FY

        return pixel_to_cam(det.u, det.v, det.depth, EE_FX, EE_FY, EE_CX, EE_CY)
