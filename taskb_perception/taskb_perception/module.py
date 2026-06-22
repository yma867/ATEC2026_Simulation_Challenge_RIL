"""Task B 感知主模块 — 在 solution.predicts 里每帧调用 update(obs)。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .collector_nav import AxisNavController, AxisNavTarget
from .config import PerceptionConfig
from .detector import ObjectDetector
from .ee_det3d import EEDetector, EEDetection3D
from .math3d import transform_point_cam_to_base
from .obs_utils import parse_depth, parse_proprio, parse_rgb
from .tracker import Tracker3D
from .types import ObjectClass, ObjectTrack, PerceptionOutput, TrackStatus


class PerceptionModule:
    """head / EE RGB-D 检测 + 3D 跟踪 + 抓取点输出（base 坐标系）。"""

    def __init__(self, cfg: PerceptionConfig | None = None):
        self.cfg = cfg or PerceptionConfig()
        self.detector = ObjectDetector(self.cfg)
        self.ee_detector = EEDetector(self.cfg) if self.cfg.enable_ee_nav_detect else None
        self.axis_nav = AxisNavController(self.cfg)
        self.tracker = Tracker3D(self.cfg)
        self.step = 0
        self._last_output: PerceptionOutput | None = None
        self._last_ee_detections: list[EEDetection3D] = []

    def reset(self):
        self.step = 0
        self.tracker.reset()
        self.axis_nav.reset()
        self._last_output = None
        self._last_ee_detections = []

    def compute_nav_velocity(self, obs: dict[str, Any]) -> np.ndarray:
        """导航速度 [vx, vy, wz]。默认光轴中线对准；pos_b_3d 需自行用 tracker。"""
        if self.cfg.nav_method == "axis_align":
            return self.axis_nav.compute_velocity(obs)
        return np.zeros(3, dtype=np.float32)

    def complete_nav_target(self) -> None:
        """当前 EE 导航锁目标已完成，释放锁以便处理下一个。"""
        self.axis_nav.complete_current()

    def reset_nav_locks(self) -> None:
        self.axis_nav.reset()

    def detect_ee_nav(self, obs: dict[str, Any]) -> list[EEDetection3D]:
        """EE 远距离导航：视角内所有 detect → depth → pos_b。"""
        if self.ee_detector is None or not self.ee_detector.ready:
            return []
        image_obs = obs.get("image") or {}
        ee_rgb = parse_rgb(image_obs, "ee_rgb")
        ee_depth = parse_depth(image_obs, "ee_depth")
        self._last_ee_detections = self.ee_detector.detect_all(ee_rgb, ee_depth)
        return self._last_ee_detections

    def lock_target(self, track_id: int) -> ObjectTrack | None:
        """抓取层进入 PRE_GRASP 时调用，冻结该目标位姿。"""
        return self.tracker.lock_target(track_id)

    def unlock_target(self):
        self.tracker.unlock()

    def mark_grasped(self, track_id: int):
        self.tracker.mark_grasped(track_id)

    def mark_placed(self, track_id: int):
        self.tracker.mark_placed(track_id)

    def update(self, obs: dict[str, Any]) -> PerceptionOutput:
        self.step += 1
        image_obs = obs.get("image") or {}
        proprio = parse_proprio(obs["proprio"])

        head_rgb = parse_rgb(image_obs, "head_rgb")
        head_depth = parse_depth(image_obs, "head_depth")

        detections_b: list[tuple[np.ndarray, Any, float, str]] = []
        src = self.cfg.nav_detect_source.lower()

        if src in ("head", "both") and head_rgb is not None:
            dets = self.detector.detect(head_rgb, head_depth, source="head")
            cam_ext = self.cfg.robot.head_cam
            for det in dets:
                p_cam = self.detector.detection_to_cam(det)
                p_b = transform_point_cam_to_base(p_cam, cam_ext.pos_b, cam_ext.quat_b)
                detections_b.append((p_b, det.obj_class, det.confidence, "head"))

        if src in ("ee", "both") and self.cfg.enable_ee_nav_detect:
            ee_dets = self.detect_ee_nav(obs)
            for det in ee_dets:
                obj_class = ObjectClass(det.obj_class) if det.obj_class in ObjectClass._value2member_map_ else ObjectClass.UNKNOWN
                detections_b.append((det.pos_b, obj_class, det.confidence, "ee"))

        # 近距离 head 跟踪时可选用 ee detect refine（与 EE 远距离导航不同）
        dist_hint = self._nearest_active_dist()
        if (
            dist_hint is not None
            and dist_hint < self.cfg.use_ee_refine_dist
            and src == "head"
            and self.ee_detector is not None
            and self.ee_detector.ready
        ):
            ee_rgb = parse_rgb(image_obs, "ee_rgb")
            ee_depth = parse_depth(image_obs, "ee_depth")
            if ee_rgb is not None and ee_depth is not None:
                for det in self.ee_detector.detect_all(ee_rgb, ee_depth):
                    obj_class = ObjectClass(det.obj_class) if det.obj_class in ObjectClass._value2member_map_ else ObjectClass.UNKNOWN
                    detections_b.append((det.pos_b, obj_class, det.confidence * 1.05, "ee_refine"))

        tracks = self.tracker.update(detections_b, self.step)
        active = [t for t in tracks if t.status == TrackStatus.ACTIVE]
        active.sort(key=lambda t: t.distance_xy())

        locked = None
        for t in tracks:
            if t.status == TrackStatus.LOCKED:
                locked = t
                break

        out = PerceptionOutput(
            step=self.step,
            tracks=tracks,
            next_target=active[0] if active else None,
            locked_target=locked,
            num_active=len(active),
        )
        self._last_output = out
        return out

    def _nearest_active_dist(self) -> float | None:
        if self._last_output is None or self._last_output.next_target is None:
            return None
        return self._last_output.next_target.distance_xy()

    @property
    def last_axis_nav_target(self) -> AxisNavTarget | None:
        return self.axis_nav.nav_target

    @property
    def nav_completed_count(self) -> int:
        return self.axis_nav.completed_count

    @property
    def last_ee_detections(self) -> list[EEDetection3D]:
        return list(self._last_ee_detections)

    @property
    def last_output(self) -> PerceptionOutput | None:
        return self._last_output
