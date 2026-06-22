"""Task B 感知层 — 供导航 / 抓取队友对接。"""

from .collector_nav import (
    AxisNavController,
    AxisNavTarget,
    B2LocomotionDriver,
    NavTargetLock,
    NavTargetMemoryBank,
    draw_axis_nav_on_rgb,
    velocity_from_axis_target,
)
from .config import (
    B2_PIPER_PROFILE,
    PerceptionConfig,
    TARGET_BIN_RADIUS,
    TARGET_BIN_XY,
)
from .ee_det3d import EEDetection3D, EEDetector, draw_ee_detections, ee_bbox_to_base, ee_pixel_depth_to_base
from .ee_seg_align import EESegAligner, EESegCenter, compute_align_error, mask_centroid_from_yolo
from .module import PerceptionModule
from .types import ObjectClass, ObjectTrack, PerceptionOutput, TrackStatus

__all__ = [
    "PerceptionModule",
    "PerceptionConfig",
    "PerceptionOutput",
    "ObjectTrack",
    "ObjectClass",
    "TrackStatus",
    "EEDetector",
    "ee_bbox_to_base",
    "ee_pixel_depth_to_base",
    "EEDetection3D",
    "EESegAligner",
    "EESegCenter",
    "mask_centroid_from_yolo",
    "compute_align_error",
    "AxisNavController",
    "AxisNavTarget",
    "NavTargetLock",
    "NavTargetMemoryBank",
    "draw_axis_nav_on_rgb",
    "velocity_from_axis_target",
    "TARGET_BIN_XY",
    "TARGET_BIN_RADIUS",
    "B2_PIPER_PROFILE",
    "B2LocomotionDriver",
]
