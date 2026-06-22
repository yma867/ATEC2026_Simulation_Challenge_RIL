"""感知层对外数据结构 — 导航 / 抓取队友只需 import 这些类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ObjectClass(str, Enum):
    SUGAR = "sugar"
    MUSTARD = "mustard"
    BANANA = "banana"
    UNKNOWN = "unknown"


class TrackStatus(str, Enum):
    ACTIVE = "active"       # 正常跟踪中
    LOCKED = "locked"         # 已被抓取层锁定（位姿不再更新）
    GRASPED = "grasped"       # 已抓取
    PLACED = "placed"         # 已放入垃圾桶
    LOST = "lost"             # 连续丢失


@dataclass
class ObjectTrack:
    """单个物体的跟踪结果（主坐标系：机器人 base 系）。"""

    track_id: int
    obj_class: ObjectClass
    # base 系下物体中心 (x, y, z)，单位 m
    pos_b: np.ndarray
    # base 系下抓取姿态 (w, x, y, z)
    grasp_quat_b: np.ndarray
    # base 系下预抓取点（物体上方）
    pre_grasp_pos_b: np.ndarray
    # base 系下抓取点（略低于 pre_grasp）
    grasp_pos_b: np.ndarray
    confidence: float
    status: TrackStatus = TrackStatus.ACTIVE
    last_seen_step: int = 0
    missed_frames: int = 0
    # 供调试：检测来源
    source: str = "head"

    def distance_xy(self) -> float:
        return float(np.linalg.norm(self.pos_b[:2]))


@dataclass
class PerceptionOutput:
    """每帧感知模块的完整输出。"""

    step: int
    tracks: list[ObjectTrack] = field(default_factory=list)
    # 推荐下一个去捡的目标（距机器人最近且 ACTIVE）
    next_target: Optional[ObjectTrack] = None
    locked_target: Optional[ObjectTrack] = None
    num_active: int = 0

    def get_track(self, track_id: int) -> Optional[ObjectTrack]:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None
