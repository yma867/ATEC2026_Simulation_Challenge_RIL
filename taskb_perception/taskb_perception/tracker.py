"""3D 多目标跟踪 + EMA 滤波。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PerceptionConfig
from .grasp_pose import compute_grasp_quat_for_object, compute_grasp_waypoints
from .types import ObjectClass, ObjectTrack, TrackStatus


@dataclass
class _InternalTrack:
    track_id: int
    obj_class: ObjectClass
    pos_b: np.ndarray
    grasp_quat_b: np.ndarray
    confidence: float
    last_seen_step: int = 0
    missed_frames: int = 0
    status: TrackStatus = TrackStatus.ACTIVE
    source: str = "head"
    locked_snapshot: ObjectTrack | None = None


class Tracker3D:
    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        self._tracks: dict[int, _InternalTrack] = {}
        self._next_id = 1
        self._locked_id: int | None = None

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._locked_id = None

    def lock_target(self, track_id: int) -> ObjectTrack | None:
        t = self._tracks.get(track_id)
        if t is None:
            return None
        self._locked_id = track_id
        t.status = TrackStatus.LOCKED
        snap = self._to_public(t)
        t.locked_snapshot = snap
        return snap

    def unlock(self):
        if self._locked_id is not None and self._locked_id in self._tracks:
            if self._tracks[self._locked_id].status == TrackStatus.LOCKED:
                self._tracks[self._locked_id].status = TrackStatus.ACTIVE
        self._locked_id = None

    def mark_grasped(self, track_id: int):
        if track_id in self._tracks:
            self._tracks[track_id].status = TrackStatus.GRASPED
        if self._locked_id == track_id:
            self._locked_id = None

    def mark_placed(self, track_id: int):
        if track_id in self._tracks:
            self._tracks[track_id].status = TrackStatus.PLACED

    def update(
        self,
        detections: list[tuple[np.ndarray, ObjectClass, float, str]],
        step: int,
    ) -> list[ObjectTrack]:
        """detections: [(pos_b, class, conf, source), ...]"""
        if self._locked_id is not None:
            locked = self._tracks.get(self._locked_id)
            if locked and locked.locked_snapshot is not None:
                public = [self._to_public(t) for t in self._tracks.values()]
                return public

        used_tracks: set[int] = set()
        alpha = self.cfg.ema_alpha

        # 贪心匹配：每个 detection 找最近 track
        for pos_b, obj_class, conf, source in detections:
            if conf < self.cfg.min_confidence:
                continue
            best_id, best_dist = None, self.cfg.match_dist_thresh
            for tid, tr in self._tracks.items():
                if tid in used_tracks:
                    continue
                if tr.status in (TrackStatus.GRASPED, TrackStatus.PLACED, TrackStatus.LOST):
                    continue
                if tr.obj_class != obj_class and tr.obj_class != ObjectClass.UNKNOWN:
                    continue
                dist = float(np.linalg.norm(tr.pos_b - pos_b))
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                grasp_q = compute_grasp_quat_for_object(obj_class)
                tr = _InternalTrack(
                    track_id=self._next_id,
                    obj_class=obj_class,
                    pos_b=pos_b.copy(),
                    grasp_quat_b=grasp_q,
                    confidence=conf,
                    last_seen_step=step,
                    source=source,
                )
                self._tracks[self._next_id] = tr
                used_tracks.add(self._next_id)
                self._next_id += 1
            else:
                tr = self._tracks[best_id]
                tr.pos_b = (1 - alpha) * tr.pos_b + alpha * pos_b
                tr.confidence = 0.7 * tr.confidence + 0.3 * conf
                tr.last_seen_step = step
                tr.missed_frames = 0
                tr.source = source
                used_tracks.add(best_id)

        # 未匹配 track 计丢失
        for tid, tr in self._tracks.items():
            if tid in used_tracks:
                continue
            if tr.status in (TrackStatus.GRASPED, TrackStatus.PLACED, TrackStatus.LOCKED):
                continue
            tr.missed_frames += 1
            if tr.missed_frames > self.cfg.max_missed_frames:
                tr.status = TrackStatus.LOST

        return [self._to_public(t) for t in self._tracks.values() if t.status != TrackStatus.LOST]

    def _to_public(self, t: _InternalTrack) -> ObjectTrack:
        if t.status == TrackStatus.LOCKED and t.locked_snapshot is not None:
            return t.locked_snapshot
        pre, grasp = compute_grasp_waypoints(t.pos_b, t.grasp_quat_b)
        return ObjectTrack(
            track_id=t.track_id,
            obj_class=t.obj_class,
            pos_b=t.pos_b.copy(),
            grasp_quat_b=t.grasp_quat_b.copy(),
            pre_grasp_pos_b=pre,
            grasp_pos_b=grasp,
            confidence=t.confidence,
            status=t.status,
            last_seen_step=t.last_seen_step,
            missed_frames=t.missed_frames,
            source=t.source,
        )
