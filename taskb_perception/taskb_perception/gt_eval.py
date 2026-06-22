"""Task B 仿真 GT：物体世界坐标 → base 系，与 EE detect 反算 pos_b 对比。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from isaaclab.utils.math import subtract_frame_transforms

from .config import NUM_OBJECTS
from .yolo_labels import CLASS_NAMES, object_index_to_class

CLASS_NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


@dataclass
class ObjectGT:
    obj_idx: int
    class_id: int
    class_name: str
    pos_w: np.ndarray
    pos_b: np.ndarray


@dataclass
class MatchResult:
    detection_index: int
    gt_obj_idx: int
    class_name: str
    pos_b_est: np.ndarray
    pos_b_gt: np.ndarray
    err_xy: float
    err_z: float
    err_3d: float
    confidence: float


@dataclass
class EvalSummary:
    matches: list[MatchResult] = field(default_factory=list)
    unmatched_detections: int = 0
    mean_err_xy: float = 0.0
    mean_err_z: float = 0.0
    mean_err_3d: float = 0.0
    max_err_xy: float = 0.0
    max_err_3d: float = 0.0

    @property
    def num_matched(self) -> int:
        return len(self.matches)


def read_ee_cam_extrinsic_in_base(env) -> tuple[np.ndarray, np.ndarray]:
    """从仿真读取 ee_camera 相对 robot base 的实时外参 (pos_b, quat_b wxyz)。"""
    robot = env.scene["robot"]
    ee_cam = env.scene.sensors["ee_camera"]
    base_pos = robot.data.root_pos_w[0:1]
    base_quat = robot.data.root_quat_w[0:1]
    cam_pos_w = ee_cam.data.pos_w[0:1]
    cam_quat_w = None
    for attr in ("quat_w_world", "quat_w_ros", "quat_w"):
        if hasattr(ee_cam.data, attr):
            cam_quat_w = getattr(ee_cam.data, attr)[0:1]
            break
    if cam_quat_w is None:
        raise AttributeError("ee_camera has no quat_w_world / quat_w_ros / quat_w")
    pos_b_t, quat_b_t = subtract_frame_transforms(base_pos, base_quat, cam_pos_w, cam_quat_w)
    pos_b = pos_b_t[0].detach().cpu().numpy().astype(np.float32)
    quat_b = quat_b_t[0].detach().cpu().numpy().astype(np.float32)
    return pos_b, quat_b


def reevaluate_with_live_extrinsic(detections, env) -> EvalSummary:
    """用仿真实时 ee 外参重算 pos_b，再与 GT 匹配（隔离 config 外参误差）。"""
    from .math3d import transform_point_cam_to_base

    cam_pos_b, cam_quat_b = read_ee_cam_extrinsic_in_base(env)
    adjusted = []
    for det in detections:
        pos_b = transform_point_cam_to_base(det.p_cam, cam_pos_b, cam_quat_b)
        clone = type(det)(
            u=det.u,
            v=det.v,
            depth_m=det.depth_m,
            bbox=det.bbox,
            obj_class=det.obj_class,
            confidence=det.confidence,
            p_cam=det.p_cam,
            pos_b=pos_b.astype(np.float32),
        )
        adjusted.append(clone)
    return match_detections_to_gt(adjusted, read_objects_gt_base(env))


def read_objects_gt_base(env) -> list[ObjectGT]:
    """读取 18 个物体相对 robot base 的 GT 位置（米）。"""
    robot = env.scene["robot"]
    base_pos = robot.data.root_pos_w[0:1]
    base_quat = robot.data.root_quat_w[0:1]
    out: list[ObjectGT] = []

    for obj_idx in range(1, NUM_OBJECTS + 1):
        obj = env.scene[f"object_{obj_idx}"]
        pos_w = obj.data.root_pos_w[0:1]
        pos_b_t, _ = subtract_frame_transforms(base_pos, base_quat, pos_w, None)
        pos_b = pos_b_t[0].detach().cpu().numpy().astype(np.float32)
        pos_w_np = pos_w[0].detach().cpu().numpy().astype(np.float32)
        cls_id = object_index_to_class(obj_idx)
        out.append(
            ObjectGT(
                obj_idx=obj_idx,
                class_id=cls_id,
                class_name=CLASS_NAMES[cls_id],
                pos_w=pos_w_np,
                pos_b=pos_b,
            )
        )
    return out


def project_pos_w_to_ee_uv(pos_w: np.ndarray, ee_cam) -> tuple[float, float] | None:
    """GT 世界坐标 → EE 图像像素 (u,v)，不可见返回 None。"""
    from .dataset_io import _project_world_to_pixel, _read_cam_pose, _read_intrinsics

    cam_pos, cam_quat = _read_cam_pose(ee_cam)
    fx, fy, cx, cy = _read_intrinsics(ee_cam)
    uv, valid = _project_world_to_pixel(
        np.asarray(pos_w, dtype=np.float32).reshape(1, 3),
        cam_pos,
        cam_quat,
        fx,
        fy,
        cx,
        cy,
    )
    if not bool(valid[0]):
        return None
    u, v = float(uv[0, 0]), float(uv[0, 1])
    if u < 0 or v < 0 or u >= 640 or v >= 480:
        return None
    return u, v


def _class_id_from_det(det) -> int | None:
    if hasattr(det.obj_class, "value"):
        name = det.obj_class.value
    else:
        name = str(det.obj_class)
    return CLASS_NAME_TO_ID.get(name)


def match_detections_to_gt(detections, gt_objects: list[ObjectGT]) -> EvalSummary:
    """按类别 + base 系 xy 最近邻贪心匹配。"""
    summary = EvalSummary()
    if not detections:
        return summary

    gt_pool = list(gt_objects)
    order = sorted(range(len(detections)), key=lambda i: detections[i].confidence, reverse=True)

    for di in order:
        det = detections[di]
        cls_id = _class_id_from_det(det)
        if cls_id is None:
            summary.unmatched_detections += 1
            continue

        candidates = [g for g in gt_pool if g.class_id == cls_id]
        if not candidates:
            summary.unmatched_detections += 1
            continue

        est = det.pos_b.astype(np.float32)
        best = min(candidates, key=lambda g: float(np.linalg.norm(est[:2] - g.pos_b[:2])))
        gt_pool.remove(best)

        err_xy = float(np.linalg.norm(est[:2] - best.pos_b[:2]))
        err_z = float(abs(est[2] - best.pos_b[2]))
        err_3d = float(np.linalg.norm(est - best.pos_b))
        summary.matches.append(
            MatchResult(
                detection_index=di,
                gt_obj_idx=best.obj_idx,
                class_name=best.class_name,
                pos_b_est=est,
                pos_b_gt=best.pos_b.copy(),
                err_xy=err_xy,
                err_z=err_z,
                err_3d=err_3d,
                confidence=float(det.confidence),
            )
        )

    if summary.matches:
        summary.mean_err_xy = float(np.mean([m.err_xy for m in summary.matches]))
        summary.mean_err_z = float(np.mean([m.err_z for m in summary.matches]))
        summary.mean_err_3d = float(np.mean([m.err_3d for m in summary.matches]))
        summary.max_err_xy = float(np.max([m.err_xy for m in summary.matches]))
        summary.max_err_3d = float(np.max([m.err_3d for m in summary.matches]))
    return summary


def format_eval_report(
    summary: EvalSummary,
    prefix: str = "[gt-eval]",
    *,
    live_summary: EvalSummary | None = None,
) -> str:
    lines = [f"{prefix} matched={summary.num_matched} unmatched_det={summary.unmatched_detections}"]
    if summary.num_matched == 0:
        lines.append(f"{prefix} 无匹配，无法计算误差")
        return "\n".join(lines)

    lines.append(
        f"{prefix} mean err_xy={summary.mean_err_xy*100:.1f}cm "
        f"err_z={summary.mean_err_z*100:.1f}cm err_3d={summary.mean_err_3d*100:.1f}cm"
    )
    lines.append(
        f"{prefix} max  err_xy={summary.max_err_xy*100:.1f}cm "
        f"err_3d={summary.max_err_3d*100:.1f}cm"
    )
    for m in summary.matches:
        lines.append(
            f"{prefix}  obj_{m.gt_obj_idx:02d} {m.class_name} conf={m.confidence:.2f} "
            f"est=({m.pos_b_est[0]:+.2f},{m.pos_b_est[1]:+.2f},{m.pos_b_est[2]:+.2f}) "
            f"gt=({m.pos_b_gt[0]:+.2f},{m.pos_b_gt[1]:+.2f},{m.pos_b_gt[2]:+.2f}) "
            f"err_xy={m.err_xy*100:.1f}cm err_3d={m.err_3d*100:.1f}cm"
        )
    if live_summary is not None and live_summary.num_matched:
        lines.append(
            f"{prefix} [live ee extrinsic] mean err_xy={live_summary.mean_err_xy*100:.1f}cm "
            f"err_z={live_summary.mean_err_z*100:.1f}cm err_3d={live_summary.mean_err_3d*100:.1f}cm"
        )
    return "\n".join(lines)


def draw_gt_eval_on_rgb(
    rgb: np.ndarray,
    detections,
    summary: EvalSummary,
    gt_all: list[ObjectGT],
    ee_cam,
) -> np.ndarray:
    """检测框 + GT 投影点（品红圈）+ est/GT 连线。"""
    import cv2

    from .ee_det3d import draw_ee_detections

    vis = draw_ee_detections(rgb, detections, show_pos_b=False)
    gt_by_idx = {g.obj_idx: g for g in gt_all}

    for m in summary.matches:
        det = detections[m.detection_index]
        u_est, v_est = int(det.u), int(det.v)
        g = gt_by_idx[m.gt_obj_idx]
        uv_gt = project_pos_w_to_ee_uv(g.pos_w, ee_cam)
        if uv_gt is None:
            continue
        u_gt, v_gt = int(uv_gt[0]), int(uv_gt[1])
        cv2.circle(vis, (u_gt, v_gt), 7, (255, 0, 255), 2)
        cv2.line(vis, (u_est, v_est), (u_gt, v_gt), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"obj{m.gt_obj_idx} e{m.err_xy*100:.0f}cm",
            (u_gt + 6, v_gt - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if summary.num_matched:
        cv2.putText(
            vis,
            f"mean xy={summary.mean_err_xy*100:.1f}cm 3d={summary.mean_err_3d*100:.1f}cm",
            (8, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


class RunningEvalStats:
    """多帧累计误差。"""

    def __init__(self):
        self._xy: list[float] = []
        self._z: list[float] = []
        self._d3: list[float] = []
        self.frames = 0

    def update(self, summary: EvalSummary) -> None:
        self.frames += 1
        for m in summary.matches:
            self._xy.append(m.err_xy)
            self._z.append(m.err_z)
            self._d3.append(m.err_3d)

    def format(self) -> str:
        if not self._d3:
            return "[gt-eval] cumulative: no matches yet"
        return (
            f"[gt-eval] cumulative n={len(self._d3)} frames={self.frames} "
            f"mean_xy={np.mean(self._xy)*100:.1f}cm "
            f"mean_z={np.mean(self._z)*100:.1f}cm "
            f"mean_3d={np.mean(self._d3)*100:.1f}cm "
            f"max_3d={np.max(self._d3)*100:.1f}cm"
        )
