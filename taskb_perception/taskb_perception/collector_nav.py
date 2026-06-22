"""Task B 数据采集用的简单巡逻移动（B2 locomotion policy + 走向物体）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _wrap_pi(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def compute_goto_velocity_cmd(
    robot_xy: np.ndarray,
    robot_yaw: float,
    target_xy: np.ndarray,
    max_vx: float = 0.55,
    max_wz: float = 0.75,
) -> np.ndarray:
    """Body 系速度指令 [vx, vy, wz]，供 B2 flat policy 使用。"""
    delta = target_xy - robot_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    desired_yaw = float(np.arctan2(delta[1], delta[0]))
    yaw_err = _wrap_pi(desired_yaw - robot_yaw)

    wz = float(np.clip(yaw_err * 1.2, -max_wz, max_wz))
    if abs(yaw_err) > 0.8:
        vx = 0.15
    else:
        vx = float(np.clip(0.25 + 0.35 * dist, 0.2, max_vx))
    return np.array([vx, 0.0, wz], dtype=np.float32)


def resolve_policy_path(explicit: str | None, repo_root: Path) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for rel in (
        "demo/policy.pt",
        "atec_robot_model/baseline/unitree_b2_flat/policy.pt",
    ):
        p = repo_root / rel
        if p.is_file():
            return p
    return None


class B2LocomotionDriver:
    """加载 B2 flat JIT policy，把速度指令转成 env action（与 demo/solution_rl.py 一致）。"""

    LEG_DIM = 12
    ARM_DIM = 8
    LEG_IDX = list(range(12))
    ARM_IDX = list(range(12, 20))

    def __init__(self, device: str, policy_path: Path):
        self.device = device
        self.policy = torch.jit.load(str(policy_path), map_location=device)
        self.policy.eval()

        self.train_to_env_scale = torch.tensor(
            [0.25, 0.5, 0.5] * 4, device=device, dtype=torch.float32
        ).view(1, -1)
        self.env_to_train_scale = torch.tensor(
            [4.0, 2.0, 2.0] * 4, device=device, dtype=torch.float32
        ).view(1, -1)
        self.arm_default = torch.zeros((1, self.ARM_DIM), device=device)

    def compute_action(self, obs: dict, vel_cmd_b: np.ndarray) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        idx = 0
        idx += 3  # base_lin_vel
        base_ang_vel = proprio[:, idx : idx + 3]
        idx += 3  # skip env velocity_commands
        idx += 3
        projected_gravity = proprio[:, idx : idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx : idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.LEG_IDX]
        joint_vel_leg = joint_vel_all[:, self.LEG_IDX]
        actions_env_leg = actions_all[:, self.LEG_IDX]
        actions_train_leg = actions_env_leg * self.env_to_train_scale

        cmd = torch.tensor(vel_cmd_b, device=self.device, dtype=torch.float32).view(1, 3)
        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                cmd,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

        with torch.inference_mode():
            action_train = self.policy(policy_obs)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        leg_env = action_train * self.train_to_env_scale
        action_env = torch.zeros((1, action_dim), device=self.device)
        action_env[:, self.LEG_IDX] = leg_env
        action_env[:, self.ARM_IDX] = self.arm_default
        return action_env


class ObjectPatrolController:
    """按距离排序，依次走向每个物体；到达后原地小转一圈增加视角。"""

    def __init__(
        self,
        num_objects: int = 18,
        arrive_dist: float = 1.3,
        spin_steps: int = 12,
        spin_wz: float = 0.6,
    ):
        self.num_objects = num_objects
        self.arrive_dist = arrive_dist
        self.spin_steps = spin_steps
        self.spin_wz = spin_wz
        self.targets_xy: list[np.ndarray] = []
        self.target_idx = 0
        self.spin_left = 0
        self.visits = 0

    def reset(self, env) -> None:
        robot = env.scene["robot"]
        rxy = robot.data.root_pos_w[0, :2].detach().cpu().numpy()
        targets: list[tuple[float, np.ndarray]] = []
        for i in range(1, self.num_objects + 1):
            obj = env.scene[f"object_{i}"]
            oxy = obj.data.root_pos_w[0, :2].detach().cpu().numpy()
            d = float(np.linalg.norm(oxy - rxy))
            targets.append((d, oxy))
        targets.sort(key=lambda x: x[0])
        self.targets_xy = [t[1] for t in targets]
        self.target_idx = 0
        self.spin_left = 0
        self.visits = 0

    @property
    def done(self) -> bool:
        return self.visits >= len(self.targets_xy)

    def get_velocity_command(self, env) -> np.ndarray:
        robot = env.scene["robot"]
        rpos = robot.data.root_pos_w[0].detach().cpu().numpy()
        rq = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = yaw_from_quat_wxyz(rq)

        if self.done or not self.targets_xy:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if self.spin_left > 0:
            self.spin_left -= 1
            return np.array([0.0, 0.0, self.spin_wz], dtype=np.float32)

        target = self.targets_xy[self.target_idx]
        dist = float(np.linalg.norm(target - rpos[:2]))
        if dist <= self.arrive_dist:
            self.visits += 1
            self.target_idx += 1
            self.spin_left = self.spin_steps
            if self.target_idx < len(self.targets_xy):
                return np.array([0.0, 0.0, self.spin_wz], dtype=np.float32)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return compute_goto_velocity_cmd(rpos[:2], yaw, target)


# ---------------------------------------------------------------------------
# 光轴中线导航：2D 检测 → 目标 u 对齐 cx → 直行（不反算 base 3D）
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402


@dataclass
class AxisNavTarget:
    """光轴中线对准目标（仅 2D）。"""

    u: float
    v: float
    bbox: tuple[int, int, int, int]
    obj_class: str
    confidence: float
    source: str
    axis_u: float
    axis_v: float
    err_u: float
    err_v: float
    on_axis: bool
    depth_m: float = 0.0
    locked: bool = False
    lock_id: int | None = None
    stale: bool = False

    @property
    def pixel_error_u(self) -> float:
        return self.err_u


@dataclass
class NavTargetLock:
    """Memory Bank 单条记录：首帧锁定的目标模板（质心、bbox、类别）。"""

    lock_id: int
    obj_class: str
    lock_confidence: float
    last_u: float
    last_v: float
    last_bbox: tuple[int, int, int, int]
    last_depth_m: float = 0.0
    missed_frames: int = 0
    arrived: bool = False


@dataclass
class NavTargetMemoryBank:
    """EE 导航目标记忆库。

    - active：当前锁定的目标（上一帧预测位置）
    - completed：已完成目标历史，用于排除重复锁定
    """

    active: NavTargetLock | None = None
    completed: list[NavTargetLock] | None = None

    @property
    def completed_count(self) -> int:
        return len(self.completed or [])


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-6))


def compute_axis_align_error(
    u: float,
    v: float,
    axis_u: float,
    axis_v: float,
    tol_u: float,
) -> tuple[float, float, bool]:
    err_u = float(u - axis_u)
    err_v = float(v - axis_v)
    on_axis = abs(err_u) <= tol_u
    return err_u, err_v, on_axis


def velocity_from_axis_target(target: AxisNavTarget | None, cfg=None) -> np.ndarray:
    from .config import PerceptionConfig

    cfg = cfg or PerceptionConfig()
    if target is None:
        return np.zeros(3, dtype=np.float32)

    tol = float(cfg.nav_axis_tol_u_px)
    max_vx = float(cfg.nav_axis_forward_vx)
    max_wz = float(cfg.nav_axis_max_wz)
    gain = float(cfg.nav_axis_turn_gain)
    slow_vx = float(cfg.nav_axis_slow_vx)
    align_first = bool(cfg.nav_axis_align_before_forward)

    err_u = target.err_u
    wz = float(np.clip(-gain * err_u, -max_wz, max_wz))

    if target.on_axis:
        vx = max_vx
    elif align_first:
        vx = slow_vx if abs(err_u) < tol * 2.5 else 0.0
    else:
        vx = slow_vx

    if target.depth_m > 0.05 and target.depth_m < cfg.nav_axis_arrive_depth_m:
        vx = min(vx, 0.15)
        if target.depth_m < cfg.nav_axis_arrive_depth_m * 0.85:
            vx = 0.0
            wz = 0.0

    return np.array([vx, 0.0, wz], dtype=np.float32)


def draw_axis_nav_on_rgb(
    rgb: np.ndarray,
    target: AxisNavTarget | None,
    *,
    all_targets: list[AxisNavTarget] | None = None,
    source: str | None = None,
    copy: bool = True,
) -> np.ndarray:
    import cv2

    from .config import CAM_CX, CAM_CY

    vis = rgb.copy() if copy else rgb
    h, w = vis.shape[:2]
    if target is None:
        cx, cy = int(CAM_CX), int(CAM_CY)
    else:
        cx, cy = int(target.axis_u), int(target.axis_v)
    cv2.line(vis, (cx, 0), (cx, h - 1), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

    pool = list(all_targets or [])
    if source is not None:
        pool = [t for t in pool if t.source == source]
    if target is not None and target not in pool:
        pool.append(target)

    for t in pool:
        is_nav = target is not None and t is target
        u, v = int(t.u), int(t.v)
        if is_nav and getattr(t, "locked", False):
            color = (255, 0, 255) if t.on_axis else (255, 80, 255)
        elif is_nav and t.on_axis:
            color = (0, 255, 0)
        elif is_nav:
            color = (255, 120, 0)
        else:
            color = (100, 180, 255)
        thickness = 2 if is_nav else 1
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(vis, (u, v), 5 if is_nav else 4, color, -1)
        if is_nav:
            cv2.line(vis, (u, v), (cx, v), color, 2, cv2.LINE_AA)
        label = f"{t.obj_class} {t.confidence:.2f}"
        if is_nav and getattr(t, "locked", False):
            lid = getattr(t, "lock_id", None)
            label = f"LOCK#{lid} {label}" if lid is not None else f"LOCK {label}"
        if getattr(t, "stale", False):
            label += " stale"
        if is_nav:
            label += f" err_u={t.err_u:+.0f}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    n = len(pool)
    status = "ON_AXIS→FWD" if target and target.on_axis else "ALIGN"
    nav_tag = ""
    if target is not None:
        for i, t in enumerate(pool):
            if t is target:
                nav_tag = f" nav=#{i + 1}"
                break
    lock_tag = ""
    if target is not None and getattr(target, "locked", False):
        lock_tag = f" LOCK#{getattr(target, 'lock_id', '?')}"
    cv2.putText(
        vis,
        f"axis_nav det={n}{nav_tag}{lock_tag} {status if target else ''}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


class AxisNavController:
    """ee 2D detect → 光轴中线对准导航；支持首帧置信度锁定 + 逐个处理。"""

    def __init__(self, cfg=None):
        from .config import PerceptionConfig
        from .detector import ObjectDetector
        from .ee_det3d import EEDetector

        self.cfg = cfg or PerceptionConfig()
        self._head_det = ObjectDetector(self.cfg)
        self._ee_det = EEDetector(self.cfg) if self.cfg.enable_ee_nav_detect else None
        self.last_targets: list[AxisNavTarget] = []
        self.nav_target: AxisNavTarget | None = None
        self._lock: NavTargetLock | None = None
        self._lock_seq = 0
        self._completed_locks: list[NavTargetLock] = []

    @property
    def nav_lock(self) -> NavTargetLock | None:
        return self._lock

    @property
    def memory_bank(self) -> NavTargetMemoryBank:
        return NavTargetMemoryBank(active=self._lock, completed=list(self._completed_locks))

    @property
    def completed_count(self) -> int:
        return len(self._completed_locks)

    def reset(self) -> None:
        self.last_targets = []
        self.nav_target = None
        self._lock = None
        self._lock_seq = 0
        self._completed_locks = []

    def complete_current(self) -> None:
        """标记当前锁目标已完成，释放锁以便处理下一个。"""
        if self._lock is not None:
            self._lock.arrived = True
            self._completed_locks.append(self._lock)
        self._lock = None
        self.nav_target = None

    def acquire_lock(self, targets: list[AxisNavTarget] | None = None) -> AxisNavTarget | None:
        """按置信度锁定 EE 最高框（排除已完成目标）。"""
        pool = targets if targets is not None else self.last_targets
        ee_targets = [t for t in pool if t.source == "ee"]
        pick = self._pick_new_lock_candidate(ee_targets)
        if pick is None:
            return None
        self._establish_lock(pick)
        self.nav_target = self._decorate_target(pick, locked=True, stale=False)
        return self.nav_target

    def _pick_target(self, targets: list[AxisNavTarget]) -> AxisNavTarget | None:
        if not targets:
            return None
        return min(targets, key=lambda t: (-t.confidence, abs(t.err_u)))

    def _pick_new_lock_candidate(self, ee_targets: list[AxisNavTarget]) -> AxisNavTarget | None:
        if not ee_targets:
            return None
        candidates = [t for t in ee_targets if not self._is_excluded(t)]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.confidence)

    def _is_excluded(self, target: AxisNavTarget) -> bool:
        for done in self._completed_locks:
            dist = float(np.hypot(target.u - done.last_u, target.v - done.last_v))
            if target.obj_class == done.obj_class and dist <= self.cfg.nav_lock_exclude_dist_px:
                return True
            if bbox_iou(target.bbox, done.last_bbox) >= self.cfg.nav_lock_exclude_min_iou:
                return True
        return False

    def _establish_lock(self, target: AxisNavTarget) -> None:
        self._lock_seq += 1
        self._lock = NavTargetLock(
            lock_id=self._lock_seq,
            obj_class=target.obj_class,
            lock_confidence=target.confidence,
            last_u=target.u,
            last_v=target.v,
            last_bbox=target.bbox,
            last_depth_m=target.depth_m,
            missed_frames=0,
        )

    def _update_lock_from(self, target: AxisNavTarget) -> None:
        if self._lock is None:
            return
        self._lock.last_u = target.u
        self._lock.last_v = target.v
        self._lock.last_bbox = target.bbox
        self._lock.last_depth_m = target.depth_m
        self._lock.missed_frames = 0

    def _decorate_target(
        self,
        target: AxisNavTarget,
        *,
        locked: bool,
        stale: bool,
    ) -> AxisNavTarget:
        return AxisNavTarget(
            u=target.u,
            v=target.v,
            bbox=target.bbox,
            obj_class=target.obj_class,
            confidence=target.confidence,
            source=target.source,
            axis_u=target.axis_u,
            axis_v=target.axis_v,
            err_u=target.err_u,
            err_v=target.err_v,
            on_axis=target.on_axis,
            depth_m=target.depth_m,
            locked=locked,
            lock_id=self._lock.lock_id if self._lock is not None else None,
            stale=stale,
        )

    def _stale_target_from_lock(self) -> AxisNavTarget | None:
        if self._lock is None:
            return None
        lock = self._lock
        tol = float(self.cfg.nav_axis_tol_u_px)
        axis_u = float(self.cfg.nav_axis_target_u)
        axis_v = float(self.cfg.nav_axis_target_v)
        err_u, err_v, on_axis = compute_axis_align_error(lock.last_u, lock.last_v, axis_u, axis_v, tol)
        base = AxisNavTarget(
            u=lock.last_u,
            v=lock.last_v,
            bbox=lock.last_bbox,
            obj_class=lock.obj_class,
            confidence=lock.lock_confidence,
            source="ee",
            axis_u=axis_u,
            axis_v=axis_v,
            err_u=err_u,
            err_v=err_v,
            on_axis=on_axis,
            depth_m=lock.last_depth_m,
        )
        return self._decorate_target(base, locked=True, stale=True)

    def _match_locked(self, candidates: list[AxisNavTarget]) -> AxisNavTarget | None:
        """用 Memory Bank 中 active 条目的预测位置，在本帧检出中找最近匹配。"""
        if self._lock is None:
            return None
        lock = self._lock
        best: AxisNavTarget | None = None
        best_score = -1.0
        max_dist = float(self.cfg.nav_lock_match_max_dist_px)
        min_score = float(self.cfg.nav_lock_match_min_score)
        for target in candidates:
            if target.source != "ee":
                continue
            dist = float(np.hypot(target.u - lock.last_u, target.v - lock.last_v))
            if dist > max_dist:
                continue
            iou = bbox_iou(lock.last_bbox, target.bbox)
            class_w = 1.0 if target.obj_class == lock.obj_class else 0.35
            score = class_w * (iou + 1.0 / (1.0 + dist / 40.0))
            if score > best_score:
                best_score = score
                best = target
        if best is None or best_score < min_score:
            return None
        return best

    def _resolve_nav_target(self, ee_targets: list[AxisNavTarget], *, manage_lock: bool) -> None:
        if not self.cfg.nav_lock_enabled:
            self.nav_target = self._pick_target(ee_targets or self.last_targets)
            return

        if self._lock is None:
            if manage_lock and self.cfg.nav_lock_auto_acquire:
                pick = self._pick_new_lock_candidate(ee_targets)
                if pick is not None:
                    self._establish_lock(pick)
                    self.nav_target = self._decorate_target(pick, locked=True, stale=False)
                else:
                    self.nav_target = None
            else:
                self.nav_target = None
            return

        matched = self._match_locked(ee_targets)
        if matched is not None:
            self._update_lock_from(matched)
            self.nav_target = self._decorate_target(matched, locked=True, stale=False)
            return

        self._lock.missed_frames += 1
        if self._lock.missed_frames <= int(self.cfg.nav_lock_max_missed):
            self.nav_target = self._stale_target_from_lock()
            return

        self._lock = None
        self.nav_target = None
        if manage_lock and self.cfg.nav_lock_auto_acquire:
            pick = self._pick_new_lock_candidate(ee_targets)
            if pick is not None:
                self._establish_lock(pick)
                self.nav_target = self._decorate_target(pick, locked=True, stale=False)

    def _check_arrived(self, target: AxisNavTarget | None) -> bool:
        if target is None or self._lock is None:
            return False
        if target.depth_m <= 0.05:
            return False
        return target.depth_m < float(self.cfg.nav_axis_arrive_depth_m) * 0.85

    def detect_targets(self, obs: dict, *, manage_lock: bool = False) -> list[AxisNavTarget]:
        from .detector import Detection2D
        from .obs_utils import parse_depth, parse_rgb

        image_obs = obs.get("image") or {}
        src = self.cfg.nav_detect_source.lower()
        tol = float(self.cfg.nav_axis_tol_u_px)
        axis_u = float(self.cfg.nav_axis_target_u)
        axis_v = float(self.cfg.nav_axis_target_v)
        out: list[AxisNavTarget] = []

        def _to_target(det: Detection2D, source: str) -> AxisNavTarget:
            err_u, err_v, on_axis = compute_axis_align_error(det.u, det.v, axis_u, axis_v, tol)
            cls = det.obj_class.value if hasattr(det.obj_class, "value") else str(det.obj_class)
            return AxisNavTarget(
                u=det.u,
                v=det.v,
                bbox=det.bbox,
                obj_class=cls,
                confidence=det.confidence,
                source=source,
                axis_u=axis_u,
                axis_v=axis_v,
                err_u=err_u,
                err_v=err_v,
                on_axis=on_axis,
                depth_m=float(det.depth) if det.depth > 0 else 0.0,
            )

        if src in ("head", "both"):
            head_rgb = parse_rgb(image_obs, "head_rgb")
            head_depth = parse_depth(image_obs, "head_depth")
            if head_rgb is not None:
                for det in self._head_det.detect(
                    head_rgb, head_depth, source="head", require_depth=False
                ):
                    out.append(_to_target(det, "head"))

        if src in ("ee", "both") and self._ee_det is not None and self._ee_det.ready:
            ee_rgb = parse_rgb(image_obs, "ee_rgb")
            ee_depth = parse_depth(image_obs, "ee_depth")
            if ee_rgb is not None:
                for det in self._ee_det.detect_2d(ee_rgb, ee_depth):
                    out.append(_to_target(det, "ee"))

        out.sort(key=lambda t: (-t.confidence, abs(t.err_u)))
        self.last_targets = out

        ee_targets = [t for t in out if t.source == "ee"]
        if self.cfg.nav_lock_enabled and src in ("ee", "both"):
            self._resolve_nav_target(ee_targets, manage_lock=manage_lock)
        else:
            pool = ee_targets if src == "ee" else out
            self.nav_target = self._pick_target(pool)
        return out

    def compute_velocity(self, obs: dict) -> np.ndarray:
        self.detect_targets(obs, manage_lock=True)
        vel = velocity_from_axis_target(self.nav_target, self.cfg)
        if (
            self.cfg.nav_lock_enabled
            and self.cfg.nav_lock_auto_next
            and self._check_arrived(self.nav_target)
        ):
            self.complete_current()
        return vel
