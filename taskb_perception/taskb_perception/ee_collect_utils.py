"""EE 相机姿态：stowed=臂收在狗身朝前 | overhead=抓取俯视。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from isaaclab.utils.math import quat_apply

# 抓取朝下（wxyz）
EE_TOPDOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

B2_PIPER_ARM_JOINTS = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
    "arm_joint7",
    "arm_joint8",
]
B2_PIPER_ARM_IK_JOINTS = B2_PIPER_ARM_JOINTS[:6]
B2_PIPER_GRIPPER_JOINTS = B2_PIPER_ARM_JOINTS[6:]

# 前伸到狗身前方、夹爪朝下（弧度）。臂座在背上，必须前伸才能看地面物体。
B2_PIPER_OVERHEAD_JOINTS = np.array(
    [0.0, 1.42, -1.18, 0.0, 1.08, 0.0, 0.035, -0.035],
    dtype=np.float32,
)

# 臂平收在狗背上、EE 朝前平视（与 overhead 明显不同；勿用 USD default，那往往是前伸俯视）
B2_PIPER_STOWED_JOINTS = np.array(
    [0.0, -0.62, 0.95, 0.0, -0.72, 0.0, 0.035, -0.035],
    dtype=np.float32,
)

ARM_POSE_STOWED = "stowed"
ARM_POSE_OVERHEAD = "overhead"
ARM_POSE_ZERO = "zero"
ARM_POSE_USD_DEFAULT = "usd_default"

STOWED_MAX_DOWN_DEG = 35.0  # 光轴相对向下超过此角度则判定仍在俯视


def read_arm_joints_from_robot(robot) -> np.ndarray:
    """读取 USD default_joint_pos（仅作参考，不等于 stowed）。"""
    arm_ids, _ = robot.find_joints(B2_PIPER_ARM_JOINTS)
    default = robot.data.default_joint_pos[0]
    return np.array([float(default[jid]) for jid in arm_ids], dtype=np.float32)


def read_current_arm_joints(robot) -> np.ndarray:
    arm_ids, _ = robot.find_joints(B2_PIPER_ARM_JOINTS)
    cur = robot.data.joint_pos[0]
    return np.array([float(cur[jid]) for jid in arm_ids], dtype=np.float32)


def write_arm_joints_to_sim(robot, joints: np.ndarray) -> None:
    """直接把臂关节写入仿真（reset 后立刻到位，不等 PD 慢慢收）。"""
    arm_ids, _ = robot.find_joints(B2_PIPER_ARM_JOINTS)
    jpos = robot.data.joint_pos.clone()
    jvel = robot.data.joint_vel.clone()
    for i, jid in enumerate(arm_ids):
        jpos[:, jid] = float(joints[i])
        jvel[:, jid] = 0.0
    robot.write_joint_state_to_sim(jpos, jvel)
    if hasattr(robot, "set_joint_position_target"):
        robot.set_joint_position_target(jpos)


def resolve_arm_joint_preset(robot, pose: str) -> np.ndarray:
    """stowed=收在狗身朝前 | overhead=抓取俯视 | zero=全0 | usd_default=USD默认。"""
    if pose == ARM_POSE_STOWED:
        return B2_PIPER_STOWED_JOINTS.copy()
    if pose == ARM_POSE_OVERHEAD:
        return B2_PIPER_OVERHEAD_JOINTS.copy()
    if pose == ARM_POSE_ZERO:
        return np.zeros(len(B2_PIPER_ARM_JOINTS), dtype=np.float32)
    if pose == ARM_POSE_USD_DEFAULT:
        return read_arm_joints_from_robot(robot)
    raise ValueError(f"未知 arm_pose: {pose!r}，可选 stowed|overhead|zero|usd_default")

# IK 模式备用：base 系前方 + 低于背部的点（勿放 z>0 在背上）
DEFAULT_EE_POS_B = np.array([0.62, 0.0, -0.30], dtype=np.float32)

EE_CAM_W, EE_CAM_H = 640, 480
EE_FX = 15.0 / 20.955 * EE_CAM_W
EE_FY = EE_FX
EE_CX, EE_CY = EE_CAM_W / 2.0, EE_CAM_H / 2.0


def _cam_optical_axis_world(cam) -> np.ndarray:
    quat = None
    for attr in ("quat_w_ros", "quat_w_world", "quat_w"):
        if hasattr(cam.data, attr):
            quat = getattr(cam.data, attr)[0]
            break
    if quat is None:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    device = quat.device
    axis = quat_apply(
        quat.unsqueeze(0),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device),
    ).squeeze()
    v = axis.detach().cpu().numpy()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)


def ee_camera_down_angle_deg(cam) -> float:
    forward = _cam_optical_axis_world(cam)
    down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cos = float(np.clip(np.dot(forward, down), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def ee_camera_is_topdown(cam, tol_deg: float = 8.0) -> tuple[bool, float]:
    angle = ee_camera_down_angle_deg(cam)
    return angle <= tol_deg, angle


def ee_camera_elevation_deg(cam) -> float:
    """光轴相对水平面的仰角（度）。≈0 朝前平视，负值朝下，正值朝上。"""
    forward = _cam_optical_axis_world(cam)
    horiz = float(np.hypot(forward[0], forward[1]))
    return float(np.degrees(np.arctan2(forward[2], max(horiz, 1e-6))))


def format_ee_camera_view(ee_cam) -> str:
    down = ee_camera_down_angle_deg(ee_cam)
    elev = ee_camera_elevation_deg(ee_cam)
    return f"仰角={elev:.1f}° 相对向下={down:.1f}°"


def verify_arm_pose_view(ee_cam, pose: str, *, max_down_deg: float = STOWED_MAX_DOWN_DEG) -> None:
    """启动后检查 EE 视角是否符合 arm_pose 预期。"""
    down = ee_camera_down_angle_deg(ee_cam)
    elev = ee_camera_elevation_deg(ee_cam)
    view = format_ee_camera_view(ee_cam)
    if pose == ARM_POSE_STOWED and down < max_down_deg:
        print(
            f"[WARN] arm_pose=stowed 但 EE 仍在俯视 ({view})。"
            f" 若仍看地面，按 Y 保存关节角，并用 --arm-joints 加载"
        )
    elif pose == ARM_POSE_OVERHEAD and down > 25.0:
        print(f"[WARN] arm_pose=overhead 但 EE 未朝下 ({view})")
    else:
        print(f"[arm] EE 视角 OK: {view}")


def try_bind_viewport_to_ee_camera(ee_cam) -> str | None:
    """Isaac 主窗口切到 ee_camera。"""
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None or not hasattr(ee_cam, "_sensor_prims") or not ee_cam._sensor_prims:
            return None
        prim_path = ee_cam._sensor_prims[0].GetPath().pathString
        if hasattr(viewport, "set_active_camera"):
            viewport.set_active_camera(prim_path)
        else:
            viewport.camera_path = prim_path
        return prim_path
    except Exception as exc:
        print(f"[view] 绑定 ee_camera 失败: {exc}")
        return None


def apply_arm_pose_preset(robot, arm_ctrl: "EEPresetArmController", pose: str) -> np.ndarray:
    """解析 preset → 写仿真 → 更新 controller 目标。"""
    joints = resolve_arm_joint_preset(robot, pose)
    write_arm_joints_to_sim(robot, joints)
    arm_ctrl.set_joints(joints)
    return joints


ARM_TUNE_HELP = """
臂 / EE 视角手动微调（先点 Isaac 仿真窗口）:
  I/K   joint2 抬高/放低     J/L   joint3 收/伸
  U/O   joint5 腕俯仰 +/-    Z/X   joint1 底座旋转
  [ / ] joint4 +/-            ; / ' joint6 +/-
  B     细调模式(步长变小)    Y     打印关节角并保存 JSON
  H     显示本帮助
加载已保存角度:  --arm-joints snapshots/ee_rgbd/arm_joints_tuned.json
完全手调启动:    --manual-arm  (不施加 stowed preset)
"""


def load_arm_joints(source: str) -> np.ndarray:
    """JSON / npy / 逗号分隔 8 个数 → (8,) float32。"""
    import json
    from pathlib import Path

    src = source.strip()
    path = Path(src)
    if path.is_file():
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32).reshape(-1)
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "joints" in raw:
                arr = np.asarray(raw["joints"], dtype=np.float32)
            else:
                arr = np.asarray(raw, dtype=np.float32)
    else:
        arr = np.asarray([float(x.strip()) for x in src.split(",")], dtype=np.float32)
    if arr.size != 8:
        raise ValueError(f"需要 8 个关节角，实际 {arr.size}: {source!r}")
    return arr.reshape(8)


def init_arm_from_cli(
    robot,
    arm_ctrl: "EEPresetArmController",
    *,
    arm_pose: str,
    arm_joints: str | None = None,
    manual_arm: bool = False,
) -> np.ndarray:
    """按 CLI 初始化臂：自定义关节 > manual > preset。"""
    if arm_joints:
        joints = load_arm_joints(arm_joints)
        write_arm_joints_to_sim(robot, joints)
        arm_ctrl.set_joints(joints)
        print(f"[arm] 已加载 --arm-joints: {joints}")
        return joints
    if manual_arm:
        joints = read_current_arm_joints(robot)
        arm_ctrl.set_joints(joints)
        print(f"[arm] --manual-arm: 从当前仿真关节开始，请用 I/K/J/L… 调整")
        return joints
    return apply_arm_pose_preset(robot, arm_ctrl, arm_pose)


def collect_arm_nudge_from_keyboard(
    inp,
    kb,
    key,
    *,
    step: float = 0.04,
    fine: bool = False,
) -> np.ndarray:
    """读取本帧键盘臂微调增量，返回 length-8 deltas。"""
    s = step * (0.25 if fine else 1.0)
    deltas = np.zeros(8, dtype=np.float32)

    def down(k) -> bool:
        return inp.get_keyboard_value(kb, k) > 0

    if down(key.I):
        deltas[1] += s
    if down(key.K):
        deltas[1] -= s
    if down(key.J):
        deltas[2] -= s
    if down(key.L):
        deltas[2] += s
    if down(key.U):
        deltas[4] += s
    if down(key.O):
        deltas[4] -= s
    if down(key.Z):
        deltas[0] += s
    if down(key.X):
        deltas[0] -= s
    if down(key.LEFT_BRACKET):
        deltas[3] -= s
    if down(key.RIGHT_BRACKET):
        deltas[3] += s
    if down(key.SEMICOLON):
        deltas[5] -= s
    if down(key.APOSTROPHE):
        deltas[5] += s
    return deltas


def apply_arm_keyboard_nudge(
    arm_ctrl: "EEPresetArmController",
    deltas: np.ndarray,
    *,
    overhead: bool = False,
) -> bool:
    """应用键盘增量；overhead 时 j3 缩放与采集脚本一致。"""
    if not np.any(np.abs(deltas) > 1e-9):
        return False
    d = deltas.copy()
    if overhead:
        d[2] *= -0.4
    arm_ctrl.nudge_joints(d)
    return True


def print_arm_tune_report(
    arm_ctrl: "EEPresetArmController",
    ee_cam=None,
    *,
    save_path: str | Path | None = None,
) -> Path | None:
    """打印当前关节角；可选保存 JSON 供 --arm-joints 加载。"""
    import json
    from pathlib import Path

    joints = arm_ctrl.target_joints
    print("[arm-tune] 当前 8 关节 (rad):")
    print(f"  {np.array2string(joints, precision=4, separator=', ')}")
    print("[arm-tune] 可复制到 ee_collect_utils.py → B2_PIPER_STOWED_JOINTS:")
    print(f"  np.array({joints.tolist()}, dtype=np.float32)")
    payload: dict = {"joints": joints.tolist()}
    if ee_cam is not None:
        payload["view"] = {
            "elevation_deg": ee_camera_elevation_deg(ee_cam),
            "down_angle_deg": ee_camera_down_angle_deg(ee_cam),
        }
        print(f"[arm-tune] EE 视角: {format_ee_camera_view(ee_cam)}")
    saved: Path | None = None
    if save_path is not None:
        saved = Path(save_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[arm-tune] 已保存 → {saved}")
        print(f"[arm-tune] 下次加载: --arm-joints {saved}")
    return saved


@dataclass
class EETopdownTarget:
    pos_b: np.ndarray
    quat_wxyz: np.ndarray = field(default_factory=lambda: EE_TOPDOWN_QUAT_WXYZ.copy())

    def as_tensors(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.tensor(self.pos_b, dtype=torch.float32, device=device).view(1, 3)
        quat = torch.tensor(self.quat_wxyz, dtype=torch.float32, device=device).view(1, 4)
        return pos, quat


class EEPresetArmController:
    """关节角预设：臂前伸到狗前方，避免停在背上拍狗皮。"""

    ARM_IDX = list(range(12, 20))
    ACTION_SCALE = 0.5

    def __init__(self, robot, device: str, joint_targets: np.ndarray | None = None):
        self.robot = robot
        self.device = device
        self.arm_ids, arm_names = robot.find_joints(B2_PIPER_ARM_JOINTS)
        if len(self.arm_ids) != 8:
            raise RuntimeError(f"期望 8 个臂关节，实际 {arm_names}")
        self._default_joint_pos = robot.data.default_joint_pos[0].clone()
        self._target = (
            joint_targets.copy() if joint_targets is not None else B2_PIPER_STOWED_JOINTS.copy()
        )

    @property
    def target_joints(self) -> np.ndarray:
        return self._target.copy()

    def set_joints(self, joints: np.ndarray) -> None:
        self._target = np.asarray(joints, dtype=np.float32).copy()

    def nudge(self, d_j2: float = 0.0, d_j3: float = 0.0, d_j5: float = 0.0, d_j1: float = 0.0) -> None:
        self._target[0] += d_j1
        self._target[1] += d_j2
        self._target[2] += d_j3
        self._target[4] += d_j5

    def nudge_joint(self, joint_index: int, delta: float) -> None:
        if 0 <= joint_index < len(self._target):
            self._target[joint_index] += delta

    def nudge_joints(self, deltas: np.ndarray) -> None:
        for i, d in enumerate(np.asarray(deltas, dtype=np.float32).reshape(-1)):
            if i < len(self._target) and abs(float(d)) > 1e-9:
                self._target[i] += float(d)

    def compute_arm_action(self) -> torch.Tensor:
        full = self._default_joint_pos.clone()
        for i, jid in enumerate(self.arm_ids):
            full[jid] = float(self._target[i])
        delta = (full - self._default_joint_pos) / self.ACTION_SCALE
        return delta[self.ARM_IDX]


class EEIkArmController:
    """笛卡尔 IK（备用）。目标必须在狗前方且低于背部。"""

    ARM_IDX = list(range(12, 20))
    ACTION_SCALE = 0.5

    def __init__(self, robot, device: str, target: EETopdownTarget):
        from atec_rl_lab.utils.cartesian_controller import CartesianController

        self.robot = robot
        self.device = device
        self.target = target
        self.arm_ids, _ = robot.find_joints(B2_PIPER_ARM_IK_JOINTS)
        self.gripper_ids, _ = robot.find_joints(B2_PIPER_GRIPPER_JOINTS)
        self.ik = CartesianController(
            robot=robot,
            ee_body_name="gripper_base",
            arm_joint_names=B2_PIPER_ARM_IK_JOINTS,
            num_envs=1,
            device=device,
            command_type="pose",
            max_joint_delta=0.12,
        )
        self.ik.reset()
        self._default_joint_pos = robot.data.default_joint_pos[0].clone()

    def nudge_target(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.target.pos_b = self.target.pos_b + np.array([dx, dy, dz], dtype=np.float32)

    def compute_arm_action(self) -> torch.Tensor:
        pos_b, quat_b = self.target.as_tensors(self.device)
        arm_jpos = self.ik.compute_base(pos_b, quat_b).squeeze(0)
        full = self.robot.data.joint_pos[0].clone()
        for i, jid in enumerate(self.arm_ids):
            full[jid] = arm_jpos[i]
        for gid in self.gripper_ids:
            full[gid] = self._default_joint_pos[gid]
        delta = (full - self._default_joint_pos) / self.ACTION_SCALE
        return delta[self.ARM_IDX]
