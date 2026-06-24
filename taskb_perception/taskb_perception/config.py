"""Task B 感知默认配置（默认 B2+Piper，可在 PerceptionModule 构造时覆盖）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Task B 环境常量（与 env_cfg.py 一致）
TARGET_BIN_XY = (-3.0, -10.0)
TARGET_BIN_RADIUS = 1.0
NUM_OBJECTS = 18

# 抓取几何（与 Task E state_machine 对齐，可按实际调）
PRE_GRASP_CLEARANCE = 0.12
GRASP_Z_OFFSET = 0.09
DEFAULT_TOPDOWN_QUAT_W = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # wxyz

# 相机内参 — 来自 envs_base_cfg head_camera / ee_camera
# fx = focal_length / horizontal_aperture * width
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FX = 24.0 / 20.955 * CAM_WIDTH   # ≈ 732.5
CAM_FY = CAM_FX
CAM_CX = CAM_WIDTH / 2.0
CAM_CY = CAM_HEIGHT / 2.0

# 深度有效范围 (m)
DEPTH_MIN = 0.05
DEPTH_MAX = 8.0


@dataclass
class CameraExtrinsic:
    """相机相对 robot base 的外参（pos + quat wxyz）。"""

    pos_b: np.ndarray
    quat_b: np.ndarray  # camera orientation in base frame, wxyz


@dataclass
class RobotProfile:
    name: str
    head_cam: CameraExtrinsic
    ee_cam: CameraExtrinsic | None = None


def _quat_from_euler_xyz(x: float, y: float, z: float) -> np.ndarray:
    """scipy 等价：intrinsic xyz euler → wxyz。"""
    cx, sx = np.cos(x / 2), np.sin(x / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cz, sz = np.cos(z / 2), np.sin(z / 2)
    # xyz intrinsic
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return np.array([qw, qx, qy, qz], dtype=np.float32)


# B2+Piper：来自 source/atec_rl_lab/.../robots/b2.py
B2_PIPER_PROFILE = RobotProfile(
    name="b2_piper",
    head_cam=CameraExtrinsic(
        pos_b=np.array([0.4216, 0.0250, 0.0619], dtype=np.float32),
        quat_b=_quat_from_euler_xyz(0.0, np.pi / 6, 0.0),
    ),
    ee_cam=CameraExtrinsic(
        pos_b=np.array([-0.05, 0.0, 0.06], dtype=np.float32),
        # gripper 上相机随臂动 — 这里只是近似，精对齐请用 ee 检测分支
        quat_b=_quat_from_euler_xyz(0.0, 0.0, -np.pi / 2),
    ),
)


# 训练产出（相对仓库根目录）— 不要带仓库目录名前缀，避免重复拼接 repo_root
WEIGHT_HEAD_DET = "taskb_perception/runs/train_fast/head_det/weights/best.pt"
WEIGHT_EE_DET = "taskb_perception/runs/train_fast/ee_det/weights/best.pt"
WEIGHT_EE_SEG = "taskb_perception/ee_seg/ee_seg/weights/best.pt"


@dataclass
class PerceptionConfig:
    robot: RobotProfile = field(default_factory=lambda: B2_PIPER_PROFILE)

    # YOLO 权重（相对仓库根目录）
    yolo_head_weights: str | None = WEIGHT_HEAD_DET   # head_rgb 导航/远距离
    yolo_ee_det_weights: str | None = WEIGHT_EE_DET   # ee_rgb 远距离检索
    yolo_ee_seg_weights: str | None = WEIGHT_EE_SEG   # ee 垂直俯视分割/对准
    yolo_weights: str | None = WEIGHT_HEAD_DET        # 兼容旧字段 = head
    conf_threshold: float = 0.35
    use_color_fallback: bool = True    # 没 YOLO 权重时用颜色+深度 blob

    # 跟踪 / 滤波
    ema_alpha: float = 0.35          # 越大越跟新观测
    max_missed_frames: int = 15      # 约 0.3s @ 50Hz
    match_dist_thresh: float = 0.25  # 3D 关联阈值 (m)
    min_confidence: float = 0.30

    # 锁定
    lock_on_request: bool = True

    # 相机选择距离阈值 (m, base 系 xy 距离)
    use_ee_refine_dist: float = 1.0

    # 导航检测来源：head / ee / both（默认仅 ee，避免 head/ee 跨视角 ID 对齐）
    nav_detect_source: str = "ee"
    enable_ee_nav_detect: bool = True

    # EE 导航目标锁：首帧按置信度锁定，跟踪同一框，完成后再锁下一个
    nav_lock_enabled: bool = True
    nav_lock_auto_acquire: bool = True          # 无锁时自动锁当前最高 conf
    nav_lock_auto_next: bool = True             # 到达 depth 阈值后自动完成并锁下一个
    nav_lock_match_max_dist_px: float = 140.0   # 帧间质心最大跳变（像素）
    nav_lock_match_min_score: float = 0.25      # IoU+距离 综合匹配阈值
    nav_lock_max_missed: int = 45               # 连续丢失帧数后放弃当前锁
    nav_lock_exclude_dist_px: float = 48.0        # 已完成目标排除半径（像素）
    nav_lock_exclude_min_iou: float = 0.35      # 已完成目标排除 IoU

    # 导航方式：axis_align=光轴中线对准+直行 | pos_b_3d=depth反算 base（旧）
    nav_method: str = "axis_align"
    nav_axis_target_u: float = CAM_WIDTH / 2.0
    nav_axis_target_v: float = CAM_HEIGHT / 2.0
    nav_axis_tol_u_px: float = 12.0
    nav_axis_turn_gain: float = 0.022  # wz ≈ -gain * err_u (rad/s per px)
    nav_axis_max_wz: float = 0.65
    nav_axis_forward_vx: float = 0.45
    nav_axis_slow_vx: float = 0.10
    nav_axis_align_before_forward: bool = True
    nav_axis_arrive_depth_m: float = 1.25  # 仅 depth 停车，不反算 XY

    # EE 垂直 seg 对准：质心应对齐的像素点（默认图像中心，可标定）
    ee_align_target_u: float = CAM_WIDTH / 2.0
    ee_align_target_v: float = CAM_HEIGHT / 2.0
    ee_align_tol_px: float = 8.0
    ee_align_gain_m_per_px: float = 0.0004

    # YOLO class_id -> "sugar" | "mustard" | "banana"
    class_names: dict[int, str] = field(default_factory=lambda: {
        0: "sugar",
        1: "mustard",
        2: "banana",
    })
