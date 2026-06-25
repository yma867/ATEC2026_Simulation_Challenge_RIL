"""
Task B 操作层 — 感知 taskb_perception + RL腿控 + YOLO导航

状态机
  SEARCH → APPROACH → CROUCHING → PREGRASP → GRASP_ARM → STAND_UP → CARRY → DROP → SEARCH

运行
  python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import torch

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

try:
    from arm_grasp import ArmGraspController
except ImportError:
    from demo.arm_grasp import ArmGraspController

try:
    from solution_gt import B2_PIPER_LEG_JOINT_NAMES, B2PiperActor, LegPostureController, rgb_to_bgr_uint8
except ImportError:
    from demo.solution_gt import B2_PIPER_LEG_JOINT_NAMES, B2PiperActor, LegPostureController, rgb_to_bgr_uint8

_REPO_ROOT = os.path.dirname(_DEMO_DIR)
_PERCEPTION_DIR = os.path.join(_REPO_ROOT, "taskb_perception")
if os.path.isdir(_PERCEPTION_DIR) and _PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, _PERCEPTION_DIR)

from taskb_perception.config import CAM_CX, CAM_CY, CAM_FX, CAM_FY, TARGET_BIN_RADIUS as BIN_RADIUS, TARGET_BIN_XY  # noqa: E402
from taskb_perception import AxisNavController, PerceptionConfig  # noqa: E402
from taskb_perception.obs_utils import parse_depth, pixel_to_cam, sample_depth_median  # noqa: E402  # noqa: E402
from taskb_perception.types import ObjectClass  # noqa: E402
from taskb_perception.math3d import quat_multiply, quat_rotate_vector, transform_point_cam_to_base  # noqa: E402
from taskb_perception.yolo_labels import CLASS_NAMES, NUM_OBJECTS, object_index_to_class  # noqa: E402

BIN_CENTER = np.array([TARGET_BIN_XY[0], TARGET_BIN_XY[1], 0.0], dtype=np.float32)

# 视角跟随开关
# True  = 视角跟随机器人移动（锁定）
# False = 视角自由，用户可用鼠标拖动 / WASD 移动（推荐调试时关闭）
# 环境变量: ATEC_TASKB_CAMERA_FOLLOW=1 或 0
ATEC_CAMERA_FOLLOW_ROBOT = os.getenv("ATEC_TASKB_CAMERA_FOLLOW", "0").lower() in {
    "1", "true", "yes", "on",
}

try:
    from rgbd_pure_dual_pipeline import RgbdPureDualPipeline  # noqa: E402
except ImportError:
    RgbdPureDualPipeline = None


class _StreamToLogger:
    def __init__(self, log_fp, fallback_stream):
        self._log_fp = log_fp
        self._fallback_stream = fallback_stream

    def write(self, data):
        if not data:
            return 0
        try:
            written = self._log_fp.write(data)
            self._log_fp.flush()
            return written
        except Exception:
            return self._fallback_stream.write(data)

    def flush(self):
        try:
            self._log_fp.flush()
        except Exception:
            self._fallback_stream.flush()

    def isatty(self):
        return False


class AlgSolution:
    ACTION_SCALE = 0.5
    LEG_JOINT_NAMES = list(B2_PIPER_LEG_JOINT_NAMES)
    ARM_JOINT_NAMES = [
        "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
        "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8",
    ]
    EE_BODY_NAME = "gripper_base"
    CROUCH_ARM_JOINT2_POS = 3.14
    CROUCH_ARM_JOINT3_POS = -1.8
    CROUCH_ARM_JOINT5_POS = -0.8
    PRE_CROUCH_ARM_MAX_STEP = 0.04
    PRE_CROUCH_ARM_READY_RAD = 0.20
    PRE_CROUCH_ARM_TIMEOUT_STEPS = 40
    ARM_CROUCH_ALPHA_STEP = 0.08
    PREGRASP_ARM_SETTLE_STEPS = 15
    PREGRASP_ARM_TIMEOUT_STEPS = 60
    PREGRASP_HOLD_TIMEOUT_STEPS = 70
    PRE_CROUCH_SETTLE_MIN_STEPS = 12
    PRE_CROUCH_SETTLE_TIMEOUT_STEPS = 20
    PRE_CROUCH_SETTLE_YAW_RATE_THRESH = 0.12
    PREGRASP_REACH_STABLE_STEPS = 5
    ARM_DEFAULT_POS = {
        "arm_joint1": 0.0,
        "arm_joint2": 2.13,
        "arm_joint3": -1.20,
        "arm_joint4": 0.0,
        "arm_joint5": -0.4, #-0.8
        "arm_joint6": 0.0,
        "arm_joint7": 0.0,
        "arm_joint8": 0.0,
    }

    GRASP_DEPTH_M = 1.10
    WARMUP_STEPS = 90
    SEARCH_VX = 0.5
    SEARCH_WZ = 0.35
    YAW_TURN_THRESH = 0.45
    NAV_WZ_GAIN = 0.85
    NAV_WZ_MAX = 0.25
    NAV_VX_MIN = 0.2
    NAV_VX_MAX = 0.8
    BIN_ARRIVE_M = 1.05

    DROP_OVER_Z = 0.48
    DROP_RELEASE_Z = 0.20
    DROP_RETREAT_Z = 0.58
    DROP_OPEN_STEPS = 40
    _shared_log_file_path: str | None = None
    _shared_log_fp = None
    _stdout_proxy = None
    _stderr_proxy = None

    def __init__(self, env=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = 0.02
        self._log_file_path = self._init_logging()

        policy_path = self._resolve_policy_path()
        self.policy = self._load_leg_policy(policy_path)
        self._leg_mode = "rl" if self.policy is not None else "scripted"

        self.nav_cfg = PerceptionConfig(
            nav_method="axis_align",
            nav_detect_source="both",
            enable_ee_nav_detect=True,
            nav_lock_enabled=True,
            nav_lock_auto_acquire=True,
            conf_threshold=0.35,
            use_color_fallback=True,
        )
        self.axis_nav = AxisNavController(self.nav_cfg)
        self._last_nav_vel = np.zeros(3, dtype=np.float32)
        print("[TaskB-RL] ✓ YOLO 光轴导航已启用 (axis_align + Memory Bank)", flush=True)

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.leg_action_dim = len(self.leg_joint_indices)
        self.total_action_dim = len(self.leg_joint_indices) + len(self.arm_joint_indices)
        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5] * 4, device=self.device, dtype=torch.float32,
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0] * 4, device=self.device, dtype=torch.float32,
        ).view(1, -1)
        self._velocity_commands = torch.tensor([[0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
        arm_default_pos = [self.ARM_DEFAULT_POS.get(name, 0.0) for name in self.ARM_JOINT_NAMES]
        self._arm_default_action = torch.tensor(
            arm_default_pos, device=self.device, dtype=torch.float32,
        ).view(1, -1)

        self.control_cfg = SimpleNamespace(
            use_squat_test=False,
            use_effort_leg_control=False,
            squat_transition_time=float(os.getenv("ATEC_TASKB_SQUAT_TRANSITION_TIME", "3.0")),
            debug_interval=max(int(os.getenv("ATEC_TASKB_SQUAT_DEBUG_INTERVAL", "10")), 1),
            max_squat_action_delta=float(os.getenv("ATEC_TASKB_MAX_SQUAT_ACTION_DELTA", "0.08")),
        )
        self._leg_posture_controller = LegPostureController(
            leg_joint_names=list(self.LEG_JOINT_NAMES),
            crouch_drop_height=float(os.getenv("ATEC_TASKB_CROUCH_DROP_HEIGHT", "0.10")),
            crouch_duration=float(os.getenv("ATEC_TASKB_CROUCH_DURATION", "2.0")),
            stand_up_duration=float(os.getenv("ATEC_TASKB_STAND_UP_DURATION", "2.0")),
            foot_pos_tol=float(os.getenv("ATEC_TASKB_CROUCH_FOOT_TOL", "0.1")),
            body_height_tol=float(os.getenv("ATEC_TASKB_CROUCH_HEIGHT_TOL", "0.02")),
            ik_damping=float(os.getenv("ATEC_TASKB_CROUCH_IK_DAMPING", "0.1")),
            max_joint_step=float(os.getenv("ATEC_TASKB_CROUCH_MAX_JOINT_STEP", "0.08")),
        )
        self._leg_control_initialized = False
        self._leg_joint_ids = None
        self._leg_joint_names_in_robot = None
        self._leg_default_dof_pos = None
        self._leg_joint_pos_limits = None
        self._leg_torque_limits = None
        self._leg_p_gains = None
        self._leg_d_gains = None
        self._env_leg_to_robot_indices = None
        self._robot_leg_to_env_indices = None
        self._last_leg_action_override = None

        self.leg_actor_obs_dim = None
        self.sit_down_min_steps = max(1, int(os.getenv("ATEC_TASKB_SIT_DOWN_MIN_STEPS", "30")))
        self.sit_down_stable_steps_required = max(1, int(os.getenv("ATEC_TASKB_SIT_DOWN_STABLE_STEPS", "15")))
        self.SIT_DOWN_TIMEOUT_STEPS = max(1, int(os.getenv("ATEC_TASKB_SIT_DOWN_TIMEOUT", "200")))
        self.sit_down_roll_pitch_thresh = float(os.getenv("ATEC_TASKB_SIT_DOWN_RP_THRESH", "0.18"))
        self.sit_down_height_vel_thresh = float(os.getenv("ATEC_TASKB_SIT_DOWN_ZVEL_THRESH", "0.15"))
        self.sit_down_ang_vel_thresh = float(os.getenv("ATEC_TASKB_SIT_DOWN_ANGVEL_THRESH", "0.4"))
        self._sit_down_step_count = 0
        self._sit_down_stable_count = 0
        self.sit_down_actor = None
        self.sit_down_actor_obs_dim = None
        self._load_sit_down_actor_model()

        self._task_state = "SEARCH"
        self._step = 0
        self._arm_grasp: ArmGraspController | None = None
        self._camera_follow_enabled = ATEC_CAMERA_FOLLOW_ROBOT

        # 速度平滑状态变量
        self._cmd_vx_filt = 0.0
        self._cmd_wz_filt = 0.0
        self.CMD_ALPHA = 0.25
        self.MAX_DVX = 0.1
        self.MAX_DWZ = 0.1

        self._hold_carry_pose = False
        self._carry_arm_jpos: torch.Tensor | None = None
        self._carry_gripper: torch.Tensor | None = None
        self._pending_grasp_status: str | None = None
        self._pending_grasp_target: dict[str, Any] | None = None
        self._pending_grasp_pos_w: np.ndarray | None = None
        self._pending_grasp_quat_w: np.ndarray | None = None
        self._crouch_arm_hold_jpos: torch.Tensor | None = None
        self._arm_crouch_alpha: float = 0.0
        self._pre_crouch_settle_steps = 0
        self._pregrasp_reach_stable_count = 0
        self._pregrasp_arm_settle_count = 0
        self._pregrasp_arm_step = 0
        self._pregrasp_arm_preset_done = False
        self._pregrasp_ee_quat_w = None

        self._drop_phase = ""
        self._drop_wait = 0
        self._objects_dropped = 0
        self._post_drop_stand_until = 0
        self._last_score = 0.0

        self._camera_debug_interval = 5
        self._camera_debug_enabled = True
        self._camera_debug_warned_keys: set[str] = set()
        self._camera_debug_depth_keys = ("distance_to_image_plane", "distance_to_camera", "depth")

        self._yolo_log_enabled = os.getenv("ATEC_TASKB_YOLO_LOG", "1").lower() in {"1", "true", "yes", "on"}
        self._yolo_log_interval = max(1, int(os.getenv("ATEC_TASKB_YOLO_LOG_INTERVAL", "20")))
        self._yolo_log_dir = None
        self._yolo_log_json_path = None
        self._yolo_detection_records: list[dict] = []
        if self._yolo_log_enabled:
            self._init_yolo_log_dir()

        # YOLO target memory
        self._last_seen_target = None
        self._last_seen_step = -1
        self._lost_target_step = -1

        # Active visual target tracking
        self._active_target = None
        self._processed_targets = []
        self._target_stage = "NONE"  # NONE / EE_TRACK / HEAD_APPROACH / EE_BASE_ALIGN / PRE_CROUCH
        self._head_confirm_count = 0
        self._pre_crouch_confirm_count = 0
        self._pre_crouch_wait_steps = 0
        self._target_lost_count = 0
        self._candidate_target = None
        self._candidate_seen_count = 0
        self._search_no_target_count = 0
        self.SEARCH_SPIN_AFTER_NO_TARGET_STEPS = 5
        self.SEARCH_SPIN_WZ = 0.4
        self.ACTIVE_TARGET_ACQUIRE_STEPS = 3
        self.TARGET_LOST_KEEP_STEPS = 35
        self.TARGET_MAX_DEPTH_JUMP_M = 0.8
        self.TARGET_MAX_ERRU_JUMP_PX = 160.0
        self.TARGET_AREA_RATIO_MIN = 0.4
        self.TARGET_AREA_RATIO_MAX = 2.8
        self.HANDOVER_DEPTH_M = 0.4
        self.HEAD_PRE_CROUCH_DEPTH_M = 0.6
        self.HEAD_PRE_CROUCH_DEPTH_EPS_M = 0.05
        self.HEAD_LOST_DIRECT_CROUCH_DEPTH_M = 0.7
        self.HEAD_LOST_DIRECT_CROUCH_STEPS = 6
        self.HEAD_APPROACH_LOST_GIVEUP_STEPS = 40
        self.EE_TRACK_LOST_GIVEUP_STEPS = 80
        self.HEAD_PRE_CROUCH_ERR_PX = 45.0
        self.FINAL_APPROACH_TRIGGER_M = 0.8
        self.FINAL_CROUCH_DEPTH_M = 0.8
        self.FINAL_APPROACH_KEEP_STEPS = 8
        self.HEAD_CONFIRM_STEPS = 5
        self.PRE_CROUCH_CONFIRM_STEPS = 7
        self._verbose_reject_logs = os.getenv("ATEC_TASKB_VERBOSE_REJECT_LOGS", "0").lower() in {"1", "true", "yes", "on"}
        self._reject_log_last_step: dict[tuple[str, str, str], int] = {}
        self._final_approach_vx = 0.0
        self._final_approach_wz = 0.0

        # pregrasp / crouch waiting
        self._entered_crouch_from_yolo = False
        self.WAIT_GRASP_TIMEOUT_STEPS = 200
        self._crouch_wait_start_step = -1
        self._ee_align_step = 0
        self._ee_align_max_steps = 120
        self._ee_align_gain_m_per_px = 0.0004

        print(
            f"[TaskB-RL] full loop: search→crouch→grasp→stand→carry→drop | "
            f"bin={BIN_CENTER.tolist()} r={BIN_RADIUS} leg={self._leg_mode}",
            flush=True,
        )
        if self._leg_mode == "scripted":
            print("[TaskB-RL] *** leg=scripted 站不稳! 请确保 demo/policy.pt 存在且非 LFS 指针 ***", flush=True)

    def _init_logging(self) -> str:
        if self.__class__._shared_log_file_path is not None and self.__class__._shared_log_fp is not None:
            return self.__class__._shared_log_file_path

        log_dir = os.path.join(_REPO_ROOT, "logs", "solution_rl")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = os.path.join(log_dir, f"solution_rl_{timestamp}.log")

        terminal_stream = getattr(sys, "__stdout__", sys.stdout)
        terminal_stream.write(f"Log file: {log_file_path}\n")
        terminal_stream.flush()

        log_fp = open(log_file_path, "a", encoding="utf-8", buffering=1)
        self.__class__._shared_log_file_path = log_file_path
        self.__class__._shared_log_fp = log_fp
        self.__class__._stdout_proxy = _StreamToLogger(log_fp, terminal_stream)
        self.__class__._stderr_proxy = _StreamToLogger(log_fp, getattr(sys, "__stderr__", sys.stderr))
        sys.stdout = self.__class__._stdout_proxy
        sys.stderr = self.__class__._stderr_proxy
        return log_file_path

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def _resolve_policy_path(self) -> str:
        candidates = [
            os.path.join(_DEMO_DIR, "model_4999.pt"),
            os.path.join(_REPO_ROOT, "logs", "rsl_rl", "unitree_b2_piper_flat", "2026-06-02_14-40-32", "model_4999.pt"),
            os.path.join(_REPO_ROOT, "atec_robot_model", "baseline", "unitree_b2_flat", "policy.pt"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def _load_leg_policy(self, policy_path: str):
        if not os.path.isfile(policy_path):
            print(f"[TaskB-RL] WARN: 无 policy checkpoint，腿用简易步态", flush=True)
            return None
        size = os.path.getsize(policy_path)
        try:
            checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
            if "model_state_dict" not in checkpoint:
                print(f"[TaskB-RL] ERROR: checkpoint 格式不正确，缺少 model_state_dict", flush=True)
                return None
            state_dict = checkpoint["model_state_dict"]
            actor_input_dim = state_dict["actor.0.weight"].shape[1]
            actor_output_dim = state_dict["actor.6.bias"].shape[0]
            actor = B2PiperActor(actor_input_dim, actor_output_dim).to(self.device)
            actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
            actor.load_state_dict(actor_state, strict=True)
            actor.eval()
            self.leg_actor_obs_dim = actor_input_dim
            print(
                f"[TaskB-RL] loaded policy from {policy_path} ({size // 1024} KB)"
                f" input={actor_input_dim}, output={actor_output_dim}",
                flush=True,
            )
            return actor
        except Exception as exc:
            self.leg_actor_obs_dim = None
            print(
                f"[TaskB-RL] ERROR: policy 无法加载 ({exc})\n"
                f"  路径: {policy_path}  大小: {size} bytes",
                flush=True,
            )
            return None

    def _load_sit_down_actor_model(self) -> None:
        checkpoint_path = os.path.join(_DEMO_DIR, "sit_down.pt")
        if not os.path.exists(checkpoint_path):
            print(f"[TaskB-RL] WARN: sit-down checkpoint not found: {checkpoint_path}", flush=True)
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            actor_input_dim = state_dict["actor.0.weight"].shape[1]
            actor_output_dim = state_dict["actor.6.bias"].shape[0]
            self.sit_down_actor = B2PiperActor(actor_input_dim, actor_output_dim).to(self.device)
            actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
            self.sit_down_actor.load_state_dict(actor_state, strict=True)
            self.sit_down_actor.eval()
            self.sit_down_actor_obs_dim = actor_input_dim
            print(f"[TaskB-RL] sit-down actor loaded: input={actor_input_dim}, output={actor_output_dim}", flush=True)
        except Exception as exc:
            print(f"[TaskB-RL] WARN: failed to load sit-down actor: {exc}", flush=True)
            self.sit_down_actor = None
            self.sit_down_actor_obs_dim = None

    def _scripted_leg_action(self, action_dim: int) -> torch.Tensor:
        vx = float(self._velocity_commands[0, 0].item())
        wz = float(self._velocity_commands[0, 2].item())
        a = torch.zeros(1, action_dim, device=self.device, dtype=torch.float32)
        t = self._step * self.dt
        amp = 0.35 if vx < 0.05 and abs(wz) < 0.05 else 1.0
        s1, s2 = amp * math.sin(t * 3.0), amp * math.sin(t * 3.0 + math.pi)

        def leg(ih: int, it: int, ic: int, s: float) -> None:
            a[0, ih] = 0.25 * s
            a[0, it] = 0.55 + 0.45 * s
            a[0, ic] = -1.35 - 0.35 * abs(s)

        leg(0, 1, 2, s1)
        leg(3, 4, 5, s2)
        leg(6, 7, 8, s2)
        leg(9, 10, 11, s1)

        if wz > 0.08:
            a[0, 0] += 0.35
            a[0, 6] += 0.35
            a[0, 3] -= 0.35
            a[0, 9] -= 0.35
        elif wz < -0.08:
            a[0, 0] -= 0.35
            a[0, 6] -= 0.35
            a[0, 3] += 0.35
            a[0, 9] += 0.35
        if vx < 0.08:
            a[:, :12] *= 0.5
        a[:, self.arm_joint_indices] = self._arm_default_action
        return a

    def _init_yolo_log_dir(self) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._yolo_log_dir = os.path.join(_REPO_ROOT, "logs", "yolo_detection", timestamp)
        os.makedirs(self._yolo_log_dir, exist_ok=True)
        self._yolo_log_json_path = os.path.join(self._yolo_log_dir, "detections.json")
        self._yolo_detection_records = []
        print(f"[TaskB-RL] YOLO detection log dir: {self._yolo_log_dir}", flush=True)

    def _save_yolo_detection(self, obs: dict) -> None:
        if not self._yolo_log_enabled or self._yolo_log_dir is None:
            return
        if self._step % self._yolo_log_interval != 0:
            return
        if cv2 is None:
            return

        targets = self.axis_nav.last_targets
        nav_target = self.axis_nav.nav_target
        if not targets and nav_target is None:
            return

        frame_dir = os.path.join(self._yolo_log_dir, f"frame_{self._step:06d}")
        os.makedirs(frame_dir, exist_ok=True)

        scene = self._scene()
        ee_camera = self._get_scene_camera(scene, "ee_camera")
        head_camera = self._get_scene_camera(scene, "head_camera")
        ee_rgb = self._get_camera_output(ee_camera, "ee_camera", "rgb") if ee_camera is not None else None
        head_rgb = self._get_camera_output(head_camera, "head_camera", "rgb") if head_camera is not None else None

        detection_info = {
            "step": self._step,
            "timestamp": time.time(),
            "task_state": self._task_state,
            "targets_count": len(targets),
            "targets": [],
            "nav_target": None,
        }

        for t in targets:
            detection_info["targets"].append({
                "u": float(t.u),
                "v": float(t.v),
                "bbox": list(t.bbox),
                "obj_class": t.obj_class,
                "confidence": float(t.confidence),
                "source": t.source,
                "err_u": float(t.err_u),
                "err_v": float(t.err_v),
                "on_axis": t.on_axis,
                "depth_m": float(t.depth_m),
                "locked": t.locked,
                "lock_id": t.lock_id,
            })

        if nav_target is not None:
            detection_info["nav_target"] = {
                "u": float(nav_target.u),
                "v": float(nav_target.v),
                "bbox": list(nav_target.bbox),
                "obj_class": nav_target.obj_class,
                "confidence": float(nav_target.confidence),
                "source": nav_target.source,
                "err_u": float(nav_target.err_u),
                "on_axis": nav_target.on_axis,
                "depth_m": float(nav_target.depth_m),
                "locked": nav_target.locked,
                "lock_id": nav_target.lock_id,
            }

        if ee_rgb is not None:
            ee_bgr = np.ascontiguousarray(rgb_to_bgr_uint8(ee_rgb))
            if targets or nav_target:
                for t in targets:
                    if t.source == "ee":
                        x1, y1, x2, y2 = t.bbox
                        color = (0, 255, 0) if t.locked else (255, 0, 0)
                        cv2.rectangle(ee_bgr, (x1, y1), (x2, y2), color, 2)
                        label = f"{t.obj_class}:{t.confidence:.2f}"
                        cv2.putText(ee_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.circle(ee_bgr, (int(t.u), int(t.v)), 5, (0, 0, 255), -1)
                if nav_target is not None and nav_target.source == "ee":
                    x1, y1, x2, y2 = nav_target.bbox
                    cv2.rectangle(ee_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.circle(ee_bgr, (int(nav_target.u), int(nav_target.v)), 8, (255, 255, 0), -1)
            cv2.imwrite(os.path.join(frame_dir, "ee_rgb.jpg"), ee_bgr)

        if head_rgb is not None:
            head_bgr = np.ascontiguousarray(rgb_to_bgr_uint8(head_rgb))
            for t in targets:
                if t.source == "head":
                    x1, y1, x2, y2 = t.bbox
                    cv2.rectangle(head_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{t.obj_class}:{t.confidence:.2f}"
                    cv2.putText(head_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(frame_dir, "head_rgb.jpg"), head_bgr)

        self._yolo_detection_records.append(detection_info)
        with open(self._yolo_log_json_path, "w") as f:
            json.dump(self._yolo_detection_records, f, indent=2)

    @property
    def camera_follow_enabled(self) -> bool:
        return self._camera_follow_enabled

    @camera_follow_enabled.setter
    def camera_follow_enabled(self, value: bool) -> None:
        self._camera_follow_enabled = bool(value)

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return None

    def reset(self) -> None:
        if self.perception is not None:
            self.perception.reset()
        self.axis_nav.reset()
        self._task_state = "SEARCH"
        self._step = 0
        self._hold_carry_pose = False
        self._carry_arm_jpos = None
        self._carry_gripper = None
        self._pending_grasp_status = None
        self._pending_grasp_target = None
        self._pending_grasp_pos_w = None
        self._pending_grasp_quat_w = None
        self._crouch_arm_hold_jpos = None
        self._arm_crouch_alpha = 0.0
        self._pre_crouch_settle_steps = 0
        self._pregrasp_reach_stable_count = 0
        self._pregrasp_arm_settle_count = 0
        self._pregrasp_arm_step = 0
        self._pregrasp_arm_preset_done = False
        self._pregrasp_ee_quat_w = None
        self._drop_phase = ""
        self._drop_wait = 0
        self._post_drop_stand_until = 0
        self._crouch_wait_start_step = -1
        self._reset_sit_down_tracking()
        self._leg_posture_controller.reset()
        if self._arm_grasp is not None:
            self._arm_grasp.reset()

        # Reset target memory
        self._last_seen_target = None
        self._last_seen_step = -1
        self._lost_target_step = -1
        self._active_target = None
        self._processed_targets = []
        self._target_stage = "NONE"
        self._head_confirm_count = 0
        self._pre_crouch_confirm_count = 0
        self._pre_crouch_wait_steps = 0
        self._target_lost_count = 0
        self._candidate_target = None
        self._candidate_seen_count = 0
        self._search_no_target_count = 0
        self._entered_crouch_from_yolo = False
        self._reject_log_last_step = {}
        self._final_approach_vx = 0.0
        self._final_approach_wz = 0.0

    def _clear_target_lock(self) -> None:
        self._last_seen_target = None
        self._last_seen_step = -1
        self._lost_target_step = -1
        self._clear_active_target()
        self._final_approach_vx = 0.0
        self._final_approach_wz = 0.0
        self._pre_crouch_wait_steps = 0
        self._pre_crouch_settle_steps = 0

    @staticmethod
    def _bbox_area(t) -> float:
        bbox = getattr(t, "bbox", None)
        if bbox is None:
            bbox = t.get("bbox") if isinstance(t, dict) else None
        if bbox is None or len(bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    def _target_snapshot(self, t) -> dict[str, Any]:
        if isinstance(t, dict):
            obj_class = str(t.get("obj_class", "unknown"))
            bbox = tuple(int(v) for v in t.get("bbox", (0, 0, 0, 0)))
            depth = float(t.get("depth", t.get("depth_m", 0.0)))
            err_u = float(t.get("err_u", 0.0))
            source = str(t.get("source", ""))
            u = float(t.get("u", 0.0))
            v = float(t.get("v", 0.0))
            confidence = float(t.get("confidence", 0.0))
        else:
            obj_class = str(getattr(t, "obj_class", "unknown"))
            bbox = tuple(int(v) for v in getattr(t, "bbox", (0, 0, 0, 0)))
            depth = float(getattr(t, "depth_m", 0.0))
            err_u = float(getattr(t, "err_u", 0.0))
            source = str(getattr(t, "source", ""))
            u = float(getattr(t, "u", 0.0))
            v = float(getattr(t, "v", 0.0))
            confidence = float(getattr(t, "confidence", 0.0))
        return {
            "class_hist": {obj_class: 1},
            "dominant_class": obj_class,
            "source": source,
            "bbox": bbox,
            "u": u,
            "v": v,
            "err_u": err_u,
            "depth": depth,
            "bbox_area": self._bbox_area({"bbox": bbox}),
            "confidence": confidence,
            "last_seen_step": self._step,
            "min_depth_seen": depth if depth > 0.05 else 99.0,
            "seen_count": 1,
        }

    def _update_active_target(self, t) -> None:
        snap = self._target_snapshot(t)
        if self._active_target is None:
            self._active_target = snap
            self._target_stage = "EE_TRACK"
            self._target_lost_count = 0
            self._last_seen_target = dict(snap)
            self._last_seen_step = self._step
            self._lost_target_step = -1
            self._log(
                f"[TaskB-RL] active target acquired cls={snap['dominant_class']} "
                f"src={snap['source']} depth={snap['depth']:.2f} area={snap['bbox_area']:.0f}"
            )
            return

        hist = dict(self._active_target.get("class_hist") or {})
        hist[snap["dominant_class"]] = hist.get(snap["dominant_class"], 0) + 1
        dominant_class = max(hist.items(), key=lambda kv: kv[1])[0]
        self._active_target.update(snap)
        self._active_target["class_hist"] = hist
        self._active_target["dominant_class"] = dominant_class
        self._active_target["seen_count"] = int(self._active_target.get("seen_count", 0)) + 1
        self._active_target["min_depth_seen"] = min(
            float(self._active_target.get("min_depth_seen", 99.0)),
            snap["depth"] if snap["depth"] > 0.05 else 99.0,
        )
        self._target_lost_count = 0
        self._last_seen_target = dict(self._active_target)
        self._last_seen_step = self._step
        self._lost_target_step = -1

    def _reject_target(self, t, reason: str) -> None:
        if not self._verbose_reject_logs:
            return
        cls = str(getattr(t, "obj_class", "unknown"))
        src = str(getattr(t, "source", ""))
        depth = float(getattr(t, "depth_m", 0.0))
        key = (src, cls, reason)
        last_step = self._reject_log_last_step.get(key, -10**9)
        if self._step - last_step < 10:
            return
        self._reject_log_last_step[key] = self._step
        self._log(f"[TaskB-RL] candidate rejected src={src} cls={cls} depth={depth:.2f} reason={reason}")

    @staticmethod
    def _depth_patch_median(depth_patch: np.ndarray) -> float:
        valid = depth_patch[(depth_patch > 0.01) & np.isfinite(depth_patch)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))

    def _is_valid_head_depth_object(self, t, obs: dict) -> bool:
        image_obs = obs.get("image") or {}
        head_depth = parse_depth(image_obs, "head_depth")
        if head_depth is None or head_depth.ndim != 2:
            self._reject_target(t, "head_depth_missing")
            return False
        h, w = head_depth.shape
        x1, y1, x2, y2 = [int(v) for v in getattr(t, "bbox", (0, 0, 0, 0))]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            self._reject_target(t, "head_invalid_bbox")
            return False
        bw = x2 - x1
        bh = y2 - y1
        if y2 >= h - 6:
            self._reject_target(t, "head_touch_bottom")
            return False
        if bh <= 0 or bw / max(bh, 1) > 3.6 or bh / max(bw, 1) > 4.5 or bw * bh > 0.28 * h * w:
            self._reject_target(t, "head_bbox_shape")
            return False

        crop = head_depth[y1:y2, x1:x2]
        valid_mask = (crop > 0.01) & np.isfinite(crop)
        valid_ratio = float(np.mean(valid_mask)) if crop.size > 0 else 0.0
        if valid_ratio <= 0.5:
            self._reject_target(t, f"head_valid_ratio={valid_ratio:.2f}")
            return False

        cx = int(round(float(getattr(t, "u", (x1 + x2) * 0.5))))
        cy = int(round(float(getattr(t, "v", (y1 + y2) * 0.5))))
        cx = max(x1, min(cx, x2 - 1))
        cy = max(y1, min(cy, y2 - 1))

        inner_rx = max(3, bw // 4)
        inner_ry = max(3, bh // 4)
        ix1, ix2 = max(0, cx - inner_rx), min(w, cx + inner_rx + 1)
        iy1, iy2 = max(0, cy - inner_ry), min(h, cy + inner_ry + 1)
        center_patch = head_depth[iy1:iy2, ix1:ix2]
        center_median = self._depth_patch_median(center_patch)
        if not (0.35 <= center_median <= 1.2):
            self._reject_target(t, f"head_depth_center={center_median:.2f}")
            return False

        ring_pad_x = max(6, bw // 5)
        ring_pad_y = max(6, bh // 5)
        ox1, ox2 = max(0, x1 - ring_pad_x), min(w, x2 + ring_pad_x)
        oy1, oy2 = max(0, y1 - ring_pad_y), min(h, y2 + ring_pad_y)
        outer = head_depth[oy1:oy2, ox1:ox2]
        ring_mask = np.ones_like(outer, dtype=bool)
        ring_mask[(y1 - oy1):(y2 - oy1), (x1 - ox1):(x2 - ox1)] = False
        ring_vals = outer[ring_mask]
        ring_vals = ring_vals[(ring_vals > 0.01) & np.isfinite(ring_vals)]
        ring_median = float(np.median(ring_vals)) if ring_vals.size > 0 else 0.0
        if ring_median > 0.01 and abs(center_median - ring_median) < 0.04:
            self._reject_target(t, f"head_low_fg_delta={abs(center_median - ring_median):.3f}")
            return False
        return True

    def _match_active_target(self, candidates, source: str):
        if self._active_target is None:
            return None
        last_err_u = float(self._active_target.get("err_u", 0.0))
        last_depth = float(self._active_target.get("depth", 0.0))
        last_area = max(float(self._active_target.get("bbox_area", 0.0)), 1.0)
        min_depth_seen = float(self._active_target.get("min_depth_seen", 99.0))
        dominant_class = str(self._active_target.get("dominant_class", "unknown"))
        last_source = str(self._active_target.get("source", ""))
        best = None
        best_score = float("inf")
        for t in candidates:
            if str(getattr(t, "source", "")) != source:
                continue
            if self._is_processed_target(t):
                self._reject_target(t, "processed_blacklist")
                continue
            depth = float(getattr(t, "depth_m", 0.0))
            err_u = float(getattr(t, "err_u", 0.0))
            area = max(self._bbox_area(t), 1.0)
            if last_depth > 0.05 and depth > 0.05 and abs(depth - last_depth) > self.TARGET_MAX_DEPTH_JUMP_M:
                self._reject_target(t, f"depth_jump={abs(depth - last_depth):.2f}")
                continue
            if abs(err_u - last_err_u) > self.TARGET_MAX_ERRU_JUMP_PX:
                self._reject_target(t, f"err_u_jump={abs(err_u - last_err_u):.1f}")
                continue
            area_ratio = area / last_area
            area_min = self.TARGET_AREA_RATIO_MIN
            area_max = self.TARGET_AREA_RATIO_MAX
            if last_source and source != last_source:
                area_min = 0.15
                area_max = 6.0
            if not (area_min <= area_ratio <= area_max):
                self._reject_target(t, f"area_ratio={area_ratio:.2f}")
                continue
            if min_depth_seen < 1.2 and depth > min_depth_seen + 0.8:
                self._reject_target(t, f"too_far_after_close min={min_depth_seen:.2f} now={depth:.2f}")
                continue
            class_penalty = 0.0 if str(getattr(t, "obj_class", "")) == dominant_class else 0.12
            score = (
                abs(err_u - last_err_u) / self.TARGET_MAX_ERRU_JUMP_PX
                + (abs(depth - last_depth) / max(self.TARGET_MAX_DEPTH_JUMP_M, 1e-6) if last_depth > 0.05 and depth > 0.05 else 0.0)
                + abs(math.log(max(area_ratio, 1e-6)))
                + class_penalty
            )
            if score < best_score:
                best = t
                best_score = score
        return best

    def _is_processed_target(self, t) -> bool:
        if not self._processed_targets:
            return False
        err_u = float(getattr(t, "err_u", 0.0))
        depth = float(getattr(t, "depth_m", 0.0))
        area = max(self._bbox_area(t), 1.0)
        obj_class = str(getattr(t, "obj_class", ""))
        kept = []
        blocked = False
        for entry in self._processed_targets:
            until = int(entry.get("until_step", 10**9))
            if until < self._step:
                continue
            kept.append(entry)
            area_ratio = area / max(float(entry.get("bbox_area", 1.0)), 1.0)
            same_geom = (
                abs(err_u - float(entry.get("err_u", 0.0))) <= 140.0
                and (depth <= 0.05 or abs(depth - float(entry.get("depth", depth))) <= 0.6)
                and 0.45 <= area_ratio <= 2.4
            )
            same_class = obj_class == str(entry.get("dominant_class", ""))
            if same_geom and (same_class or float(entry.get("depth", 99.0)) < 1.4):
                blocked = True
        self._processed_targets = kept
        return blocked

    def _mark_active_target_processed(self, success: bool) -> None:
        if self._active_target is None:
            return
        record = dict(self._active_target)
        record["success"] = bool(success)
        record["processed_step"] = self._step
        record["until_step"] = self._step + (10**9 if success else 400)
        self._processed_targets.append(record)
        self._log(
            f"[TaskB-RL] active target marked processed success={success} "
            f"cls={record.get('dominant_class')} depth={float(record.get('depth', 0.0)):.2f}"
        )

    def _clear_active_target(self) -> None:
        self._active_target = None
        self._target_stage = "NONE"
        self._head_confirm_count = 0
        self._pre_crouch_confirm_count = 0
        self._target_lost_count = 0
        self._candidate_target = None
        self._candidate_seen_count = 0
        self._search_no_target_count = 0

    def _get_valid_ee_target(self):
        ee_candidates = [t for t in self.axis_nav.last_targets if getattr(t, "source", "") == "ee"]
        if self._active_target is None:
            filtered = [t for t in ee_candidates if not self._is_processed_target(t)]
            if not filtered:
                self._candidate_target = None
                self._candidate_seen_count = 0
                return None
            best = min(filtered, key=lambda t: (abs(float(t.err_u)), -float(t.confidence)))
            if self._candidate_target is not None:
                prev_err = float(self._candidate_target.get("err_u", 0.0))
                prev_depth = float(self._candidate_target.get("depth", 0.0))
                prev_area = max(float(self._candidate_target.get("bbox_area", 1.0)), 1.0)
                area_ratio = max(self._bbox_area(best), 1.0) / prev_area
                stable = (
                    abs(float(best.err_u) - prev_err) <= 90.0
                    and (prev_depth <= 0.05 or abs(float(best.depth_m) - prev_depth) <= 0.5)
                    and 0.55 <= area_ratio <= 1.8
                )
                if stable:
                    self._candidate_seen_count += 1
                else:
                    self._candidate_seen_count = 1
            else:
                self._candidate_seen_count = 1
            self._candidate_target = self._target_snapshot(best)
            if self._candidate_seen_count >= self.ACTIVE_TARGET_ACQUIRE_STEPS:
                self._update_active_target(best)
                self._target_stage = "EE_TRACK"
                self._candidate_target = None
                self._candidate_seen_count = 0
                return best
            return None
        return self._match_active_target(ee_candidates, "ee")

    def _get_valid_head_target(self, obs: dict):
        head_candidates = [t for t in self.axis_nav.last_targets if getattr(t, "source", "") == "head"]
        matched = self._match_active_target(head_candidates, "head")
        if matched is None:
            return None
        return matched

    def _head_assisted_depth(self, ee_target, head_target, include_active: bool = True) -> float:
        depths = []
        if ee_target is not None:
            depth = float(getattr(ee_target, "depth_m", 0.0))
            if depth > 0.05:
                depths.append(depth)
        if head_target is not None:
            depth = float(getattr(head_target, "depth_m", 0.0))
            if depth > 0.05:
                depths.append(depth)
        if include_active and self._active_target is not None:
            depth = float(self._active_target.get("depth", 0.0))
            if depth > 0.05:
                depths.append(depth)
        if not depths:
            return 0.0
        return float(min(depths))

    def _scene(self):
        if self.env is None:
            return None
        env_u = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(env_u, "scene"):
            return env_u.scene
        if hasattr(env_u, "_env") and hasattr(env_u._env, "scene"):
            return env_u._env.scene
        return None

    def _robot(self):
        scene = self._scene()
        if scene is None:
            return None
        try:
            robot = scene["robot"]
        except Exception:
            return None
        if isinstance(robot, (list, tuple)):
            robot = robot[0] if robot else None
        if robot is not None and not hasattr(robot, "_env"):
            setattr(robot, "_env", self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env)
        return robot

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._camera_debug_warned_keys:
            return
        self._camera_debug_warned_keys.add(key)
        print(message, flush=True)

    def _ensure_arm_controller(self) -> ArmGraspController | None:
        if self._arm_grasp is not None:
            return self._arm_grasp
        robot = self._robot()
        if robot is None:
            return None
        try:
            self._arm_grasp = ArmGraspController(
                robot=robot,
                device=self.device,
                arm_joint_names=self.ARM_JOINT_NAMES[:6],
                gripper_joint_names=self.ARM_JOINT_NAMES[6:],
                ee_body_name=self.EE_BODY_NAME,
                action_scale=self.ACTION_SCALE,
            )
            self._arm_grasp.b2_piper_arm_defaults = dict(self.ARM_DEFAULT_POS)
        except Exception as exc:
            print(f"[TaskB-RL] ArmGraspController init failed: {exc}", flush=True)
            self._arm_grasp = None
        return self._arm_grasp

    def _set_velocity_commands(self, vx: float, vy: float, wz: float) -> None:
        self._velocity_commands = torch.tensor([[float(vx), float(vy), float(wz)]], device=self.device, dtype=torch.float32)

    def _extract_policy_obs(self, obs: dict, base_cmd: np.ndarray, obs_dim: int | None = None) -> torch.Tensor:
        proprio = torch.as_tensor(obs["proprio"], device=self.device, dtype=torch.float32)
        idx = 0
        idx += 3
        base_ang_vel = proprio[:, idx: idx + 3]
        idx += 3
        idx += 3
        projected_gravity = proprio[:, idx: idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx: idx + self.total_action_dim]
        idx += self.total_action_dim
        joint_vel_all = proprio[:, idx: idx + self.total_action_dim]
        idx += self.total_action_dim
        actions_all = proprio[:, idx: idx + self.total_action_dim]

        joint_pos_leg = joint_pos_all[:, :self.leg_action_dim]
        joint_vel_leg = joint_vel_all[:, :self.leg_action_dim]
        actions_leg_env = actions_all[:, :self.leg_action_dim]
        actions_leg_train = actions_leg_env * self.env_to_train_action_scale.to(dtype=proprio.dtype)

        if obs_dim is None:
            if self.policy is not None:
                obs_dim = int(getattr(self.policy.actor[0], "in_features", 45))
            else:
                obs_dim = 45

        components = [base_ang_vel * 0.25, projected_gravity]
        if obs_dim >= 45:
            velocity_commands = torch.as_tensor(base_cmd, device=self.device, dtype=proprio.dtype).view(1, 3)
            if proprio.shape[0] > 1:
                velocity_commands = velocity_commands.repeat(proprio.shape[0], 1)
            components.append(velocity_commands)
            # 训练时包含 base_height_command（1维），需要添加
            if obs_dim >= 46:
                # 使用默认高度命令 0.45m（训练时的典型值）
                base_height_command = torch.full((proprio.shape[0], 1), 0.45, device=self.device, dtype=proprio.dtype)
                components.append(base_height_command)
        components.extend([joint_pos_leg, joint_vel_leg * 0.05, actions_leg_train])
        return torch.nan_to_num(torch.cat(components, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor) -> torch.Tensor:
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        action_env = torch.zeros((action_train.shape[0], self.total_action_dim), device=self.device, dtype=torch.float32)
        action_env[:, :self.leg_action_dim] = action_train * self.train_to_env_action_scale
        action_env[:, self.leg_action_dim:] = 0.0
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _generate_action_tensor(self, obs: dict, base_cmd: np.ndarray, actor=None, obs_dim: int | None = None) -> torch.Tensor:
        if actor is None:
            actor = self.policy
        if obs is None or actor is None:
            return self._scripted_leg_action(self.total_action_dim)
        if obs_dim is None:
            obs_dim = int(getattr(actor.actor[0], "in_features", 45))
        with torch.inference_mode():
            action_train = actor(self._extract_policy_obs(obs, base_cmd, obs_dim=obs_dim))
        return self._map_policy_action_to_env_action(action_train)

    def _generate_sit_down_action_tensor(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        return self._generate_action_tensor(
            obs,
            zero_cmd,
            actor=self.sit_down_actor,
            obs_dim=self.sit_down_actor_obs_dim,
        )

    def _leg_action(self, obs: dict, base_cmd: np.ndarray) -> torch.Tensor:
        if self.policy is None:
            return self._scripted_leg_action(self.total_action_dim)
        return self._generate_action_tensor(obs, base_cmd, obs_dim=self.leg_actor_obs_dim)

    def _reset_sit_down_tracking(self) -> None:
        self._sit_down_step_count = 0
        self._sit_down_stable_count = 0

    def _get_base_rpy_height(self, robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quat = robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        return roll, pitch, robot.data.root_pos_w[:, 2]

    def _is_sit_down_stable(self, robot) -> bool:
        roll, pitch, _ = self._get_base_rpy_height(robot)
        lin_vel_z = torch.abs(robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy = torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1)
        stable = (
            (torch.abs(roll) <= self.sit_down_roll_pitch_thresh)
            & (torch.abs(pitch) <= self.sit_down_roll_pitch_thresh)
            & (lin_vel_z <= self.sit_down_height_vel_thresh)
            & (ang_vel_xy <= self.sit_down_ang_vel_thresh)
        )
        return bool(torch.all(stable))

    def _ensure_leg_control_initialized(self, robot) -> None:
        if self._leg_control_initialized or robot is None:
            return
        self._leg_joint_ids, self._leg_joint_names_in_robot = robot.find_joints(self.LEG_JOINT_NAMES)
        self._leg_joint_names_in_robot = list(self._leg_joint_names_in_robot)
        robot_name_to_local_idx = {name: idx for idx, name in enumerate(self._leg_joint_names_in_robot)}
        self._env_leg_to_robot_indices = torch.tensor(
            [robot_name_to_local_idx[name] for name in self.LEG_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        self._robot_leg_to_env_indices = torch.empty_like(self._env_leg_to_robot_indices)
        self._robot_leg_to_env_indices[self._env_leg_to_robot_indices] = torch.arange(
            len(self.LEG_JOINT_NAMES), device=self.device, dtype=torch.long,
        )
        self._leg_default_dof_pos = robot.data.default_joint_pos[:, self._leg_joint_ids].clone()
        if hasattr(robot.data, "soft_joint_pos_limits"):
            self._leg_joint_pos_limits = robot.data.soft_joint_pos_limits[:, self._leg_joint_ids, :].clone()
        elif hasattr(robot.data, "joint_pos_limits"):
            self._leg_joint_pos_limits = robot.data.joint_pos_limits[:, self._leg_joint_ids, :].clone()
        else:
            min_pos = torch.full_like(self._leg_default_dof_pos, -10.0)
            max_pos = torch.full_like(self._leg_default_dof_pos, 10.0)
            self._leg_joint_pos_limits = torch.stack((min_pos, max_pos), dim=-1)
        self._leg_posture_controller._ensure_initialized(robot)
        self._leg_control_initialized = True

    def _reorder_robot_leg_to_env(self, tensor_robot_order: torch.Tensor) -> torch.Tensor:
        return tensor_robot_order[:, self._robot_leg_to_env_indices]

    def _compute_leg_position_actions(self, target_dof_pos: torch.Tensor, robot) -> torch.Tensor:
        self._ensure_leg_control_initialized(robot)
        joint_limits = self._leg_joint_pos_limits.to(device=target_dof_pos.device, dtype=target_dof_pos.dtype)
        target_dof_pos = torch.clamp(target_dof_pos, joint_limits[..., 0], joint_limits[..., 1])
        target_dof_pos_env = self._reorder_robot_leg_to_env(target_dof_pos)
        current_dof_pos_env = self._reorder_robot_leg_to_env(robot.data.joint_pos[:, self._leg_joint_ids])
        leg_actions = (target_dof_pos_env - current_dof_pos_env) / self.train_to_env_action_scale.to(dtype=target_dof_pos.dtype)
        max_delta = float(self.control_cfg.max_squat_action_delta)
        if self._last_leg_action_override is None or self._last_leg_action_override.shape != leg_actions.shape:
            self._last_leg_action_override = leg_actions.clone()
        else:
            delta = torch.clamp(leg_actions - self._last_leg_action_override, -max_delta, max_delta)
            leg_actions = self._last_leg_action_override + delta
            self._last_leg_action_override = leg_actions.clone()
        return leg_actions

    def _generate_control_action_tensor(self, obs: dict, base_cmd: np.ndarray, robot) -> torch.Tensor:
        action_env = self._leg_action(obs, base_cmd)
        self._ensure_leg_control_initialized(robot)
        if self._leg_posture_controller.state == "IDLE":
            self._last_leg_action_override = None
            return action_env
        _, target_dof_pos = self._leg_posture_controller.step(robot, self.dt)
        if target_dof_pos is None:
            target_dof_pos = robot.data.joint_pos[:, self._leg_joint_ids].clone()
        action_env[:, :self.leg_action_dim] = self._compute_leg_position_actions(target_dof_pos, robot)
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_scene_camera(self, scene, camera_name: str):
        if scene is None:
            self._warn_once("camera_debug_scene_missing", "[TaskB-RL] Warning: scene unavailable, disabling camera debug display.")
            return None
        try:
            camera = scene[camera_name]
        except Exception:
            self._warn_once(
                f"camera_debug_missing_{camera_name}",
                f"[TaskB-RL] Warning: camera '{camera_name}' not found in scene.",
            )
            return None
        if isinstance(camera, (list, tuple)):
            camera = camera[0] if camera else None
        return camera

    def _get_camera_output(self, camera, camera_name: str, output_key: str):
        output = getattr(getattr(camera, "data", None), "output", None)
        if output is None:
            self._warn_once(
                f"camera_debug_output_missing_{camera_name}",
                f"[TaskB-RL] Warning: camera '{camera_name}' has no data.output.",
            )
            return None
        if output_key not in output:
            self._warn_once(
                f"camera_debug_output_key_missing_{camera_name}_{output_key}",
                f"[TaskB-RL] Warning: camera '{camera_name}' output '{output_key}' not found.",
            )
            return None
        return output[output_key]

    def _process_camera_debug(self) -> None:
        if not self._camera_debug_enabled or self._step % self._camera_debug_interval != 0:
            return
        if cv2 is None:
            self._warn_once("camera_debug_cv2_missing", "[TaskB-RL] Warning: OpenCV unavailable, camera debug disabled.")
            self._camera_debug_enabled = False
            return
        try:
            scene = self._scene()
            head_camera = self._get_scene_camera(scene, "head_camera")
            ee_camera = self._get_scene_camera(scene, "ee_camera")
            head_rgb = self._get_camera_output(head_camera, "head_camera", "rgb") if head_camera is not None else None
            ee_rgb = self._get_camera_output(ee_camera, "ee_camera", "rgb") if ee_camera is not None else None
            if head_rgb is not None:
                cv2.imshow("head_rgb", rgb_to_bgr_uint8(head_rgb))
            if ee_rgb is not None:
                cv2.imshow("ee_rgb", rgb_to_bgr_uint8(ee_rgb))
            if head_rgb is not None or ee_rgb is not None:
                cv2.waitKey(1)
        except Exception as exc:
            self._warn_once("camera_debug_runtime_error", f"[TaskB-RL] Warning: camera debug disabled due to error: {exc}")
            self._camera_debug_enabled = False

    def _update_last_seen_target(self) -> None:
        if self._active_target is None:
            if self._lost_target_step < 0:
                self._lost_target_step = self._step
            return
        self._last_seen_target = dict(self._active_target)
        self._last_seen_step = int(self._active_target.get("last_seen_step", self._step))
        self._lost_target_step = -1

    @staticmethod
    def _nav_depth(nav: dict) -> float:
        for key in ("nav_depth_m", "depth_m", "dist_to_robot"):
            val = nav.get(key)
            if val is not None:
                try:
                    d = float(val)
                    if d > 0.05:
                        return d
                except (TypeError, ValueError):
                    pass
        return 99.0

    @staticmethod
    def _nav_yaw(nav: dict) -> float:
        for key in ("yaw_rel", "nav_yaw_rel"):
            val = nav.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _robot_xy_yaw(self, perc: dict) -> tuple[np.ndarray, float] | None:
        robot = perc.get("robot") or {}
        pos_w = robot.get("pos_world")
        if pos_w is None:
            return None
        return np.asarray(pos_w, dtype=np.float32), float(robot.get("yaw") or 0.0)

    def _dist_to_bin(self, perc: dict) -> float:
        xy = self._robot_xy_yaw(perc)
        if xy is None:
            return 99.0
        rp, _ = xy
        return float(np.linalg.norm(rp[:2] - BIN_CENTER[:2]))

    def _at_bin(self, perc: dict) -> bool:
        return self._dist_to_bin(perc) < self.BIN_ARRIVE_M

    def _cmd_approach(self, nav: dict) -> tuple[float, float]:
        depth = self._nav_depth(nav)
        yaw = self._nav_yaw(nav)
        if depth < self.GRASP_DEPTH_M:
            return 0.2, float(np.clip(yaw * self.NAV_WZ_GAIN, -self.NAV_WZ_MAX, self.NAV_WZ_MAX))
        if abs(yaw) > self.YAW_TURN_THRESH:
            return 0.0, float(np.clip(yaw * self.NAV_WZ_GAIN, -self.NAV_WZ_MAX, self.NAV_WZ_MAX))
        vx = self.NAV_VX_MIN + (self.NAV_VX_MAX - self.NAV_VX_MIN) * min(depth, 3.0) / 3.0
        wz = float(np.clip(yaw * self.NAV_WZ_GAIN * 0.7, -self.NAV_WZ_MAX * 0.6, self.NAV_WZ_MAX * 0.6))
        return float(vx), wz

    def _cmd_search(self) -> tuple[float, float]:
        return self.SEARCH_VX, self.SEARCH_WZ

    def _cmd_carry(self, perc: dict) -> tuple[float, float]:
        xy = self._robot_xy_yaw(perc)
        if xy is None:
            return 0.2, 0.0
        rp, yaw = xy
        delta = np.asarray(BIN_CENTER[:2], dtype=np.float32) - rp[:2]
        dist = float(np.linalg.norm(delta))
        if dist < self.BIN_ARRIVE_M:
            return 0.0, 0.0
        yaw_to_bin = float(math.atan2(delta[1], delta[0]))
        err = (yaw_to_bin - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > 0.35:
            return 0.2, float(np.clip(err * 0.9, -0.45, 0.45))
        vx = float(np.clip(0.2 + 0.6 * min(dist, 4.0) / 4.0, 0.2, 0.8))
        wz = float(np.clip(err * 0.6, -0.25, 0.25))
        return vx, wz

    def _cmd_visual_approach(self, target) -> tuple[float, float, float, float]:
        err_u = float(target.err_u)
        depth = float(target.depth_m)
        if abs(err_u) < 35.0:
            wz = 0.0
        elif abs(err_u) < 80.0:
            wz = float(np.clip(-0.0025 * err_u, -0.2, 0.2))
        else:
            wz = float(np.clip(-0.0040 * err_u, -0.4, 0.4))
        align_scale = float(np.clip(1.0 - abs(err_u) / 250.0, 0.3, 1.0))
        depth_scale = float(min(1.0, max(0.4, depth / 2.0)))
        vx = float(np.clip(0.8 * align_scale * depth_scale, 0.3, 0.9))
        if abs(err_u) > 150.0:
            vx = 0.0
        elif abs(err_u) > 100.0:
            vx = min(vx, 0.2)
        return vx, wz, err_u, depth

    def _choose_velocity(self, perc: dict, obs: dict) -> tuple[float, float, str]:
        if self._step < self.WARMUP_STEPS or self._step < self._post_drop_stand_until:
            return 0.0, 0.0, "STAND"
        if self._task_state in {"CROUCHING", "PREGRASP", "GRASP_ARM", "STAND_UP"}:
            return 0.0, 0.0, self._task_state
        if self._task_state == "DROP" or self._drop_phase:
            return 0.0, 0.0, "DROP"
        if self._task_state == "CARRY":
            if self._at_bin(perc):
                return 0.0, 0.0, "DROP"
            vx, wz = self._cmd_carry(perc)
            return min(vx, 0.8), wz, "CARRY"

        if self._task_state in {"SEARCH", "APPROACH"}:
            ee_target = self._get_valid_ee_target()
            head_target = self._get_valid_head_target(obs) if self._active_target is not None else None

            if self._active_target is None:
                has_ee_candidates = any(getattr(t, "source", "") == "ee" for t in self.axis_nav.last_targets)
                if has_ee_candidates:
                    self._search_no_target_count = 0
                    return 0.0, 0.0, "SEARCH_WAIT"
                self._search_no_target_count += 1
                if self._search_no_target_count >= self.SEARCH_SPIN_AFTER_NO_TARGET_STEPS:
                    return 0.0, self.SEARCH_SPIN_WZ, "SEARCH_SPIN"
                return 0.0, 0.0, "SEARCH_WAIT"

            self._search_no_target_count = 0

            if self._target_stage == "NONE":
                self._target_stage = "EE_TRACK"

            ee_depth_for_stage = float(ee_target.depth_m) if ee_target is not None else (float(self._active_target.get("depth", 0.0)) if self._active_target is not None else 0.0)
            min_depth_seen = float(self._active_target.get("min_depth_seen", 99.0))
            if (
                self._target_stage == "EE_TRACK"
                and ee_depth_for_stage > 0.05
                and ee_depth_for_stage <= self.GRASP_DEPTH_M
            ):
                self._target_stage = "HEAD_APPROACH"
                self._target_lost_count = 0
                self._log(f"[TaskB-RL] EE_TRACK -> HEAD_APPROACH depth={ee_depth_for_stage:.2f}")

            if self._target_stage == "EE_TRACK":
                if ee_target is not None:
                    self._update_active_target(ee_target)
                    vx, wz, _, depth = self._cmd_visual_approach(ee_target)
                    self._final_approach_vx = float(vx)
                    self._final_approach_wz = float(wz)
                    if depth > 0.05 and depth <= self.GRASP_DEPTH_M:
                        self._target_stage = "HEAD_APPROACH"
                        self._target_lost_count = 0
                        self._log(f"[TaskB-RL] EE_TRACK -> HEAD_APPROACH depth={depth:.2f}")
                    return vx, wz, "EE_TRACK"

                self._target_lost_count += 1
                if self._lost_target_step < 0:
                    self._lost_target_step = self._step
                ee_depth = float(self._active_target.get("depth", 0.0)) if self._active_target is not None else 0.0
                if ee_depth > 0.05 and ee_depth <= self.GRASP_DEPTH_M:
                    self._target_stage = "HEAD_APPROACH"
                    self._target_lost_count = 0
                    self._log(f"[TaskB-RL] ee depth reached head handover threshold={ee_depth:.2f}")
                    return 0.0, 0.0, "SEARCH_WAIT"
                if self._target_lost_count <= self.FINAL_APPROACH_KEEP_STEPS:
                    return self._final_approach_vx, self._final_approach_wz, "EE_TRACK"
                if self._target_lost_count >= self.EE_TRACK_LOST_GIVEUP_STEPS:
                    self._log(
                        f"[TaskB-RL] EE_TRACK lost {self._target_lost_count} steps, "
                        f"giving up target -> SEARCH_SPIN"
                    )
                    self._clear_target_lock()
                    return 0.0, self.SEARCH_SPIN_WZ, "SEARCH_SPIN"
                last_err_u = float(self._active_target.get("err_u", 0.0)) if self._active_target is not None else 0.0
                if abs(last_err_u) > 30.0:
                    lost_wz = float(np.clip(-0.0040 * last_err_u, -0.4, 0.4))
                    return 0.0, lost_wz, "EE_TRACK"
                return 0.0, 0.0, "SEARCH_WAIT"

            if self._target_stage == "HEAD_APPROACH":
                head_crouch_depth_m = self.HEAD_PRE_CROUCH_DEPTH_M + self.HEAD_PRE_CROUCH_DEPTH_EPS_M
                if head_target is not None:
                    self._update_active_target(head_target)
                    vx, wz, _, depth = self._cmd_visual_approach(head_target)
                    self._final_approach_vx = float(vx)
                    self._final_approach_wz = float(wz)
                    if depth > 0.05 and depth <= head_crouch_depth_m:
                        self._target_stage = "PRE_CROUCH"
                        self._pre_crouch_confirm_count = self.PRE_CROUCH_CONFIRM_STEPS
                        self._pre_crouch_wait_steps = 0
                        self._log(f"[TaskB-RL] HEAD_APPROACH -> PRE_CROUCH depth={depth:.2f}")
                        return 0.0, 0.0, "START_CROUCH"
                    return vx, wz, "HEAD_APPROACH"

                self._target_lost_count += 1
                if self._lost_target_step < 0:
                    self._lost_target_step = self._step
                head_depth = float(self._active_target.get("depth", 0.0)) if self._active_target is not None else 0.0
                if head_depth > 0.05 and head_depth <= head_crouch_depth_m:
                    self._target_stage = "PRE_CROUCH"
                    self._pre_crouch_confirm_count = self.PRE_CROUCH_CONFIRM_STEPS
                    self._pre_crouch_wait_steps = 0
                    self._log(f"[TaskB-RL] HEAD_APPROACH -> PRE_CROUCH depth={head_depth:.2f}")
                    return 0.0, 0.0, "START_CROUCH"
                if head_depth > 0.05 and head_depth <= self.HEAD_LOST_DIRECT_CROUCH_DEPTH_M and self._target_lost_count >= self.HEAD_LOST_DIRECT_CROUCH_STEPS:
                    self._target_stage = "PRE_CROUCH"
                    self._pre_crouch_confirm_count = self.PRE_CROUCH_CONFIRM_STEPS
                    self._pre_crouch_wait_steps = 0
                    self._log(
                        f"[TaskB-RL] HEAD_APPROACH lost {self._target_lost_count} steps, "
                        f"depth={head_depth:.2f} close enough -> PRE_CROUCH"
                    )
                    return 0.0, 0.0, "START_CROUCH"
                if self._target_lost_count <= self.FINAL_APPROACH_KEEP_STEPS:
                    return self._final_approach_vx, self._final_approach_wz, "HEAD_APPROACH"
                if ee_target is not None:
                    self._target_stage = "EE_TRACK"
                    self._target_lost_count = 0
                    return 0.0, 0.0, "EE_REACQUIRE"
                if self._target_lost_count >= self.HEAD_APPROACH_LOST_GIVEUP_STEPS:
                    self._log(
                        f"[TaskB-RL] HEAD_APPROACH lost {self._target_lost_count} steps, "
                        f"giving up target -> SEARCH_SPIN"
                    )
                    self._clear_target_lock()
                    return 0.0, self.SEARCH_SPIN_WZ, "SEARCH_SPIN"
                return 0.0, 0.0, "SEARCH_WAIT"

            if self._target_stage == "EE_BASE_ALIGN":
                center, depth_m = self._get_ee_seg_center(obs)
                if center is not None:
                    stage_depth = depth_m if depth_m > 0.05 else float(self._active_target.get("depth", 0.0))
                    ee_nav = SimpleNamespace(err_u=float(center.err_u), depth_m=float(stage_depth))
                    vx, wz, _, depth = self._cmd_visual_approach(ee_nav)
                    self._final_approach_vx = float(vx)
                    self._final_approach_wz = float(wz)
                    if center.aligned and depth > 0.05:
                        self._target_stage = "PRE_CROUCH"
                        self._pre_crouch_confirm_count = 1
                        self._log(
                            f"[TaskB-RL] EE_BASE_ALIGN -> PRE_CROUCH "
                            f"err=({center.err_u:+.1f},{center.err_v:+.1f}) depth={depth:.2f}"
                        )
                        return 0.0, 0.0, "START_CROUCH"
                    return vx, wz, "EE_BASE_ALIGN"

                self._target_lost_count += 1
                last_depth = float(self._active_target.get("depth", 0.0)) if self._active_target is not None else 0.0
                head_crouch_depth_m = self.HEAD_PRE_CROUCH_DEPTH_M + self.HEAD_PRE_CROUCH_DEPTH_EPS_M
                if last_depth > 0.05 and last_depth <= head_crouch_depth_m:
                    self._target_stage = "PRE_CROUCH"
                    self._pre_crouch_confirm_count = 1
                    self._log(f"[TaskB-RL] EE_BASE_ALIGN lost target -> PRE_CROUCH depth={last_depth:.2f}")
                    return 0.0, 0.0, "START_CROUCH"
                if self._target_lost_count <= self.FINAL_APPROACH_KEEP_STEPS:
                    return self._final_approach_vx, self._final_approach_wz, "EE_BASE_ALIGN"
                return 0.0, 0.0, "SEARCH_WAIT"

            if self._target_stage == "PRE_CROUCH":
                return 0.0, 0.0, "START_CROUCH"

            return *self._cmd_search(), "SEARCH_YOLO"

        if self.perception is not None:
            phase = str(perc.get("phase") or "approach")
            grasp = perc.get("target_grasp")
            nav = perc.get("target_nav")
            if phase == "grasp" and isinstance(grasp, dict) and grasp.get("grasp_pos_world"):
                return 0.0, 0.0, "READY_TO_GRASP"
            if isinstance(nav, dict):
                return *self._cmd_approach(nav), "APPROACH"

        return *self._cmd_search(), "SEARCH"

    def _smooth_cmd(self, vx: float, wz: float) -> tuple[float, float]:
        vx_f = (1.0 - self.CMD_ALPHA) * self._cmd_vx_filt + self.CMD_ALPHA * float(vx)
        wz_f = (1.0 - self.CMD_ALPHA) * self._cmd_wz_filt + self.CMD_ALPHA * float(wz)

        vx_f = float(np.clip(vx_f, self._cmd_vx_filt - self.MAX_DVX, self._cmd_vx_filt + self.MAX_DVX))
        wz_f = float(np.clip(wz_f, self._cmd_wz_filt - self.MAX_DWZ, self._cmd_wz_filt + self.MAX_DWZ))

        self._cmd_vx_filt = vx_f
        self._cmd_wz_filt = wz_f
        return vx_f, wz_f

    def _is_standing_phase(self) -> bool:
        return self._step < self.WARMUP_STEPS or self._step < self._post_drop_stand_until

    def _clear_pending_grasp(self) -> None:
        self._pending_grasp_status = None
        self._pending_grasp_target = None
        self._pending_grasp_pos_w = None
        self._pending_grasp_quat_w = None
        self._pregrasp_reach_stable_count = 0

    def _update_grasp(self, perc: dict) -> None:
        if self._is_standing_phase() or self._hold_carry_pose or self._drop_phase:
            return
        if self._task_state in {"GRASP_ARM", "STAND_UP", "CARRY"}:
            return
        if str(perc.get("phase") or "") != "grasp":
            return
        grasp = perc.get("target_grasp")
        if not isinstance(grasp, dict) or grasp.get("grasp_pos_world") is None:
            return
        self._pending_grasp_target = grasp
        self._pending_grasp_pos_w = np.asarray(grasp.get("pos_world") or grasp["grasp_pos_world"], dtype=np.float32)
        gq = grasp.get("grasp_quat_world")
        self._pending_grasp_quat_w = None if gq is None else np.asarray(gq, dtype=np.float32)
        if self._task_state in {"SEARCH", "APPROACH"} and self._target_stage != "PRE_CROUCH":
            return
        if self._task_state in {"CROUCHING", "PREGRASP"}:
            return
        robot = self._robot()
        if robot is None:
            return
        self._crouch_wait_start_step = -1
        self._reset_sit_down_tracking()
        self._entered_crouch_from_yolo = False
        self._leg_posture_controller.start_crouch(robot)
        self._task_state = "CROUCHING"
        print(f"[TaskB-RL] start crouch before grasp id={grasp.get('id')} class={grasp.get('class', '?')}", flush=True)

    def _lock_carry_pose(self, arm: ArmGraspController) -> None:
        arm.close_gripper()
        if arm.desired_arm_joint_pos is not None:
            self._carry_arm_jpos = arm.desired_arm_joint_pos.detach().clone()
        else:
            robot = self._robot()
            if robot is not None:
                self._carry_arm_jpos = robot.data.joint_pos[:, arm.arm_joint_ids].clone()
        self._carry_gripper = arm.gripper_close_pos.clone()
        self._hold_carry_pose = True
        arm.state = "IDLE"
        self.axis_nav.complete_current()
        print(f"[TaskB-RL] YOLO 目标完成，已处理 {self.axis_nav.completed_count} 个", flush=True)

    def _apply_carry_arm(self, action_env: torch.Tensor) -> torch.Tensor:
        if not self._hold_carry_pose:
            return action_env
        arm = self._ensure_arm_controller()
        robot = self._robot()
        if arm is None or robot is None:
            return action_env
        if self._carry_arm_jpos is not None:
            arm.desired_arm_joint_pos = self._carry_arm_jpos.to(device=action_env.device, dtype=action_env.dtype)
        if self._carry_gripper is not None:
            arm.desired_gripper_joint_pos = self._carry_gripper.to(device=action_env.device, dtype=action_env.dtype)
        return arm.apply_to_action_tensor(action_env, robot)

    def _crouch_arm_target(self, arm, device, dtype) -> torch.Tensor:
        arm_target = self._arm_default_action[:, :len(arm.arm_joint_ids)].to(device=device, dtype=dtype).clone()
        arm_target[:, 1] = self.CROUCH_ARM_JOINT2_POS
        arm_target[:, 2] = self.CROUCH_ARM_JOINT3_POS
        arm_target[:, 4] = self.CROUCH_ARM_JOINT5_POS
        return arm_target

    def _get_crouch_arm_pose_status(self, robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        arm = self._ensure_arm_controller()
        if arm is None or robot is None:
            empty = torch.empty((1, 0), device=self.device, dtype=torch.float32)
            return empty, empty, empty, False
        current = robot.data.joint_pos[:, arm.arm_joint_ids].to(device=self.device, dtype=torch.float32)
        target = self._crouch_arm_target(arm, current.device, current.dtype)
        err = torch.abs(target - current)
        ready = bool(torch.all(err <= self.PRE_CROUCH_ARM_READY_RAD))
        return current, target, err, ready

    def _crouch_arm_pose_ready(self, robot) -> bool:
        current, target, err, ready = self._get_crouch_arm_pose_status(robot)
        if ready:
            self._crouch_arm_hold_jpos = target.detach().clone()
        return ready

    def _log_pre_crouch_arm_status(self, robot, ready: bool | None = None) -> bool:
        current, target, err, computed_ready = self._get_crouch_arm_pose_status(robot)
        if ready is None:
            ready = computed_ready
        if current.shape[1] == 0:
            self._log("[TaskB-RL] PRE_CROUCH arm status unavailable")
            return False
        self._log(
            "[TaskB-RL] PRE_CROUCH arm "
            f"current={np.round(current[0].detach().cpu().numpy(), 4).tolist()} "
            f"target={np.round(target[0].detach().cpu().numpy(), 4).tolist()} "
            f"err={np.round(err[0].detach().cpu().numpy(), 4).tolist()} "
            f"ready={ready}"
        )
        if ready:
            self._crouch_arm_hold_jpos = target.detach().clone()
        return ready

    def _apply_crouch_arm_pose(self, action_env: torch.Tensor, robot) -> torch.Tensor:
        arm = self._ensure_arm_controller()
        if arm is None or robot is None:
            return action_env
        crouch_target = self._crouch_arm_target(arm, action_env.device, action_env.dtype)
        self._crouch_arm_hold_jpos = crouch_target.detach().clone()
        arm.desired_arm_joint_pos = crouch_target
        arm.open_gripper()
        # if self._step % 10 == 0:
        #     cur_jpos = robot.data.joint_pos[0, arm.arm_joint_ids].detach().cpu().numpy()
        #     tgt_jpos = crouch_target[0].detach().cpu().numpy()
        #     err_jpos = np.abs(cur_jpos - tgt_jpos)
        #     self._log(
        #         f"[CROUCH_ARM]\n"
        #         f"  current={np.round(cur_jpos, 4).tolist()}\n"
        #         f"  target ={np.round(tgt_jpos, 4).tolist()}\n"
        #         f"  error  ={np.round(err_jpos, 4).tolist()}"
        #     )
        return arm.apply_to_action_tensor(action_env, robot)

    def _get_root_yaw_rate(self, robot) -> float:
        if robot is None or not hasattr(robot, "data") or not hasattr(robot.data, "root_ang_vel_b"):
            return 0.0
        return float(robot.data.root_ang_vel_b[0, 2].detach().cpu().item())

    def _step_pre_crouch_settle(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        robot = self._robot()
        if robot is None:
            self._task_state = "SEARCH"
            self._clear_pending_grasp()
            self._clear_target_lock()
            return self._leg_action(obs, zero_cmd)

        if self.sit_down_actor is not None:
            action_env = self._generate_control_action_tensor(obs, zero_cmd, robot)
        else:
            action_env = self._generate_control_action_tensor(obs, zero_cmd, robot)

        self._pre_crouch_settle_steps += 1
        yaw_rate = self._get_root_yaw_rate(robot)
        settle_ready = (
            self._pre_crouch_settle_steps >= self.PRE_CROUCH_SETTLE_MIN_STEPS
            and abs(yaw_rate) <= self.PRE_CROUCH_SETTLE_YAW_RATE_THRESH
        )
        if self._pre_crouch_settle_steps >= self.PRE_CROUCH_SETTLE_TIMEOUT_STEPS:
            self._log(
                f"[TaskB-RL] WARN: PRE_CROUCH_SETTLE timeout={self._pre_crouch_settle_steps} "
                f"yaw_rate_z={yaw_rate:+.3f}; start crouch anyway"
            )
            settle_ready = True
        elif self._pre_crouch_settle_steps == 1 or self._pre_crouch_settle_steps % 5 == 0:
            self._log(
                f"[TaskB-RL] PRE_CROUCH_SETTLE step={self._pre_crouch_settle_steps} "
                f"yaw_rate_z={yaw_rate:+.3f} thresh={self.PRE_CROUCH_SETTLE_YAW_RATE_THRESH:.3f}"
            )

        if settle_ready:
            self._reset_sit_down_tracking()
            self._leg_posture_controller.start_crouch(robot)
            self._task_state = "CROUCHING"
            self._entered_crouch_from_yolo = True
            self._crouch_wait_start_step = -1
            self._pregrasp_reach_stable_count = 0
            self._pre_crouch_settle_steps = 0
            self._log(f"[TaskB-RL] PRE_CROUCH_SETTLE -> CROUCHING yaw_rate_z={yaw_rate:+.3f}")

        return action_env

    def _ee_seg_depth_m(self, center, ee_depth: np.ndarray | None) -> float:
        if ee_depth is None or ee_depth.ndim != 2:
            return 0.0
        if center.polygon is not None and len(center.polygon) >= 3:
            mask = np.zeros_like(ee_depth, dtype=np.uint8)
            pts = np.round(center.polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
            valid = ee_depth[(mask > 0) & (ee_depth > 0.05) & np.isfinite(ee_depth)]
            if valid.size >= 8:
                return float(np.median(valid))
        return sample_depth_median(ee_depth, int(round(center.u)), int(round(center.v)), radius=4)

    def _build_pending_grasp_from_ee_center(self, center, depth_m: float, robot, arm) -> bool:
        if depth_m <= 0.05 or robot is None or arm is None or self._active_target is None:
            return False
        p_cam = pixel_to_cam(center.u, center.v, depth_m, CAM_FX, CAM_FY, CAM_CX, CAM_CY)
        pos_b = transform_point_cam_to_base(
            p_cam,
            self.nav_cfg.robot.ee_cam.pos_b,
            self.nav_cfg.robot.ee_cam.quat_b,
        )
        base_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
        base_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()
        pos_w = base_pos_w + quat_rotate_vector(base_quat_w, pos_b)
        current_ee_quat_w = arm.get_ee_pose()[1]
        target = {
            "id": self._active_target.get("id", f"ee_seg_{self._step}"),
            "class": self._active_target.get("dominant_class", "unknown"),
            "source": "ee_seg",
            "u": float(center.u),
            "v": float(center.v),
            "depth_m": float(depth_m),
            "pos_base": np.asarray(pos_b, dtype=np.float32),
            "pos_world": np.asarray(pos_w, dtype=np.float32),
            "grasp_pos_world": np.asarray(pos_w, dtype=np.float32),
            "grasp_quat_world": np.asarray(current_ee_quat_w, dtype=np.float32),
        }
        self._pending_grasp_target = target
        self._pending_grasp_pos_w = np.asarray(pos_w, dtype=np.float32)
        self._pending_grasp_quat_w = np.asarray(current_ee_quat_w, dtype=np.float32)
        self._log(
            f"[TaskB-RL] ee_seg grasp target ready cls={target['class']} "
            f"depth={depth_m:.2f} pos_b=({pos_b[0]:.3f},{pos_b[1]:.3f},{pos_b[2]:.3f})"
        )
        return True

    def _head_target_depth_m(self, head_depth: np.ndarray | None) -> float:
        if head_depth is None or head_depth.ndim != 2 or self._active_target is None:
            return 0.0
        bbox = self._active_target.get("bbox")
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = head_depth.shape
            x1 = max(0, min(w - 1, x1))
            x2 = max(x1 + 1, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(y1 + 1, min(h, y2))
            patch = head_depth[y1:y2, x1:x2]
            valid = patch[(patch > 0.05) & np.isfinite(patch)]
            if valid.size >= 8:
                return float(np.median(valid))
        u = float(self._active_target.get("u", 0.0))
        v = float(self._active_target.get("v", 0.0))
        if u <= 0.0 and v <= 0.0:
            return 0.0
        return sample_depth_median(head_depth, int(round(u)), int(round(v)), radius=4)

    def _get_head_cam_extrinsic(self, robot):
        scene = self._scene()
        head_cam = self._get_scene_camera(scene, "head_camera")
        cfg_pos_b = np.asarray(self.nav_cfg.robot.head_cam.pos_b, dtype=np.float32)
        cfg_quat_b = np.asarray(self.nav_cfg.robot.head_cam.quat_b, dtype=np.float32)
        if head_cam is None or robot is None:
            return cfg_pos_b, cfg_quat_b, {"selected_source": "cfg_offset"}
        from isaaclab.utils.math import subtract_frame_transforms
        base_pos = robot.data.root_pos_w[0:1]
        base_quat = robot.data.root_quat_w[0:1]
        base_pos_np = base_pos[0].detach().cpu().numpy().astype(np.float32)
        base_quat_np = base_quat[0].detach().cpu().numpy().astype(np.float32)
        fk_cam_pos_w = base_pos_np + quat_rotate_vector(base_quat_np, cfg_pos_b)
        fk_cam_quat_w = quat_multiply(base_quat_np, cfg_quat_b)
        debug = {
            "selected_source": "cfg_offset",
            "base_pos_w": base_pos_np,
            "base_quat_w": base_quat_np,
            "cfg_pos_b": cfg_pos_b,
            "cfg_quat_b": cfg_quat_b,
            "fk_cam_pos_w": fk_cam_pos_w.astype(np.float32),
            "fk_cam_quat_w": fk_cam_quat_w.astype(np.float32),
            "sensor_candidates": {},
        }
        if hasattr(head_cam.data, "pos_w"):
            cam_pos_w = head_cam.data.pos_w[0:1]
            debug["sensor_pos_w"] = cam_pos_w[0].detach().cpu().numpy().astype(np.float32)
            for attr in ("quat_w_ros", "quat_w_world", "quat_w"):
                if not hasattr(head_cam.data, attr):
                    continue
                cam_quat_w = getattr(head_cam.data, attr)[0:1]
                pos_b_t, quat_b_t = subtract_frame_transforms(base_pos, base_quat, cam_pos_w, cam_quat_w)
                debug["sensor_candidates"][attr] = {
                    "pos_b": pos_b_t[0].detach().cpu().numpy().astype(np.float32),
                    "quat_b": quat_b_t[0].detach().cpu().numpy().astype(np.float32),
                    "quat_w": cam_quat_w[0].detach().cpu().numpy().astype(np.float32),
                }
        return cfg_pos_b, cfg_quat_b, debug

    def _build_pending_grasp_from_head_target(self, obs: dict, robot, arm) -> bool:
        if robot is None or arm is None or self._active_target is None:
            return False
        if str(self._active_target.get("source", "")) != "head":
            return False
        head_depth = parse_depth((obs.get("image") or {}), "head_depth")
        depth_m = self._head_target_depth_m(head_depth)
        if depth_m <= 0.05:
            return False
        u = float(self._active_target.get("u", 0.0))
        v = float(self._active_target.get("v", 0.0))
        p_cam = pixel_to_cam(u, v, depth_m, CAM_FX, CAM_FY, CAM_CX, CAM_CY)
        cam_pos_b, cam_quat_b, cam_debug = self._get_head_cam_extrinsic(robot)
        pos_b = transform_point_cam_to_base(p_cam, cam_pos_b, cam_quat_b)
        base_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
        base_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()
        pos_w = base_pos_w + quat_rotate_vector(base_quat_w, pos_b)
        pos_w[2] = 0.05
        pos_b = quat_rotate_vector(
            np.array([base_quat_w[0], -base_quat_w[1], -base_quat_w[2], -base_quat_w[3]], dtype=np.float32),
            np.asarray(pos_w, dtype=np.float32) - np.asarray(base_pos_w, dtype=np.float32),
        )
        candidate_3d_lines = []
        if cam_debug is not None:
            for name, candidate in cam_debug.get("sensor_candidates", {}).items():
                cand_pos_b = transform_point_cam_to_base(p_cam, candidate["pos_b"], candidate["quat_b"])
                cand_pos_w = base_pos_w + quat_rotate_vector(base_quat_w, cand_pos_b)
                candidate_3d_lines.append(
                    f"  3D_{name}: pos_b=({cand_pos_b[0]:.3f},{cand_pos_b[1]:.3f},{cand_pos_b[2]:.3f}) "
                    f"pos_w=({cand_pos_w[0]:.3f},{cand_pos_w[1]:.3f},{cand_pos_w[2]:.3f})"
                )
        target = {
            "id": self._active_target.get("id", f"head_rgbd_{self._step}"),
            "class": self._active_target.get("dominant_class", "unknown"),
            "source": "head_rgbd",
            "u": u,
            "v": v,
            "depth_m": float(depth_m),
            "pos_base": np.asarray(pos_b, dtype=np.float32),
            "pos_world": np.asarray(pos_w, dtype=np.float32),
            "grasp_pos_world": np.asarray(pos_w, dtype=np.float32),
            "grasp_quat_world": None,
        }
        self._pending_grasp_target = target
        self._pending_grasp_pos_w = np.asarray(pos_w, dtype=np.float32)
        self._pending_grasp_quat_w = None
        gt_pos_w, gt_err = self._nearest_gt_world_pos(target["class"], np.asarray(pos_w, dtype=np.float32))
        gt_info = " gt_w=(n/a)"
        gt_base_info = ""
        if gt_pos_w is not None and gt_err is not None:
            gt_info = (
                f" gt_w=({gt_pos_w[0]:.3f},{gt_pos_w[1]:.3f},{gt_pos_w[2]:.3f})"
                f" err=({gt_err[0]:+.3f},{gt_err[1]:+.3f},{gt_err[2]:+.3f})"
            )
            gt_pos_b = quat_rotate_vector(
                np.array([base_quat_w[0], -base_quat_w[1], -base_quat_w[2], -base_quat_w[3]], dtype=np.float32),
                np.asarray(gt_pos_w, dtype=np.float32) - np.asarray(base_pos_w, dtype=np.float32),
            )
            gt_base_info = (
                f" gt_b=({gt_pos_b[0]:.3f},{gt_pos_b[1]:.3f},{gt_pos_b[2]:.3f})"
                f" sel_err_b=({pos_b[0]-gt_pos_b[0]:+.3f},{pos_b[1]-gt_pos_b[1]:+.3f},{pos_b[2]-gt_pos_b[2]:+.3f})"
            )
        cam_world_info = ""
        if cam_debug is not None:
            cam_world_info = (
                f"  selected_extrinsic[{cam_debug.get('selected_source', 'unknown')}]: "
                f"pos_b=({cam_pos_b[0]:.4f},{cam_pos_b[1]:.4f},{cam_pos_b[2]:.4f}) "
                f"quat_b=({cam_quat_b[0]:.4f},{cam_quat_b[1]:.4f},{cam_quat_b[2]:.4f},{cam_quat_b[3]:.4f})\n"
                f"  base_w:   pos=({cam_debug['base_pos_w'][0]:.4f},{cam_debug['base_pos_w'][1]:.4f},{cam_debug['base_pos_w'][2]:.4f}) "
                f"quat=({cam_debug['base_quat_w'][0]:.4f},{cam_debug['base_quat_w'][1]:.4f},{cam_debug['base_quat_w'][2]:.4f},{cam_debug['base_quat_w'][3]:.4f})\n"
                f"  camera_fk_w: pos=({cam_debug['fk_cam_pos_w'][0]:.4f},{cam_debug['fk_cam_pos_w'][1]:.4f},{cam_debug['fk_cam_pos_w'][2]:.4f}) "
                f"quat=({cam_debug['fk_cam_quat_w'][0]:.4f},{cam_debug['fk_cam_quat_w'][1]:.4f},{cam_debug['fk_cam_quat_w'][2]:.4f},{cam_debug['fk_cam_quat_w'][3]:.4f})\n"
            )
            if "sensor_pos_w" in cam_debug:
                cam_world_info += (
                    f"  camera_sensor_w: pos=({cam_debug['sensor_pos_w'][0]:.4f},{cam_debug['sensor_pos_w'][1]:.4f},{cam_debug['sensor_pos_w'][2]:.4f})\n"
                )
            for name, candidate in cam_debug.get("sensor_candidates", {}).items():
                cam_world_info += (
                    f"  extrinsic_{name}: pos_b=({candidate['pos_b'][0]:.4f},{candidate['pos_b'][1]:.4f},{candidate['pos_b'][2]:.4f}) "
                    f"quat_b=({candidate['quat_b'][0]:.4f},{candidate['quat_b'][1]:.4f},{candidate['quat_b'][2]:.4f},{candidate['quat_b'][3]:.4f})\n"
                )
        self._log(
            f"[TaskB-RL] head rgbd grasp target ready cls={target['class']}\n"
            f"  RGBD: u={u:.1f} v={v:.1f} depth={depth_m:.3f} p_cam=({p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f})\n"
            f"{cam_world_info}"
            f"  3D_selected: pos_b=({pos_b[0]:.3f},{pos_b[1]:.3f},{pos_b[2]:.3f}) pos_w=({pos_w[0]:.3f},{pos_w[1]:.3f},{pos_w[2]:.3f}){gt_info}{gt_base_info}"
            f"{'' if not candidate_3d_lines else chr(10) + chr(10).join(candidate_3d_lines)}"
        )
        return True

    def _nearest_gt_world_pos(self, class_name: str, est_pos_w: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        scene = self._scene()
        if scene is None:
            return None, None
        try:
            class_id = CLASS_NAMES.index(class_name)
        except ValueError:
            return None, None
        best_pos_w = None
        best_dist = None
        for obj_idx in range(1, NUM_OBJECTS + 1):
            if object_index_to_class(obj_idx) != class_id:
                continue
            try:
                obj = scene[f"object_{obj_idx}"]
            except Exception:
                continue
            if not hasattr(obj, "data") or not hasattr(obj.data, "root_pos_w"):
                continue
            pos_w = obj.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
            dist = float(np.linalg.norm(pos_w - est_pos_w))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_pos_w = pos_w
        if best_pos_w is None:
            return None, None
        return best_pos_w, est_pos_w - best_pos_w

    def _get_ee_seg_center(self, obs: dict) -> tuple[Any | None, float]:
        if not self._ee_seg.ready or self._active_target is None:
            return None, 0.0
        scene = self._scene()
        ee_camera = self._get_scene_camera(scene, "ee_camera")
        ee_rgb = self._get_camera_output(ee_camera, "ee_camera", "rgb") if ee_camera is not None else None
        ee_depth = parse_depth((obs.get("image") or {}), "ee_depth")
        if ee_rgb is None:
            return None, 0.0
        if isinstance(ee_rgb, torch.Tensor):
            ee_rgb = ee_rgb.detach().cpu().numpy()
        ee_rgb = np.squeeze(ee_rgb)
        if ee_rgb.ndim == 3 and ee_rgb.shape[0] in (3, 4) and ee_rgb.shape[-1] not in (3, 4):
            ee_rgb = np.transpose(ee_rgb, (1, 2, 0))
        if ee_rgb.shape[-1] == 4:
            ee_rgb = ee_rgb[..., :3]
        if ee_rgb.dtype != np.uint8:
            if ee_rgb.max() <= 1.0:
                ee_rgb = (ee_rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                ee_rgb = ee_rgb.astype(np.uint8)
        ee_rgb = np.ascontiguousarray(ee_rgb)
        target_class_name = str(self._active_target.get("dominant_class", "unknown"))
        try:
            target_class = ObjectClass(target_class_name)
        except ValueError:
            target_class = None
        center = self._ee_seg.best_center(ee_rgb, target_class=target_class)
        if center is None:
            return None, 0.0
        return center, self._ee_seg_depth_m(center, ee_depth)

    def _step_crouch(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        robot = self._robot()
        if robot is None:
            self._task_state = "SEARCH"
            self._clear_pending_grasp()
            self._clear_target_lock()
            return self._leg_action(obs, zero_cmd)

        if self.sit_down_actor is not None:
            action_env = self._generate_sit_down_action_tensor(obs)
            self._sit_down_step_count += 1
            if self._is_sit_down_stable(robot):
                self._sit_down_stable_count += 1
            else:
                self._sit_down_stable_count = 0
            crouch_ready = (
                self._sit_down_step_count >= self.sit_down_min_steps
                and self._sit_down_stable_count >= self.sit_down_stable_steps_required
            )
            if crouch_ready:
                self._leg_posture_controller.state = "HOLDING_CROUCH"
            elif self._sit_down_step_count >= self.SIT_DOWN_TIMEOUT_STEPS:
                self._log(
                    f"[TaskB-RL] WARN: sit_down timeout {self._sit_down_step_count} steps "
                    f"stable={self._sit_down_stable_count}; force crouch_ready"
                )
                crouch_ready = True
                self._leg_posture_controller.state = "HOLDING_CROUCH"
            elif self._step % 20 == 0:
                roll, pitch, root_h = self._get_base_rpy_height(robot)
                self._log(
                    f"[TaskB-RL] CROUCHING step={self._sit_down_step_count} "
                    f"stable={self._sit_down_stable_count}/{self.sit_down_stable_steps_required} "
                    f"roll={float(roll[0]):.3f} pitch={float(pitch[0]):.3f} h={float(root_h[0]):.4f}"
                )
        else:
            action_env = self._generate_control_action_tensor(obs, zero_cmd, robot)
            crouch_ready = self._leg_posture_controller.state == "HOLDING_CROUCH"

        if crouch_ready:
            self._task_state = "PREGRASP"
            self._ee_align_step = 0
            self._arm_crouch_alpha = 0.0
            self._pregrasp_reach_stable_count = 0
            self._pregrasp_arm_settle_count = 0
            self._pregrasp_arm_step = 0
            self._pregrasp_arm_preset_done = False
            self._pregrasp_ee_quat_w = None
            if self._crouch_wait_start_step < 0:
                self._crouch_wait_start_step = self._step
            self._log("[TaskB-RL] crouch ready -> PREGRASP")

        return action_env

    def _step_pregrasp(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        robot = self._robot()
        if robot is None:
            self._task_state = "SEARCH"
            self._clear_pending_grasp()
            self._clear_target_lock()
            return self._leg_action(obs, zero_cmd)

        if self.sit_down_actor is not None:
            action_env = self._generate_sit_down_action_tensor(obs)
        else:
            action_env = self._generate_control_action_tensor(obs, zero_cmd, robot)

        arm = self._ensure_arm_controller()
        self._pregrasp_arm_step += 1
        if self._pregrasp_arm_step <= self.PREGRASP_ARM_SETTLE_STEPS:
            action_env = self._apply_crouch_arm_pose(action_env, robot)
            if self._step % 10 == 0:
                self._log(
                    f"[TaskB-RL] PREGRASP arm settle {self._pregrasp_arm_step}/{self.PREGRASP_ARM_SETTLE_STEPS}"
                )
            return action_env
        if not self._pregrasp_arm_preset_done:
            arm_ready = self._crouch_arm_pose_ready(robot)
            arm_timed_out = self._pregrasp_arm_step >= self.PREGRASP_ARM_TIMEOUT_STEPS
            if not arm_ready and not arm_timed_out:
                action_env = self._apply_crouch_arm_pose(action_env, robot)
                if self._step % 10 == 0:
                    self._log_pre_crouch_arm_status(robot)
                return action_env
            self._pregrasp_arm_preset_done = True
            if arm_timed_out and not arm_ready:
                self._log(
                    f"[TaskB-RL] PREGRASP arm preset timeout {self._pregrasp_arm_step} steps, skip ready check"
                )
            else:
                self._log(
                    f"[TaskB-RL] PREGRASP arm preset ready at step {self._pregrasp_arm_step}"
                )
            if arm is not None:
                self._pregrasp_ee_quat_w = arm.get_ee_pose()[1].copy()
                self._log(f"[TaskB-RL] PREGRASP captured ee_quat_w={np.round(self._pregrasp_ee_quat_w, 4)}")

        action_env = self._apply_crouch_arm_pose(action_env, robot)

        pregrasp_hold_steps = self._pregrasp_arm_step - self.PREGRASP_ARM_TIMEOUT_STEPS
        if pregrasp_hold_steps >= self.PREGRASP_HOLD_TIMEOUT_STEPS:
            self._log(
                f"[TaskB-RL] PREGRASP hold timeout {pregrasp_hold_steps}/{self.PREGRASP_HOLD_TIMEOUT_STEPS} steps, "
                f"giving up target -> STAND_UP"
            )
            self._pending_grasp_status = "failed"
            self._mark_active_target_processed(success=False)
            self._entered_crouch_from_yolo = False
            self._reset_sit_down_tracking()
            self._leg_posture_controller.start_stand_up(robot)
            self._task_state = "STAND_UP"
            return action_env

        if self._step % 20 == 0:
            self._log(
                f"[TaskB-RL] PREGRASP holding crouch arm pose (no IK) "
                f"hold={pregrasp_hold_steps}/{self.PREGRASP_HOLD_TIMEOUT_STEPS}"
            )
        return action_env

    def _step_arm_grasp(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        arm = self._arm_grasp
        robot = self._robot()
        if arm is None or robot is None:
            self._pending_grasp_status = "failed"
            if robot is not None:
                self._mark_active_target_processed(success=False)
                self._leg_posture_controller.start_stand_up(robot)
                self._task_state = "STAND_UP"
                return self._generate_control_action_tensor(obs, zero_cmd, robot)
            self._task_state = "SEARCH"
            self._clear_pending_grasp()
            self._clear_target_lock()
            return self._leg_action(obs, zero_cmd)

        action_env = self._generate_sit_down_action_tensor(obs) if self.sit_down_actor is not None else self._generate_control_action_tensor(obs, zero_cmd, robot)
        done, success = arm.step(robot, self._scene(), self.dt)
        action_env = arm.apply_to_action_tensor(action_env, robot)
        if done:
            self._pending_grasp_status = "grasped" if success else "failed"
            self._reset_sit_down_tracking()
            self._mark_active_target_processed(success=success)
            if success:
                self._lock_carry_pose(arm)
                self._entered_crouch_from_yolo = False
            self._leg_posture_controller.start_stand_up(robot)
            self._task_state = "STAND_UP"
            print(f"[TaskB-RL] arm grasp finished, success={success}. Starting stand up.", flush=True)
        return action_env

    def _step_stand_up(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        robot = self._robot()
        if robot is None:
            next_state = "CARRY" if self._pending_grasp_status == "grasped" else "SEARCH"
            self._task_state = next_state
            if next_state == "SEARCH":
                self._clear_target_lock()
            self._clear_pending_grasp()
            return self._leg_action(obs, zero_cmd)

        if self.sit_down_actor is not None:
            action_env = self._generate_sit_down_action_tensor(obs)
            self._sit_down_step_count += 1
            if self._is_sit_down_stable(robot):
                self._sit_down_stable_count += 1
            else:
                self._sit_down_stable_count = 0
            stand_up_done = (
                self._sit_down_step_count >= self.sit_down_min_steps
                and self._sit_down_stable_count >= self.sit_down_stable_steps_required
            )
        else:
            action_env = self._generate_control_action_tensor(obs, zero_cmd, robot)
            stand_up_done = self._leg_posture_controller.state == "IDLE"

        action_env = self._apply_carry_arm(action_env)
        if stand_up_done:
            next_state = "CARRY" if self._pending_grasp_status == "grasped" else "SEARCH"
            self._clear_active_target()
            if next_state == "SEARCH":
                self._hold_carry_pose = False
                self._carry_arm_jpos = None
                self._carry_gripper = None
                self._clear_target_lock()
                if self._arm_grasp is not None:
                    self._arm_grasp.reset()
            self._task_state = next_state
            self._entered_crouch_from_yolo = False
            self._reset_sit_down_tracking()
            self._clear_pending_grasp()
            print(f"[TaskB-RL] stand up complete → {next_state}", flush=True)
        return action_env

    @staticmethod
    def _bin_pos(z: float) -> np.ndarray:
        return np.array([float(BIN_CENTER[0]), float(BIN_CENTER[1]), float(z)], dtype=np.float32)

    def _step_drop(self, action_env: torch.Tensor) -> torch.Tensor:
        arm = self._ensure_arm_controller()
        robot = self._robot()
        if arm is None or robot is None:
            self._finish_drop()
            return action_env
        if not self._drop_phase:
            self._drop_phase = "MOVE_OVER"
            self._drop_wait = 0
            print(f"[TaskB-RL] DROP start @ bin dist={self.BIN_ARRIVE_M}m", flush=True)

        ee_pos, _ = arm.get_ee_pose()
        if self._drop_phase == "MOVE_OVER":
            arm.close_gripper()
            tgt = self._bin_pos(self.DROP_OVER_Z)
            arm.move_ee_to_pose(tgt, None)
            if arm.ee_reached(ee_pos, tgt):
                self._drop_phase = "LOWER"
                print("[TaskB-RL] DROP: over bin → lower", flush=True)
        elif self._drop_phase == "LOWER":
            arm.close_gripper()
            tgt = self._bin_pos(self.DROP_RELEASE_Z)
            arm.move_ee_to_pose(tgt, None)
            if arm.ee_reached(ee_pos, tgt):
                self._drop_phase = "OPEN"
                self._drop_wait = 0
                print("[TaskB-RL] DROP: release", flush=True)
        elif self._drop_phase == "OPEN":
            arm.open_gripper()
            self._drop_wait += 1
            if self._drop_wait >= self.DROP_OPEN_STEPS:
                self._drop_phase = "RETREAT"
        elif self._drop_phase == "RETREAT":
            arm.open_gripper()
            tgt = self._bin_pos(self.DROP_RETREAT_Z)
            arm.move_ee_to_pose(tgt, None)
            if arm.ee_reached(ee_pos, tgt):
                self._drop_phase = "DONE"
        elif self._drop_phase == "DONE":
            self._finish_drop()
            return action_env
        return arm.apply_to_action_tensor(action_env, robot)

    def _finish_drop(self) -> None:
        self._objects_dropped += 1
        self._drop_phase = ""
        self._drop_wait = 0
        self._hold_carry_pose = False
        self._carry_arm_jpos = None
        self._carry_gripper = None
        self._task_state = "SEARCH"
        self._clear_pending_grasp()
        self._clear_target_lock()
        self._entered_crouch_from_yolo = False
        if self._arm_grasp is not None:
            self._arm_grasp.reset()
        self._post_drop_stand_until = self._step + self.WARMUP_STEPS // 2
        print(f"[TaskB-RL] DROP done (total={self._objects_dropped}) → SEARCH next object", flush=True)

    def _log_status(self, perc: dict, vx: float, wz: float, state: str) -> None:
        if self._step % 10 != 0:
            return
        nav = perc.get("target_nav") or {}
        nd = self._nav_depth(nav) if nav else 0.0
        if self._active_target is not None:
            yolo_info = (
                f"active={self._active_target.get('dominant_class', '-')} "
                f"src={self._active_target.get('source', '-')} "
                f"err_u={float(self._active_target.get('err_u', 0.0)):+.0f}px "
                f"depth={float(self._active_target.get('depth', 0.0)):.2f}m "
                f"stage={self._target_stage}"
            )
        else:
            yolo_info = "active=none"

        last_seen_info = ""
        if self._last_seen_target is not None:
            last_seen_age = self._step - self._last_seen_step
            last_seen_info = (
                f" last_seen_depth={float(self._last_seen_target['depth']):.2f}m "
                f"last_seen_age={last_seen_age}"
            )

        locked_info = (
            f" target_stage={self._target_stage} lost={self._target_lost_count} "
            f"head_confirm={self._head_confirm_count}/{self.HEAD_CONFIRM_STEPS} "
            f"pre_crouch={self._pre_crouch_confirm_count}/{self.PRE_CROUCH_CONFIRM_STEPS}"
        )

        print(
            f"[TaskB] step={self._step} state={state} drop={self._drop_phase or '-'} "
            f"perc={perc.get('phase')} ee={len(perc.get('ee_objects') or [])} "
            f"nav_d={nd:.2f} bin_d={self._dist_to_bin(perc):.2f} dropped={self._objects_dropped} "
            f"cmd=({vx:.2f},{wz:.2f}) {yolo_info}{locked_info}{last_seen_info}",
            flush=True,
        )

    def predicts(self, obs, current_score):
        score_delta = current_score - self._last_score
        if score_delta > 0.001:
            print(f"[+{score_delta:.2f}] Score: {current_score:.2f} | Step: {self._step}", flush=True)
        self._last_score = current_score

        self._process_camera_debug()

        self._last_nav_vel = self.axis_nav.compute_velocity(obs)
        self._update_last_seen_target()
        self._save_yolo_detection(obs)
        perc = self.perception.process(obs, self.dt) if self.perception is not None else {}

        vx, wz, nav_state = self._choose_velocity(perc, obs)

        if nav_state in {"EE_TRACK", "FINAL_APPROACH", "HEAD_APPROACH", "EE_BASE_ALIGN", "SEARCH_SPIN"}:
            vx, wz = self._smooth_cmd(vx, wz)
        else:
            self._cmd_vx_filt = float(vx)
            self._cmd_wz_filt = float(wz)

        if self._task_state in {"SEARCH", "APPROACH"}:
            if nav_state == "EE_TRACK":
                vx = float(np.clip(vx, 0.0, 0.8))
            elif vx > 0.0:
                vx = float(np.clip(vx, 0.2, 0.8))
            wz = float(np.clip(wz, -0.5, 0.5))

        if nav_state == "START_CROUCH":
            robot = self._robot()
            if robot is not None and self._task_state not in {"PRE_CROUCH_SETTLE", "CROUCHING", "PREGRASP", "GRASP_ARM", "STAND_UP"}:
                self._pre_crouch_settle_steps = 0
                self._pre_crouch_wait_steps = 0
                self._arm_crouch_alpha = 0.0
                self._crouch_arm_hold_jpos = None
                self._task_state = "PRE_CROUCH_SETTLE"
                self._log("[TaskB-RL] PRE_CROUCH -> PRE_CROUCH_SETTLE")
            vx, wz = 0.0, 0.0

        protected_states = {"PRE_CROUCH_SETTLE", "CROUCHING", "PREGRASP", "GRASP_ARM", "STAND_UP", "CARRY", "DROP"}

        if nav_state in {"EE_TRACK", "FINAL_APPROACH", "HEAD_APPROACH", "EE_BASE_ALIGN", "PRE_CROUCH"}:
            if self._task_state == "SEARCH":
                self._task_state = "APPROACH"
        elif nav_state in {"SEARCH_WAIT", "SEARCH_SPIN"}:
            if self._task_state not in protected_states:
                self._task_state = "SEARCH"
        elif self._task_state not in protected_states and nav_state != "STAND":
            self._task_state = nav_state

        self._set_velocity_commands(vx, 0.0, wz)
        self._update_grasp(perc)

        base_cmd = np.array([vx, 0.0, wz], dtype=np.float32)
        robot = self._robot()

        if self._task_state == "PRE_CROUCH_SETTLE":
            action_env = self._step_pre_crouch_settle(obs)
        elif self._task_state == "CROUCHING":
            action_env = self._step_crouch(obs)
        elif self._task_state == "PREGRASP":
            action_env = self._step_pregrasp(obs)
        elif self._task_state == "GRASP_ARM":
            action_env = self._step_arm_grasp(obs)
        elif self._task_state == "STAND_UP":
            action_env = self._step_stand_up(obs)
        else:
            if robot is not None:
                action_env = self._generate_control_action_tensor(obs, base_cmd, robot)
            else:
                action_env = self._leg_action(obs, base_cmd)
            if self._task_state == "DROP" or self._drop_phase:
                action_env = self._step_drop(action_env)
            elif self._task_state == "CARRY":
                action_env = self._apply_carry_arm(action_env)

        self._log_status(perc, vx, wz, nav_state if nav_state == "STAND" else self._task_state)
        self._step += 1
        return {"action": action_env.detach().cpu().numpy().tolist(), "giveup": False}