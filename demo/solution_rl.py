"""
Task B 操作层 — 感知 taskb_perception + RL腿控 + YOLO导航

状态机
  SEARCH → APPROACH → CROUCHING → GRASP_ARM → STAND_UP → CARRY → DROP → SEARCH

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

from taskb_perception.config import TARGET_BIN_RADIUS as BIN_RADIUS, TARGET_BIN_XY  # noqa: E402
from taskb_perception import AxisNavController, PerceptionConfig  # noqa: E402

BIN_CENTER = np.array([TARGET_BIN_XY[0], TARGET_BIN_XY[1], 0.0], dtype=np.float32)

try:
    from rgbd_pure_dual_pipeline import RgbdPureDualPipeline  # noqa: E402
except ImportError:
    RgbdPureDualPipeline = None


class AlgSolution:
    ACTION_SCALE = 0.5
    LEG_JOINT_NAMES = list(B2_PIPER_LEG_JOINT_NAMES)
    ARM_JOINT_NAMES = [
        "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
        "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8",
    ]
    EE_BODY_NAME = "gripper_base"

    GRASP_DEPTH_M = 1.10
    WARMUP_STEPS = 90
    SEARCH_VX = 0.0
    SEARCH_WZ = 0.12
    YAW_TURN_THRESH = 0.45
    NAV_WZ_GAIN = 0.85
    NAV_WZ_MAX = 0.25
    NAV_VX_MIN = 0.25
    NAV_VX_MAX = 0.45
    BIN_ARRIVE_M = 1.05

    DROP_OVER_Z = 0.48
    DROP_RELEASE_Z = 0.20
    DROP_RETREAT_Z = 0.58
    DROP_OPEN_STEPS = 40

    def __init__(self, env=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = 0.02

        policy_path = self._resolve_policy_path()
        self.policy = self._load_leg_policy(policy_path)
        self._leg_mode = "rl" if self.policy is not None else "scripted"

        if RgbdPureDualPipeline is not None:
            self.perception = RgbdPureDualPipeline()
        else:
            self.perception = None
            print("[TaskB-RL] WARN: RgbdPureDualPipeline 不可用，抓取阶段将尽量依赖 YOLO 导航", flush=True)

        self.nav_cfg = PerceptionConfig(
            nav_method="axis_align",
            nav_detect_source="ee",
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
        self._arm_default_action = torch.zeros((1, 8), device=self.device, dtype=torch.float32)

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
        self._camera_follow_enabled = True

        # 速度平滑状态变量
        self._cmd_vx_filt = 0.0
        self._cmd_wz_filt = 0.0
        self.CMD_ALPHA = 0.25
        self.MAX_DVX = 0.04
        self.MAX_DWZ = 0.04

        self._hold_carry_pose = False
        self._carry_arm_jpos: torch.Tensor | None = None
        self._carry_gripper: torch.Tensor | None = None
        self._pending_grasp_status: str | None = None
        self._pending_grasp_target: dict[str, Any] | None = None
        self._pending_grasp_pos_w: np.ndarray | None = None
        self._pending_grasp_quat_w: np.ndarray | None = None

        self._drop_phase = ""
        self._drop_wait = 0
        self._objects_dropped = 0
        self._post_drop_stand_until = 0

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

        print(
            f"[TaskB-RL] full loop: search→crouch→grasp→stand→carry→drop | "
            f"bin={BIN_CENTER.tolist()} r={BIN_RADIUS} leg={self._leg_mode}",
            flush=True,
        )
        if self._leg_mode == "scripted":
            print("[TaskB-RL] *** leg=scripted 站不稳! 请确保 demo/policy.pt 存在且非 LFS 指针 ***", flush=True)

    def _resolve_policy_path(self) -> str:
        candidates = [
            os.path.join(_DEMO_DIR, "policy.pt"),
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
        self._drop_phase = ""
        self._drop_wait = 0
        self._post_drop_stand_until = 0
        self._reset_sit_down_tracking()
        self._leg_posture_controller.reset()
        if self._arm_grasp is not None:
            self._arm_grasp.reset()

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
            return 0.06, float(np.clip(yaw * self.NAV_WZ_GAIN, -self.NAV_WZ_MAX, self.NAV_WZ_MAX))
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
            return 0.15, 0.0
        rp, yaw = xy
        delta = np.asarray(BIN_CENTER[:2], dtype=np.float32) - rp[:2]
        dist = float(np.linalg.norm(delta))
        if dist < self.BIN_ARRIVE_M:
            return 0.0, 0.0
        yaw_to_bin = float(math.atan2(delta[1], delta[0]))
        err = (yaw_to_bin - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > 0.35:
            return 0.10, float(np.clip(err * 0.9, -0.45, 0.45))
        vx = float(np.clip(0.12 + 0.25 * min(dist, 4.0) / 4.0, 0.12, 0.45))
        wz = float(np.clip(err * 0.6, -0.25, 0.25))
        return vx, wz

    def _choose_velocity(self, perc: dict) -> tuple[float, float, str]:
        if self._step < self.WARMUP_STEPS or self._step < self._post_drop_stand_until:
            return 0.0, 0.0, "STAND"
        if self._task_state in {"CROUCHING", "GRASP_ARM", "STAND_UP"}:
            return 0.0, 0.0, self._task_state
        if self._task_state == "DROP" or self._drop_phase:
            return 0.0, 0.0, "DROP"
        if self._task_state == "CARRY":
            if self._at_bin(perc):
                return 0.0, 0.0, "DROP"
            vx, wz = self._cmd_carry(perc)
            return min(vx, 0.32), wz, "CARRY"

        if self._task_state in {"SEARCH", "APPROACH"}:
            nav_target = self.axis_nav.nav_target
            if nav_target is not None:
                depth = float(nav_target.depth_m)
                err_u = float(nav_target.err_u)

                if depth > 2.0:
                    dead_px = 35.0
                    wz_gain = 0.004
                    wz_max = 0.22

                    if abs(err_u) < dead_px:
                        wz = 0.0
                    else:
                        wz = -wz_gain * err_u
                    wz = float(np.clip(wz, -wz_max, wz_max))

                    if abs(err_u) < 80.0:
                        vx = 0.80
                    elif abs(err_u) < 150.0:
                        vx = 0.60
                    else:
                        vx = 0.40

                    return vx, wz, "NAV_YOLO"

                elif depth > self.GRASP_DEPTH_M:
                    dead_px = 25.0
                    wz_gain = 0.005
                    wz_max = 0.25

                    if abs(err_u) < dead_px:
                        wz = 0.0
                    else:
                        wz = -wz_gain * err_u
                    wz = float(np.clip(wz, -wz_max, wz_max))

                    if abs(err_u) < 60.0:
                        vx = 0.40
                    elif abs(err_u) < 120.0:
                        vx = 0.30
                    else:
                        vx = 0.20

                    return vx, wz, "NAV_YOLO"

                elif depth > 0.05:
                    dead_px = 20.0
                    wz_gain = 0.004
                    wz_max = 0.18

                    if abs(err_u) < dead_px:
                        if nav_target.on_axis:
                            return 0.08, 0.0, "APPROACH_SLOW"
                        return 0.06, 0.0, "APPROACH_SLOW"

                    wz = float(np.clip(-wz_gain * err_u, -wz_max, wz_max))
                    return 0.06, wz, "APPROACH_ALIGN"

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

    def _update_grasp(self, perc: dict) -> None:
        if self._is_standing_phase() or self._hold_carry_pose or self._drop_phase:
            return
        if self._task_state in {"CROUCHING", "GRASP_ARM", "STAND_UP", "CARRY"}:
            return
        if str(perc.get("phase") or "") != "grasp":
            return
        grasp = perc.get("target_grasp")
        if not isinstance(grasp, dict) or grasp.get("grasp_pos_world") is None:
            return
        robot = self._robot()
        if robot is None:
            return
        self._pending_grasp_target = grasp
        self._pending_grasp_pos_w = np.asarray(grasp.get("pos_world") or grasp["grasp_pos_world"], dtype=np.float32)
        gq = grasp.get("grasp_quat_world")
        self._pending_grasp_quat_w = None if gq is None else np.asarray(gq, dtype=np.float32)
        self._reset_sit_down_tracking()
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

    def _step_crouch(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        robot = self._robot()
        if robot is None:
            self._task_state = "SEARCH"
            self._clear_pending_grasp()
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
        else:
            action_env = self._generate_control_action_tensor(obs, zero_cmd, robot)
            crouch_ready = self._leg_posture_controller.state == "HOLDING_CROUCH"

        if crouch_ready:
            arm = self._ensure_arm_controller()
            if arm is not None and self._pending_grasp_target is not None and self._pending_grasp_pos_w is not None:
                current_ee_quat_w = arm.get_ee_pose()[1]
                arm.start_grasp(
                    self._pending_grasp_target,
                    self._pending_grasp_pos_w,
                    current_ee_quat_w=current_ee_quat_w if self._pending_grasp_quat_w is None else self._pending_grasp_quat_w,
                )
                self._task_state = "GRASP_ARM"
                print(f"[TaskB-RL] crouch ready, start arm grasp id={self._pending_grasp_target.get('id')}", flush=True)
            else:
                self._pending_grasp_status = "failed"
                self._reset_sit_down_tracking()
                self._leg_posture_controller.start_stand_up(robot)
                self._task_state = "STAND_UP"
                print("[TaskB-RL] arm controller unavailable after crouch, start stand up.", flush=True)
        return action_env

    def _step_arm_grasp(self, obs: dict) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        arm = self._arm_grasp
        robot = self._robot()
        if arm is None or robot is None:
            self._pending_grasp_status = "failed"
            if robot is not None:
                self._leg_posture_controller.start_stand_up(robot)
                self._task_state = "STAND_UP"
                return self._generate_control_action_tensor(obs, zero_cmd, robot)
            self._task_state = "SEARCH"
            self._clear_pending_grasp()
            return self._leg_action(obs, zero_cmd)

        action_env = self._generate_sit_down_action_tensor(obs) if self.sit_down_actor is not None else self._generate_control_action_tensor(obs, zero_cmd, robot)
        done, success = arm.step(robot, self._scene(), self.dt)
        action_env = arm.apply_to_action_tensor(action_env, robot)
        if done:
            self._pending_grasp_status = "grasped" if success else "failed"
            self._reset_sit_down_tracking()
            if success:
                self._lock_carry_pose(arm)
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
            if next_state == "SEARCH":
                self._hold_carry_pose = False
                self._carry_arm_jpos = None
                self._carry_gripper = None
                if self._arm_grasp is not None:
                    self._arm_grasp.reset()
            self._task_state = next_state
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
        if self._arm_grasp is not None:
            self._arm_grasp.reset()
        self._post_drop_stand_until = self._step + self.WARMUP_STEPS // 2
        print(f"[TaskB-RL] DROP done (total={self._objects_dropped}) → SEARCH next object", flush=True)

    def _log_status(self, perc: dict, vx: float, wz: float, state: str) -> None:
        if self._step % 60 != 0:
            return
        nav = perc.get("target_nav") or {}
        nd = self._nav_depth(nav) if nav else 0.0
        yolo_target = self.axis_nav.nav_target
        if yolo_target is not None:
            yolo_info = (
                f"YOLO={yolo_target.obj_class} conf={yolo_target.confidence:.2f} "
                f"err_u={yolo_target.err_u:+.0f}px on_axis={yolo_target.on_axis} "
                f"depth={yolo_target.depth_m:.2f}m lock={yolo_target.locked}"
            )
        else:
            yolo_info = "YOLO=no_target"
        print(
            f"[TaskB] step={self._step} state={state} drop={self._drop_phase or '-'} "
            f"perc={perc.get('phase')} ee={len(perc.get('ee_objects') or [])} "
            f"nav_d={nd:.2f} bin_d={self._dist_to_bin(perc):.2f} dropped={self._objects_dropped} "
            f"cmd=({vx:.2f},{wz:.2f}) {yolo_info}",
            flush=True,
        )

    def predicts(self, obs, current_score):
        if current_score > 1:
            return {"action": [], "giveup": True}

        self._process_camera_debug()

        self.axis_nav.detect_targets(obs, manage_lock=True)
        self._save_yolo_detection(obs)
        self._last_nav_vel = self.axis_nav.compute_velocity(obs)
        perc = self.perception.process(obs, self.dt) if self.perception is not None else {}

        vx, wz, nav_state = self._choose_velocity(perc)

        # 对导航状态进行速度平滑
        if nav_state in {"NAV_YOLO", "APPROACH_ALIGN", "APPROACH_SLOW", "SEARCH_YOLO"}:
            vx, wz = self._smooth_cmd(vx, wz)
        else:
            self._cmd_vx_filt = float(vx)
            self._cmd_wz_filt = float(wz)

        if nav_state in {"APPROACH_SLOW", "APPROACH_ALIGN"} and self._task_state == "SEARCH":
            self._task_state = "APPROACH"
            print("[TaskB-RL] YOLO: 到达目标距离 → 准备抓取", flush=True)

        # 修复状态机污染问题：导航子状态不覆盖主任务状态
        if nav_state in {"NAV_YOLO", "APPROACH_ALIGN", "APPROACH_SLOW", "SEARCH_YOLO"}:
            if self._task_state == "SEARCH":
                self._task_state = "APPROACH"
        elif self._task_state not in {"CROUCHING", "GRASP_ARM", "STAND_UP"} and nav_state != "STAND":
            self._task_state = nav_state
        self._set_velocity_commands(vx, 0.0, wz)
        self._update_grasp(perc)

        base_cmd = np.array([vx, 0.0, wz], dtype=np.float32)
        robot = self._robot()

        if self._task_state == "CROUCHING":
            action_env = self._step_crouch(obs)
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
