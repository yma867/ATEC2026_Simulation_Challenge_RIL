import math
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCEPTION_DIR = os.path.join(REPO_ROOT, "taskb_perception")
if os.path.isdir(PERCEPTION_DIR) and PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, PERCEPTION_DIR)

try:
    from config import BIN_CENTER, ROBOT_INIT_POS, ROBOT_INIT_YAW
except Exception:
    BIN_CENTER = np.array([-3.0, -10.0, 0.0], dtype=np.float32)
    ROBOT_INIT_POS = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
    ROBOT_INIT_YAW = 0.0

try:
    from perception_pipeline import PerceptionPipeline
except Exception:
    PerceptionPipeline = None

try:
    from atec_rl_lab.assets.robots.b2 import UNITREE_B2_PIPER_CFG

    B2_PIPER_LEG_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
    B2_PIPER_ARM_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.arm_joint_names)
    B2_PIPER_TOTAL_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.joint_names)
except Exception:
    B2_PIPER_LEG_JOINT_NAMES = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    B2_PIPER_ARM_JOINT_NAMES = [
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
        "arm_joint7",
        "arm_joint8",
    ]
    B2_PIPER_TOTAL_JOINT_NAMES = B2_PIPER_LEG_JOINT_NAMES + B2_PIPER_ARM_JOINT_NAMES


class B2PiperActor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class AlgSolution:
    def __init__(self):
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.nav_debug = os.getenv("ATEC_TASKB_NAV_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
        self.nav_debug_every = max(1, int(os.getenv("ATEC_TASKB_NAV_DEBUG_EVERY", "25")))
        self.use_perception = os.getenv("ATEC_TASKB_USE_PERCEPTION", "1").lower() in {"1", "true", "yes", "on"}
        self.target_mode = os.getenv("ATEC_TASKB_TARGET_MODE", "object").lower()
        self._nav_debug_step = 0
        self._step_count = 0
        self._perception_error_printed = False
        self._odom_pos = np.asarray(ROBOT_INIT_POS, dtype=np.float32).copy()
        self._odom_yaw = float(ROBOT_INIT_YAW)
        self._last_base_cmd = np.zeros(3, dtype=np.float32)

        self.checkpoint_path = os.path.join(
            REPO_ROOT,
            "logs",
            "rsl_rl",
            "unitree_b2_piper_flat",
            "2026-06-02_14-40-32",
            "model_4999.pt",
        )
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")

        self.perception = PerceptionPipeline(device=self.device_str) if self.use_perception and PerceptionPipeline is not None else None

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        self.leg_joint_names = list(B2_PIPER_LEG_JOINT_NAMES)
        self.arm_joint_names = list(B2_PIPER_ARM_JOINT_NAMES)
        self.leg_action_dim = len(self.leg_joint_names)
        self.arm_action_dim = len(self.arm_joint_names)
        self.total_action_dim = len(B2_PIPER_TOTAL_JOINT_NAMES)

        actor_input_dim = state_dict["actor.0.weight"].shape[1]
        actor_output_dim = state_dict["actor.6.bias"].shape[0]
        self.actor = B2PiperActor(actor_input_dim, actor_output_dim).to(self.device)
        actor_state = {key: value for key, value in state_dict.items() if key.startswith("actor.")}
        self.actor.load_state_dict(actor_state, strict=True)
        self.actor.eval()

        self.leg_action_scale = torch.tensor(
            [0.25 if "hip_joint" in name else 0.5 for name in self.leg_joint_names],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self.leg_action_scale_inv = torch.reciprocal(self.leg_action_scale)
        
        # Define default arm joint positions for known robots
        # This matches the configuration in env_cfg.py for TaskB B2Piper
        b2_piper_arm_defaults = {
            "arm_joint1": 0.0,
            "arm_joint2": 2.13,
            "arm_joint3": -1.20,
            "arm_joint4": 0.0,
            "arm_joint5": -0.8,
            "arm_joint6": 0.0,
            "arm_joint7": 0.0,
            "arm_joint8": 0.0,
        }
        
        # Build arm default action vector, fallback to 0.0 for unknown joints
        arm_default_pos = [b2_piper_arm_defaults.get(name, 0.0) for name in self.arm_joint_names]
        
        self.arm_default_action = torch.tensor(
            arm_default_pos,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        if actor_input_dim != 45 or actor_output_dim != self.leg_action_dim:
            raise ValueError(
                f"Checkpoint shape mismatch: actor input={actor_input_dim}, output={actor_output_dim}, "
                f"expected input=45, output={self.leg_action_dim}"
            )

        self.stand_still_steps = max(0, int(float(os.getenv("ATEC_TASKB_STAND_SECONDS", "2.0")) / 0.02))
        self.stop_distance = float(os.getenv("ATEC_TASKB_STOP_DISTANCE", "0.9"))
        self.object_stop_distance = float(os.getenv("ATEC_TASKB_OBJECT_STOP_DISTANCE", "0.7"))
        self.stop_tolerance = float(os.getenv("ATEC_TASKB_STOP_TOLERANCE", "0.2"))
        self.object_stop_tolerance = float(os.getenv("ATEC_TASKB_OBJECT_STOP_TOLERANCE", "0.15"))
        self.object_yaw_tolerance = float(os.getenv("ATEC_TASKB_OBJECT_YAW_TOLERANCE", "0.15"))
        self.slow_down_radius = float(os.getenv("ATEC_TASKB_SLOW_DOWN_RADIUS", "1.5"))
        self.max_lin_vel = float(os.getenv("ATEC_TASKB_MAX_LIN_VEL", "0.8"))
        self.max_lat_vel = float(os.getenv("ATEC_TASKB_MAX_LAT_VEL", "0.4"))
        self.max_ang_vel = float(os.getenv("ATEC_TASKB_MAX_ANG_VEL", "0.8"))
        self.heading_kp = float(os.getenv("ATEC_TASKB_HEADING_KP", "1.2"))
        self.min_heading_lin_scale = float(os.getenv("ATEC_TASKB_MIN_HEADING_LIN_SCALE", "0.25"))
        self.default_bin_center = np.asarray(BIN_CENTER, dtype=np.float32)
        self.dt = 0.02

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return None

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _safe_numpy(vector, fallback: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector if vector is not None else fallback, dtype=np.float32)
        if arr.shape[0] < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: arr.shape[0]] = arr
            arr = padded
        return np.nan_to_num(arr[:3], nan=0.0, posinf=0.0, neginf=0.0)

    def _update_local_odometry(self, obs: dict[str, Any]) -> dict[str, Any]:
        proprio = torch.as_tensor(obs["proprio"], device="cpu", dtype=torch.float32).squeeze(0).numpy()
        lin_vel = np.nan_to_num(proprio[0:3], nan=0.0, posinf=0.0, neginf=0.0)
        yaw_rate = float(np.nan_to_num(proprio[5], nan=0.0, posinf=0.0, neginf=0.0))

        self._odom_yaw = self._wrap_angle(self._odom_yaw + yaw_rate * self.dt)
        cos_yaw = math.cos(self._odom_yaw)
        sin_yaw = math.sin(self._odom_yaw)
        vel_world = np.array(
            [
                cos_yaw * lin_vel[0] - sin_yaw * lin_vel[1],
                sin_yaw * lin_vel[0] + cos_yaw * lin_vel[1],
            ],
            dtype=np.float32,
        )
        self._odom_pos[:2] += vel_world * self.dt
        self._odom_pos[2] = float(ROBOT_INIT_POS[2])

        return {
            "robot": {
                "pos_world": self._odom_pos.tolist(),
                "yaw": float(self._odom_yaw),
            },
            "bin": {
                "center_world": self.default_bin_center.tolist(),
            },
        }

    def _select_target(self, perception_output: dict[str, Any]) -> dict[str, Any] | None:
        target = perception_output.get("target")
        if target is not None:
            return target

        objects = perception_output.get("objects_detailed") or perception_output.get("objects_remaining") or []
        candidates = []
        for obj in objects:
            if obj.get("in_bin", False):
                continue
            pos_robot = obj.get("pos_robot")
            if pos_robot is None:
                continue
            pos = self._safe_numpy(pos_robot, np.zeros(3, dtype=np.float32))
            if pos[0] <= 0.05:
                continue
            dist = float(obj.get("dist_to_robot", obj.get("dist", np.linalg.norm(pos[:2]))))
            if np.isfinite(dist):
                candidates.append((dist, obj))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _compute_object_cmd(self, target: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        pos_robot = self._safe_numpy(target.get("pos_robot"), np.zeros(3, dtype=np.float32))
        if pos_robot[0] <= 0.05:
            return np.zeros(3, dtype=np.float32), {
                "phase": "search",
                "goal_dist": 0.0,
                "target_dist": 0.0,
                "heading_error": 0.0,
                "target": target,
            }

        target_dist = float(np.linalg.norm(pos_robot[:2]))
        heading_error = float(math.atan2(pos_robot[1], pos_robot[0]))
        forward_error = float(pos_robot[0] - self.object_stop_distance)
        lateral_error = float(pos_robot[1])
        goal_dist = float(np.linalg.norm([forward_error, lateral_error]))

        if target_dist <= self.object_stop_distance + self.object_stop_tolerance:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = float(np.clip(heading_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))
            phase = "align"
            if abs(heading_error) <= self.object_yaw_tolerance:
                ang_z = 0.0
                phase = "stopped"
        else:
            approach_scale = float(np.clip(goal_dist / self.slow_down_radius, 0.0, 1.0))
            heading_scale = max(self.min_heading_lin_scale, math.cos(min(abs(heading_error), math.pi / 2.0)))
            lin_x = float(np.clip(forward_error * approach_scale * heading_scale, -self.max_lin_vel, self.max_lin_vel))
            lin_y = float(np.clip(lateral_error * approach_scale * 0.4, -self.max_lat_vel, self.max_lat_vel))
            ang_z = float(np.clip(heading_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))
            phase = "approach"

        base_cmd = np.array([lin_x, lin_y, ang_z], dtype=np.float32)
        base_cmd = np.nan_to_num(base_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        return base_cmd, {
            "phase": phase,
            "goal_dist": goal_dist,
            "target_dist": target_dist,
            "heading_error": heading_error,
            "target": target,
        }

    def _get_navigation_input(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        local_nav = self._update_local_odometry(obs)
        if self.perception is None:
            return local_nav, None

        try:
            perception_output = self.perception.process(obs, dt=self.dt)
        except Exception as exc:
            if not self._perception_error_printed:
                print(f"[TaskB-PERCEPTION] disabled after error: {type(exc).__name__}: {exc}", flush=True)
                self._perception_error_printed = True
            return local_nav, None

        target = self._select_target(perception_output)
        return perception_output, target

    def _compute_base_cmd(self, nav_input: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        robot_info = nav_input.get("robot") or {}
        bin_info = nav_input.get("bin") or {}

        robot_pos = self._safe_numpy(robot_info.get("pos_world"), np.zeros(3, dtype=np.float32))
        robot_yaw = float(robot_info.get("yaw", 0.0))
        if not np.isfinite(robot_yaw):
            robot_yaw = 0.0

        bin_center = self._safe_numpy(bin_info.get("center_world"), self.default_bin_center)
        vector_to_bin = bin_center[:2] - robot_pos[:2]
        dist_to_bin = float(np.linalg.norm(vector_to_bin))
        if not np.isfinite(dist_to_bin) or dist_to_bin < 1e-6:
            return np.zeros(3, dtype=np.float32), {"heading": 0.0, "goal_dist": 0.0, "bin_dist": dist_to_bin}

        stop_point = bin_center[:2] - self.stop_distance * vector_to_bin / dist_to_bin
        vector_to_goal = stop_point - robot_pos[:2]
        goal_dist = float(np.linalg.norm(vector_to_goal))

        if not np.isfinite(goal_dist) or goal_dist <= self.stop_tolerance:
            return np.zeros(3, dtype=np.float32), {
                "heading": 0.0,
                "goal_dist": max(goal_dist, 0.0),
                "bin_dist": dist_to_bin,
            }

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        goal_x_robot = cos_yaw * vector_to_goal[0] + sin_yaw * vector_to_goal[1]
        goal_y_robot = -sin_yaw * vector_to_goal[0] + cos_yaw * vector_to_goal[1]

        desired_heading = math.atan2(vector_to_goal[1], vector_to_goal[0])
        heading_error = self._wrap_angle(desired_heading - robot_yaw)
        approach_scale = float(np.clip(goal_dist / self.slow_down_radius, 0.0, 1.0))
        heading_scale = max(self.min_heading_lin_scale, math.cos(min(abs(heading_error), math.pi / 2.0)))

        lin_x = float(np.clip(goal_x_robot * approach_scale * heading_scale, -self.max_lin_vel, self.max_lin_vel))
        lin_y = float(np.clip(goal_y_robot * approach_scale * 0.3, -self.max_lat_vel, self.max_lat_vel))
        ang_z = float(np.clip(heading_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))

        if goal_dist < self.stop_tolerance * 2.0:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0

        base_cmd = np.array([lin_x, lin_y, ang_z], dtype=np.float32)
        base_cmd = np.nan_to_num(base_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        return base_cmd, {"heading": desired_heading, "heading_error": heading_error, "goal_dist": goal_dist, "bin_dist": dist_to_bin}

    def _extract_policy_obs(self, obs: dict[str, Any], base_cmd: np.ndarray) -> torch.Tensor:
        proprio = torch.as_tensor(obs["proprio"], device=self.device, dtype=torch.float32)
        expected_dim = 3 + 3 + 3 + 3 + self.total_action_dim + self.total_action_dim + self.total_action_dim
        if proprio.shape[-1] != expected_dim:
            raise ValueError(f"Unexpected proprio dim: got {proprio.shape[-1]}, expected {expected_dim}")

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + self.total_action_dim]
        idx += self.total_action_dim

        joint_vel_all = proprio[:, idx:idx + self.total_action_dim]
        idx += self.total_action_dim

        actions_all = proprio[:, idx:idx + self.total_action_dim]

        joint_pos_leg = joint_pos_all[:, :self.leg_action_dim]
        joint_vel_leg = joint_vel_all[:, :self.leg_action_dim]
        actions_leg_env = actions_all[:, :self.leg_action_dim]
        actions_leg_train = actions_leg_env * self.leg_action_scale_inv.to(dtype=proprio.dtype)

        velocity_commands = torch.as_tensor(base_cmd, device=self.device, dtype=proprio.dtype).view(1, 3)
        if proprio.shape[0] > 1:
            velocity_commands = velocity_commands.repeat(proprio.shape[0], 1)

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_leg_train,
            ],
            dim=-1,
        )
        return torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}")

        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, self.total_action_dim), device=self.device, dtype=torch.float32)
        action_env[:, :self.leg_action_dim] = action_train * self.leg_action_scale
        action_env[:, self.leg_action_dim:] = self.arm_default_action.repeat(num_envs, 1)
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _maybe_print_nav_debug(self, base_cmd: np.ndarray, nav_info: dict[str, float]) -> None:
        if not self.nav_debug:
            return

        self._nav_debug_step += 1
        if self._nav_debug_step % self.nav_debug_every != 0:
            return

        goal_dist = float(nav_info.get("goal_dist", 0.0))
        bin_dist = float(nav_info.get("bin_dist", 0.0))
        heading_error = float(nav_info.get("heading_error", 0.0))
        phase = str(nav_info.get("phase", "nav"))
        target = nav_info.get("target")
        target_id = None if target is None else target.get("id")
        target_class = None if target is None else target.get("class")
        target_pos = None if target is None else target.get("pos_robot")
        target_dist = float(nav_info.get("target_dist", 0.0))
        target_pos_text = None if target_pos is None else np.asarray(target_pos, dtype=np.float32).round(3).tolist()
        stopped = phase == "stopped" or (goal_dist <= self.stop_tolerance and np.allclose(base_cmd[:2], 0.0))
        print(
            "[TaskB-TARGET] "
            f"phase={phase} "
            f"target_id={target_id} class={target_class} target_dist={target_dist:.3f} "
            f"target_pos_robot={target_pos_text} "
            f"base_cmd={np.asarray(base_cmd, dtype=np.float32).round(3).tolist()} "
            f"goal_dist={goal_dist:.3f} bin_dist={bin_dist:.3f} "
            f"heading_error={heading_error:.3f} stopped={stopped}",
            flush=True,
        )

    def predicts(self, obs, current_score):
        self._step_count += 1
        nav_input, target = self._get_navigation_input(obs)

        if self._step_count <= self.stand_still_steps:
            base_cmd = np.zeros(3, dtype=np.float32)
            nav_info = {"phase": "stand", "goal_dist": 0.0, "bin_dist": 0.0, "heading_error": 0.0, "target": target}
        elif self.target_mode == "object" and target is not None:
            base_cmd, nav_info = self._compute_object_cmd(target)
        elif self.target_mode == "object":
            base_cmd = np.zeros(3, dtype=np.float32)
            nav_info = {"phase": "search", "goal_dist": 0.0, "bin_dist": 0.0, "heading_error": 0.0, "target": None}
        else:
            base_cmd, nav_info = self._compute_base_cmd(nav_input)
            nav_info["phase"] = "nav"

        self._last_base_cmd = base_cmd
        self._maybe_print_nav_debug(base_cmd, nav_info)
        policy_obs = self._extract_policy_obs(obs, base_cmd)

        with torch.inference_mode():
            action_train = self.actor(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)

        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train)
        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}


if __name__ == "__main__":
    solution = AlgSolution()
    dummy_obs = {
        "proprio": torch.zeros(1, 72, dtype=torch.float32),
        "image": {
            "head_rgb": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
            "head_depth": torch.ones(1, 480, 640, 1, dtype=torch.float32),
        },
    }
    try:
        result = solution.predicts(dummy_obs, current_score=0.0)
        print("action_dim", len(result["action"][0]))
        print("giveup", result["giveup"])
    except Exception as exc:
        print(type(exc).__name__, exc)