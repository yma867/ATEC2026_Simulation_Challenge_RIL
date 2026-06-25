import glob
import math
import os
from typing import Any

import torch
import torch.nn.functional as F


class AlgSolution:
    """TaskD box-pushing state machine.

    Uses the selected rough DreamWaQ policy with the existing TaskD box-pushing
    state machine.  A TaskD-observation DreamWaQ checkpoint remains available
    through ATEC_POLICY_MODE=dreamwaq_taskd.
    """

    LEG_ACTION_DIM = 12
    CONTROL_HZ = 50.0
    PARKOUR_NUM_PROP = 53
    PARKOUR_NUM_SCAN = 132
    PARKOUR_NUM_PRIV_EXPLICIT = 9
    PARKOUR_NUM_PRIV_LATENT = 29
    PARKOUR_HISTORY_LENGTH = 10
    PARKOUR_DEPTH_SHAPE = (58, 87)

    # Task env leg actions use a larger scale than UnitreeB2Rough training.
    TRAIN_TO_TASK_ACTION_SCALE = (
        0.25,
        0.5,
        0.5,
        0.25,
        0.5,
        0.5,
        0.25,
        0.5,
        0.5,
        0.25,
        0.5,
        0.5,
    )
    TASK_TO_TRAIN_ACTION_SCALE = (
        4.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
    )

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.policy_mode = os.environ.get("ATEC_POLICY_MODE", "dreamwaq").strip().lower()
        self.use_parkour_cross = os.environ.get("ATEC_ENABLE_PARKOUR_CROSS", "0") == "1"
        self.policy_path = self._find_policy_path()
        self.policy = torch.jit.load(self.policy_path, map_location=self.device)
        self.policy.eval()

        self.parkour_policy = None
        if self.policy_mode == "parkour_student" or self.use_parkour_cross:
            self.parkour_policy_path = (
                self.policy_path if self.policy_mode == "parkour_student" else self._find_parkour_policy_path()
            )
            self.parkour_policy = (
                self.policy
                if self.policy_mode == "parkour_student"
                else torch.jit.load(self.parkour_policy_path, map_location=self.device)
            )
            self.parkour_policy.eval()
            self.depth_encoder_path = self._find_parkour_depth_encoder_path(self.parkour_policy_path)
            self.depth_encoder = torch.jit.load(self.depth_encoder_path, map_location=self.device)
            self.depth_encoder.eval()
            self.parkour_policy_obs_dim = (
                self.PARKOUR_NUM_PROP
                + self.PARKOUR_NUM_SCAN
                + self.PARKOUR_NUM_PRIV_EXPLICIT
                + self.PARKOUR_NUM_PRIV_LATENT
                + self.PARKOUR_HISTORY_LENGTH * self.PARKOUR_NUM_PROP
            )
            self._parkour_history = torch.zeros(
                1,
                self.PARKOUR_HISTORY_LENGTH,
                self.PARKOUR_NUM_PROP,
                device=self.device,
                dtype=torch.float32,
            )
            self._parkour_depth_latent = torch.zeros(1, 32, device=self.device, dtype=torch.float32)
            self._parkour_yaw = torch.zeros(1, 2, device=self.device, dtype=torch.float32)
            self._last_parkour_action = torch.zeros(1, self.LEG_ACTION_DIM, device=self.device, dtype=torch.float32)
            self._last_parkour_raw_action = torch.zeros(1, self.LEG_ACTION_DIM, device=self.device, dtype=torch.float32)
            self._last_parkour_clipped_action = torch.zeros(
                1, self.LEG_ACTION_DIM, device=self.device, dtype=torch.float32
            )
            self._last_parkour_scan_stats: tuple[float, float] | None = None
            self._last_parkour_depth_stats: tuple[float, float] | None = None
        if self.policy_mode == "parkour_student":
            self.policy_obs_dim = self.parkour_policy_obs_dim
        elif self.policy_mode in {"dreamwaq", "dreamwaq_taskd"}:
            self.policy_obs_dim = self._repair_policy_history_buffer()
        else:
            self.policy_obs_dim = 45
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        self.step_count = 0
        self.debug = os.environ.get("ATEC_DEBUG_SOLUTION", "0") == "1"
        self.debug_log_path = os.environ.get("ATEC_DEBUG_LOG", "").strip()
        if self.debug_log_path:
            try:
                with open(self.debug_log_path, "w", encoding="utf-8") as file:
                    file.write("")
            except OSError as exc:
                print(f"[solution] WARNING: cannot write debug log {self.debug_log_path!r}: {exc}")
                self.debug_log_path = ""
        self.phase = "route_behind_box"
        self.phase_start_step = 0
        self.current_phase = "init"
        self.current_command = (0.0, 0.0, 0.0)
        self.current_target: tuple[float, float] | None = None

        self._last_phase = None
        self._task_name = ""
        self._target_yaw: float | None = None
        self._latest_robot_pos: tuple[float, float, float] | None = None
        self._latest_box_pos: tuple[float, float, float] | None = None
        self._latest_robot_yaw: float | None = None
        self._parkour_heading_target: float | None = None
        self._parkour_heading_error = 0.0
        self._parkour_path_origin: tuple[float, float] | None = None
        self._parkour_path_heading: float | None = None
        self._parkour_cross_track_error = 0.0
        self._parkour_switch_step: int | None = None
        self._latest_robot_rot_w_b: torch.Tensor | None = None
        self._latest_box_yaw: float | None = None
        self._retreat_left_target_y: float | None = None
        self._warned_task_mismatch = False
        self._printed_obs_shapes = False

        self._stage = None
        self._robot_prim_path = None
        self._box_prim_path = None
        self._xform_cache = None

        self.train_to_task_action_scale = torch.tensor(
            self.TRAIN_TO_TASK_ACTION_SCALE, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.task_to_train_action_scale = torch.tensor(
            self.TASK_TO_TRAIN_ACTION_SCALE, device=self.device, dtype=torch.float32
        ).view(1, -1)

        self.height_scan_xy = self._build_grid_xy(size_x=2.4, size_y=1.6, resolution=0.08)
        self.depth_feature_dim = 12 * 16
        self.base_actor_obs_dim = 45
        self.depth_camera_pos_b = torch.tensor((0.33, 0.0, 0.08), device=self.device, dtype=torch.float32)
        self.depth_camera_rot_b = self._quat_wxyz_to_matrix(
            torch.tensor(
                (-0.4055798351764679, 0.5792279839515686, -0.5792279839515686, 0.4055797755718231),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.depth_camera_dirs_b = self._build_depth_camera_dirs_b()
        self.depth_ray_t = torch.linspace(
            0.05,
            2.5,
            int(os.environ.get("ATEC_DEPTH_RAY_STEPS", "80")),
            device=self.device,
            dtype=torch.float32,
        )

        print(f"[solution] using {self.policy_mode} loco policy: {self.policy_path}")
        if self.policy_mode == "parkour_student":
            print(f"[solution] using parkour depth encoder: {self.depth_encoder_path}")
        elif self.use_parkour_cross:
            print(f"[solution] shadowing parkour crossing policy: {self.parkour_policy_path}")
            print(f"[solution] using parkour depth encoder: {self.depth_encoder_path}")
        print(f"[solution] policy obs dim: {self.policy_obs_dim}")
        if os.environ.get("ATEC_FORCE_CMD", "").strip():
            print("[solution] ATEC_FORCE_CMD is set; TaskD box-pushing state machine will be bypassed.")

    def _debug_log(self, message: str) -> None:
        print(message)
        if not self.debug_log_path:
            return
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as file:
                file.write(message + "\n")
        except OSError:
            self.debug_log_path = ""

    def _repair_policy_history_buffer(self) -> int:
        state = self.policy.state_dict()
        obs_dim = int(getattr(self.policy, "obs_dim", state["obs_history"].shape[-1]))
        encoder_in_dim = int(state["encoder.0.weight"].shape[1])
        history_length = max(1, encoder_in_dim // obs_dim)
        current_history = getattr(self.policy, "obs_history", None)
        if current_history is None or tuple(current_history.shape) != (1, history_length, obs_dim):
            self.policy.obs_history = torch.zeros(1, history_length, obs_dim, device=self.device)
            print(f"[solution] repaired DreamWaQ history buffer: length={history_length}, obs_dim={obs_dim}")
        return obs_dim

    def _find_policy_path(self) -> str:
        env_path = os.environ.get("ATEC_LOCO_POLICY")
        candidates = []
        if env_path:
            candidates.append(env_path)

        if self.policy_mode == "parkour_student":
            return self._find_parkour_policy_path(env_path)

        if self.policy_mode not in {"dreamwaq", "dreamwaq_taskd"}:
            candidates.append(os.path.join(self.repo_root, "demo", "policy.pt"))
            candidates.extend(
                sorted(
                    glob.glob(
                        os.path.join(
                            self.repo_root,
                            "logs",
                            "rsl_rl",
                            "unitree_b2_rough",
                            "*",
                            "exported",
                            "policy.pt",
                        )
                    ),
                    reverse=True,
                )
            )
            for path in candidates:
                if path and os.path.exists(path):
                    return path
            raise FileNotFoundError(
                "Cannot find official rough policy. Set ATEC_LOCO_POLICY=/path/to/rough/policy.pt"
            )

        experiment_name = (
            "unitree_b2_dreamwaq_taskd_obs_dreamwaq"
            if self.policy_mode == "dreamwaq_taskd"
            else "unitree_b2_dreamwaq_rough_dreamwaq"
        )

        if self.policy_mode == "dreamwaq":
            candidates.append(
                os.path.join(
                    self.repo_root,
                    "logs",
                    "rsl_rl",
                    "unitree_b2_rough_dreamwaq",
                    "2026-06-08_18-21-27",
                    "exported",
                    "policy.pt",
                )
            )

        if self.policy_mode == "dreamwaq_taskd":
            # Stable local bundle copied from the selected server checkpoint.
            candidates.append(
                os.path.join(
                    self.repo_root,
                    "exports",
                    "taskd_obs_latest",
                    "policy.pt",
                )
            )

        candidates.append(
            os.path.join(
                self.repo_root,
                "logs",
                "rsl_rl",
                experiment_name,
                "exported",
                "policy.pt",
            )
        )
        candidates.append(
            os.path.join(
                self.repo_root,
                "logs",
                "rsl_rl",
                experiment_name,
                "latest",
                "exported",
                "policy.pt",
            )
        )
        candidates.extend(
            sorted(
                glob.glob(
                    os.path.join(
                        self.repo_root,
                        "logs",
                        "rsl_rl",
                        experiment_name,
                        "*",
                        "exported",
                        "policy.pt",
                    )
                ),
                reverse=True,
            )
        )

        for path in candidates:
            if path and os.path.exists(path):
                return path

        raise FileNotFoundError(
            "Cannot find DreamWaQ rough policy. Set "
            "ATEC_LOCO_POLICY=/path/to/dreamwaq/exported/policy.pt"
        )

    def _find_parkour_policy_path(self, explicit_path: str | None = None) -> str:
        candidates = []
        env_path = explicit_path or os.environ.get("ATEC_PARKOUR_POLICY")
        if env_path:
            candidates.append(env_path)
        candidates.append(
            os.path.join(
                os.path.dirname(self.repo_root),
                "Isaaclab_Parkour",
                "logs",
                "rsl_rl",
                "unitree_b2_parkour",
                "latest_student",
                "exported_deploy",
                "policy.pt",
            )
        )
        candidates.extend(
            sorted(
                glob.glob(
                    os.path.join(
                        os.path.dirname(self.repo_root),
                        "Isaaclab_Parkour",
                        "logs",
                        "rsl_rl",
                        "unitree_b2_parkour",
                        "*",
                        "exported_deploy",
                        "policy.pt",
                    )
                ),
                reverse=True,
            )
        )
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Cannot find Isaaclab_Parkour student deploy policy. Set "
            "ATEC_PARKOUR_POLICY=/path/to/exported_deploy/policy.pt"
        )

    def _find_parkour_depth_encoder_path(self, policy_path: str) -> str:
        env_path = os.environ.get("ATEC_PARKOUR_DEPTH_ENCODER")
        candidates = []
        if env_path:
            candidates.append(env_path)
        policy_dir = os.path.dirname(policy_path)
        candidates.extend(
            [
                os.path.join(policy_dir, "depth_latest.pt"),
                os.path.join(policy_dir, "depth_encoder.pt"),
                os.path.join(policy_dir, "depth.pt"),
            ]
        )
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Cannot find Isaaclab_Parkour depth encoder. Set "
            "ATEC_PARKOUR_DEPTH_ENCODER=/path/to/exported_deploy/depth_latest.pt"
        )

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        if self.policy_mode == "parkour_student":
            # Match the Parkour B2 training action contract exactly. Safety for
            # the first TaskD frames is handled by the warm-up ramp below.
            leg_clip = float(os.environ.get("ATEC_LEG_ACTION_CLIP", "4.8"))
            leg_scale = float(os.environ.get("ATEC_LEG_ACTION_SCALE", "0.25"))
            return {"leg": {"mode": "position", "scale": leg_scale, "clip": [-leg_clip, leg_clip]}}
        leg_clip = float(os.environ.get("ATEC_LEG_ACTION_CLIP", "3.0"))
        return {"leg": {"clip": [-leg_clip, leg_clip]}}

    def _get_stage(self):
        if self._stage is not None:
            return self._stage
        try:
            import omni.usd

            self._stage = omni.usd.get_context().get_stage()
        except Exception:
            self._stage = None
        return self._stage

    def _find_prim_path(self, suffix: str) -> str | None:
        stage = self._get_stage()
        if stage is None:
            return None
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.endswith(suffix):
                return path
        return None

    def _get_world_pos(self, prim_path: str | None) -> tuple[float, float, float] | None:
        if prim_path is None:
            return None
        stage = self._get_stage()
        if stage is None:
            return None
        try:
            from pxr import UsdGeom

            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None
            if self._xform_cache is None:
                self._xform_cache = UsdGeom.XformCache()
            self._xform_cache.Clear()
            translation = self._xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            return float(translation[0]), float(translation[1]), float(translation[2])
        except Exception:
            return None

    def _get_world_yaw(self, prim_path: str | None) -> float | None:
        if prim_path is None:
            return None
        stage = self._get_stage()
        if stage is None:
            return None
        try:
            from pxr import UsdGeom

            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None
            if self._xform_cache is None:
                self._xform_cache = UsdGeom.XformCache()
            self._xform_cache.Clear()
            matrix = self._xform_cache.GetLocalToWorldTransform(prim)
            return math.atan2(float(matrix[0][1]), float(matrix[0][0]))
        except Exception:
            return None

    def _update_stage_poses(self) -> None:
        if self._robot_prim_path is None:
            self._robot_prim_path = (
                self._find_prim_path("/Robot/base_link")
                or self._find_prim_path("/Robot/base")
                or self._find_prim_path("/Robot")
            )
        if self._box_prim_path is None:
            self._box_prim_path = self._find_prim_path("/Box")

        robot_pos = self._get_world_pos(self._robot_prim_path)
        box_pos = self._get_world_pos(self._box_prim_path)
        box_yaw = self._get_world_yaw(self._box_prim_path)
        if robot_pos is not None:
            self._latest_robot_pos = robot_pos
        if box_pos is not None:
            self._latest_box_pos = box_pos
        if box_yaw is not None:
            self._latest_box_yaw = box_yaw

    def _phase_elapsed(self) -> float:
        return (self.step_count - self.phase_start_step) / self.CONTROL_HZ

    def _set_phase(self, phase: str) -> None:
        if phase != self.phase:
            if self.debug:
                self._debug_log(f"[solution] phase {self.phase} -> {phase} at step={self.step_count}")
            self.phase = phase
            self.phase_start_step = self.step_count
            self._retreat_left_target_y = None
            if phase == "done" and hasattr(self.policy, "reset"):
                self.policy.reset()

    def _clip(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _quat_wxyz_to_yaw(self, quat: list[float]) -> float:
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _quat_wxyz_to_rpy(self, quat: list[float]) -> tuple[float, float, float]:
        w, x, y, z = quat
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi * 0.5, sinp)
        else:
            pitch = math.asin(sinp)
        yaw = self._quat_wxyz_to_yaw(quat)
        return roll, pitch, yaw

    def _quat_wxyz_to_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        quat = quat.to(device=self.device, dtype=torch.float32)
        quat = quat / torch.clamp(torch.linalg.norm(quat), min=1.0e-8)
        w, x, y, z = quat.unbind()
        return torch.stack(
            (
                torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
                torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
                torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
            )
        )

    def _robot_tilt_rad(self) -> float:
        if self._latest_robot_rot_w_b is None:
            return 0.0
        upright_cos = float(self._latest_robot_rot_w_b[2, 2].detach().cpu())
        upright_cos = max(-1.0, min(1.0, upright_cos))
        return math.acos(upright_cos)

    def _build_depth_camera_dirs_b(self) -> torch.Tensor:
        width = 80
        height = 48
        focal_length = 11.041
        horizontal_aperture = 20.955
        vertical_aperture = 12.240
        fx = focal_length / horizontal_aperture * width
        fy = focal_length / vertical_aperture * height
        u = torch.arange(width, device=self.device, dtype=torch.float32) + 0.5
        v = torch.arange(height, device=self.device, dtype=torch.float32) + 0.5
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        x = (grid_u - width * 0.5) / fx
        y = (grid_v - height * 0.5) / fy
        z = torch.ones_like(x)
        dirs_camera = torch.stack((x, y, z), dim=-1)
        dirs_camera = dirs_camera / torch.linalg.norm(dirs_camera, dim=-1, keepdim=True)
        dirs_body = torch.matmul(dirs_camera.reshape(-1, 3), self.depth_camera_rot_b.T)
        return dirs_body / torch.linalg.norm(dirs_body, dim=-1, keepdim=True)

    def _yaw_hold_wz(self) -> float:
        if os.environ.get("ATEC_YAW_HOLD", "1") == "0":
            return 0.0
        if self._latest_robot_yaw is None:
            return 0.0
        if self._target_yaw is None:
            self._target_yaw = self._latest_robot_yaw
        yaw_error = self._wrap_angle(self._latest_robot_yaw - self._target_yaw)
        kp = float(os.environ.get("ATEC_YAW_KP", "1.2"))
        max_wz = float(os.environ.get("ATEC_YAW_MAX_WZ", "0.45"))
        return self._clip(-kp * yaw_error, max_wz)

    def _parkour_heading_signal(
        self,
        command_values: tuple[float, float, float],
        batch: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build Parkour's route-yaw input from TaskD's real base heading."""
        signal = torch.zeros((batch, 2), device=self.device, dtype=dtype)
        if os.environ.get("ATEC_PARKOUR_HEADING_HOLD", "1") == "0":
            return signal
        if self._latest_robot_yaw is None:
            return signal

        if self._parkour_heading_target is None:
            self._parkour_heading_target = self._latest_robot_yaw
        if self._parkour_path_heading is None:
            self._parkour_path_heading = self._latest_robot_yaw
        if self._parkour_path_origin is None and self._latest_robot_pos is not None:
            self._parkour_path_origin = self._latest_robot_pos[:2]

        vx, vy, commanded_wz = command_values
        straight_motion = abs(vx) > 0.05 and abs(vy) < 0.05 and abs(commanded_wz) < 1.0e-4
        if straight_motion and self._parkour_path_origin is not None and self._parkour_path_heading is not None:
            dx = self._latest_robot_pos[0] - self._parkour_path_origin[0]
            dy = self._latest_robot_pos[1] - self._parkour_path_origin[1]
            path_yaw = self._parkour_path_heading
            self._parkour_cross_track_error = -math.sin(path_yaw) * dx + math.cos(path_yaw) * dy
            cross_track_gain = float(os.environ.get("ATEC_PARKOUR_CROSS_TRACK_GAIN", "0.8"))
            lookahead = max(0.1, float(os.environ.get("ATEC_PARKOUR_LOOKAHEAD", "1.0")))
            correction = math.atan2(-cross_track_gain * self._parkour_cross_track_error, lookahead)
            self._parkour_heading_target = self._wrap_angle(path_yaw + correction)

        # A non-zero yaw command rotates the heading reference. For straight
        # commands the initial world heading remains fixed, preventing small
        # left/right gait asymmetries from accumulating into a turn.
        if abs(commanded_wz) > 1.0e-4:
            self._parkour_heading_target = self._wrap_angle(
                self._parkour_heading_target + commanded_wz / self.CONTROL_HZ
            )

        error = self._wrap_angle(self._parkour_heading_target - self._latest_robot_yaw)
        gain = float(os.environ.get("ATEC_PARKOUR_HEADING_GAIN", "1.0"))
        limit = float(os.environ.get("ATEC_PARKOUR_HEADING_CLIP", "0.8"))
        self._parkour_heading_error = self._clip(gain * error, limit)
        signal.fill_(self._parkour_heading_error)
        return signal

    def _world_velocity_to_body(self, vx_w: float, vy_w: float) -> tuple[float, float]:
        if self._latest_robot_yaw is None:
            return vx_w, vy_w
        c = math.cos(self._latest_robot_yaw)
        s = math.sin(self._latest_robot_yaw)
        return c * vx_w + s * vy_w, -s * vx_w + c * vy_w

    def _drive_to_xy(
        self,
        robot_pos: tuple[float, float, float],
        target_x: float,
        target_y: float,
        max_speed: float,
        gain: float = 0.7,
        turn_and_go: bool | None = None,
    ) -> tuple[float, float, float]:
        if turn_and_go is None:
            # TaskD box pushing is more repeatable with the original scripted
            # route: back up, sidestep, move forward to align, then push.
            # Turning toward every waypoint makes the robot approach the box
            # diagonally and breaks the side-push setup.
            turn_and_go = os.environ.get("ATEC_TURN_AND_GO_NAV", "0") == "1"

        vx_w = self._clip((target_x - robot_pos[0]) * gain, max_speed)
        vy_w = self._clip((target_y - robot_pos[1]) * gain, max_speed)
        if turn_and_go and self._latest_robot_yaw is not None:
            dx = target_x - robot_pos[0]
            dy = target_y - robot_pos[1]
            distance = math.hypot(dx, dy)
            if distance < 1.0e-4:
                return 0.0, 0.0, self._yaw_hold_wz()
            desired_yaw = math.atan2(dy, dx)
            yaw_error = self._wrap_angle(desired_yaw - self._latest_robot_yaw)
            max_wz = float(os.environ.get("ATEC_NAV_MAX_WZ", "0.65"))
            wz = self._clip(float(os.environ.get("ATEC_NAV_YAW_KP", "1.4")) * yaw_error, max_wz)
            heading_scale = max(0.22, 0.5 + 0.5 * math.cos(yaw_error))
            vx = self._clip(distance * gain, max_speed) * heading_scale
            if abs(yaw_error) > float(os.environ.get("ATEC_NAV_TURN_IN_PLACE_RAD", "1.05")):
                vx = max(vx, float(os.environ.get("ATEC_NAV_MIN_VX_WHILE_TURNING", "0.16")))
            return vx, 0.0, wz

        vx, vy = self._world_velocity_to_body(vx_w, vy_w)
        return vx, vy, self._yaw_hold_wz()

    def _taskd_command(self) -> tuple[str, tuple[float, float, float]]:
        force_cmd = os.environ.get("ATEC_FORCE_CMD", "").strip()
        if force_cmd:
            try:
                vx, vy, wz = [float(value) for value in force_cmd.split(",")]
                return "force_cmd", (vx, vy, wz)
            except Exception:
                if self.debug:
                    self._debug_log(f"[solution] invalid ATEC_FORCE_CMD={force_cmd!r}, expected vx,vy,wz")

        # play_atec_task.py injects simulator poses into obs before calling
        # predicts(). Prefer those tensor poses for the state machine. The USD
        # stage traversal is only a fallback; on some scenes it can find a prim
        # whose transform is not the robot root used by the physics view.
        if (
            os.environ.get("ATEC_USE_STAGE_POSES", "0") == "1"
            or self._latest_robot_pos is None
            or self._latest_box_pos is None
        ):
            self._update_stage_poses()
        robot_pos = self._latest_robot_pos
        box_pos = self._latest_box_pos
        if robot_pos is None or box_pos is None:
            return "forward_until_pose_ready", (0.35, 0.0, self._yaw_hold_wz())

        box_target_y = float(os.environ.get("ATEC_BOX_TARGET_Y", "-0.90"))
        box_target_x = float(os.environ.get("ATEC_BOX_TARGET_X", "-0.45"))
        finish_x = float(os.environ.get("ATEC_FINISH_TARGET_X", "3.6"))
        side_offset_y = float(os.environ.get("ATEC_SIDE_OFFSET_Y", "1.85"))
        back_offset_x = float(os.environ.get("ATEC_BACK_OFFSET_X", "2.0"))
        route_back_x = box_pos[0] - float(os.environ.get("ATEC_ROUTE_BACK_X", "1.50"))
        side_align_x_bias = float(os.environ.get("ATEC_SIDE_ALIGN_X_BIAS", "0.20"))
        behind_align_y_bias = float(os.environ.get("ATEC_BEHIND_ALIGN_Y_BIAS", "-0.10"))
        push_align_y_bias = float(os.environ.get("ATEC_PUSH_ALIGN_Y_BIAS", "0.0"))
        xy_tol = float(os.environ.get("ATEC_XY_TOL", "0.14"))
        box_y_tol = float(os.environ.get("ATEC_BOX_Y_TOL", "0.10"))
        box_x_tol = float(os.environ.get("ATEC_BOX_X_TOL", "0.08"))

        y_push_direction = float(os.environ.get("ATEC_Y_PUSH_DIRECTION", "-1.0"))
        y_push_speed = float(os.environ.get("ATEC_PUSH_Y_SPEED", "0.55"))
        x_push_speed = float(os.environ.get("ATEC_PUSH_X_SPEED", "1.40"))

        raw_side_y = box_pos[1] - y_push_direction * side_offset_y
        side_y = self._clip(raw_side_y, float(os.environ.get("ATEC_SIDE_Y_LIMIT", "2.75")))
        behind_x = box_pos[0] - back_offset_x

        def box_y_reached() -> bool:
            if y_push_direction < 0.0:
                return box_pos[1] <= box_target_y + box_y_tol
            return box_pos[1] >= box_target_y - box_y_tol

        def box_x_reached() -> bool:
            return box_pos[0] >= box_target_x - box_x_tol

        def positive_until(value: float, target: float, gain: float, limit: float, tol: float) -> float:
            if value >= target - tol:
                return 0.0
            return max(0.0, self._clip((target - value) * gain, limit))

        def negative_until(value: float, target: float, gain: float, limit: float, tol: float) -> float:
            if value <= target + tol:
                return 0.0
            return min(0.0, self._clip((target - value) * gain, limit))

        # 1) Back up from the box/front wall to create clearance.
        if self.phase == "route_behind_box":
            self.current_target = (route_back_x, robot_pos[1])
            reached = robot_pos[0] <= route_back_x + xy_tol
            vx_w = negative_until(
                robot_pos[0],
                route_back_x,
                gain=0.9,
                limit=float(os.environ.get("ATEC_ROUTE_BACK_MAX_SPEED", "0.72")),
                tol=xy_tol,
            )
            if not reached and vx_w == 0.0:
                vx_w = -float(os.environ.get("ATEC_BACKUP_MIN_SPEED", "0.12"))
            route_back_timeout = float(os.environ.get("ATEC_ROUTE_BACK_TIMEOUT_S", "4.5"))
            if reached or self._phase_elapsed() > route_back_timeout:
                self._set_phase("route_to_y_push_lane")
                vx_w = 0.0
            vx, vy = self._world_velocity_to_body(vx_w, 0.0)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        # 2) Move left/+y to the box side-push lane.
        if self.phase == "route_to_y_push_lane":
            self.current_target = (route_back_x, side_y)
            vx_w = 0.0
            side_speed = float(os.environ.get("ATEC_ROUTE_SIDE_MAX_SPEED", "0.58"))
            vy_w = positive_until(robot_pos[1], side_y, gain=0.85, limit=side_speed, tol=xy_tol)
            reached = robot_pos[1] >= side_y - xy_tol
            if reached or self._phase_elapsed() > float(os.environ.get("ATEC_ROUTE_SIDE_TIMEOUT_S", "7.0")):
                self._set_phase("align_for_y_push")
                vy_w = 0.0
            vx, vy = self._world_velocity_to_body(0.0, vy_w)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        # 3) Move forward/+x until the robot is aligned with the box side.
        if self.phase == "align_for_y_push":
            target = (box_pos[0] + side_align_x_bias, side_y)
            self.current_target = target
            align_x_tol = float(os.environ.get("ATEC_ALIGN_X_TOL", "0.08"))
            align_speed = float(os.environ.get("ATEC_ALIGN_X_MAX_SPEED", "0.65"))
            vx_w = positive_until(robot_pos[0], target[0], gain=1.25, limit=align_speed, tol=align_x_tol)
            vy_w = 0.0
            aligned_x = robot_pos[0] >= target[0] - align_x_tol
            if aligned_x or self._phase_elapsed() > 6.5:
                self._set_phase("push_box_to_center_y")
                vx_w = 0.0
                vy_w = 0.0
            vx, vy = self._world_velocity_to_body(vx_w, vy_w)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        # 4) Push the box right/-y onto the pit centerline.
        if self.phase == "push_box_to_center_y":
            self.current_target = (box_pos[0] + side_align_x_bias, box_target_y)
            if box_y_reached() or self._phase_elapsed() > float(os.environ.get("ATEC_PUSH_Y_TIMEOUT_S", "6.0")):
                self._set_phase("retreat_from_y_side")
                return self.phase, (0.0, 0.0, self._yaw_hold_wz())
            vx_w = 0.0
            vx, vy = self._world_velocity_to_body(vx_w, y_push_direction * y_push_speed)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        # 5) Back up again to clear the box before routing behind it.
        if self.phase == "retreat_from_y_side":
            self.current_target = (behind_x, side_y)
            reached = robot_pos[0] <= behind_x + float(os.environ.get("ATEC_BEHIND_X_TOL", "0.12"))
            left_clear_y = box_pos[1] + float(os.environ.get("ATEC_RETREAT_LEFT_OFFSET_Y", "0.45"))
            if self._retreat_left_target_y is None:
                left_step_y = float(os.environ.get("ATEC_RETREAT_LEFT_STEP_Y", "0.28"))
                self._retreat_left_target_y = max(left_clear_y, robot_pos[1] + left_step_y)
            left_clear_y = self._retreat_left_target_y
            left_clear_tol = float(os.environ.get("ATEC_RETREAT_LEFT_Y_TOL", "0.08"))

            if (
                robot_pos[1] < left_clear_y - left_clear_tol
                and self._phase_elapsed() < float(os.environ.get("ATEC_RETREAT_LEFT_TIMEOUT_S", "1.8"))
            ):
                self.current_target = (robot_pos[0], left_clear_y)
                vx, vy = self._world_velocity_to_body(
                    0.0,
                    float(os.environ.get("ATEC_RETREAT_LEFT_SPEED", "0.34")),
                )
                return self.phase, (vx, vy, self._yaw_hold_wz())
            vx_w = negative_until(
                robot_pos[0],
                behind_x,
                gain=0.95,
                limit=float(os.environ.get("ATEC_RETREAT_MAX_SPEED", "0.65")),
                tol=float(os.environ.get("ATEC_BEHIND_X_TOL", "0.12")),
            )
            if not reached and vx_w == 0.0:
                vx_w = -float(os.environ.get("ATEC_RETREAT_MIN_SPEED", "0.18"))
            vy_w = 0.0
            if reached or self._phase_elapsed() > float(os.environ.get("ATEC_RETREAT_TIMEOUT_S", "5.0")):
                self._set_phase("align_behind_box_for_x")
                vx_w = 0.0
            vx, vy = self._world_velocity_to_body(vx_w, vy_w)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        # 6) Move right/-y to the box rear side directly, but stop on the target
        # y line instead of walking right for a fixed distance.
        if self.phase == "align_behind_box_for_x":
            target_y = box_pos[1] + behind_align_y_bias
            self.current_target = (robot_pos[0], target_y)

            right_speed = float(os.environ.get("ATEC_BEHIND_RIGHT_SPEED", "0.32"))
            right_time = float(os.environ.get("ATEC_BEHIND_RIGHT_TIME_S", "5.5"))
            right_tol = float(os.environ.get("ATEC_BEHIND_RIGHT_Y_TOL", "0.08"))

            y_aligned = robot_pos[1] <= target_y + right_tol or self._phase_elapsed() > right_time
            if y_aligned:
                self._set_phase("push_box_to_pit_x")
                return self.phase, (0.0, 0.0, 0.0)

            vx, vy = self._world_velocity_to_body(0.0, -right_speed)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        # 7) Push forward/+x to place the box into the gap.
        if self.phase == "push_box_to_pit_x":
            self.current_target = (box_target_x, box_pos[1] + push_align_y_bias)
            if box_x_reached():
                if os.environ.get("ATEC_ENABLE_CROSS", "1") == "1":
                    self._set_phase("back_off_after_pit_push")
                    return self.phase, (0.0, 0.0, self._yaw_hold_wz())
                else:
                    self._set_phase("done")
                return self.phase, (0.0, 0.0, 0.0)
            return self.phase, (x_push_speed, 0.0, 0.0)

        # 8) Back off slightly, then align for the crossing attempt.
        if self.phase == "back_off_after_pit_push":
            if not box_x_reached():
                self._set_phase("align_behind_box_for_x")
                return self.phase, (0.0, 0.0, self._yaw_hold_wz())
            backoff_target_x = box_pos[0] - float(os.environ.get("ATEC_AFTER_PUSH_BACK_OFFSET_X", "1.10"))
            self.current_target = (backoff_target_x, robot_pos[1])
            reached = robot_pos[0] <= backoff_target_x + float(os.environ.get("ATEC_AFTER_PUSH_BACK_X_TOL", "0.10"))
            if reached or self._phase_elapsed() > float(os.environ.get("ATEC_BACK_OFF_AFTER_PUSH_S", "1.4")):
                self._set_phase("align_for_cross")
                return self.phase, (0.0, 0.0, self._yaw_hold_wz())
            vx_w = negative_until(robot_pos[0], backoff_target_x, gain=0.8, limit=0.36, tol=0.0)
            vx, vy = self._world_velocity_to_body(vx_w, 0.0)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        # 9) Align with the box/bridge and try to cross the gap.
        if self.phase == "align_for_cross":
            if not box_x_reached():
                self._set_phase("align_behind_box_for_x")
                return self.phase, (0.0, 0.0, self._yaw_hold_wz())
            target_x = box_pos[0] - float(os.environ.get("ATEC_CROSS_BACK_OFFSET_X", "1.10"))
            target_y = box_pos[1] + float(os.environ.get("ATEC_CROSS_ALIGN_Y_BIAS", "0.0"))
            self.current_target = (target_x, target_y)
            vx_w = self._clip((target_x - robot_pos[0]) * 0.75, 0.40)
            vy_w = self._clip(
                (target_y - robot_pos[1]) * 0.85,
                float(os.environ.get("ATEC_CROSS_ALIGN_Y_SPEED", "0.28")),
            )
            aligned_x = abs(robot_pos[0] - target_x) < float(os.environ.get("ATEC_CROSS_ALIGN_X_TOL", "0.18"))
            aligned_y = abs(robot_pos[1] - target_y) < float(os.environ.get("ATEC_CROSS_ALIGN_Y_TOL", "0.10"))

            if (aligned_x and aligned_y) or self._phase_elapsed() > float(
                os.environ.get("ATEC_CROSS_ALIGN_TIMEOUT_S", "2.0")
            ):
                attempt_cross = os.environ.get("ATEC_ATTEMPT_CROSS", "0") == "1" or self.use_parkour_cross
                next_phase = "cross_pit" if attempt_cross else "hold_after_align"
                self._set_phase(next_phase)
                vx_w = 0.0
                vy_w = 0.0
            vx, vy = self._world_velocity_to_body(vx_w, vy_w)
            return self.phase, (vx, vy, self._yaw_hold_wz())

        if self.phase == "hold_after_align":
            self.current_target = (robot_pos[0], robot_pos[1])
            return self.phase, (0.0, 0.0, self._yaw_hold_wz())

        if self.phase == "cross_pit":
            if not box_x_reached():
                self._set_phase("align_behind_box_for_x")
                return self.phase, (0.0, 0.0, self._yaw_hold_wz())
            self.current_target = (finish_x, box_pos[1])
            if robot_pos[0] > finish_x:
                self._set_phase("done")
                return self.phase, (0.0, 0.0, self._yaw_hold_wz())
            vx = float(os.environ.get("ATEC_CROSS_VX", "0.70"))
            return self.phase, (vx, 0.0, 0.0)
        return "done", (0.0, 0.0, 0.0)

    def _build_grid_xy(self, size_x: float, size_y: float, resolution: float) -> torch.Tensor:
        nx = int(round(size_x / resolution)) + 1
        ny = int(round(size_y / resolution)) + 1
        xs = torch.linspace(-size_x * 0.5, size_x * 0.5, nx, device=self.device)
        ys = torch.linspace(-size_y * 0.5, size_y * 0.5, ny, device=self.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

    def _taskd_terrain_height(self, x: float, y: float) -> float:
        pit_center_x = float(os.environ.get("ATEC_PIT_CENTER_X", "0.0"))
        pit_width = float(os.environ.get("ATEC_PIT_WIDTH", "1.35"))
        pit_half_y = float(os.environ.get("ATEC_PIT_HALF_Y", "3.35"))
        pit_depth = float(os.environ.get("ATEC_PIT_DEPTH", "1.0"))
        height = 0.0
        if abs(x - pit_center_x) <= pit_width * 0.5 and abs(y) <= pit_half_y:
            height = -pit_depth

        if os.environ.get("ATEC_INCLUDE_PLATFORM_IN_TERRAIN", "0") == "1":
            platform_center_y = float(os.environ.get("ATEC_PLATFORM_CENTER_Y", "2.0"))
            platform_half_y = float(os.environ.get("ATEC_PLATFORM_HALF_Y", "1.5"))
            platform_height = float(os.environ.get("ATEC_PLATFORM_HEIGHT", "1.1"))
            if abs(x - pit_center_x) <= pit_width * 0.5 and abs(y - platform_center_y) <= platform_half_y:
                height = max(height, platform_height)

        # The DreamWaQ locomotion policy uses terrain perception.  During the
        # pushing phases, treating the box as terrain makes it look like a
        # wall/step obstacle, so the policy tends to stop or lift its legs
        # instead of pushing.  Keep the default scan focused on the pit only.
        include_box = self._include_box_in_loco_terrain("height")
        if include_box and self._latest_box_pos is not None:
            bx, by, bz = self._latest_box_pos
            box_half_x = float(os.environ.get("ATEC_BOX_HALF_X", "0.40"))
            box_half_y = float(os.environ.get("ATEC_BOX_HALF_Y", "0.50"))
            box_half_z = float(os.environ.get("ATEC_BOX_HALF_Z", "0.30"))
            if abs(x - bx) <= box_half_x and abs(y - by) <= box_half_y:
                height = max(height, bz + box_half_z)
        return height

    def _include_box_in_loco_terrain(self, source: str) -> bool:
        env_name = "ATEC_INCLUDE_BOX_IN_DEPTH" if source == "depth" else "ATEC_INCLUDE_BOX_IN_HEIGHT_SCAN"
        return os.environ.get(env_name, "0") == "1"

    def _taskd_terrain_height_tensor(self, x: torch.Tensor, y: torch.Tensor, include_box: bool = False) -> torch.Tensor:
        pit_center_x = float(os.environ.get("ATEC_PIT_CENTER_X", "0.0"))
        pit_width = float(os.environ.get("ATEC_PIT_WIDTH", "1.35"))
        pit_half_y = float(os.environ.get("ATEC_PIT_HALF_Y", "3.35"))
        pit_depth = float(os.environ.get("ATEC_PIT_DEPTH", "1.0"))
        height = torch.zeros_like(x)

        in_pit = (torch.abs(x - pit_center_x) <= pit_width * 0.5) & (torch.abs(y) <= pit_half_y)
        height = torch.where(in_pit, torch.full_like(height, -pit_depth), height)

        if os.environ.get("ATEC_INCLUDE_PLATFORM_IN_TERRAIN", "0") == "1":
            platform_center_y = float(os.environ.get("ATEC_PLATFORM_CENTER_Y", "2.0"))
            platform_half_y = float(os.environ.get("ATEC_PLATFORM_HALF_Y", "1.5"))
            platform_height = float(os.environ.get("ATEC_PLATFORM_HEIGHT", "1.1"))
            in_platform = (
                (torch.abs(x - pit_center_x) <= pit_width * 0.5)
                & (torch.abs(y - platform_center_y) <= platform_half_y)
            )
            height = torch.where(in_platform, torch.maximum(height, torch.full_like(height, platform_height)), height)

        if include_box and self._latest_box_pos is not None:
            bx, by, bz = self._latest_box_pos
            box_half_x = float(os.environ.get("ATEC_BOX_HALF_X", "0.40"))
            box_half_y = float(os.environ.get("ATEC_BOX_HALF_Y", "0.50"))
            box_half_z = float(os.environ.get("ATEC_BOX_HALF_Z", "0.30"))
            in_box = (torch.abs(x - bx) <= box_half_x) & (torch.abs(y - by) <= box_half_y)
            height = torch.where(in_box, torch.maximum(height, torch.full_like(height, bz + box_half_z)), height)
        return height

    def _fit_feature_dim(self, values: torch.Tensor, target_dim: int) -> torch.Tensor:
        values = values.flatten(start_dim=1)
        if values.shape[-1] == target_dim:
            return values
        values = F.interpolate(values.unsqueeze(1), size=target_dim, mode="linear", align_corners=False)
        return values.squeeze(1)

    def _height_scan(self, proprio: torch.Tensor, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = proprio.shape[0]
        default_source = "real" if self.policy_mode == "dreamwaq_taskd" else "geom"
        source = os.environ.get("ATEC_HEIGHT_SCAN_SOURCE", default_source).strip().lower()
        extero = obs.get("extero")
        if extero is not None and (source == "real" or os.environ.get("ATEC_USE_REAL_EXTERO", "0") == "1"):
            extero_tensor = extero.to(self.device, dtype=proprio.dtype)
            extero_tensor = torch.nan_to_num(extero_tensor, nan=2.0, posinf=2.0, neginf=-2.0)
            extero_tensor = torch.clamp(extero_tensor, -2.0, 2.0)
            return self._fit_feature_dim(extero_tensor, self.height_scan_xy.shape[0])

        if self.policy_mode == "dreamwaq_taskd" and source == "real":
            raise ValueError(
                "ATEC_POLICY_MODE=dreamwaq_taskd requires TaskD obs['extero'] LiDAR height scan. "
                "Run TaskD with the official extero observation enabled, or explicitly set "
                "ATEC_HEIGHT_SCAN_SOURCE=geom/zero for debugging only."
            )

        if os.environ.get("ATEC_ZERO_HEIGHT_SCAN", "0") == "1":
            return torch.zeros((batch, self.height_scan_xy.shape[0]), device=self.device, dtype=proprio.dtype)
        if self._latest_robot_pos is None:
            return torch.zeros((batch, self.height_scan_xy.shape[0]), device=self.device, dtype=proprio.dtype)

        robot_x, robot_y, robot_z = self._latest_robot_pos
        yaw = self._latest_robot_yaw or 0.0
        c = math.cos(yaw)
        s = math.sin(yaw)
        heights = []
        for px, py in self.height_scan_xy.detach().cpu().tolist():
            wx = robot_x + c * px - s * py
            wy = robot_y + s * px + c * py
            heights.append(self._clip(robot_z - self._taskd_terrain_height(wx, wy), 2.0))
        scan = torch.tensor(heights, device=self.device, dtype=proprio.dtype).view(1, -1)
        return scan.repeat(batch, 1) if batch > 1 else scan

    def _find_depth_tensor(self, item: Any) -> torch.Tensor | None:
        if isinstance(item, torch.Tensor):
            if item.ndim >= 3 and item.dtype.is_floating_point:
                return item
            return None
        if isinstance(item, dict):
            for key, value in item.items():
                if "depth" in str(key).lower():
                    found = self._find_depth_tensor(value)
                    if found is not None:
                        return found
            for value in item.values():
                found = self._find_depth_tensor(value)
                if found is not None:
                    return found
        return None

    def _raycast_depth_features(self, proprio: torch.Tensor) -> torch.Tensor:
        batch = proprio.shape[0]
        if self._latest_robot_pos is None:
            value = float(os.environ.get("ATEC_DEPTH_FEATURE_VALUE", "0.5"))
            return torch.full((batch, self.depth_feature_dim), value, device=self.device, dtype=proprio.dtype)

        robot_pos = torch.tensor(self._latest_robot_pos, device=self.device, dtype=torch.float32)
        if self._latest_robot_rot_w_b is None:
            yaw = self._latest_robot_yaw or 0.0
            c = math.cos(yaw)
            s = math.sin(yaw)
            rot_w_b = torch.tensor(
                ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            rot_w_b = self._latest_robot_rot_w_b.to(device=self.device, dtype=torch.float32)

        camera_pos_w = robot_pos + torch.matmul(self.depth_camera_pos_b, rot_w_b.T)
        dirs_w = torch.matmul(self.depth_camera_dirs_b, rot_w_b.T)
        t_values = self.depth_ray_t.to(dtype=torch.float32)
        points = camera_pos_w.view(1, 1, 3) + dirs_w.unsqueeze(1) * t_values.view(1, -1, 1)
        terrain_h = self._taskd_terrain_height_tensor(
            points[..., 0],
            points[..., 1],
            include_box=self._include_box_in_loco_terrain("depth"),
        )
        hits = points[..., 2] <= terrain_h + float(os.environ.get("ATEC_DEPTH_HIT_EPS", "0.01"))
        has_hit = hits.any(dim=1)
        first_hit = hits.to(torch.int64).argmax(dim=1)
        depth = t_values[first_hit]
        depth = torch.where(has_hit, depth, torch.full_like(depth, 2.5))
        depth = torch.clamp(depth, 0.0, 2.5).view(1, 48, 80) / 2.5 - 0.5
        depth = F.interpolate(depth.unsqueeze(1), size=(12, 16), mode="bilinear", align_corners=False).squeeze(1)
        depth = depth.flatten(start_dim=1).to(dtype=proprio.dtype)
        return depth.repeat(batch, 1) if batch > 1 else depth

    def _depth_features(self, proprio: torch.Tensor, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        default_source = "real" if self.policy_mode == "dreamwaq_taskd" else "geom"
        source = os.environ.get("ATEC_DEPTH_SOURCE", default_source).strip().lower()
        if os.environ.get("ATEC_ZERO_DEPTH_FEATURES", "0") == "1" or source == "zero":
            return torch.zeros(
                (proprio.shape[0], self.depth_feature_dim),
                device=self.device,
                dtype=proprio.dtype,
            )
        if source == "geom":
            return self._raycast_depth_features(proprio)

        image = obs.get("image")
        if image is not None and (source == "real" or os.environ.get("ATEC_USE_REAL_DEPTH", "0") == "1"):
            depth = self._find_depth_tensor(image)
            if depth is not None:
                depth = depth.to(self.device, dtype=proprio.dtype)
                if depth.ndim == 4 and depth.shape[-1] == 1:
                    depth = depth.squeeze(-1)
                if depth.ndim == 2:
                    depth = depth.unsqueeze(0)
                if depth.ndim == 3:
                    depth = torch.nan_to_num(depth, nan=2.5, posinf=2.5, neginf=0.0)
                    depth = torch.clamp(depth, 0.0, 2.5) / 2.5 - 0.5
                    depth = F.interpolate(
                        depth.unsqueeze(1),
                        size=(12, 16),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    return depth.flatten(start_dim=1)

        if self.policy_mode == "dreamwaq_taskd" and source == "real":
            if os.environ.get("ATEC_ALLOW_NEUTRAL_DEPTH", "0") != "1":
                raise ValueError(
                    "ATEC_POLICY_MODE=dreamwaq_taskd expects TaskD camera depth in obs['image']. "
                    "Run play_atec_task.py with --enable_cameras, or set ATEC_ALLOW_NEUTRAL_DEPTH=1 "
                    "to use the trained neutral-depth dropout fallback."
                )

        # The trained policy sees neutral depth during dropout.  This fallback is
        # explicit for no-camera debugging; official TaskD deployment should pass
        # real depth by running with --enable_cameras.
        value = float(os.environ.get("ATEC_DEPTH_FEATURE_VALUE", "0.5"))
        return torch.full(
            (proprio.shape[0], self.depth_feature_dim),
            value,
            device=self.device,
            dtype=proprio.dtype,
        )

    def _parkour_height_scan(self, proprio: torch.Tensor, obs: dict[str, Any]) -> torch.Tensor:
        extero = obs.get("extero")
        if extero is None:
            raise ValueError(
                "ATEC_POLICY_MODE=parkour_student requires TaskD obs['extero'] LiDAR height scan. "
                "Run the official TaskD env without disabling extero observations."
            )
        scan = extero.to(self.device, dtype=proprio.dtype)
        scan = torch.nan_to_num(scan, nan=1.0, posinf=1.0, neginf=-1.0)
        scan = torch.clamp(scan, -1.0, 1.0)
        return self._fit_feature_dim(scan, self.PARKOUR_NUM_SCAN)

    def _extract_depth_image_tensor(self, image: Any) -> torch.Tensor | None:
        depth = self._find_depth_tensor(image)
        if depth is None:
            return None
        depth = depth.to(self.device, dtype=torch.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim != 3:
            return None
        depth = torch.nan_to_num(depth, nan=2.0, posinf=2.0, neginf=0.0)
        depth = torch.clamp(depth, 0.0, 2.0)
        # Match Parkour's RayCasterCamera preprocessing: crop, resize to 58x87,
        # then normalize by max_distance=2.0 and shift by -0.5.
        if depth.shape[-2] > 4 and depth.shape[-1] > 8:
            depth = depth[:, :-2, 4:-4]
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=self.PARKOUR_DEPTH_SHAPE,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        return depth / 2.0 - 0.5

    def _parkour_depth_image(self, obs: dict[str, Any], batch: int, dtype: torch.dtype) -> torch.Tensor:
        image = obs.get("image")
        if image is not None:
            depth = self._extract_depth_image_tensor(image)
            if depth is not None:
                depth = depth.to(self.device, dtype=dtype)
                return depth.repeat(batch, 1, 1) if depth.shape[0] == 1 and batch > 1 else depth
        if os.environ.get("ATEC_PARKOUR_ALLOW_DEPTH_FALLBACK", "0") != "1":
            raise ValueError(
                "ATEC_POLICY_MODE=parkour_student requires real depth image in obs['image']. "
                "Run play_atec_task.py with --enable_cameras. For no-camera debugging only, set "
                "ATEC_PARKOUR_ALLOW_DEPTH_FALLBACK=1."
            )
        value = float(os.environ.get("ATEC_PARKOUR_DEPTH_FALLBACK_VALUE", "0.5"))
        return torch.full(
            (batch, *self.PARKOUR_DEPTH_SHAPE),
            value,
            device=self.device,
            dtype=dtype,
        )

    def _parkour_roll_pitch_tensor(self, proprio: torch.Tensor) -> torch.Tensor:
        batch = proprio.shape[0]
        if self._latest_robot_rot_w_b is not None:
            # Use the quaternion injected by play_atec_task.py.  This matches the
            # Parkour observation more closely than reconstructing attitude from
            # projected gravity.
            rot = self._latest_robot_rot_w_b.detach().cpu()
            # Convert through matrix entries to avoid storing the raw quaternion.
            pitch = math.atan2(-float(rot[2, 0]), math.sqrt(float(rot[2, 1]) ** 2 + float(rot[2, 2]) ** 2))
            roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
            rp = torch.tensor((roll, pitch), device=self.device, dtype=proprio.dtype).view(1, 2)
            return rp.repeat(batch, 1)
        projected_gravity = proprio[:, 9:12]
        roll = torch.atan2(projected_gravity[:, 1], torch.clamp(projected_gravity[:, 2], min=-1.0, max=1.0))
        pitch = torch.atan2(-projected_gravity[:, 0], torch.linalg.norm(projected_gravity[:, 1:3], dim=-1))
        return torch.stack((roll, pitch), dim=-1)

    def _parkour_contact_fill(self, proprio: torch.Tensor) -> torch.Tensor:
        value = float(os.environ.get("ATEC_PARKOUR_CONTACT_DEFAULT", "0.5"))
        return torch.full((proprio.shape[0], 4), value, device=self.device, dtype=proprio.dtype)

    def _task_to_parkour_leg_order(self, leg_tensor: torch.Tensor) -> torch.Tensor:
        """Convert TaskD FR-leg blocks to Parkour's native joint-type order."""
        order = os.environ.get("ATEC_PARKOUR_LEG_ORDER", "native").strip().lower()
        if order in {"task", "fr_fl_rr_rl", "frflrrrl"}:
            return leg_tensor
        perm = torch.tensor((3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8), device=leg_tensor.device)
        return leg_tensor.index_select(dim=-1, index=perm)

    def _parkour_to_task_leg_order(self, leg_tensor: torch.Tensor) -> torch.Tensor:
        """Convert Parkour native joint-type order to TaskD FR-leg blocks."""
        order = os.environ.get("ATEC_PARKOUR_LEG_ORDER", "native").strip().lower()
        if order in {"task", "fr_fl_rr_rl", "frflrrrl"}:
            return leg_tensor
        perm = torch.tensor((1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10), device=leg_tensor.device)
        return leg_tensor.index_select(dim=-1, index=perm)

    def _build_parkour_prop(
        self,
        proprio: torch.Tensor,
        obs: dict[str, Any],
        action_dim: int,
        command_values: tuple[float, float, float],
    ) -> torch.Tensor:
        idx = 0
        idx += 3  # base linear velocity
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        idx += 3  # task command, replaced by state-machine command
        idx += 3  # projected gravity
        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        batch = proprio.shape[0]
        command = torch.tensor(command_values, device=self.device, dtype=proprio.dtype).view(1, 3)
        if batch > 1:
            command = command.repeat(batch, 1)

        last_action_parkour = self._last_parkour_action.to(device=self.device, dtype=proprio.dtype)
        if last_action_parkour.shape[0] != batch:
            last_action_parkour = last_action_parkour[:1].repeat(batch, 1)
        joint_pos_leg = self._task_to_parkour_leg_order(joint_pos_all[:, :self.LEG_ACTION_DIM])
        joint_vel_leg = self._task_to_parkour_leg_order(joint_vel_all[:, :self.LEG_ACTION_DIM])
        prop = torch.cat(
            (
                base_ang_vel * 0.25,
                self._parkour_roll_pitch_tensor(proprio),
                torch.zeros((batch, 1), device=self.device, dtype=proprio.dtype),
                torch.zeros((batch, 1), device=self.device, dtype=proprio.dtype),
                torch.zeros((batch, 1), device=self.device, dtype=proprio.dtype),
                torch.zeros((batch, 2), device=self.device, dtype=proprio.dtype),
                command[:, 0:1],
                torch.full(
                    (batch, 1),
                    float(os.environ.get("ATEC_PARKOUR_TERRAIN_FLAG", "0.0")),
                    device=self.device,
                    dtype=proprio.dtype,
                ),
                torch.full(
                    (batch, 1),
                    float(os.environ.get("ATEC_PARKOUR_FLAT_FLAG", "1.0")),
                    device=self.device,
                    dtype=proprio.dtype,
                ),
                joint_pos_leg,
                joint_vel_leg * 0.05,
                last_action_parkour,
                self._parkour_contact_fill(proprio),
            ),
            dim=-1,
        )
        if prop.shape[-1] != self.PARKOUR_NUM_PROP:
            raise ValueError(f"Parkour prop dim mismatch: got {prop.shape[-1]}, expected {self.PARKOUR_NUM_PROP}")
        return prop

    def _extract_parkour_obs(
        self,
        obs: dict[str, Any],
        action_dim: int,
        phase_command: tuple[str, tuple[float, float, float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = obs["proprio"].to(self.device)
        phase, command_values = phase_command if phase_command is not None else self._taskd_command()
        self.current_phase = phase
        self.current_command = command_values
        if self.debug and phase != self._last_phase:
            self._debug_log(f"[solution] step={self.step_count} phase={phase} cmd={command_values}")
            self._last_phase = phase

        prop = self._build_parkour_prop(proprio, obs, action_dim, command_values)
        prop_for_depth = prop.clone()
        prop_for_depth[:, 6:8] = 0.0
        if self.step_count <= 1 or self._parkour_history.shape[0] != proprio.shape[0]:
            self._parkour_history = torch.stack([prop_for_depth] * self.PARKOUR_HISTORY_LENGTH, dim=1)

        depth_image = self._parkour_depth_image(obs, proprio.shape[0], proprio.dtype)
        self._last_parkour_depth_stats = (float(depth_image.min()), float(depth_image.max()))
        if self.step_count % 5 == 0 or self._parkour_depth_latent.shape[0] != proprio.shape[0]:
            depth_latent_and_yaw = self.depth_encoder(depth_image, prop_for_depth)
            self._parkour_depth_latent = depth_latent_and_yaw[:, :-2].to(device=self.device, dtype=proprio.dtype)
            self._parkour_yaw = depth_latent_and_yaw[:, -2:].to(device=self.device, dtype=proprio.dtype)
            if os.environ.get("ATEC_PARKOUR_DISABLE_DEPTH_LATENT", "0") == "1":
                self._parkour_depth_latent.zero_()

        prop = prop.clone()
        # The depth network's yaw channels encode the Parkour route waypoint,
        # which TaskD does not provide. Keep the real depth latent, but do not
        # inject route-yaw predictions unless an equivalent target is supplied.
        if os.environ.get("ATEC_PARKOUR_USE_DEPTH_YAW", "0") == "1":
            prop[:, 6:8] = 1.5 * self._parkour_yaw
        else:
            prop[:, 6:8] = self._parkour_heading_signal(command_values, proprio.shape[0], proprio.dtype)
        scan = self._parkour_height_scan(proprio, obs)
        self._last_parkour_scan_stats = (float(scan.min()), float(scan.max()))
        idx = 0
        base_lin_vel = proprio[:, idx:idx + 3]
        priv_explicit = torch.cat((base_lin_vel * 2.0, 0.0 * base_lin_vel, 0.0 * base_lin_vel), dim=-1)
        priv_latent = torch.zeros(
            (proprio.shape[0], self.PARKOUR_NUM_PRIV_LATENT),
            device=self.device,
            dtype=proprio.dtype,
        )
        full_obs = torch.cat(
            (
                prop,
                scan,
                priv_explicit,
                priv_latent,
                self._parkour_history.reshape(proprio.shape[0], -1),
            ),
            dim=-1,
        )
        self._parkour_history = torch.cat(
            (self._parkour_history[:, 1:], prop_for_depth.unsqueeze(1)),
            dim=1,
        )
        return full_obs, self._parkour_depth_latent

    def _describe_value(self, value: Any) -> str:
        if isinstance(value, torch.Tensor):
            return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
        if isinstance(value, dict):
            inner = ", ".join(f"{key}: {self._describe_value(val)}" for key, val in value.items())
            return "{" + inner + "}"
        return type(value).__name__

    def _print_obs_shapes_once(self, obs: dict[str, Any]) -> None:
        if not self.debug or self._printed_obs_shapes:
            return
        self._printed_obs_shapes = True
        print("[solution] obs layout:")
        for key, value in obs.items():
            print(f"[solution]   {key}: {self._describe_value(value)}")

    def _extract_policy_obs(self, obs: dict[str, torch.Tensor], action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 0
        idx += 3  # base linear velocity, not used by actor
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        idx += 3  # task command, replaced by state-machine command
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, :self.LEG_ACTION_DIM]
        joint_vel_leg = joint_vel_all[:, :self.LEG_ACTION_DIM]
        actions_leg_train = actions_all[:, :self.LEG_ACTION_DIM] * self.task_to_train_action_scale.to(dtype=proprio.dtype)

        phase, command_values = self._taskd_command()
        self.current_phase = phase
        self.current_command = command_values
        if self.debug and phase != self._last_phase:
            self._debug_log(f"[solution] step={self.step_count} phase={phase} cmd={command_values}")
            self._last_phase = phase

        command = torch.tensor(command_values, device=self.device, dtype=proprio.dtype).view(1, 3)
        if proprio.shape[0] > 1:
            command = command.repeat(proprio.shape[0], 1)

        obs_terms = [
            base_ang_vel * 0.25,
            projected_gravity,
            command,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_leg_train,
        ]
        if self.policy_mode in {"dreamwaq", "dreamwaq_taskd"}:
            current_dim = sum(term.shape[-1] for term in obs_terms)
            remaining_dim = self.policy_obs_dim - current_dim
            if remaining_dim < 0:
                raise ValueError(
                    f"DreamWaQ base obs dim mismatch: base obs has {current_dim}, policy expects {self.policy_obs_dim}"
                )
            if remaining_dim > 0:
                height_scan = self._height_scan(proprio, obs)
                height_dim = min(height_scan.shape[-1], remaining_dim)
                obs_terms.append(self._fit_feature_dim(height_scan, height_dim))
                remaining_dim -= height_dim
            if remaining_dim > 0:
                depth_features = self._depth_features(proprio, obs)
                obs_terms.append(self._fit_feature_dim(depth_features, remaining_dim))
        return torch.cat(obs_terms, dim=-1)

    def _map_parkour_action(
        self,
        action_train: torch.Tensor,
        action_dim: int,
        standalone: bool,
    ) -> torch.Tensor:
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        raw_action_train = action_train.detach().clone()
        policy_clip = float(os.environ.get("ATEC_PARKOUR_POLICY_ACTION_CLIP", "4.8"))
        action_train = torch.clamp(action_train, -policy_clip, policy_clip)
        action_task = torch.zeros((action_train.shape[0], action_dim), device=self.device, dtype=torch.float32)
        self._last_parkour_raw_action = raw_action_train[:, :self.LEG_ACTION_DIM].detach().clone()
        parkour_leg_action = action_train[:, :self.LEG_ACTION_DIM]
        self._last_parkour_clipped_action = parkour_leg_action.detach().clone()
        self._last_parkour_action = parkour_leg_action.detach().clone()
        task_leg_action = self._parkour_to_task_leg_order(parkour_leg_action)
        # Standalone Parkour changes the TaskD action scale to 0.25. Hybrid
        # mode keeps the rough-policy default scale of 0.5, so halve the raw
        # Parkour action to preserve the same physical joint target.
        default_scale = "1.0" if standalone else "0.5"
        task_scale = float(os.environ.get("ATEC_PARKOUR_ACTION_TO_TASK_SCALE", default_scale))
        action_task[:, :self.LEG_ACTION_DIM] = task_leg_action * task_scale
        if standalone:
            warmup_steps = int(float(os.environ.get("ATEC_PARKOUR_WARMUP_S", "1.5")) * self.CONTROL_HZ)
            if warmup_steps > 0 and self.step_count <= warmup_steps:
                action_task[:, :self.LEG_ACTION_DIM] *= float(self.step_count) / float(warmup_steps)
        return action_task

    def _map_policy_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        if self.policy_mode == "parkour_student":
            return self._map_parkour_action(action_train, action_dim, standalone=True)

        policy_clip = float(os.environ.get("ATEC_POLICY_ACTION_CLIP", "3.0"))
        action_train = torch.clamp(action_train, -policy_clip, policy_clip)
        action_task = torch.zeros((action_train.shape[0], action_dim), device=self.device, dtype=torch.float32)
        action_task[:, :self.LEG_ACTION_DIM] = action_train[:, :self.LEG_ACTION_DIM] * self.train_to_task_action_scale
        return action_task

    def predicts(self, obs, current_score):
        self.step_count += 1
        self._print_obs_shapes_once(obs)
        if "_task_name" in obs:
            self._task_name = str(obs["_task_name"])
            if (
                "TaskD" in self._task_name
                and "B2" not in self._task_name
                and not self._warned_task_mismatch
            ):
                print(
                    "[solution] WARNING: this state machine is for B2/B2Piper. "
                    f"Current task is {self._task_name}; use --task ATEC-TaskD-B2Piper."
                )
                self._warned_task_mismatch = True
        if "_robot_pos" in obs:
            robot_pos = obs["_robot_pos"][0].detach().cpu().tolist()
            self._latest_robot_pos = (float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]))
        if "_box_pos" in obs:
            box_pos = obs["_box_pos"][0].detach().cpu().tolist()
            self._latest_box_pos = (float(box_pos[0]), float(box_pos[1]), float(box_pos[2]))
        if "_robot_quat" in obs:
            quat = obs["_robot_quat"][0].detach().cpu().tolist()
            self._latest_robot_yaw = self._quat_wxyz_to_yaw([float(value) for value in quat])
            self._latest_robot_rot_w_b = self._quat_wxyz_to_matrix(
                torch.tensor([float(value) for value in quat], device=self.device, dtype=torch.float32)
            )

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        if self.policy_mode == "parkour_student":
            policy_obs, parkour_depth_latent = self._extract_parkour_obs(obs, action_dim)
        else:
            policy_obs = self._extract_policy_obs(obs, action_dim)
            parkour_depth_latent = None

        box_reward_reached = current_score >= float(os.environ.get("ATEC_BOX_REWARD_DONE_SCORE", "13.5"))
        if (
            box_reward_reached
            and self.phase == "push_box_to_pit_x"
            and os.environ.get("ATEC_ENABLE_CROSS", "1") != "1"
        ):
            self._set_phase("done")
            self.current_phase = "done"
            self.current_command = (0.0, 0.0, 0.0)

        if self.phase == "done" and os.environ.get("ATEC_GIVEUP_ON_DONE", "1") == "1":
            action_task = torch.zeros((proprio.shape[0], action_dim), device=self.device, dtype=torch.float32)
            return {"action": action_task.cpu().tolist(), "giveup": True}

        expected_obs_dim = self.policy_obs_dim
        if policy_obs.shape[-1] != expected_obs_dim:
            raise ValueError(f"DreamWaQ obs dim mismatch: got {policy_obs.shape[-1]}, expected {expected_obs_dim}")

        with torch.inference_mode():
            if self.policy_mode == "parkour_student":
                action_train = self.policy(policy_obs, parkour_depth_latent)
            else:
                action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)

        action_task = self._map_policy_action(action_train.to(device=self.device, dtype=torch.float32), action_dim)
        parkour_blend = 0.0
        if self.use_parkour_cross and self.policy_mode != "parkour_student":
            entering_cross = self.current_phase == "cross_pit" and self._parkour_switch_step is None
            if entering_cross:
                self._parkour_switch_step = self.step_count
                self._parkour_heading_target = self._latest_robot_yaw
                self._parkour_path_heading = self._latest_robot_yaw
                self._parkour_path_origin = (
                    self._latest_robot_pos[:2] if self._latest_robot_pos is not None else None
                )
                self._parkour_cross_track_error = 0.0
                if self.debug:
                    self._debug_log(
                        f"[solution] switching rough -> parkour at step={self.step_count} "
                        f"pos={self._latest_robot_pos} yaw={self._latest_robot_yaw}"
                    )

            parkour_obs, parkour_depth_latent = self._extract_parkour_obs(
                obs,
                action_dim,
                phase_command=(self.current_phase, self.current_command),
            )
            if parkour_obs.shape[-1] != self.parkour_policy_obs_dim:
                raise ValueError(
                    f"Parkour obs dim mismatch: got {parkour_obs.shape[-1]}, "
                    f"expected {self.parkour_policy_obs_dim}"
                )
            with torch.inference_mode():
                parkour_action_train = self.parkour_policy(parkour_obs, parkour_depth_latent)
            parkour_action_task = self._map_parkour_action(
                parkour_action_train.to(device=self.device, dtype=torch.float32),
                action_dim,
                standalone=False,
            )

            if self.current_phase == "cross_pit" and self._parkour_switch_step is not None:
                blend_steps = max(
                    1,
                    int(float(os.environ.get("ATEC_PARKOUR_SWITCH_BLEND_S", "0.5")) * self.CONTROL_HZ),
                )
                parkour_blend = min(1.0, (self.step_count - self._parkour_switch_step) / blend_steps)
                action_task = torch.lerp(action_task, parkour_action_task, parkour_blend)
        if self.debug and self.step_count % int(os.environ.get("ATEC_DEBUG_INTERVAL", "50")) == 0:
            action_leg = action_task[:, :self.LEG_ACTION_DIM]
            base_lin_vel = proprio[:, :3]
            base_ang_vel = proprio[:, 3:6]
            projected_gravity = proprio[:, 9:12]
            extra = ""
            if self.policy_mode == "parkour_student" or self.use_parkour_cross:
                raw = self._last_parkour_raw_action
                clipped = self._last_parkour_clipped_action
                depth_latent = self._parkour_depth_latent
                extra = (
                    f" parkour_raw=({float(raw.min()):.3f},{float(raw.max()):.3f})"
                    f" parkour_clip=({float(clipped.min()):.3f},{float(clipped.max()):.3f})"
                    f" depth_latent=({float(depth_latent.min()):.3f},{float(depth_latent.max()):.3f})"
                    f" depth_yaw=({float(self._parkour_yaw[0,0]):.3f},{float(self._parkour_yaw[0,1]):.3f})"
                    f" depth={self._last_parkour_depth_stats}"
                    f" scan={self._last_parkour_scan_stats}"
                    f" heading=({self._parkour_heading_target},{self._parkour_heading_error:.3f})"
                    f" cross_track={self._parkour_cross_track_error:.3f}"
                    f" parkour_blend={parkour_blend:.2f}"
                )
            self._debug_log(
                "[solution] "
                f"step={self.step_count} task={self._task_name} phase={self.current_phase} "
                f"cmd=({self.current_command[0]:.2f},{self.current_command[1]:.2f},{self.current_command[2]:.2f}) "
                f"yaw={self._latest_robot_yaw} target={self.current_target} "
                f"robot={self._latest_robot_pos} box={self._latest_box_pos} "
                f"base_v=({float(base_lin_vel[0,0]):.3f},{float(base_lin_vel[0,1]):.3f},{float(base_lin_vel[0,2]):.3f}) "
                f"base_w=({float(base_ang_vel[0,0]):.3f},{float(base_ang_vel[0,1]):.3f},{float(base_ang_vel[0,2]):.3f}) "
                f"grav=({float(projected_gravity[0,0]):.3f},{float(projected_gravity[0,1]):.3f},{float(projected_gravity[0,2]):.3f}) "
                f"leg_action_minmax=({float(action_leg.min()):.3f},{float(action_leg.max()):.3f})"
                f"{extra}"
            )
        return {"action": action_task.cpu().tolist(), "giveup": False}
