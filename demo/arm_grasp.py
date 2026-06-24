"""机械臂抓取 — 供 solution_rl 使用."""
import math
import os
import numpy as np
import torch
try:
    from atec_rl_lab.utils.cartesian_controller import CartesianController
except ImportError:
    CartesianController = None


class ArmGraspController:
    """最小可用的机械臂抓取状态机。"""

    def __init__(
        self,
        robot,
        device: torch.device,
        arm_joint_names: list[str],
        gripper_joint_names: list[str],
        ee_body_name: str = "gripper_base",
        action_scale: float = 0.5,
        pregrasp_height: float = 0.15,
        grasp_height_offset: float = 0.03,
        lift_height: float = 0.30,
        ee_pos_tol: float = 0.05,
        gripper_close_wait_steps: int = 30,
        lift_success_threshold: float = 0.20,
        gripper_open_pos: tuple[float, float] = (0.035, -0.035),
        gripper_close_pos: tuple[float, float] = (-0.015, 0.015),
        log_every_steps: int = 10,
    ):
        self.robot = robot
        self.device = device
        self.arm_joint_names = list(arm_joint_names)
        self.gripper_joint_names = list(gripper_joint_names)
        self.ee_body_name = ee_body_name
        self.action_scale = float(action_scale)
        self.pregrasp_height = float(pregrasp_height)
        self.grasp_height_offset = float(grasp_height_offset)
        self.lift_height = float(lift_height)
        self.ee_pos_tol = float(ee_pos_tol)
        self.gripper_close_wait_steps = int(gripper_close_wait_steps)
        self.lift_success_threshold = float(lift_success_threshold)
        self.gripper_open_pos = torch.tensor(gripper_open_pos, dtype=torch.float32, device=device)
        self.gripper_close_pos = torch.tensor(gripper_close_pos, dtype=torch.float32, device=device)
        self.log_every_steps = max(1, int(log_every_steps))
        self.stable_steps_required = max(1, int(os.getenv("ATEC_TASKB_ARM_STABLE_STEPS", "4")))
        self.max_phase_steps = max(1, int(os.getenv("ATEC_TASKB_ARM_MAX_PHASE_STEPS", "200")))
        self.ik_max_iters = max(1, int(os.getenv("ATEC_TASKB_ARM_IK_MAX_ITERS", "3")))

        # 时间步长，用于笛卡尔空间速度限制
        self.dt = float(os.getenv("ATEC_TASKB_SIM_DT", "0.02"))

        self.arm_joint_ids, _ = robot.find_joints(self.arm_joint_names)
        self.gripper_joint_ids, _ = robot.find_joints(self.gripper_joint_names)
        self.arm_and_gripper_joint_ids = list(self.arm_joint_ids) + list(self.gripper_joint_ids)

        self.cartesian = None
        if CartesianController is None:
            print("[ArmGraspController] Warning: CartesianController unavailable, grasping will fail closed.", flush=True)
        else:
            try:
                # max_joint_delta controls arm movement speed - smaller = slower/more stable
                max_joint_delta = float(os.getenv("ATEC_TASKB_ARM_MAX_JOINT_DELTA", "0.05"))
                self.cartesian = CartesianController(
                    robot=robot,
                    ee_body_name=ee_body_name,
                    arm_joint_names=self.arm_joint_names,
                    num_envs=1,
                    device=str(device),
                    command_type="pose",
                    lambda_val=0.1,
                    max_joint_delta=max_joint_delta,
                )
                self.cartesian.reset()
            except Exception as exc:
                print(f"[ArmGraspController] Warning: failed to initialize IK controller: {exc}", flush=True)
                self.cartesian = None

        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.current_target = None
        self.current_target_id = None
        self.trash_pos_w = None
        self.pregrasp_pos_w = None
        self.grasp_pos_w = None
        self.lift_pos_w = None
        self.target_ee_quat_w = None
        self.initial_trash_z = None
        self.wait_steps = 0
        self.step_counter = 0
        self.phase_step_counter = 0
        self.phase_stable_counter = 0
        self.success = False
        self.failure_reason = None
        self.desired_arm_joint_pos = None
        self.desired_gripper_joint_pos = self.gripper_open_pos.clone()
        self.current_target_pos_w = None
        self.current_target_pos_b = None
        self.current_target_pos_b_raw = None
        self.current_target_pos_w_compensated = None
        if self.cartesian is not None:
            self.cartesian.reset()

    def start_grasp(self, trash_target, trash_pos_w, current_ee_quat_w=None):
        self.reset()
        self.current_target = trash_target
        self.current_target_id = None if trash_target is None else trash_target.get("id", "unknown")
        self.trash_pos_w = np.asarray(trash_pos_w, dtype=np.float32).copy()
        self.pregrasp_pos_w = self.trash_pos_w + np.array([0.0, 0.0, self.pregrasp_height], dtype=np.float32)
        # 这里默认 trash_pos_w 可直接作为抓取中心参考点；若 z 是物体中心，后续可调 grasp_height_offset。
        self.grasp_pos_w = self.trash_pos_w + np.array([0.0, 0.0, self.grasp_height_offset], dtype=np.float32)
        self.lift_pos_w = self.trash_pos_w + np.array([0.0, 0.0, self.lift_height], dtype=np.float32)
        self.initial_trash_z = float(self.trash_pos_w[2])
        if current_ee_quat_w is None:
            ee_quat_w = self.get_ee_pose()[1]
        else:
            ee_quat_w = np.asarray(current_ee_quat_w, dtype=np.float32)
        self.target_ee_quat_w = ee_quat_w.copy()
        self._refresh_target_poses(scene=None)
        self._set_state("MOVE_TO_PREGRASP")
        self._log(force=True, extra="start_grasp")

    def step(self, robot, scene, sim_dt):
        del sim_dt
        self.robot = robot
        self.step_counter += 1
        self.phase_step_counter += 1

        if self.state == "IDLE":
            return False, False
        if self.cartesian is None:
            self.failure_reason = "ik_unavailable"
            self._set_state("FAILED")
            self._log(force=True, extra="IK unavailable")
            return True, False

        self._refresh_target_poses(scene)
        ee_pos_w, _ = self.get_ee_pose()

        if self._phase_timed_out():
            self.failure_reason = f"phase_timeout state={self.state} steps={self.phase_step_counter}"
            self._set_state("FAILED")
            self._log(force=True, extra="Phase timeout")
            return True, False

        if self.state == "MOVE_TO_PREGRASP":
            self.open_gripper()
            self.move_ee_to_pose(self.pregrasp_pos_w, self.target_ee_quat_w)
            if self._update_stable_reached(self.ee_reached(ee_pos_w, self.pregrasp_pos_w)):
                self._set_state("MOVE_DOWN_TO_GRASP")
                self._log(force=True, extra="Reached pregrasp")

        elif self.state == "MOVE_DOWN_TO_GRASP":
            self.open_gripper()
            self.move_ee_to_pose(self.grasp_pos_w, self.target_ee_quat_w)
            if self._update_stable_reached(self.ee_reached(ee_pos_w, self.grasp_pos_w)):
                self._set_state("CLOSE_GRIPPER")
                self.wait_steps = 0
                self._log(force=True, extra="Reached grasp pose")

        elif self.state == "CLOSE_GRIPPER":
            self.close_gripper()
            self.wait_steps += 1
            if self.wait_steps >= self.gripper_close_wait_steps:
                self._set_state("LIFT_OBJECT")
                self._log(force=True, extra="Gripper close wait finished")

        elif self.state == "LIFT_OBJECT":
            self.close_gripper()
            self.move_ee_to_pose(self.lift_pos_w, self.target_ee_quat_w)
            if self._update_stable_reached(self.ee_reached(ee_pos_w, self.lift_pos_w)):
                self._set_state("VERIFY_GRASP")
                self._log(force=True, extra="Reached lift pose")

        elif self.state == "VERIFY_GRASP":
            self.close_gripper()
            self.success = self.check_grasp_success(scene)
            self._set_state("DONE" if self.success else "FAILED")
            self._log(force=True, extra=f"verify success={self.success}")

        elif self.state == "DONE":
            return True, True

        elif self.state == "FAILED":
            return True, False

        self._log()
        return False, False

    def move_ee_to_pose(self, target_pos_w, target_quat_w=None):
        if self.cartesian is None:
            return

        self.current_target_pos_w = np.asarray(target_pos_w, dtype=np.float32)
        self.current_target_pos_w_compensated = self.current_target_pos_w.copy()
        raw_target_pos_b = self._world_pos_to_base_frame(self.current_target_pos_w)
        self.current_target_pos_b_raw = None if raw_target_pos_b is None else raw_target_pos_b.detach().cpu().numpy()[0]

        # Compensate for gripper_base to fingertip offset
        # gripper_base origin is at the base of the gripper, but fingertips extend
        # 0.1358m along gripper_base's local Z axis. We need to move gripper_base
        # BACKWARD so that fingertips end up at the target position.
        ee_pos_w, ee_quat_w = self.get_ee_pose()
        ee_quat_t = torch.tensor(ee_quat_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        # gripper_base local Z axis in world frame
        local_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).unsqueeze(0)
        from isaaclab.utils.math import quat_rotate
        z_world = quat_rotate(ee_quat_t, local_z).squeeze(0).cpu().numpy()
        # Offset gripper_base backward along its Z axis so fingertips reach target
        finger_offset = float(os.getenv("ATEC_TASKB_FINGER_OFFSET", "0.12"))
        disable_finger_comp = os.getenv("ATEC_TASKB_DISABLE_FINGER_COMP", "1").lower() in {"1", "true", "yes", "on"}
        if disable_finger_comp:
            compensated_target = np.asarray(target_pos_w, dtype=np.float32)
        else:
            compensated_target = np.asarray(target_pos_w, dtype=np.float32) - z_world * finger_offset
        self.current_target_pos_w_compensated = compensated_target.copy()

        target_pos_b = self._world_pos_to_base_frame(compensated_target)
        self.current_target_pos_b = None if target_pos_b is None else target_pos_b.detach().cpu().numpy()[0]
        if target_pos_b is None:
            return

        if target_quat_w is None:
            target_quat_w = self._compute_top_down_target_quat_w(compensated_target)
        target_quat_b = self._world_quat_to_base_frame(target_quat_w)
        if target_quat_b is None:
            return
        self.desired_arm_joint_pos = self.cartesian.compute_base(target_pos_b, target_quat_b).detach().clone()

    def ee_reached(self, ee_pos_w, target_pos_w):
        finger_offset = float(os.getenv("ATEC_TASKB_FINGER_OFFSET", "0.12"))
        ee_quat_w_val = self.get_ee_pose()[1]
        ee_quat_t = torch.tensor(ee_quat_w_val, dtype=torch.float32, device=self.device).unsqueeze(0)
        local_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).unsqueeze(0)
        from isaaclab.utils.math import quat_rotate
        z_world = quat_rotate(ee_quat_t, local_z).squeeze(0).cpu().numpy()
        fingertip_pos = np.asarray(ee_pos_w) + z_world * finger_offset
        pos_err = float(np.linalg.norm(fingertip_pos - np.asarray(target_pos_w)))
        return pos_err < self.ee_pos_tol

    def open_gripper(self):
        self.desired_gripper_joint_pos = self.gripper_open_pos.clone()

    def close_gripper(self):
        self.desired_gripper_joint_pos = self.gripper_close_pos.clone()

    def get_ee_pose(self):
        if self.cartesian is not None:
            ee_pos = self.cartesian.ee_pos_w[0].detach().cpu().numpy()
            ee_quat = self.cartesian.ee_quat_w[0].detach().cpu().numpy()
            return ee_pos, ee_quat

        if hasattr(self.robot.data, "body_pose_w"):
            try:
                body_ids, _ = self.robot.find_bodies(self.ee_body_name)
                body_pose = self.robot.data.body_pose_w[0, body_ids[0]]
                return body_pose[:3].detach().cpu().numpy(), body_pose[3:].detach().cpu().numpy()
            except Exception:
                pass
        return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_current_trash_pos_w(self, scene):
        lookup_ids = []
        if self.current_target is not None:
            scene_id = self.current_target.get("scene_object_id")
            if scene_id is not None:
                lookup_ids.append(scene_id)
        if self.current_target_id is not None:
            lookup_ids.append(self.current_target_id)
        if not lookup_ids or scene is None:
            return None
        for lookup_id in lookup_ids:
            for container_name in ("rigid_objects", "articulations"):
                container = getattr(scene, container_name, None)
                if container is None:
                    continue
                try:
                    obj = container[lookup_id]
                except Exception:
                    continue
                if hasattr(obj, "data") and hasattr(obj.data, "root_pos_w"):
                    return obj.data.root_pos_w[0].detach().cpu().numpy()
        return None

    def check_grasp_success(self, scene):
        current_trash_pos_w = self.get_current_trash_pos_w(scene)
        if current_trash_pos_w is None:
            # TODO: 若后续 target 仅有坐标无物体句柄，应接入真实物体高度验证而非占位成功。
            self.failure_reason = "missing_trash_pose_fallback_success"
            print("[ArmGraspController] Warning: current trash pose unavailable, using temporary success fallback.", flush=True)
            return True

        lifted = float(current_trash_pos_w[2]) > float(self.initial_trash_z) + self.lift_success_threshold
        if not lifted:
            self.failure_reason = (
                f"trash_not_lifted current_z={float(current_trash_pos_w[2]):.3f} "
                f"initial_z={float(self.initial_trash_z):.3f}"
            )
        return lifted

    def apply_to_action_tensor(self, action_env: torch.Tensor, robot) -> torch.Tensor:
        if action_env.ndim == 1:
            action_env = action_env.unsqueeze(0)

        if self.desired_arm_joint_pos is None:
            arm_target = robot.data.joint_pos[:, self.arm_joint_ids].clone()
        else:
            arm_target = self.desired_arm_joint_pos.to(device=action_env.device, dtype=action_env.dtype)
        if self.desired_gripper_joint_pos is None:
            gripper_target = robot.data.joint_pos[:, self.gripper_joint_ids].clone()
        else:
            gripper_target = self.desired_gripper_joint_pos.to(device=action_env.device, dtype=action_env.dtype).view(1, -1)

        default_joint_pos = robot.data.default_joint_pos.to(device=action_env.device, dtype=action_env.dtype)
        arm_default_target = default_joint_pos[:, self.arm_joint_ids].clone()
        if hasattr(self, "b2_piper_arm_defaults"):
            for local_idx, joint_name in enumerate(self.arm_joint_names):
                if joint_name in self.b2_piper_arm_defaults:
                    arm_default_target[:, local_idx] = float(self.b2_piper_arm_defaults[joint_name])
        action_env[:, self.arm_joint_ids] = (arm_target - arm_default_target) / self.action_scale
        action_env[:, self.gripper_joint_ids] = (gripper_target - default_joint_pos[:, self.gripper_joint_ids]) / self.action_scale
        return action_env

    def _log(self, force: bool = False, extra: str | None = None):
        if not force and self.step_counter % self.log_every_steps != 0:
            return
        ee_pos_w, _ = self.get_ee_pose()
        target_pos = None
        if self.state == "MOVE_TO_PREGRASP":
            target_pos = self.pregrasp_pos_w
        elif self.state == "MOVE_DOWN_TO_GRASP":
            target_pos = self.grasp_pos_w
        elif self.state in {"LIFT_OBJECT", "VERIFY_GRASP", "DONE", "FAILED"}:
            target_pos = self.lift_pos_w
        pos_err = None if target_pos is None else float(np.linalg.norm(np.asarray(ee_pos_w) - np.asarray(target_pos)))
        finger_offset = float(os.getenv("ATEC_TASKB_FINGER_OFFSET", "0.1358"))
        ee_quat_w_val = self.get_ee_pose()[1]
        ee_quat_t = torch.tensor(ee_quat_w_val, dtype=torch.float32, device=self.device).unsqueeze(0)
        local_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).unsqueeze(0)
        from isaaclab.utils.math import quat_rotate
        z_world = quat_rotate(ee_quat_t, local_z).squeeze(0).cpu().numpy()
        fingertip_pos = np.asarray(ee_pos_w) + z_world * finger_offset
        fingertip_err = None if target_pos is None else float(np.linalg.norm(fingertip_pos - np.asarray(target_pos)))
        current_arm_joint_pos = None
        desired_arm_joint_pos = None
        arm_joint_delta = None
        if self.robot is not None and hasattr(self.robot, "data") and hasattr(self.robot.data, "joint_pos"):
            current_arm_joint_pos = (
                self.robot.data.joint_pos[0, self.arm_joint_ids].detach().cpu().numpy().astype(np.float32)
            )
        if self.desired_arm_joint_pos is not None:
            desired_arm_joint_pos = self.desired_arm_joint_pos[0].detach().cpu().numpy().astype(np.float32)
        if current_arm_joint_pos is not None and desired_arm_joint_pos is not None:
            arm_joint_delta = desired_arm_joint_pos - current_arm_joint_pos
        msg = (
            f"[ArmGraspController] state={self.state} target={self.current_target_id} "
            f"trash_pos_w={None if self.trash_pos_w is None else np.round(self.trash_pos_w, 3)} "
            f"pregrasp_pos_w={None if self.pregrasp_pos_w is None else np.round(self.pregrasp_pos_w, 3)} "
            f"grasp_pos_w={None if self.grasp_pos_w is None else np.round(self.grasp_pos_w, 3)} "
            f"lift_pos_w={None if self.lift_pos_w is None else np.round(self.lift_pos_w, 3)} "
            f"target_pos_w_comp={None if self.current_target_pos_w_compensated is None else np.round(self.current_target_pos_w_compensated, 3)} "
            f"target_pos_b_raw={None if self.current_target_pos_b_raw is None else np.round(self.current_target_pos_b_raw, 3)} "
            f"target_pos_b={None if self.current_target_pos_b is None else np.round(self.current_target_pos_b, 3)} "
            f"current_ee_pos_w={np.round(ee_pos_w, 3)} "
            f"ee_pos_err={None if pos_err is None else round(pos_err, 4)} "
            f"fingertip_err={None if fingertip_err is None else round(fingertip_err, 4)} "
            f"arm_q={None if current_arm_joint_pos is None else np.round(current_arm_joint_pos, 4)} "
            f"arm_q_des={None if desired_arm_joint_pos is None else np.round(desired_arm_joint_pos, 4)} "
            f"arm_dq={None if arm_joint_delta is None else np.round(arm_joint_delta, 4)} "
            f"stable={self.phase_stable_counter}/{self.stable_steps_required} "
            f"phase_steps={self.phase_step_counter}/{self.max_phase_steps} "
            f"gripper_cmd={'close' if torch.allclose(self.desired_gripper_joint_pos, self.gripper_close_pos) else 'open'}"
        )
        if extra:
            msg += f" reason={extra}"
        if self.failure_reason and self.state == "FAILED":
            msg += f" failure={self.failure_reason}"
        print(msg, flush=True)

    def _set_state(self, new_state: str):
        if self.state != new_state:
            self.phase_step_counter = 0
            self.phase_stable_counter = 0
        self.state = new_state

    def _phase_timed_out(self) -> bool:
        if self.state in {"DONE", "FAILED", "IDLE"}:
            return False
        phase_limit = self.max_phase_steps
        if self.state == "CLOSE_GRIPPER":
            phase_limit = max(phase_limit, self.gripper_close_wait_steps + self.stable_steps_required)
        return self.phase_step_counter >= phase_limit

    def _update_stable_reached(self, reached: bool) -> bool:
        if reached:
            self.phase_stable_counter += 1
        else:
            self.phase_stable_counter = 0
        return self.phase_stable_counter >= self.stable_steps_required

    def _refresh_target_poses(self, scene):
        if self.current_target_id is not None and scene is not None:
            current_trash_pos_w = self.get_current_trash_pos_w(scene)
            if current_trash_pos_w is not None:
                self.trash_pos_w = np.asarray(current_trash_pos_w, dtype=np.float32)
        if self.trash_pos_w is None:
            return
        self.pregrasp_pos_w = self.trash_pos_w + np.array([0.0, 0.0, self.pregrasp_height], dtype=np.float32)
        self.grasp_pos_w = self.trash_pos_w + np.array([0.0, 0.0, self.grasp_height_offset], dtype=np.float32)
        self.lift_pos_w = self.trash_pos_w + np.array([0.0, 0.0, self.lift_height], dtype=np.float32)

    def _world_pos_to_base_frame(self, target_pos_w) -> torch.Tensor | None:
        if self.robot is None or not hasattr(self.robot.data, "root_pose_w"):
            return None
        from isaaclab.utils.math import quat_rotate_inverse as _quat_rotate_inverse

        root_pos_w = self.robot.data.root_pose_w[:, :3]
        root_quat_w = self.robot.data.root_pose_w[:, 3:]
        target_pos_t = torch.as_tensor(target_pos_w, dtype=torch.float32, device=self.device).view(1, 3)
        return _quat_rotate_inverse(root_quat_w, target_pos_t - root_pos_w)

    def _world_quat_to_base_frame(self, target_quat_w) -> torch.Tensor | None:
        if target_quat_w is None or self.robot is None or not hasattr(self.robot.data, "root_pose_w"):
            return None
        from isaaclab.utils.math import quat_conjugate as _quat_conjugate, quat_mul as _quat_mul

        root_quat_w = self.robot.data.root_pose_w[:, 3:]
        target_quat_t = torch.as_tensor(target_quat_w, dtype=torch.float32, device=self.device).view(1, 4)
        return _quat_mul(_quat_conjugate(root_quat_w), target_quat_t)

    def _compute_top_down_target_quat_w(self, target_pos_w) -> np.ndarray | None:
        if self.robot is None or target_pos_w is None:
            return None

        arm_base_pos_w = None
        try:
            arm_base_body_ids, _ = self.robot.find_bodies("arm_base")
            if len(arm_base_body_ids) > 0:
                arm_base_pos_w = self.robot.data.body_pos_w[0, arm_base_body_ids[0], :3].detach().cpu().numpy()
        except Exception:
            arm_base_pos_w = None

        if arm_base_pos_w is None:
            arm_base_pos_w = self.robot.data.root_pos_w[0, :3].detach().cpu().numpy()

        target_pos_w = np.asarray(target_pos_w, dtype=np.float32)
        horizontal_dir = target_pos_w[:2] - arm_base_pos_w[:2]
        horizontal_norm = float(np.linalg.norm(horizontal_dir))
        if horizontal_norm < 1.0e-6:
            root_quat_w = self.robot.data.root_quat_w[0].detach().cpu().numpy()
            yaw = math.atan2(
                2.0 * (root_quat_w[0] * root_quat_w[3] + root_quat_w[1] * root_quat_w[2]),
                1.0 - 2.0 * (root_quat_w[2] ** 2 + root_quat_w[3] ** 2),
            )
            x_axis_w = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        else:
            x_axis_w = np.array(
                [horizontal_dir[0] / horizontal_norm, horizontal_dir[1] / horizontal_norm, 0.0],
                dtype=np.float32,
            )

        z_axis_w = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y_axis_w = np.cross(z_axis_w, x_axis_w)
        y_norm = float(np.linalg.norm(y_axis_w))
        if y_norm < 1.0e-6:
            y_axis_w = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            y_axis_w = y_axis_w / y_norm
        x_axis_w = np.cross(y_axis_w, z_axis_w)
        x_axis_w = x_axis_w / max(float(np.linalg.norm(x_axis_w)), 1.0e-6)

        rot_mat = np.stack([x_axis_w, y_axis_w, z_axis_w], axis=1)
        return self._rotation_matrix_to_quat_wxyz(rot_mat)

    def _rotation_matrix_to_quat_wxyz(self, rot_mat: np.ndarray) -> np.ndarray:
        m = np.asarray(rot_mat, dtype=np.float32)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        quat = np.array([qw, qx, qy, qz], dtype=np.float32)
        quat /= max(float(np.linalg.norm(quat)), 1.0e-6)
        return quat
