"""
TaskD Solution - 推物块到坑里的解决方案
专门为 ATEC-TaskD-B2Piper 设计
参考 solution_gt.py 的策略模型调用方式
"""

import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from atec_rl_lab.assets.robots.b2 import UNITREE_B2_PIPER_CFG
except ImportError:
    sys.path.insert(0, os.path.join(REPO_ROOT, "source", "atec_rl_lab"))
    from atec_rl_lab.assets.robots.b2 import UNITREE_B2_PIPER_CFG

B2_PIPER_LEG_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
B2_PIPER_ARM_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.arm_joint_names)
B2_PIPER_TOTAL_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.joint_names)


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
    def __init__(self, env=None):
        self.env = env
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.camera_follow_enabled = True
        
        # 任务状态
        self._task_state = "STANDING"  # STANDING, APPROACH_BOX, PUSH_TO_PIT, DONE
        self._step_count = 0
        
        # 站稳计数器
        self._standing_steps = 0
        self._standing_duration = 100  # 站稳需要的步数（约2秒）
        
        # 日志
        self._log_file_path = self._init_logging()
        
        # 加载预训练模型
        self._load_actor_model()
        
        # 控制参数
        self.max_vx = 0.6
        self.max_vy = 0.3
        self.max_yaw_rate = 0.5
        self.kp_pos = 1.0
        self.kp_yaw = 2.0
        self.dt = 0.02
        
        # 目标位置（坑的位置）
        self.pit_center_x = 6.0  # terrain size[0] / 2
        self.pit_center_y = 4.0  # terrain size[1] / 2
        
        # 物块目标位置（推到坑附近）
        self.target_box_x = self.pit_center_x
        self.target_box_y = self.pit_center_y - 0.5
        
        # 机器人与物块的相对位置（从后方推）
        self.push_distance = 0.5
        self.push_offset_y = 0.0
        
        # 状态标志
        self._box_pushed = False
        self._arrived_at_push_position = False
        self._push_started = False
        self._push_steps = 0
        self._max_push_steps = 300

    def _init_logging(self) -> str:
        import time
        log_dir = os.path.join(REPO_ROOT, "logs", "solution_taskd")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = os.path.join(log_dir, f"solution_taskd_{timestamp}.log")
        print(f"[TaskD] Log file: {log_file_path}", flush=True)
        return log_file_path

    def _log(self, message: str) -> None:
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        with open(self._log_file_path, "a") as f:
            f.write(log_line)

    def _load_actor_model(self):
        """加载预训练的腿部控制模型（参考 solution_gt.py）"""
        self.checkpoint_path = os.path.join(
            REPO_ROOT,
            "logs",
            "rsl_rl",
            "unitree_b2_piper_flat",
            "2026-06-02_14-40-32",
            "model_4999.pt",
        )
        
        if not os.path.exists(self.checkpoint_path):
            print(f"[TaskD] Warning: Checkpoint not found: {self.checkpoint_path}")
            self.actor = None
            return
        
        try:
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
            
            # 腿部动作缩放（参考 solution_gt.py）
            self.leg_action_scale = torch.tensor(
                [0.25 if "hip_joint" in name else 0.5 for name in self.leg_joint_names],
                device=self.device,
                dtype=torch.float32,
            ).view(1, -1)
            self.leg_action_scale_inv = torch.reciprocal(self.leg_action_scale)
            
            # 机械臂默认位置（参考 solution_gt.py，arm_joint2 设为 3.14 让手臂抬起）
            self.b2_piper_arm_defaults = {
                "arm_joint1": 0.0,
                "arm_joint2": 3.14,  # 抬起手臂，避免碰撞
                "arm_joint3": -1.20,
                "arm_joint4": 0.0,
                "arm_joint5": -0.8,
                "arm_joint6": 0.0,
                "arm_joint7": 0.0,
                "arm_joint8": 0.0,
            }
            arm_default_pos = [self.b2_piper_arm_defaults.get(name, 0.0) for name in self.arm_joint_names]
            self.arm_default_action = torch.tensor(
                arm_default_pos,
                device=self.device,
                dtype=torch.float32,
            ).view(1, -1)
            
            print(f"[TaskD] Actor model loaded successfully: input={actor_input_dim}, output={actor_output_dim}")
            
        except Exception as e:
            print(f"[TaskD] Failed to load actor model: {e}")
            self.actor = None

    def reset(self, **kwargs) -> None:
        self._task_state = "STANDING"
        self._step_count = 0
        self._standing_steps = 0
        self._box_pushed = False
        self._arrived_at_push_position = False
        self._push_started = False
        self._push_steps = 0
        print("[TaskD] Reset task state to STANDING", flush=True)

    def get_action_spec(self) -> dict | None:
        return None

    def _get_scene(self):
        if self.env is None:
            return None
        env_unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(env_unwrapped, "scene"):
            return env_unwrapped.scene
        if hasattr(env_unwrapped, "_env") and hasattr(env_unwrapped._env, "scene"):
            return env_unwrapped._env.scene
        return None

    def _get_robot(self):
        scene = self._get_scene()
        if scene is None:
            return None
        try:
            robot = scene["robot"]
        except Exception:
            return None
        if isinstance(robot, (list, tuple)):
            robot = robot[0] if robot else None
        return robot

    def _get_box_position(self):
        """获取物块的真实位置"""
        scene = self._get_scene()
        if scene is None:
            return None
        try:
            box = scene["box"]
            if isinstance(box, (list, tuple)):
                box = box[0]
            if hasattr(box, "data") and hasattr(box.data, "root_pos_w"):
                pos = box.data.root_pos_w.cpu().numpy()[0]
                return np.array(pos, dtype=np.float32)
        except Exception as e:
            print(f"[TaskD] Failed to get box position: {e}", flush=True)
        return None

    def _get_robot_pose(self):
        """获取机器人的真实位姿"""
        robot = self._get_robot()
        if robot is None:
            return None, None
        try:
            pos = robot.data.root_pos_w.cpu().numpy()[0]
            pos = np.array(pos, dtype=np.float32)
            quat = robot.data.root_quat_w.cpu().numpy()[0]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = float(math.atan2(siny_cosp, cosy_cosp))
            return pos, yaw
        except Exception as e:
            print(f"[TaskD] Failed to get robot pose: {e}", flush=True)
            return None, None

    def _compute_base_command(self, robot_pos, robot_yaw, box_pos):
        """计算底盘速度命令"""
        if box_pos is None or robot_pos is None:
            return np.zeros(3, dtype=np.float32)
        
        if self._task_state == "STANDING":
            # 站稳阶段：发送零速度命令让机器人保持站立
            self._standing_steps += 1
            print(f"[TaskD] STANDING - step {self._standing_steps}/{self._standing_duration}", flush=True)
            
            if self._standing_steps >= self._standing_duration:
                self._task_state = "APPROACH_BOX"
                print("[TaskD] State changed to APPROACH_BOX", flush=True)
            
            return np.zeros(3, dtype=np.float32)
        
        elif self._task_state == "APPROACH_BOX":
            # 计算推物位置（物块后方）
            push_position_x = box_pos[0] - self.push_distance
            push_position_y = box_pos[1] + self.push_offset_y
            
            # 计算到推物位置的误差
            dx = push_position_x - robot_pos[0]
            dy = push_position_y - robot_pos[1]
            pos_error = np.sqrt(dx**2 + dy**2)
            
            # 计算目标朝向（朝向物块）
            target_yaw = math.atan2(box_pos[1] - robot_pos[1], box_pos[0] - robot_pos[0])
            yaw_error = self._wrap_angle(target_yaw - robot_yaw)
            
            print(f"[TaskD] APPROACH_BOX - pos_error={pos_error:.2f}, yaw_error={yaw_error:.2f}", flush=True)
            
            # 先转向，再前进
            if abs(yaw_error) > 0.3:
                vx = 0.0
                vy = 0.0
                yaw_rate = self.kp_yaw * yaw_error
                yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
            else:
                # 计算速度
                vx_w = self.kp_pos * dx
                vy_w = self.kp_pos * dy
                
                # 转换到机器人坐标系
                cos_yaw = math.cos(robot_yaw)
                sin_yaw = math.sin(robot_yaw)
                vx = cos_yaw * vx_w + sin_yaw * vy_w
                vy = -sin_yaw * vx_w + cos_yaw * vy_w
                
                # 限制速度
                vx = max(-self.max_vx, min(self.max_vx, vx))
                vy = max(-self.max_vy, min(self.max_vy, vy))
                
                # 朝向微调
                yaw_rate = self.kp_yaw * yaw_error * 0.3
                yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
            
            # 检查是否到达推物位置
            if pos_error < 0.3 and abs(yaw_error) < 0.2:
                self._arrived_at_push_position = True
                self._task_state = "PUSH_TO_PIT"
                print("[TaskD] State changed to PUSH_TO_PIT", flush=True)
            
            return np.array([vx, vy, yaw_rate], dtype=np.float32)
        
        elif self._task_state == "PUSH_TO_PIT":
            self._push_started = True
            self._push_steps += 1
            
            # 持续向前推
            vx = self.max_vx * 0.8
            vy = 0.0
            yaw_rate = 0.0
            
            # 检查物块是否已经接近坑
            if box_pos[0] > self.pit_center_x - 1.0:
                self._box_pushed = True
                self._task_state = "DONE"
                print("[TaskD] State changed to DONE - Box pushed to pit!", flush=True)
            
            # 最大推步数保护
            if self._push_steps >= self._max_push_steps:
                self._task_state = "DONE"
                print("[TaskD] State changed to DONE - Max push steps reached", flush=True)
            
            print(f"[TaskD] PUSH_TO_PIT - step={self._push_steps}, box_x={box_pos[0]:.2f}", flush=True)
            
            return np.array([vx, vy, yaw_rate], dtype=np.float32)
        
        else:  # DONE
            return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _extract_policy_obs(self, obs: dict, base_cmd: np.ndarray, obs_dim: int | None = None) -> torch.Tensor:
        """提取策略观测（参考 solution_gt.py）"""
        if not isinstance(obs, dict) or "proprio" not in obs:
            return None
        
        proprio = torch.as_tensor(obs["proprio"], device=self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        
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
        
        if obs_dim is None:
            obs_dim = int(getattr(self.actor.actor[0], "in_features", 45)) if self.actor is not None else 45
        
        components = [
            base_ang_vel * 0.25,
            projected_gravity,
        ]
        if obs_dim >= 45:
            velocity_commands = torch.as_tensor(base_cmd, device=self.device, dtype=proprio.dtype).view(1, 3)
            if proprio.shape[0] > 1:
                velocity_commands = velocity_commands.repeat(proprio.shape[0], 1)
            components.append(velocity_commands)
        components.extend([
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_leg_train,
        ])
        
        policy_obs = torch.cat(components, dim=-1)
        return torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor) -> torch.Tensor:
        """将策略动作映射到环境动作（参考 solution_gt.py）"""
        if self.actor is None:
            return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)
        
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        
        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, self.total_action_dim), device=self.device, dtype=torch.float32)
        
        # 腿部动作
        action_env[:, :self.leg_action_dim] = action_train * self.leg_action_scale
        
        # 机械臂动作（保持默认位置）
        action_env[:, self.leg_action_dim:] = self.arm_default_action
        
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _generate_action_tensor(self, obs, base_cmd) -> torch.Tensor:
        """生成环境动作张量"""
        if self.actor is None:
            return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)
        
        try:
            policy_obs = self._extract_policy_obs(obs, base_cmd)
            if policy_obs is None:
                return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)
            
            with torch.inference_mode():
                action_train = self.actor(policy_obs)
            
            return self._map_policy_action_to_env_action(action_train)
        except Exception as e:
            print(f"[TaskD] Actor inference error: {e}", flush=True)
            return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)

    def predicts(self, obs, total_reward=0.0):
        """主预测函数"""
        self._step_count += 1
        
        # 获取机器人和物块位置
        robot_pos, robot_yaw = self._get_robot_pose()
        box_pos = self._get_box_position()
        
        # 计算底盘命令
        base_cmd = self._compute_base_command(robot_pos, robot_yaw, box_pos)
        
        # 使用策略网络生成动作
        action_env = self._generate_action_tensor(obs, base_cmd)
        action_np = action_env.detach().cpu().numpy()[0]
        
        # 检查是否完成
        giveup = self._task_state == "DONE"
        
        return {"action": action_np, "giveup": giveup}