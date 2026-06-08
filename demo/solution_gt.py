# /home/ril/myq/ATEC2026_Simulation_Challenge_RIL/demo/solution_gt.py
"""
TaskB GT Navigation Solution - 基于 Ground Truth 的导航方案
直接从仿真环境读取物体位置，绕过视觉识别

环境变量控制:
    ATEC_TASKB_USE_GT_NAV=1    - 启用 GT 导航模式
    ATEC_TASKB_GT_STOP_DIST=0.5 - 停止距离（米）
    ATEC_TASKB_NAV_MODE=order  - 导航模式: "nearest"(找最近目标)、"order"(按编号顺序 object1-18) 或 "keyboard"(键盘手动控制)
"""

import atexit
import math
import os
import select
import sys
import termios
import time
import tty
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

# 导航模式选择开关
# 可选值: "nearest" - 找最近的目标; "order" - 按编号顺序 object1-18
#        "keyboard" - Isaac/Omniverse 键盘手动控制（终端输入兜底）
NAV_MODE = os.getenv("ATEC_TASKB_NAV_MODE", "nearest").lower()
assert NAV_MODE in ["nearest", "order", "keyboard"], (
    f"Invalid NAV_MODE: {NAV_MODE}. Must be 'nearest', 'order' or 'keyboard'"
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KeyboardController:
    """键盘控制器：优先使用 Isaac/Omniverse 事件，失败时回退到终端 stdin。"""

    def __init__(self):
        self.forward_speed = float(os.getenv("ATEC_TASKB_KEYBOARD_LINEAR_SPEED", "1.0"))
        self.side_speed = float(os.getenv("ATEC_TASKB_KEYBOARD_SIDE_SPEED", "0.4"))
        self.turn_speed = float(os.getenv("ATEC_TASKB_KEYBOARD_TURN_SPEED", "0.6"))
        self.command_hold_time = float(os.getenv("ATEC_TASKB_KEYBOARD_HOLD_TIME", "0.25"))
        self.pressed_keys: set[str] = set()
        self.quit_requested = False
        self.backend = None

        # Isaac/Omniverse keyboard backend state.
        self._carb = None
        self._input_interface = None
        self._keyboard = None
        self._keyboard_subscription = None
        self._isaac_warned = False
        self._isaac_help_printed = False

        # Terminal fallback backend state.
        self._terminal_enabled = False
        self._old_termios = None
        self._terminal_warned = False
        self._terminal_help_printed = False
        self._printed_help = False
        self._restore_registered = False
        self._last_terminal_key_time = 0.0

    def setup(self):
        """初始化可用键盘后端。"""
        if self.backend is not None:
            return

        if self._setup_isaac_keyboard():
            self.backend = "isaac"
            return

        if self._setup_terminal_keyboard():
            self.backend = "terminal"
            return

        self.backend = "disabled"

    def _setup_isaac_keyboard(self) -> bool:
        """订阅 Isaac Sim / Omniverse Kit app window 键盘事件。"""
        try:
            import carb
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("omni.appwindow.get_default_app_window() returned None")

            keyboard = app_window.get_keyboard()
            if keyboard is None:
                raise RuntimeError("default app window has no keyboard")

            input_interface = carb.input.acquire_input_interface()
            subscription = input_interface.subscribe_to_keyboard_events(
                keyboard,
                self._on_isaac_keyboard_event,
            )
            if subscription is None:
                raise RuntimeError("subscribe_to_keyboard_events() returned None")

            self._carb = carb
            self._input_interface = input_interface
            self._keyboard = keyboard
            self._keyboard_subscription = subscription

            if not self._restore_registered:
                atexit.register(self.restore)
                self._restore_registered = True

            if not self._isaac_help_printed:
                print(
                    "[KEYBOARD] Isaac/Omniverse keyboard control enabled: "
                    "W/Up=forward, S/Down=backward, Z=strafe left, X=strafe right, "
                    "A/Left=turn left, D/Right=turn right, Q/Esc=quit.",
                    flush=True,
                )
                self._isaac_help_printed = True
            return True
        except Exception as e:
            if not self._isaac_warned:
                print(
                    f"[KEYBOARD] Isaac/Omniverse keyboard API unavailable ({e}); "
                    "falling back to terminal stdin. Terminal mode needs the terminal window focused.",
                    flush=True,
                )
                self._isaac_warned = True
            return False

    def _setup_terminal_keyboard(self) -> bool:
        """进入终端 cbreak 模式；仅作为 Isaac 键盘事件不可用时的兜底。"""
        if not sys.stdin.isatty():
            if not self._terminal_warned:
                print(
                    "[KEYBOARD] stdin is not a TTY; keyboard control is disabled. "
                    "Run from an interactive terminal or use Isaac/Omniverse keyboard events.",
                    flush=True,
                )
                self._terminal_warned = True
            return False

        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._terminal_enabled = True

            if not self._restore_registered:
                atexit.register(self.restore)
                self._restore_registered = True

            if not self._terminal_help_printed:
                print(
                    "[KEYBOARD] Terminal keyboard fallback enabled: "
                    "W/Up=forward, S/Down=backward, Z=strafe left, X=strafe right, "
                    "A/Left=turn left, D/Right=turn right, Q/Esc=quit. Terminal window must have focus.",
                    flush=True,
                )
                self._terminal_help_printed = True
            return True
        except Exception as e:
            if not self._terminal_warned:
                print(f"[KEYBOARD] Failed to initialize terminal keyboard input: {e}", flush=True)
                self._terminal_warned = True
            return False

    def restore(self):
        """释放键盘订阅并恢复终端输入状态。"""
        self._keyboard_subscription = None
        self._input_interface = None
        self._keyboard = None
        self._carb = None
        self.pressed_keys.clear()

        if self._old_termios is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
            finally:
                self._old_termios = None
                self._terminal_enabled = False

    def _on_isaac_keyboard_event(self, event) -> bool:
        """Isaac/Omniverse 键盘事件回调，维护按下/释放状态。"""
        key = self._normalize_isaac_key(getattr(event, "input", None))
        if key is None:
            return True

        event_type = getattr(event, "type", None)
        if self._is_isaac_event_type(event_type, "KEY_PRESS", "KEY_REPEAT"):
            if key in {"q", "esc"}:
                self.quit_requested = True
            else:
                self.pressed_keys.add(key)
        elif self._is_isaac_event_type(event_type, "KEY_RELEASE"):
            self.pressed_keys.discard(key)

        return True

    def _is_isaac_event_type(self, event_type, *names: str) -> bool:
        """兼容不同 Isaac/Kit 版本的 keyboard event type 表示。"""
        if self._carb is not None:
            keyboard_event_type = getattr(self._carb.input, "KeyboardEventType", None)
            for name in names:
                expected = getattr(keyboard_event_type, name, None)
                if expected is None:
                    continue
                if event_type == expected:
                    return True
                try:
                    if int(event_type) == int(expected):
                        return True
                except Exception:
                    pass

        return self._enum_name(event_type) in set(names)

    def _enum_name(self, value) -> str:
        """兼容 carb enum / int / str 的事件名读取。"""
        name = getattr(value, "name", None)
        if name:
            return str(name).upper()
        return str(value).split(".")[-1].upper()

    def _normalize_isaac_key(self, key_input) -> str | None:
        """将 Isaac keyboard input 规范化为 w/s/a/d/q/esc/up/down/left/right。"""
        if key_input is None:
            return None

        candidates = []
        name = getattr(key_input, "name", None)
        if name:
            candidates.append(str(name))
        candidates.append(str(key_input))

        for candidate in candidates:
            token = candidate.split(".")[-1].split(":")[-1].strip().lower()
            mapping = {
                "w": "w",
                "s": "s",
                "a": "a",
                "d": "d",
                "z": "z",
                "x": "x",
                "q": "q",
                "escape": "esc",
                "esc": "esc",
                "up": "up",
                "down": "down",
                "left": "left",
                "right": "right",
                "arrow_up": "up",
                "arrow_down": "down",
                "arrow_left": "left",
                "arrow_right": "right",
            }
            if token in mapping:
                return mapping[token]
        return None

    def _read_terminal_key(self) -> str | None:
        """非阻塞读取一个普通按键或方向键。"""
        if not self._terminal_enabled:
            return None

        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return None

            ch = sys.stdin.read(1)
            if ch != "\x1b":
                return ch

            readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not readable:
                return "esc"

            ch2 = sys.stdin.read(1)
            if ch2 != "[":
                return "esc"

            readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not readable:
                return "esc"

            ch3 = sys.stdin.read(1)
            arrow_keys = {"A": "up", "B": "down", "C": "right", "D": "left"}
            return arrow_keys.get(ch3, "esc")
        except Exception as e:
            print(f"[KEYBOARD] Error reading keyboard input: {e}", flush=True)
            self.restore()
            return None

    def _poll_terminal_keyboard(self):
        """终端兜底模式无法获得释放事件，使用短时保持模拟按住。"""
        key = self._read_terminal_key()
        now = time.time()

        if key is not None:
            key_lower = key.lower()
            if key_lower in {"w", "s", "a", "d", "z", "x", "up", "down", "left", "right"}:
                self.pressed_keys = {key_lower}
                self._last_terminal_key_time = now
            elif key_lower in {"q", "esc"}:
                self.quit_requested = True

        if now - self._last_terminal_key_time > self.command_hold_time:
            self.pressed_keys.clear()

        return key

    def get_command(self) -> tuple[np.ndarray, bool]:
        """返回 (base_cmd, should_quit)。base_cmd 格式为 [vx, vy, yaw_rate]。"""
        self.setup()

        if self.backend == "terminal":
            self._poll_terminal_keyboard()

        cmd = np.zeros(3, dtype=np.float32)
        if "w" in self.pressed_keys or "up" in self.pressed_keys:
            cmd[0] += self.forward_speed
        if "s" in self.pressed_keys or "down" in self.pressed_keys:
            cmd[0] -= self.forward_speed
        if "z" in self.pressed_keys:
            cmd[1] += self.side_speed
        if "x" in self.pressed_keys:
            cmd[1] -= self.side_speed
        if "a" in self.pressed_keys or "left" in self.pressed_keys:
            cmd[2] += self.turn_speed
        if "d" in self.pressed_keys or "right" in self.pressed_keys:
            cmd[2] -= self.turn_speed

        return cmd, self.quit_requested


def run_keyboard_control_mode(solution: "AlgSolution", obs) -> dict:
    """
    键盘控制模式入口。

    该脚本的仿真 step 由 scripts/play_atec_task.py 统一执行，因此这里复用
    predicts() 的返回格式，把键盘按键转换为底盘速度命令，再由现有 actor
    生成环境需要的关节动作。
    """
    if getattr(solution, "_keyboard_controller", None) is None:
        solution._keyboard_controller = KeyboardController()

    base_cmd, should_quit = solution._keyboard_controller.get_command()

    if should_quit:
        solution._keyboard_controller.restore()
        print("[KEYBOARD] Quit requested, leaving keyboard control mode.", flush=True)
        return {"action": solution._get_stand_still_action(obs), "giveup": True}

    return {"action": solution._generate_action(obs, base_cmd), "giveup": False}

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


class PregraspNavigator:
    """Pregrasp导航控制器 - 负责计算pregrasp站位点和速度命令"""
    
    def __init__(self, nav_mode: str = "nearest"):
        # 导航参数
        self.stand_off = 0.6        # 机器人底盘中心距离垃圾的目标距离（米）
        self.kp_pos = 1.2           # 位置P控制增益
        self.kp_yaw = 2.0           # 朝向P控制增益
        self.max_vx = 0.8           # 最大x速度（机体系）
        self.max_vy = 0.25          # 最大y速度（机体系）
        self.max_yaw_rate = 0.8     # 最大角速度
        self.pos_tol = 0.32         # 位置到达阈值（米）
        self.yaw_tol = 0.087        # 朝向到达阈值（弧度，约5度）
        self.slow_radius = 0.5      # 减速半径（米）
        
        # 导航模式: "nearest" - 找最近目标; "order" - 按编号顺序 object1-18
        self.nav_mode = nav_mode
        
        # 导航状态
        self.nav_state = "SELECT_TARGET"  # SELECT_TARGET, COMPUTE_PREGRASP_POSE, NAVIGATE_TO_PREGRASP, ALIGN_TO_TRASH, READY_TO_GRASP, MARK_DONE, DONE
        
        # 当前目标
        self.current_target = None
        self.goal_xy = np.array([0.0, 0.0])
        self.goal_yaw = 0.0
        self.done_target_ids = set()
        
        # 垃圾目标列表（状态追踪）
        self.trash_targets = []
        
        # 按编号顺序模式下，记录当前应处理的编号
        self._current_order_idx = 1
    
    def _wrap_angle(self, angle: float) -> float:
        """角度归一化到 [-pi, pi]"""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
    
    def _world_to_body_velocity(self, vx_world: float, vy_world: float, robot_yaw: float) -> tuple:
        """将世界坐标系速度转换为机器人机体系速度"""
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        vx_body = cos_yaw * vx_world + sin_yaw * vy_world
        vy_body = -sin_yaw * vx_world + cos_yaw * vy_world
        return vx_body, vy_body
    
    def compute_pregrasp_pose(self, robot_pos_w: np.ndarray, trash_pos_w: np.ndarray) -> tuple:
        """
        计算pregrasp站位点
        - goal_xy: 机器人需要到达的xy位置
        - goal_yaw: 机器人到达后应面朝的方向（朝向垃圾）
        """
        robot_xy = robot_pos_w[:2]
        trash_xy = trash_pos_w[:2]
        
        # 计算从垃圾指向机器人的方向
        direction = robot_xy - trash_xy
        norm = np.linalg.norm(direction)
        
        # 保护：避免除零
        if norm < 0.01:
            # 如果机器人已经非常接近垃圾，使用默认方向（例如x轴正方向）
            direction = np.array([1.0, 0.0])
            norm = 1.0
        
        direction_normalized = direction / norm
        
        # 计算目标站位点：在垃圾前方 stand_off 距离处
        goal_xy = trash_xy + direction_normalized * self.stand_off
        
        # 计算目标朝向：面朝垃圾
        dx = trash_pos_w[0] - goal_xy[0]
        dy = trash_pos_w[1] - goal_xy[1]
        goal_yaw = math.atan2(dy, dx)
        
        return goal_xy, goal_yaw
    
    def select_nearest_target(self, trash_targets: list, robot_pos_w: np.ndarray) -> dict | None:
        """
        选择距离机器人最近的pending状态垃圾
        返回：垃圾目标字典 或 None（没有pending垃圾）
        """
        robot_xy = robot_pos_w[:2]
        candidates = []
        
        for trash in trash_targets:
            if trash.get("status", "pending") != "pending":
                continue
            if trash.get("id") in self.done_target_ids:
                continue
            trash_xy = trash["pos_w"][:2]
            dist = np.linalg.norm(trash_xy - robot_xy)
            if np.isfinite(dist):
                candidates.append((dist, trash))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def select_order_target(self, trash_targets: list) -> dict | None:
        """
        按编号顺序选择目标（object_1-18）
        返回：垃圾目标字典 或 None（没有未处理的垃圾）
        """
        # 从当前编号开始查找
        for idx in range(self._current_order_idx, 19):
            target_id = f"object_{idx}"
            for trash in trash_targets:
                if trash.get("id") == target_id:
                    # 检查是否已处理或已入桶
                    if trash.get("status", "pending") == "pending" and trash.get("id") not in self.done_target_ids:
                        return trash
            # 如果当前编号的目标不存在或已处理，继续下一个
            self._current_order_idx = idx + 1
        
        # 如果从当前编号开始没找到，从头开始检查是否有遗漏
        if self._current_order_idx > 1:
            for idx in range(1, self._current_order_idx):
                target_id = f"object_{idx}"
                for trash in trash_targets:
                    if trash.get("id") == target_id:
                        if trash.get("status", "pending") == "pending" and trash.get("id") not in self.done_target_ids:
                            return trash
        
        return None
    
    def compute_velocity_command(self, robot_pos_w: np.ndarray, robot_yaw: float) -> tuple:
        """
        计算速度命令 (vx, vy, yaw_rate)
        返回：(vx_body, vy_body, yaw_rate), nav_info
        """
        robot_xy = robot_pos_w[:2]
        
        # 计算位置误差（世界坐标系）
        error_xy_w = self.goal_xy - robot_xy
        pos_error_norm = np.linalg.norm(error_xy_w)
        
        # 计算朝向误差
        yaw_error = self._wrap_angle(self.goal_yaw - robot_yaw)
        
        if self.nav_state == "NAVIGATE_TO_PREGRASP":
            # 导航到pregrasp位置：先对准朝向，再移动
            # 如果朝向误差大于阈值，先原地转向
            if abs(yaw_error) > 0.3:  # 约17度
                # 先转向对准目标方向
                vx_body = 0.0
                vy_body = 0.0
                yaw_rate = self.kp_yaw * yaw_error
                yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
                arrived = False
            else:
                # 朝向已对准，开始移动
                # 计算世界坐标系速度
                vx_w = self.kp_pos * error_xy_w[0]
                vy_w = self.kp_pos * error_xy_w[1]
                
                # 减速处理：接近目标时降低速度
                if pos_error_norm < self.slow_radius:
                    decel_ratio = pos_error_norm / self.slow_radius
                    vx_w *= decel_ratio
                    vy_w *= decel_ratio
                
                # 转换到机体系
                vx_body, vy_body = self._world_to_body_velocity(vx_w, vy_w, robot_yaw)
                vx_body = max(-self.max_vx, min(self.max_vx, vx_body))
                vy_body = max(-self.max_vy, min(self.max_vy, vy_body))
                
                # 小角度微调朝向
                yaw_rate = self.kp_yaw * yaw_error * 0.3  # 降低转向增益
                yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
                
                # 检查是否到达位置
                arrived = pos_error_norm < self.pos_tol
            
        elif self.nav_state == "ALIGN_TO_TRASH":
            # 原地转向对准垃圾
            vx_body = 0.0
            vy_body = 0.0
            yaw_rate = self.kp_yaw * yaw_error
            yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
            
            # 检查是否转向完成
            arrived = abs(yaw_error) < self.yaw_tol
            
        else:
            # 停止状态
            vx_body = 0.0
            vy_body = 0.0
            yaw_rate = 0.0
            arrived = False
        
        nav_info = {
            "pos_error": float(pos_error_norm),
            "yaw_error": float(yaw_error),
            "arrived": arrived,
            "goal_xy": self.goal_xy.copy(),
            "goal_yaw": float(self.goal_yaw)
        }
        
        return (vx_body, vy_body, yaw_rate), nav_info
    
    def update(self, robot_pos_w: np.ndarray, robot_yaw: float, trash_targets: list) -> tuple:
        """
        更新导航状态机
        返回：(vx, vy, yaw_rate), nav_info
        """
        self.trash_targets = trash_targets
        
        if self.nav_state == "SELECT_TARGET":
            # 根据导航模式选择目标
            if self.nav_mode == "order":
                self.current_target = self.select_order_target(trash_targets)
            else:
                self.current_target = self.select_nearest_target(trash_targets, robot_pos_w)
            
            if self.current_target is None:
                # 没有pending垃圾，任务完成
                self.nav_state = "DONE"
                print("[PregraspNavigator] All trash processed, DONE", flush=True)
                return (0.0, 0.0, 0.0), {"state": "DONE", "arrived": True}
            
            print(f"[PregraspNavigator] Selected target: {self.current_target['id']} (mode: {self.nav_mode})", flush=True)
            self.nav_state = "COMPUTE_PREGRASP_POSE"
            return (0.0, 0.0, 0.0), {"state": "SELECT_TARGET", "arrived": False}
        
        if self.nav_state == "COMPUTE_PREGRASP_POSE":
            # 计算pregrasp站位点
            self.goal_xy, self.goal_yaw = self.compute_pregrasp_pose(
                robot_pos_w, 
                self.current_target["pos_w"]
            )
            print(f"[PregraspNavigator] Computed pregrasp pose: goal_xy={self.goal_xy}, goal_yaw={math.degrees(self.goal_yaw):.1f}°", flush=True)
            self.nav_state = "NAVIGATE_TO_PREGRASP"
            return (0.0, 0.0, 0.0), {"state": "COMPUTE_PREGRASP_POSE", "arrived": False}
        
        if self.nav_state == "NAVIGATE_TO_PREGRASP":
            # 导航到pregrasp位置
            cmd, nav_info = self.compute_velocity_command(robot_pos_w, robot_yaw)
            nav_info["state"] = "NAVIGATE_TO_PREGRASP"
            
            if nav_info["arrived"]:
                print(f"[PregraspNavigator] Arrived at pregrasp position, starting alignment", flush=True)
                self.nav_state = "ALIGN_TO_TRASH"
                return (0.0, 0.0, 0.0), {"state": "NAVIGATE_TO_PREGRASP", "arrived": True}
            
            return cmd, nav_info
        
        if self.nav_state == "ALIGN_TO_TRASH":
            # 原地转向对准垃圾
            cmd, nav_info = self.compute_velocity_command(robot_pos_w, robot_yaw)
            nav_info["state"] = "ALIGN_TO_TRASH"
            
            if nav_info["arrived"]:
                print(f"[PregraspNavigator] Aligned to trash, ready to grasp", flush=True)
                self.nav_state = "READY_TO_GRASP"
                return (0.0, 0.0, 0.0), {"state": "ALIGN_TO_TRASH", "arrived": True}
            
            return cmd, nav_info
        
        if self.nav_state == "READY_TO_GRASP":
            # 准备抓取状态：停止底盘
            # TODO: 调用机械臂抓取接口
            return (0.0, 0.0, 0.0), {"state": "READY_TO_GRASP", "arrived": True}
        
        if self.nav_state == "MARK_DONE":
            # 标记当前垃圾为已处理
            if self.current_target:
                self.current_target["status"] = "done"
                self.done_target_ids.add(self.current_target["id"])
                print(f"[PregraspNavigator] Marked {self.current_target['id']} as done", flush=True)
            self.nav_state = "SELECT_TARGET"
            return (0.0, 0.0, 0.0), {"state": "MARK_DONE", "arrived": True}
        
        if self.nav_state == "DONE":
            # 所有任务完成
            return (0.0, 0.0, 0.0), {"state": "DONE", "arrived": True}
        
        # 默认：停止
        return (0.0, 0.0, 0.0), {"state": self.nav_state, "arrived": False}


class AlgSolution:
    """基于 Ground Truth 的导航解决方案"""
    
    def __init__(self, env=None):
        """
        Args:
            env: 仿真环境实例，用于获取物体 Ground Truth 位置
        """
        self.env = env
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        
        # 导航参数（通过环境变量配置）
        self.gt_stop_dist = float(os.getenv("ATEC_TASKB_GT_STOP_DIST", "0.5"))
        
        # 速度范围：根据 velocity_env_cfg.py 配置
        self.lin_vel_range = (-1.0, 1.0)  # lin_vel_x, lin_vel_y
        self.ang_vel_range = (-1.0, 1.0)   # ang_vel_z
        self.max_lin_vel = 1.0
        self.max_ang_vel = 1.0
        self.heading_kp = float(os.getenv("ATEC_TASKB_HEADING_KP", "0.8"))
        
        # 机器人状态
        self.robot_pos = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
        self.robot_yaw = 0.0
        self._step_count = 0
        self.dt = 0.02

        # 在 __init__ 中添加标志位
        self._printed_objects_info = False
        
        # 垃圾桶位置（与环境配置一致）
        self.bin_center = np.array([-3.0, -10.0, 0.0], dtype=np.float32)
        self.bin_radius = 1.0
        
        # 初始化 Pregrasp 导航器（传递导航模式；keyboard 模式下不会进入自动导航状态机）
        self._pregrasp_navigator = PregraspNavigator(nav_mode=NAV_MODE if NAV_MODE != "keyboard" else "nearest")
        self._keyboard_controller = None
        
        # 加载预训练模型（用于腿部控制）
        self._load_actor_model()
        
        # End-effector 摄像头配置
        self._enable_ee_camera = os.getenv("ATEC_TASKB_ENABLE_EE_CAM", "1").lower() in {"1", "true", "yes", "on"}
        self._ee_cam_save_interval = int(os.getenv("ATEC_TASKB_EE_CAM_SAVE_INTERVAL", "50"))  # 每50帧保存一次
        self._ee_cam_display = os.getenv("ATEC_TASKB_EE_CAM_DISPLAY", "1").lower() in {"1", "true", "yes", "on"}
        
        # 创建图像保存目录
        self._ee_cam_save_dir = os.path.join(REPO_ROOT, "logs", "ee_camera")
        os.makedirs(self._ee_cam_save_dir, exist_ok=True)
        
        # 摄像头状态
        self._last_ee_cam_save_time = time.time()
        self._ee_cam_frame_count = 0
        
        print(f"[GT-NAV] GT Navigation enabled")
        print(f"[GT-NAV] NAV_MODE: {NAV_MODE}")
        print(f"[GT-NAV] Stop distance: {self.gt_stop_dist}m")
        print(f"[GT-NAV] Linear vel range: {self.lin_vel_range}")
        print(f"[GT-NAV] Angular vel range: {self.ang_vel_range}")
        print(f"[GT-NAV] Device: {self.device_str}")
        print(f"[GT-NAV] EE Camera: {'enabled' if self._enable_ee_camera else 'disabled'}")
        print(f"[GT-NAV] EE Camera Save Interval: {self._ee_cam_save_interval} frames")
        print(f"[GT-NAV] EE Camera Display: {'enabled' if self._ee_cam_display else 'disabled'}")
        print(f"[GT-NAV] EE Camera Save Dir: {self._ee_cam_save_dir}")

    def _load_actor_model(self):
        """加载预训练的腿部控制模型"""
        self.checkpoint_path = os.path.join(
            REPO_ROOT,
            "logs",
            "rsl_rl",
            "unitree_b2_piper_flat",
            "2026-06-02_14-40-32",
            "model_4999.pt",
        )
        
        if not os.path.exists(self.checkpoint_path):
            print(f"[GT-NAV] Warning: Checkpoint not found: {self.checkpoint_path}")
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
            
            # 腿部动作缩放
            self.leg_action_scale = torch.tensor(
                [0.25 if "hip_joint" in name else 0.5 for name in self.leg_joint_names],
                device=self.device,
                dtype=torch.float32,
            ).view(1, -1)
            self.leg_action_scale_inv = torch.reciprocal(self.leg_action_scale)
            
            # 机械臂默认位置
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
            arm_default_pos = [b2_piper_arm_defaults.get(name, 0.0) for name in self.arm_joint_names]
            self.arm_default_action = torch.tensor(
                arm_default_pos,
                device=self.device,
                dtype=torch.float32,
            ).view(1, -1)
            
            print(f"[GT-NAV] Actor model loaded successfully")
            
        except Exception as e:
            print(f"[GT-NAV] Failed to load actor model: {e}")
            self.actor = None

    def get_action_spec(self) -> dict | None:
        """返回动作规格（与 solution_rl.py 保持一致）"""
        return None

    def _process_ee_camera(self, obs):
        """
        处理 end-effector 摄像头图像：显示画面并定期保存
        """
        if not self._enable_ee_camera:
            return
        
        # 检查是否有 ee_rgb 图像数据
        if obs is None or "image" not in obs or "ee_rgb" not in obs["image"]:
            return
        
        try:
            # 获取 ee_rgb 图像数据
            ee_rgb = obs["image"]["ee_rgb"]
            
            # 处理 tensor 或 numpy 数组
            if isinstance(ee_rgb, torch.Tensor):
                img_np = ee_rgb.cpu().numpy()
            else:
                img_np = np.array(ee_rgb)
            
            # 移除 batch 维度（如果存在）
            if img_np.ndim == 4 and img_np.shape[0] == 1:
                img_np = img_np[0]
            
            # 确保是 (H, W, 3) 的 uint8 格式
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            
            # OpenCV 使用 BGR 格式，需要将 RGB 转换为 BGR
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            self._ee_cam_frame_count += 1
            
            # 显示摄像头画面
            if self._ee_cam_display:
                cv2.imshow("End-Effector Camera", img_bgr)
                cv2.waitKey(1)  # 1ms 等待，允许窗口更新
            
            # 定期保存图像
            if self._ee_cam_frame_count % self._ee_cam_save_interval == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                frame_num = self._ee_cam_frame_count
                save_path = os.path.join(self._ee_cam_save_dir, f"ee_cam_{timestamp}_{frame_num:06d}.png")
                
                # 使用已转换的 BGR 格式保存
                cv2.imwrite(save_path, img_bgr)
                
                if self._step_count % 20 == 0:
                    print(f"[GT-NAV] Saved EE camera image: {save_path}", flush=True)
        
        except Exception as e:
            if self._step_count % 100 == 0:
                print(f"[GT-NAV] Error processing EE camera: {e}", flush=True)

    def _wrap_angle(self, angle: float) -> float:
        """角度归一化到 [-pi, pi]"""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _get_object_index(self, obj_name: str) -> int | None:
        """从 object 名称中提取编号，仅接受 object1-object18。"""
        obj_name_lower = obj_name.lower()
        if "object" not in obj_name_lower:
            return None

        obj_idx_str = ''.join([c for c in obj_name if c.isdigit()])
        if not obj_idx_str:
            return None

        obj_idx = int(obj_idx_str)
        if 1 <= obj_idx <= 18:
            return obj_idx
        return None

    def _get_robot_pose(self, obs):
        """从观测中获取机器人位姿（仅使用 Ground Truth，失败则报错）"""
        # 直接从环境获取真实位置，不使用里程计回退
        if self.env is None:
            raise RuntimeError("[GT-NAV] env is None, cannot get Ground Truth robot pose")
        
        env_unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        scene = None
        
        if hasattr(env_unwrapped, 'scene'):
            scene = env_unwrapped.scene
        elif hasattr(env_unwrapped, '_env') and hasattr(env_unwrapped._env, 'scene'):
            scene = env_unwrapped._env.scene
        
        if scene is None:
            raise RuntimeError("[GT-NAV] Cannot access scene from env, cannot get Ground Truth robot pose")
        
        # 直接尝试获取 robot，不使用 'in' 操作符（scene.__contains__ 实现有问题）
        try:
            robot = scene['robot']
        except KeyError as e:
            raise RuntimeError(f"[GT-NAV] 'robot' not found in scene, cannot get Ground Truth robot pose. Error: {e}")
        # 处理 scene['robot'] 返回数组的情况
        if isinstance(robot, (list, tuple)):
            robot = robot[0]
        
        if not (hasattr(robot, 'data') and hasattr(robot.data, 'root_pos_w')):
            raise RuntimeError("[GT-NAV] Cannot access robot.data.root_pos_w, cannot get Ground Truth position")
        
        if not (hasattr(robot, 'data') and hasattr(robot.data, 'root_quat_w')):
            raise RuntimeError("[GT-NAV] Cannot access robot.data.root_quat_w, cannot get Ground Truth orientation")
        
        # 获取真实位置
        pos_world = robot.data.root_pos_w.cpu().numpy()[0]
        self.robot_pos = np.array(pos_world, dtype=np.float32)
        self.robot_pos[2] = 0.68  # 保持高度不变
        
        # 从四元数提取 yaw
        quat = robot.data.root_quat_w.cpu().numpy()[0]
        
        # Isaac Sim 的四元数格式通常是 [w, x, y, z] 或 [x, y, z, w]
        # 尝试两种格式，选择更合理的结果
        # 格式1: [w, x, y, z]
        w1, x1, y1, z1 = quat[0], quat[1], quat[2], quat[3]
        siny_cosp1 = 2 * (w1 * z1 + x1 * y1)
        cosy_cosp1 = 1 - 2 * (y1 * y1 + z1 * z1)
        yaw1 = math.atan2(siny_cosp1, cosy_cosp1)
        
        # 格式2: [x, y, z, w]
        w2, x2, y2, z2 = quat[3], quat[0], quat[1], quat[2]
        siny_cosp2 = 2 * (w2 * z2 + x2 * y2)
        cosy_cosp2 = 1 - 2 * (y2 * y2 + z2 * z2)
        yaw2 = math.atan2(siny_cosp2, cosy_cosp2)
        
        # 使用格式1 (w, x, y, z) - 从调试输出看这是正确的格式
        self.robot_yaw = yaw1
        
        # 调试：每100步打印一次位置和朝向信息
        if self._step_count % 100 == 0:
            print(f"[GT-NAV DEBUG] Robot pos: {self.robot_pos}, quat: {quat}, yaw1: {math.degrees(yaw1):.1f}°, yaw2: {math.degrees(yaw2):.1f}°", flush=True)
        
        return self.robot_pos, self.robot_yaw

    # 修改 _get_gt_objects 方法，只在第一次打印详细信息
    def _get_gt_objects(self):
        """从环境中获取所有物体的 Ground Truth 位置"""
        objects = []
        
        if self.env is None:
            return objects
        
        try:
            # 获取 scene 对象
            env_unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            scene = None
            
            # 尝试不同的方式获取 scene
            if hasattr(env_unwrapped, 'scene'):
                scene = env_unwrapped.scene
            elif hasattr(env_unwrapped, '_env') and hasattr(env_unwrapped._env, 'scene'):
                scene = env_unwrapped._env.scene
            
            if scene is None:
                print("[GT-NAV] Warning: Cannot access scene", flush=True)
                return objects
            
            # 遍历所有物体容器
            obj_containers = []
            if hasattr(scene, 'rigid_objects'):
                obj_containers.append(scene.rigid_objects)
            if hasattr(scene, 'articulations'):
                obj_containers.append(scene.articulations)
            
            robot_pos, _ = self._get_robot_pose(None)
            
            for container in obj_containers:
                for obj_name, obj_articulation in container.items():
                    # 只处理 object1-object18
                    obj_idx = self._get_object_index(obj_name)
                    if obj_idx is None:
                        continue
                    
                    # 获取物体世界坐标
                    try:
                        if hasattr(obj_articulation, 'data') and hasattr(obj_articulation.data, 'root_pos_w'):
                            pos_world = obj_articulation.data.root_pos_w.cpu().numpy()[0]
                        elif hasattr(obj_articulation, 'root_pos_w'):
                            pos_world = obj_articulation.root_pos_w.cpu().numpy()[0]
                        else:
                            continue
                    except Exception:
                        continue
                    
                    # 判断物体类别（根据名称索引）
                    obj_class = "sugar_box"
                    if obj_idx <= 6:
                        obj_class = "sugar_box"
                    elif obj_idx <= 12:
                        obj_class = "mustard_bottle"
                    else:
                        obj_class = "banana"
                    
                    # 判断是否已入桶
                    dist_to_bin = float(np.linalg.norm(pos_world[:2] - self.bin_center[:2]))
                    in_bin = dist_to_bin < self.bin_radius
                    
                    # 计算到机器人的距离
                    dist_to_robot = float(np.linalg.norm(pos_world[:2] - robot_pos[:2]))
                    
                    objects.append({
                        "id": obj_name,
                        "class": obj_class,
                        "pos_world": pos_world,
                        "in_bin": in_bin,
                        "dist_to_bin": dist_to_bin,
                        "dist_to_robot": dist_to_robot,
                    })
            
            # 只在第一次找到物体时打印详细信息
            if not self._printed_objects_info and len(objects) > 0:
                print(f"\n[GT-NAV] ========== Initial Object Scan ==========", flush=True)
                print(f"[GT-NAV] Found {len(objects)} objects", flush=True)
                
                # 按距离排序
                sorted_objects = sorted(objects, key=lambda x: x['dist_to_robot'])
                
                for i, obj in enumerate(sorted_objects):
                    pos_str = f"({obj['pos_world'][0]:.3f}, {obj['pos_world'][1]:.3f}, {obj['pos_world'][2]:.3f})"
                    status = "IN_BIN" if obj['in_bin'] else f"dist={obj['dist_to_robot']:.2f}m"
                    print(f"[GT-NAV] {i+1:2d}. {obj['id']:10s} {obj['class']:15s} pos={pos_str} {status}", flush=True)
                
                print(f"[GT-NAV] =========================================\n", flush=True)
                self._printed_objects_info = True
            
            # 移除每次都打印的语句
            
        except Exception as e:
            print(f"[GT-NAV] Error getting GT objects: {e}", flush=True)
        
        return objects
    def _select_nearest_target(self, objects):
        """选择最近的未入桶物体"""
        robot_pos, _ = self._get_robot_pose(None)
        
        candidates = []
        for obj in objects:
            if obj["in_bin"]:
                continue
            dist = np.linalg.norm(obj["pos_world"][:2] - robot_pos[:2])
            if np.isfinite(dist):
                candidates.append((dist, obj))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _compute_nav_cmd(self, target, phase):
        """计算导航指令（分离速度和转向控制）"""
        robot_pos, robot_yaw = self._get_robot_pose(None)
        
        # 计算相对位置
        dx = target["pos_world"][0] - robot_pos[0]
        dy = target["pos_world"][1] - robot_pos[1]
        dist = math.hypot(dx, dy)
        
        # 计算方向角
        desired_heading = math.atan2(dy, dx)
        heading_error = self._wrap_angle(desired_heading - robot_yaw)
        
        if phase == "approaching":
            # 阶段1：只控制速度，直行到目标0.5m处
            # 如果当前距离已经小于0.5m，需要后退到0.5m之外
            target_dist = self.gt_stop_dist
            dist_error = dist - target_dist
            
            if dist_error <= 0:
                # 已经在0.5m内，需要后退
                lin_x = -0.4  # 后退速度
                if dist < 0.2:
                    # 太近了，多退一些
                    lin_x = -0.5
            else:
                # 还没到目标距离，前进
                max_speed = 0.6
                slow_speed = 0.3
                decel_dist = 1.0
                
                if dist_error > decel_dist:
                    lin_x = max_speed
                else:
                    ratio = dist_error / decel_dist
                    lin_x = slow_speed + (max_speed - slow_speed) * ratio
            
            lin_x = max(-0.3, min(0.6, lin_x))
            
            # 只直行，不转向
            ang_z = 0.0
            
            # 检查是否到达目标距离
            if dist_error > 0 and dist_error < 0.05:
                current_phase = "stopped"
            elif dist_error <= 0 and abs(dist_error) < 0.05:
                current_phase = "stopped"
            else:
                current_phase = "approaching"
            
        elif phase == "turning":
            # 阶段2：只控制转向，面朝目标
            lin_x = 0.0  # 停止移动
            
            ang_kp = self.heading_kp * 0.8
            ang_z = np.clip(heading_error * ang_kp, self.ang_vel_range[0], self.ang_vel_range[1])
            
            # 检查朝向是否到位（误差小于5度）
            if abs(heading_error) < math.radians(5):
                current_phase = "completed"
            else:
                current_phase = "turning"
        else:
            lin_x = 0.0
            ang_z = 0.0
            current_phase = "stopped"
        
        return np.array([lin_x, 0.0, ang_z], dtype=np.float32), {
            "phase": current_phase,
            "dist": float(dist),
            "heading_error": float(heading_error),
            "target_class": target["class"]
        }

    def _extract_policy_obs(self, obs: dict[str, Any], base_cmd: np.ndarray) -> torch.Tensor:
        """提取策略观测（与 solution_rl.py 兼容）"""
        proprio = torch.as_tensor(obs["proprio"], device=self.device, dtype=torch.float32)
        
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
        """将策略动作映射到环境动作"""
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}")
        
        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, self.total_action_dim), device=self.device, dtype=torch.float32)
        action_env[:, :self.leg_action_dim] = action_train * self.leg_action_scale
        action_env[:, self.leg_action_dim:] = self.arm_default_action.repeat(num_envs, 1)
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_stand_still_action(self, obs):
        """获取站立不动的动作"""
        if self.actor is not None and obs is not None:
            zero_cmd = np.zeros(3, dtype=np.float32)
            return self._generate_action(obs, zero_cmd)
        else:
            return np.zeros(self.total_action_dim if hasattr(self, 'total_action_dim') else 20, dtype=np.float32).tolist()

    def _generate_action(self, obs, base_cmd):
        """生成机器人动作"""
        if obs is None or self.actor is None:
            return np.zeros(self.total_action_dim if hasattr(self, 'total_action_dim') else 20, dtype=np.float32).tolist()
        
        try:
            policy_obs = self._extract_policy_obs(obs, base_cmd)
            with torch.inference_mode():
                action_train = self.actor(policy_obs)
            
            if action_train.ndim == 1:
                action_train = action_train.unsqueeze(0)
            
            action_env = self._map_policy_action_to_env_action(action_train)
            return action_env.cpu().numpy().tolist()
        except Exception as e:
            print(f"[GT-NAV] Actor inference error: {e}", flush=True)
            return np.zeros(self.total_action_dim if hasattr(self, 'total_action_dim') else 20, dtype=np.float32).tolist()

    def predicts(self, obs, current_score):
        """
        决策入口 - 返回动作
        实现逻辑：使用 PregraspNavigator 导航到每个垃圾的 pregrasp 位置 → 对准朝向 → 准备抓取 → 标记完成 → 重复
        """
        self._step_count += 1
        
        # 处理 end-effector 摄像头图像
        self._process_ee_camera(obs)

        if NAV_MODE == "keyboard":
            return run_keyboard_control_mode(self, obs)
        
        # 更新机器人位姿
        self.robot_pos, self.robot_yaw = self._get_robot_pose(obs)
        
        # 获取 GT 物体位置并转换为 trash_targets 格式
        objects = self._get_gt_objects()
        trash_targets = []
        for obj in objects:
            # 检查是否已被处理（通过状态追踪）
            status = "pending"
            if obj.get("in_bin", False):
                status = "done"
            
            trash_targets.append({
                "id": obj["id"],
                "pos_w": obj["pos_world"],
                "status": status,
                "class": obj["class"]
            })
        
        # 使用 PregraspNavigator 计算速度命令
        (vx_body, vy_body, yaw_rate), nav_info = self._pregrasp_navigator.update(
            self.robot_pos, 
            self.robot_yaw, 
            trash_targets
        )
        
        # 构建基础命令 [vx, vy, yaw_rate]
        base_cmd = np.array([vx_body, vy_body, yaw_rate], dtype=np.float32)
        
        # 打印导航信息
        if self._step_count % 20 == 0:
            print(
                f"[GT-NAV] Step={self._step_count:4d} "
                f"State={nav_info['state']} "
                f"PosErr={nav_info.get('pos_error', 0):.2f}m "
                f"YawErr={math.degrees(nav_info.get('yaw_error', 0)):.1f}° "
                f"Cmd={base_cmd.round(3)}",
                flush=True
            )
        
        # 处理 READY_TO_GRASP 状态
        if nav_info["state"] == "READY_TO_GRASP":
            # 等待一段时间后标记为完成（模拟抓取过程）
            if not hasattr(self, '_grasp_wait_timer'):
                self._grasp_wait_timer = 0.0
            
            self._grasp_wait_timer += self.dt
            
            if self._grasp_wait_timer >= 3.0:  # 等待3秒模拟抓取
                self._grasp_wait_timer = 0.0
                self._pregrasp_navigator.nav_state = "MARK_DONE"
                print(f"[GT-NAV] Grasp completed, marking target as done", flush=True)
            
            return {"action": self._get_stand_still_action(obs), "giveup": False}
        
        # 处理 DONE 状态
        if nav_info["state"] == "DONE":
            return {"action": self._get_stand_still_action(obs), "giveup": False}
        
        # 使用 actor 网络生成动作
        return {"action": self._generate_action(obs, base_cmd), "giveup": False}


if __name__ == "__main__":
    """最小语法验证"""
    print("=" * 60)
    print("  AlgSolution GT Navigation - Syntax Test")
    print("=" * 60)
    
    # 测试 PregraspNavigator 初始化
    navigator = PregraspNavigator()
    print("✓ PregraspNavigator initialized")
    
    # 测试 PregraspNavigator._wrap_angle
    angle = navigator._wrap_angle(3.5 * math.pi)
    assert -math.pi <= angle <= math.pi, f"Angle wrap failed: {angle}"
    print("✓ PregraspNavigator._wrap_angle: OK")
    
    # 测试 PregraspNavigator.compute_pregrasp_pose
    robot_pos = np.array([1.0, 0.0, 0.68])
    trash_pos = np.array([0.0, 0.0, 0.1])
    goal_xy, goal_yaw = navigator.compute_pregrasp_pose(robot_pos, trash_pos)
    expected_dist = np.linalg.norm(goal_xy - trash_pos[:2])
    assert abs(expected_dist - navigator.stand_off) < 0.01, f"Stand-off distance incorrect: {expected_dist}"
    print(f"✓ PregraspNavigator.compute_pregrasp_pose: goal_xy={goal_xy}, goal_yaw={math.degrees(goal_yaw):.1f}°")
    
    # 测试 PregraspNavigator.select_nearest_target
    test_trash = [
        {"id": "trash1", "pos_w": np.array([1.0, 0.0, 0.1]), "status": "pending"},
        {"id": "trash2", "pos_w": np.array([2.0, 0.0, 0.1]), "status": "pending"},
        {"id": "trash3", "pos_w": np.array([0.5, 0.0, 0.1]), "status": "done"},
    ]
    nearest = navigator.select_nearest_target(test_trash, np.array([0.0, 0.0, 0.68]))
    assert nearest["id"] == "trash1", f"Nearest selection failed: {nearest}"
    print("✓ PregraspNavigator.select_nearest_target: OK")
    
    # 测试 PregraspNavigator.update 状态机
    test_trash_active = [
        {"id": "trash1", "pos_w": np.array([2.0, 0.0, 0.1]), "status": "pending"},
    ]
    # 第一次调用：SELECT_TARGET -> COMPUTE_PREGRASP_POSE
    cmd, info = navigator.update(np.array([0.0, 0.0, 0.68]), 0.0, test_trash_active)
    assert info["state"] == "SELECT_TARGET", f"Initial state error: {info['state']}"
    # 第二次调用：COMPUTE_PREGRASP_POSE -> NAVIGATE_TO_PREGRASP
    cmd, info = navigator.update(np.array([0.0, 0.0, 0.68]), 0.0, test_trash_active)
    assert info["state"] == "COMPUTE_PREGRASP_POSE", f"State error: {info['state']}"
    # 第三次调用：NAVIGATE_TO_PREGRASP
    cmd, info = navigator.update(np.array([0.0, 0.0, 0.68]), 0.0, test_trash_active)
    assert info["state"] == "NAVIGATE_TO_PREGRASP", f"State error: {info['state']}"
    print("✓ PregraspNavigator.update state machine: OK")
    
    # 测试 AlgSolution 初始化
    solution = AlgSolution()
    print("✓ AlgSolution initialized")
    
    # 测试 get_action_spec
    spec = solution.get_action_spec()
    print(f"✓ get_action_spec: {spec}")
    
    # 测试 predicts
    dummy_obs = {
        "proprio": np.zeros((1, 72), dtype=np.float32),
        "image": {
            "head_rgb": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "head_depth": np.ones((1, 480, 640, 1), dtype=np.float32),
        },
    }
    result = solution.predicts(dummy_obs, 0.0)
    assert "action" in result, "Missing 'action' in result"
    assert "giveup" in result, "Missing 'giveup' in result"
    # 获取实际的动作维度
    action_dim = len(result["action"]) if isinstance(result["action"], list) else result["action"].shape[-1]
    assert action_dim > 0, f"Action dimension must be > 0, got: {action_dim}"
    print(f"✓ predicts: OK (action_dim={action_dim})")
    
    print("\n" + "=" * 60)
    print("  ✅ All syntax tests passed!")
    print("=" * 60)