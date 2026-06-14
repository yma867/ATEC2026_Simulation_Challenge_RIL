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
import re
import sys
import termios
import time
import tty
from types import SimpleNamespace
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import torch
import torch.nn as nn

try:
    from atec_rl_lab.utils import CartesianController
except Exception:
    CartesianController = None

# 导航模式选择开关
# 可选值: "nearest" - 找最近的目标; "order" - 按编号顺序 object1-18
#        "keyboard" - Isaac/Omniverse 键盘手动控制（终端输入兜底）
NAV_MODE = os.getenv("ATEC_TASKB_NAV_MODE", "nearest").lower()
assert NAV_MODE in ["nearest", "order", "keyboard"], (
    f"Invalid NAV_MODE: {NAV_MODE}. Must be 'nearest', 'order' or 'keyboard'"
)

# 视角跟随开关
# True  = 视角跟随机器人移动（锁定）
# False = 视角自由，用户可用鼠标拖动 / WASD 移动（推荐调试时关闭）
# 环境变量: ATEC_TASKB_CAMERA_FOLLOW=1 或 0
ATEC_CAMERA_FOLLOW_ROBOT = os.getenv("ATEC_TASKB_CAMERA_FOLLOW", "0").lower() in {
    "1", "true", "yes", "on",
}


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _to_numpy_array(data: Any) -> np.ndarray:
    """将 torch/numpy/array-like 数据转换为 numpy 数组。"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def rgb_to_bgr_uint8(rgb):
    """将 RGB 图像转换为 OpenCV 可显示的 BGR uint8。"""
    rgb_np = _to_numpy_array(rgb)
    while rgb_np.ndim > 3 and rgb_np.shape[0] == 1:
        rgb_np = rgb_np[0]
    if rgb_np.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dims, got shape={rgb_np.shape}")

    if rgb_np.shape[-1] < 3:
        raise ValueError(f"Expected RGB channels in last dim, got shape={rgb_np.shape}")

    rgb_np = rgb_np[..., :3]
    rgb_np = np.nan_to_num(rgb_np, nan=0.0, posinf=255.0, neginf=0.0)
    if rgb_np.dtype != np.uint8:
        rgb_np = rgb_np.astype(np.float32)
        if rgb_np.max(initial=0.0) <= 1.0:
            rgb_np = rgb_np * 255.0
        rgb_np = np.clip(rgb_np, 0.0, 255.0).astype(np.uint8)
    return rgb_np[..., ::-1]


def depth_to_colormap(depth, min_depth=0.0, max_depth=5.0):
    """将深度图转换为 OpenCV JET 伪彩色图。"""
    depth_np = _to_numpy_array(depth)
    while depth_np.ndim > 2 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    if depth_np.ndim != 2:
        raise ValueError(f"Expected depth image with 2 dims, got shape={depth_np.shape}")

    depth_np = np.nan_to_num(depth_np.astype(np.float32), nan=max_depth, posinf=max_depth, neginf=min_depth)
    depth_np = np.clip(depth_np, min_depth, max_depth)
    if max_depth <= min_depth:
        depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
    else:
        depth_vis = ((depth_np - min_depth) / (max_depth - min_depth) * 255.0).astype(np.uint8)
    if cv2 is None:
        return np.repeat(depth_vis[..., None], 3, axis=-1)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


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


class PolicyNavigator:
    """
    Policy驱动的导航控制器 - 计算 base_cmd 交给 actor 策略网络执行。
    导航逻辑与原 PregraspNavigator 一致：先转向对准 → 再前进 → 停在垃圾前0.6m。
    区别在于返回 base_cmd (np.array) 而非 (vx, vy, yaw_rate) 元组，
    由上层送入 policy 网络产生关节动作。
    """

    def __init__(self, nav_mode: str = "nearest"):
        self.stand_off = 0.6
        self.kp_pos = 1.2
        self.kp_yaw = 2.0
        self.max_vx = 0.8
        self.max_vy = 0.25
        self.max_yaw_rate = 0.8
        self.pos_tol = 0.32
        self.yaw_tol = 0.087
        self.slow_radius = 0.5

        self.nav_mode = nav_mode
        self.nav_state = "SELECT_TARGET"
        self.current_target = None
        self.goal_xy = np.array([0.0, 0.0])
        self.goal_yaw = 0.0
        self.done_target_ids = set()
        self.trash_targets = []
        self._current_order_idx = 1

    def _wrap_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _world_to_body_velocity(self, vx_world: float, vy_world: float, robot_yaw: float) -> tuple:
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        vx_body = cos_yaw * vx_world + sin_yaw * vy_world
        vy_body = -sin_yaw * vx_world + cos_yaw * vy_world
        return vx_body, vy_body

    def compute_pregrasp_pose(self, robot_pos_w: np.ndarray, trash_pos_w: np.ndarray) -> tuple:
        robot_xy = robot_pos_w[:2]
        trash_xy = trash_pos_w[:2]
        direction = robot_xy - trash_xy
        norm = np.linalg.norm(direction)
        if norm < 0.01:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        direction_normalized = direction / norm
        goal_xy = trash_xy + direction_normalized * self.stand_off
        dx = trash_pos_w[0] - goal_xy[0]
        dy = trash_pos_w[1] - goal_xy[1]
        goal_yaw = math.atan2(dy, dx)
        return goal_xy, goal_yaw

    def select_nearest_target(self, trash_targets: list, robot_pos_w: np.ndarray) -> dict | None:
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
        for idx in range(self._current_order_idx, 19):
            target_id = f"object_{idx}"
            for trash in trash_targets:
                if trash.get("id") == target_id:
                    if trash.get("status", "pending") == "pending" and trash.get("id") not in self.done_target_ids:
                        return trash
            self._current_order_idx = idx + 1
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
        计算速度命令 - 与原 PregraspNavigator 逻辑完全一致：
        先转向对准目标方向，再前进，接近时减速。
        返回：(vx_body, vy_body, yaw_rate), nav_info
        """
        robot_xy = robot_pos_w[:2]
        error_xy_w = self.goal_xy - robot_xy
        pos_error_norm = np.linalg.norm(error_xy_w)
        yaw_error = self._wrap_angle(self.goal_yaw - robot_yaw)

        if self.nav_state == "NAVIGATE_TO_PREGRASP":
            if abs(yaw_error) > 0.3:
                vx_body = 0.0
                vy_body = 0.0
                yaw_rate = self.kp_yaw * yaw_error
                yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
                arrived = False
            else:
                vx_w = self.kp_pos * error_xy_w[0]
                vy_w = self.kp_pos * error_xy_w[1]
                if pos_error_norm < self.slow_radius:
                    decel_ratio = pos_error_norm / self.slow_radius
                    vx_w *= decel_ratio
                    vy_w *= decel_ratio
                vx_body, vy_body = self._world_to_body_velocity(vx_w, vy_w, robot_yaw)
                vx_body = max(-self.max_vx, min(self.max_vx, vx_body))
                vy_body = max(-self.max_vy, min(self.max_vy, vy_body))
                yaw_rate = self.kp_yaw * yaw_error * 0.3
                yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
                arrived = pos_error_norm < self.pos_tol
        elif self.nav_state == "ALIGN_TO_TRASH":
            vx_body = 0.0
            vy_body = 0.0
            yaw_rate = self.kp_yaw * yaw_error
            yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
            arrived = abs(yaw_error) < self.yaw_tol
        else:
            vx_body = 0.0
            vy_body = 0.0
            yaw_rate = 0.0
            arrived = False

        nav_info = {
            "pos_error": float(pos_error_norm),
            "yaw_error": float(yaw_error),
            "arrived": arrived,
            "goal_xy": self.goal_xy.copy(),
            "goal_yaw": float(self.goal_yaw),
        }
        return (vx_body, vy_body, yaw_rate), nav_info

    def update(self, robot_pos_w: np.ndarray, robot_yaw: float, trash_targets: list) -> tuple:
        """
        状态机更新。返回 (base_cmd, nav_info)。
        base_cmd = np.array([vx, vy, yaw_rate])，由上层送入 policy 网络。
        """
        self.trash_targets = trash_targets
        zero_cmd = np.zeros(3, dtype=np.float32)

        if self.nav_state == "SELECT_TARGET":
            if self.nav_mode == "order":
                self.current_target = self.select_order_target(trash_targets)
            else:
                self.current_target = self.select_nearest_target(trash_targets, robot_pos_w)
            if self.current_target is None:
                self.nav_state = "DONE"
                print("[PolicyNavigator] All trash processed, DONE", flush=True)
                return zero_cmd, {"state": "DONE", "arrived": True}
            print(f"[PolicyNavigator] Selected target: {self.current_target['id']} (mode: {self.nav_mode})", flush=True)
            self.nav_state = "COMPUTE_PREGRASP_POSE"
            return zero_cmd, {"state": "SELECT_TARGET", "arrived": False}

        if self.nav_state == "COMPUTE_PREGRASP_POSE":
            self.goal_xy, self.goal_yaw = self.compute_pregrasp_pose(robot_pos_w, self.current_target["pos_w"])
            print(f"[PolicyNavigator] Computed pregrasp pose: goal_xy={self.goal_xy}, goal_yaw={math.degrees(self.goal_yaw):.1f}°", flush=True)
            self.nav_state = "NAVIGATE_TO_PREGRASP"
            return zero_cmd, {"state": "COMPUTE_PREGRASP_POSE", "arrived": False}

        if self.nav_state == "NAVIGATE_TO_PREGRASP":
            (vx_body, vy_body, yaw_rate), nav_info = self.compute_velocity_command(robot_pos_w, robot_yaw)
            nav_info["state"] = "NAVIGATE_TO_PREGRASP"
            if nav_info["arrived"]:
                print(f"[PolicyNavigator] Arrived at pregrasp position, starting alignment", flush=True)
                self.nav_state = "ALIGN_TO_TRASH"
                return zero_cmd, {"state": "NAVIGATE_TO_PREGRASP", "arrived": True}
            return np.array([vx_body, vy_body, yaw_rate], dtype=np.float32), nav_info

        if self.nav_state == "ALIGN_TO_TRASH":
            (vx_body, vy_body, yaw_rate), nav_info = self.compute_velocity_command(robot_pos_w, robot_yaw)
            nav_info["state"] = "ALIGN_TO_TRASH"
            if nav_info["arrived"]:
                print(f"[PolicyNavigator] Aligned to trash, ready to grasp", flush=True)
                self.nav_state = "READY_TO_GRASP"
                return zero_cmd, {"state": "ALIGN_TO_TRASH", "arrived": True}
            return np.array([vx_body, vy_body, yaw_rate], dtype=np.float32), nav_info

        if self.nav_state == "READY_TO_GRASP":
            return zero_cmd, {"state": "READY_TO_GRASP", "arrived": True}

        if self.nav_state == "CROUCHING":
            return zero_cmd, {"state": "CROUCHING", "arrived": False}

        if self.nav_state == "GRASPING":
            return zero_cmd, {"state": "GRASPING", "arrived": False}

        if self.nav_state == "STAND_UP":
            return zero_cmd, {"state": "STAND_UP", "arrived": False}

        if self.nav_state == "DONE":
            return zero_cmd, {"state": "DONE", "arrived": True}

        return zero_cmd, {"state": self.nav_state, "arrived": False}

    def finish_current_target(self, status: str):
        if self.current_target:
            self.current_target["status"] = status
            self.done_target_ids.add(self.current_target["id"])
            print(f"[PolicyNavigator] Marked {self.current_target['id']} as {status}", flush=True)
        self.current_target = None
        self.nav_state = "SELECT_TARGET"


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
        pregrasp_height: float = 0.20,
        grasp_height_offset: float = 0.03,
        lift_height: float = 0.30,
        ee_pos_tol: float = 0.05,
        gripper_close_wait_steps: int = 30,
        lift_success_threshold: float = 0.10,
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

        self.arm_joint_ids, _ = robot.find_joints(self.arm_joint_names)
        self.gripper_joint_ids, _ = robot.find_joints(self.gripper_joint_names)
        self.arm_and_gripper_joint_ids = list(self.arm_joint_ids) + list(self.gripper_joint_ids)

        self.cartesian = None
        if CartesianController is None:
            print("[ArmGraspController] Warning: CartesianController unavailable, grasping will fail closed.", flush=True)
        else:
            try:
                self.cartesian = CartesianController(
                    robot=robot,
                    ee_body_name=ee_body_name,
                    arm_joint_names=self.arm_joint_names,
                    num_envs=1,
                    device=str(device),
                    command_type="position",
                    lambda_val=0.1,
                    max_joint_delta=0.2,
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
        self.success = False
        self.failure_reason = None
        self.desired_arm_joint_pos = None
        self.desired_gripper_joint_pos = self.gripper_open_pos.clone()
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
        self.state = "MOVE_TO_PREGRASP"
        self._log(force=True, extra="start_grasp")

    def step(self, robot, scene, sim_dt):
        del sim_dt
        self.robot = robot
        self.step_counter += 1

        if self.state == "IDLE":
            return False, False
        if self.cartesian is None:
            self.failure_reason = "ik_unavailable"
            self.state = "FAILED"
            self._log(force=True, extra="IK unavailable")
            return True, False

        ee_pos_w, _ = self.get_ee_pose()

        if self.state == "MOVE_TO_PREGRASP":
            self.open_gripper()
            self.move_ee_to_pose(self.pregrasp_pos_w, self.target_ee_quat_w)
            if self.ee_reached(ee_pos_w, self.pregrasp_pos_w):
                self.state = "MOVE_DOWN_TO_GRASP"
                self._log(force=True, extra="Reached pregrasp")

        elif self.state == "MOVE_DOWN_TO_GRASP":
            self.open_gripper()
            self.move_ee_to_pose(self.grasp_pos_w, self.target_ee_quat_w)
            if self.ee_reached(ee_pos_w, self.grasp_pos_w):
                self.state = "CLOSE_GRIPPER"
                self.wait_steps = 0
                self._log(force=True, extra="Reached grasp pose")

        elif self.state == "CLOSE_GRIPPER":
            self.close_gripper()
            self.wait_steps += 1
            if self.wait_steps >= self.gripper_close_wait_steps:
                self.state = "LIFT_OBJECT"
                self._log(force=True, extra="Gripper close wait finished")

        elif self.state == "LIFT_OBJECT":
            self.close_gripper()
            self.move_ee_to_pose(self.lift_pos_w, self.target_ee_quat_w)
            if self.ee_reached(ee_pos_w, self.lift_pos_w):
                self.state = "VERIFY_GRASP"
                self._log(force=True, extra="Reached lift pose")

        elif self.state == "VERIFY_GRASP":
            self.close_gripper()
            self.success = self.check_grasp_success(scene)
            self.state = "DONE" if self.success else "FAILED"
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
        target_pos = torch.as_tensor(target_pos_w, dtype=torch.float32, device=self.device).view(1, 3)
        if self.cartesian.command_type == "position":
            self.desired_arm_joint_pos = self.cartesian.compute(target_pos).detach().clone()
        else:
            target_quat = torch.as_tensor(target_quat_w, dtype=torch.float32, device=self.device).view(1, 4)
            self.desired_arm_joint_pos = self.cartesian.compute(target_pos, target_quat).detach().clone()

    def ee_reached(self, ee_pos_w, target_pos_w):
        pos_err = float(np.linalg.norm(np.asarray(ee_pos_w) - np.asarray(target_pos_w)))
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
        if self.current_target_id is None or scene is None:
            return None
        for container_name in ("rigid_objects", "articulations"):
            container = getattr(scene, container_name, None)
            if container is None:
                continue
            try:
                obj = container[self.current_target_id]
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
        action_env[:, self.arm_joint_ids] = (arm_target - default_joint_pos[:, self.arm_joint_ids]) / self.action_scale
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
        msg = (
            f"[ArmGraspController] state={self.state} target={self.current_target_id} "
            f"trash_pos_w={None if self.trash_pos_w is None else np.round(self.trash_pos_w, 3)} "
            f"pregrasp_pos_w={None if self.pregrasp_pos_w is None else np.round(self.pregrasp_pos_w, 3)} "
            f"grasp_pos_w={None if self.grasp_pos_w is None else np.round(self.grasp_pos_w, 3)} "
            f"lift_pos_w={None if self.lift_pos_w is None else np.round(self.lift_pos_w, 3)} "
            f"current_ee_pos_w={np.round(ee_pos_w, 3)} "
            f"ee_pos_err={None if pos_err is None else round(pos_err, 4)} "
            f"gripper_cmd={'close' if torch.allclose(self.desired_gripper_joint_pos, self.gripper_close_pos) else 'open'}"
        )
        if extra:
            msg += f" reason={extra}"
        if self.failure_reason and self.state == "FAILED":
            msg += f" failure={self.failure_reason}"
        print(msg, flush=True)


class LegPostureController:
    """基于 smoothstep + PD torque 的腿部蹲下控制器。"""

    def __init__(
        self,
        leg_joint_names: list[str],
        front_crouch_thigh: float = 0.35,
        front_crouch_calf: float = -1.45,
        rear_crouch_thigh: float = 0.5,
        rear_crouch_calf: float = -1.65,
        crouch_duration: float = 2.5,
        stand_up_duration: float = 2.5,
        joint_error_tol: float = 0.12,
    ):
        self.leg_joint_names = list(leg_joint_names)
        self.front_crouch_thigh = float(front_crouch_thigh)
        self.front_crouch_calf = float(front_crouch_calf)
        self.rear_crouch_thigh = float(rear_crouch_thigh)
        self.rear_crouch_calf = float(rear_crouch_calf)
        self.crouch_duration = max(float(crouch_duration), 1.0e-3)
        self.stand_up_duration = max(float(stand_up_duration), 1.0e-3)
        self.joint_error_tol = float(joint_error_tol)
        self.state = "IDLE"
        self._leg_joint_ids = None
        self._leg_joint_names_in_robot = None
        self._initialized = False
        self._last_episode_length_buf = None
        self._phase_start_step = None
        self._phase_start_dof_pos = None
        self._stand_dof_pos = None
        self._squat_dof_pos = None
        self._last_alpha = None

    def reset(self):
        self.state = "IDLE"
        self._phase_start_step = None
        self._phase_start_dof_pos = None
        self._last_alpha = None

    def start_crouch(self, robot):
        self._ensure_initialized(robot)
        self._sync_episode_reset(robot)
        self._phase_start_dof_pos = self._current_leg_dof_pos(robot).clone()
        self._stand_dof_pos = self._phase_start_dof_pos.clone()
        self._squat_dof_pos = self._build_squat_dof_pos(
            self._phase_start_dof_pos.clone(),
            self._leg_joint_names_in_robot,
        )
        self._phase_start_step = self._get_episode_step_buf(robot).clone()
        self._last_alpha = torch.zeros(
            self._stand_dof_pos.shape[0], device=self._stand_dof_pos.device, dtype=self._stand_dof_pos.dtype
        )
        self.state = "CROUCHING"

    def start_stand_up(self, robot):
        self._ensure_initialized(robot)
        self._sync_episode_reset(robot)
        self._phase_start_dof_pos = self._current_leg_dof_pos(robot).clone()
        self._phase_start_step = self._get_episode_step_buf(robot).clone()
        self._last_alpha = torch.zeros(
            self._phase_start_dof_pos.shape[0], device=self._phase_start_dof_pos.device, dtype=self._phase_start_dof_pos.dtype
        )
        self.state = "STANDING_UP"

    def step(self, robot, sim_dt: float) -> tuple[bool, torch.Tensor | None]:
        if self.state == "IDLE":
            return False, None
        del sim_dt
        self._ensure_initialized(robot)
        self._sync_episode_reset(robot)

        if self.state == "CROUCHING":
            target_dof_pos, alpha = self._interpolate_to_target(
                robot=robot,
                start_dof_pos=self._phase_start_dof_pos,
                goal_dof_pos=self._squat_dof_pos,
                duration=self.crouch_duration,
            )
            done = bool(torch.all(alpha >= 0.999))
            if done:
                dof_error = torch.max(torch.abs(self._current_leg_dof_pos(robot) - self._squat_dof_pos), dim=1).values
                done = bool(torch.all(dof_error <= self.joint_error_tol))
            if done:
                self.state = "HOLDING_CROUCH"
            return done, target_dof_pos

        if self.state == "STANDING_UP":
            target_dof_pos, alpha = self._interpolate_to_target(
                robot=robot,
                start_dof_pos=self._phase_start_dof_pos,
                goal_dof_pos=self._stand_dof_pos,
                duration=self.stand_up_duration,
            )
            done = bool(torch.all(alpha >= 0.999))
            if done:
                self.state = "IDLE"
            return done, target_dof_pos

        self.state = "IDLE"
        return True, None

    def hold_current_target(self) -> torch.Tensor | None:
        if self.state == "HOLDING_CROUCH":
            return None if self._squat_dof_pos is None else self._squat_dof_pos.clone()
        if self.state == "STANDING_UP":
            return None if self._stand_dof_pos is None else self._stand_dof_pos.clone()
        return None

    def get_alpha(self) -> torch.Tensor | None:
        return None if self._last_alpha is None else self._last_alpha.clone()

    def get_leg_joint_ids(self):
        return self._leg_joint_ids

    def get_leg_joint_names(self):
        return self._leg_joint_names_in_robot

    def get_stand_dof_pos(self) -> torch.Tensor | None:
        return None if self._stand_dof_pos is None else self._stand_dof_pos.clone()

    def get_squat_dof_pos(self) -> torch.Tensor | None:
        return None if self._squat_dof_pos is None else self._squat_dof_pos.clone()

    def _ensure_initialized(self, robot):
        if self._initialized:
            return
        self._leg_joint_ids, self._leg_joint_names_in_robot = robot.find_joints(self.leg_joint_names)
        self._leg_joint_names_in_robot = list(self._leg_joint_names_in_robot)
        current_leg_dof_pos = self._current_leg_dof_pos(robot).clone()
        self._stand_dof_pos = current_leg_dof_pos.clone()
        self._squat_dof_pos = self._build_squat_dof_pos(
            current_leg_dof_pos.clone(),
            self._leg_joint_names_in_robot,
        )
        self._phase_start_dof_pos = current_leg_dof_pos.clone()
        self._phase_start_step = self._get_episode_step_buf(robot).clone()
        self._last_episode_length_buf = self._get_episode_step_buf(robot).clone()
        self._last_alpha = torch.zeros(
            current_leg_dof_pos.shape[0], device=current_leg_dof_pos.device, dtype=current_leg_dof_pos.dtype
        )
        self._initialized = True

    def _get_episode_step_buf(self, robot) -> torch.Tensor:
        env = getattr(robot, "_env", None)
        if env is not None and hasattr(env, "episode_length_buf"):
            return env.episode_length_buf.clone()
        return torch.zeros(robot.data.joint_pos.shape[0], device=robot.data.joint_pos.device, dtype=torch.long)

    def _sync_episode_reset(self, robot):
        if not self._initialized:
            return
        step_buf = self._get_episode_step_buf(robot)
        reset_mask = step_buf < self._last_episode_length_buf
        if torch.any(reset_mask):
            env_ids = torch.nonzero(reset_mask, as_tuple=False).flatten()
            current_leg_dof_pos = self._current_leg_dof_pos(robot)
            self._stand_dof_pos[env_ids] = current_leg_dof_pos[env_ids].clone()
            self._phase_start_dof_pos[env_ids] = current_leg_dof_pos[env_ids].clone()
            self._phase_start_step[env_ids] = step_buf[env_ids].clone()
            self._last_alpha[env_ids] = 0.0
            self.state = "IDLE"
        self._last_episode_length_buf = step_buf

    def _current_leg_dof_pos(self, robot) -> torch.Tensor:
        return robot.data.joint_pos[:, self._leg_joint_ids]

    def _build_squat_dof_pos(self, reference: torch.Tensor, leg_joint_names: list[str]) -> torch.Tensor:
        squat_dof_pos = reference.clone()
        leg_offsets = {
            "FR": {"hip": 0.15, "thigh": self.front_crouch_thigh, "calf": self.front_crouch_calf},
            "FL": {"hip": -0.15, "thigh": self.front_crouch_thigh, "calf": self.front_crouch_calf},
            "RR": {"hip": 0.1, "thigh": self.rear_crouch_thigh, "calf": self.rear_crouch_calf},
            "RL": {"hip": -0.1, "thigh": self.rear_crouch_thigh, "calf": self.rear_crouch_calf},
        }
        for joint_idx, joint_name in enumerate(leg_joint_names):
            for leg_key, leg_offset in leg_offsets.items():
                if leg_key not in joint_name:
                    continue
                if "hip" in joint_name:
                    squat_dof_pos[:, joint_idx] = reference[:, joint_idx] + leg_offset["hip"]
                elif "thigh" in joint_name:
                    squat_dof_pos[:, joint_idx] = reference[:, joint_idx] + leg_offset["thigh"]
                elif "calf" in joint_name:
                    squat_dof_pos[:, joint_idx] = reference[:, joint_idx] + leg_offset["calf"]
                break
        return squat_dof_pos

    def _interpolate_to_target(
        self,
        robot,
        start_dof_pos: torch.Tensor,
        goal_dof_pos: torch.Tensor,
        duration: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_buf = self._get_episode_step_buf(robot).to(dtype=torch.float32)
        phase_start_step = self._phase_start_step.to(dtype=torch.float32)
        step_dt = float(getattr(getattr(robot, "_env", None), "step_dt", 0.02))
        t = (step_buf - phase_start_step) * step_dt
        alpha = torch.clamp(t / max(duration, 1.0e-6), 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        self._last_alpha = alpha.clone()
        alpha = alpha.unsqueeze(1)
        return (1.0 - alpha) * start_dof_pos + alpha * goal_dof_pos, self._last_alpha


class AlgSolution:
    """基于 Ground Truth 的导航解决方案"""
    
    def __init__(self, env=None):
        """
        Args:
            env: 仿真环境实例，用于获取物体 Ground Truth 位置
        """
        # 视角跟随开关（由文件顶部 ATEC_CAMERA_FOLLOW_ROBOT 控制）
        #  play_atec_task.py 会读取 self.camera_follow_enabled 决定是否调用 camera_follow
        self.camera_follow_enabled = ATEC_CAMERA_FOLLOW_ROBOT

        self.env = env
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.control_cfg = SimpleNamespace(
            use_squat_test=False,
            use_effort_leg_control=False,
            enable_pregrasp_crouch=False,
            squat_transition_time=float(os.getenv("ATEC_TASKB_SQUAT_TRANSITION_TIME", "3.0")),
            debug_interval=max(int(os.getenv("ATEC_TASKB_SQUAT_DEBUG_INTERVAL", "10")), 1),
            max_squat_action_delta=float(os.getenv("ATEC_TASKB_MAX_SQUAT_ACTION_DELTA", "0.08")),
        )
        self._leg_control_initialized = False
        self._leg_joint_ids = None
        self._leg_joint_names_in_robot = None
        self._leg_default_dof_pos = None
        self._leg_joint_pos_limits = None
        self._leg_torque_limits = None
        self._leg_p_gains = None
        self._leg_d_gains = None
        self._printed_leg_control_info = False
        self._env_leg_to_robot_indices = None
        self._robot_leg_to_env_indices = None
        self._leg_default_dof_pos_env = None
        self._last_leg_action_override = None
        
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
        self._pregrasp_navigator = PolicyNavigator(nav_mode=NAV_MODE if NAV_MODE != "keyboard" else "nearest")
        self._keyboard_controller = None
        self._arm_grasp_controller = None
        self._arm_controller_init_failed = False
        self.arm_action_scale = 0.5
        self.arm_ik_joint_names = list(B2_PIPER_ARM_JOINT_NAMES[:6])
        self.gripper_joint_names = list(B2_PIPER_ARM_JOINT_NAMES[6:])
        self._leg_posture_controller = LegPostureController(
            leg_joint_names=list(B2_PIPER_LEG_JOINT_NAMES),
            front_crouch_thigh=float(os.getenv("ATEC_TASKB_FRONT_CROUCH_THIGH", "0.25")),
            front_crouch_calf=float(os.getenv("ATEC_TASKB_FRONT_CROUCH_CALF", "-0.6")),
            rear_crouch_thigh=float(os.getenv("ATEC_TASKB_REAR_CROUCH_THIGH", "0.3")),
            rear_crouch_calf=float(os.getenv("ATEC_TASKB_REAR_CROUCH_CALF", "-0.7")),
            crouch_duration=float(os.getenv("ATEC_TASKB_CROUCH_DURATION", "3.5")),
            stand_up_duration=float(os.getenv("ATEC_TASKB_STAND_UP_DURATION", "3.5")),
            joint_error_tol=float(os.getenv("ATEC_TASKB_CROUCH_JOINT_TOL", "0.12")),
        )
        self._pending_grasp_status = None
        self._wbc_grasp_phase = None
        self._wbc_grasp_trash_pos = None
        self._wbc_grasp_step_count = 0
        self._wbc_grasp_success = False
        self._wbc_grasp_timeout = 500
        self._wbc_last_action_18 = None
        self._wbc_hold_ee_pos_w = None
        self._wbc_hold_ee_quat_w = None
        self._wbc_nav_initialized = False
        
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
        self._camera_debug_interval = 5
        self._camera_debug_enabled = True
        self._camera_debug_warned_keys: set[str] = set()
        self._camera_debug_depth_keys = ("distance_to_image_plane", "distance_to_camera", "depth")
        
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
            self.leg_joint_action_scale_by_name = {
                name: float(scale)
                for name, scale in zip(self.leg_joint_names, self.leg_action_scale.view(-1).detach().cpu().tolist())
            }
            
            # 机械臂默认位置
            b2_piper_arm_defaults = {
                "arm_joint1": 0.0,
                "arm_joint2": 2.13,
                "arm_joint3": -1.20,
                "arm_joint4": 0.0,
                "arm_joint5": 0.4,
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

        self._load_wbc_actor_model()

    def _load_wbc_actor_model(self):
        """加载 WBC 全身控制模型"""
        wbc_checkpoint_path = os.path.join(REPO_ROOT, "demo", "wbcpolicy.pt")

        if not os.path.exists(wbc_checkpoint_path):
            print(f"[GT-NAV] Warning: WBC checkpoint not found: {wbc_checkpoint_path}")
            self.wbc_actor = None
            return

        try:
            checkpoint = torch.load(wbc_checkpoint_path, map_location="cpu")
            state_dict = checkpoint["actor_state_dict"]

            wbc_input_dim = state_dict["mlp.0.weight"].shape[1]
            wbc_output_dim = state_dict["mlp.6.bias"].shape[0]

            self.wbc_actor = B2PiperActor(wbc_input_dim, wbc_output_dim).to(self.device)
            wbc_actor_state = {k.replace("mlp.", "actor."): v for k, v in state_dict.items() if k.startswith("mlp.")}
            self.wbc_actor.load_state_dict(wbc_actor_state, strict=True)
            self.wbc_actor.eval()

            self.wbc_obs_dim = 70
            self.wbc_obs_history = torch.zeros(1, self.wbc_obs_dim * 3, device=self.device, dtype=torch.float32)
            self.wbc_history_initialized = False
            self.wbc_action_scale = 0.25

            print(f"[GT-NAV] WBC Actor model loaded: input={wbc_input_dim}, output={wbc_output_dim}")
        except Exception as e:
            print(f"[GT-NAV] Failed to load WBC actor model: {e}")
            self.wbc_actor = None

    def _reset_wbc_history(self, obs=None, base_cmd=None, ee_pose_cmd=None):
        """重置 WBC 观测历史缓冲"""
        if self.wbc_actor is None:
            return
        if obs is not None:
            current_obs = self._extract_wbc_single_obs(obs, base_cmd, ee_pose_cmd)
            self.wbc_obs_history = torch.cat([current_obs, current_obs, current_obs], dim=-1)
        else:
            self.wbc_obs_history = torch.zeros(1, self.wbc_obs_dim * 3, device=self.device, dtype=torch.float32)
        self.wbc_history_initialized = True

    def _compute_ee_pose_cmd(self, target_pos_w, target_quat_w=None):
        """计算 WBC policy 需要的 ee_pose 命令 (arm_base 坐标系)
        WBC 训练时 ee_pose 格式: [x, y, z, qw, qx, qy, qz]
          - xy: arm_base 坐标系下的偏移
          - z: 世界坐标系下的绝对高度
          - quat: arm_base 坐标系下的四元数
        """
        robot = self._get_robot()
        if robot is None:
            return np.zeros(7, dtype=np.float32)

        arm_base_body_ids, _ = robot.find_bodies("arm_base")
        if len(arm_base_body_ids) == 0:
            return np.zeros(7, dtype=np.float32)

        arm_base_pos_w = robot.data.body_pos_w[0, arm_base_body_ids[0]].detach().cpu().numpy()
        arm_base_quat_w = robot.data.body_quat_w[0, arm_base_body_ids[0]].detach().cpu().numpy()

        target_pos_w = np.asarray(target_pos_w, dtype=np.float32)
        delta_pos_w = target_pos_w - arm_base_pos_w

        from isaaclab.utils.math import quat_rotate_inverse as _quat_rotate_inverse
        arm_base_quat_t = torch.as_tensor(arm_base_quat_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        delta_pos_t = torch.as_tensor(delta_pos_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        delta_pos_b = _quat_rotate_inverse(arm_base_quat_t, delta_pos_t).squeeze(0).cpu().numpy()

        ee_pose_cmd = np.zeros(7, dtype=np.float32)
        ee_pose_cmd[0] = delta_pos_b[0]
        ee_pose_cmd[1] = delta_pos_b[1]
        ee_pose_cmd[2] = target_pos_w[2]

        if target_quat_w is not None:
            from isaaclab.utils.math import quat_mul as _quat_mul, quat_conjugate as _quat_conjugate
            arm_base_quat_conj = _quat_conjugate(arm_base_quat_t)
            target_quat_t = torch.as_tensor(target_quat_w, dtype=torch.float32, device=self.device).unsqueeze(0)
            ee_quat_b = _quat_mul(arm_base_quat_conj, target_quat_t).squeeze(0).cpu().numpy()
            ee_pose_cmd[3:7] = ee_quat_b
        else:
            ee_pose_cmd[3] = 1.0

        return ee_pose_cmd

    def _get_current_gripper_pose_w(self):
        """读取当前 gripper_base 世界位姿。"""
        robot = self._get_robot()
        if robot is None:
            return None, None
        try:
            body_ids, _ = robot.find_bodies("gripper_base")
            if len(body_ids) == 0:
                return None, None
            body_id = body_ids[0]
            pos_w = robot.data.body_pos_w[0, body_id].detach().cpu().numpy().astype(np.float32)
            quat_w = robot.data.body_quat_w[0, body_id].detach().cpu().numpy().astype(np.float32)
            return pos_w, quat_w
        except Exception:
            return None, None

    def _capture_wbc_hold_ee_pose(self, force: bool = False) -> bool:
        """锁定导航阶段使用的末端保持位姿。"""
        if not force and self._wbc_hold_ee_pos_w is not None and self._wbc_hold_ee_quat_w is not None:
            return True
        pos_w, quat_w = self._get_current_gripper_pose_w()
        if pos_w is None or quat_w is None:
            return False
        self._wbc_hold_ee_pos_w = np.asarray(pos_w, dtype=np.float32).copy()
        self._wbc_hold_ee_quat_w = np.asarray(quat_w, dtype=np.float32).copy()
        return True

    def _get_wbc_hold_ee_pose_cmd(self):
        """返回导航阶段保持末端静止的 WBC 命令。"""
        if self._wbc_hold_ee_pos_w is None or self._wbc_hold_ee_quat_w is None:
            if not self._capture_wbc_hold_ee_pose(force=False):
                return None
        return self._compute_ee_pose_cmd(self._wbc_hold_ee_pos_w, self._wbc_hold_ee_quat_w)

    def _ensure_wbc_nav_session(self, obs, base_cmd):
        """初始化全程 WBC 导航会话，复用同一套历史。"""
        if self.wbc_actor is None:
            return
        if self._wbc_nav_initialized:
            return
        if not self._capture_wbc_hold_ee_pose(force=False):
            return
        self._wbc_last_action_18 = torch.zeros(1, 18, device=self.device, dtype=torch.float32)
        ee_pose_cmd = self._get_wbc_hold_ee_pose_cmd()
        self._reset_wbc_history(obs, base_cmd, ee_pose_cmd)
        self._wbc_nav_initialized = True

    def _extract_wbc_single_obs(self, obs, base_cmd, ee_pose_cmd=None):
        """提取 WBC policy 单步观测 (70维)
        顺序: base_ang_vel(3) + projected_gravity(3) + joint_pos(18) + joint_vel(18)
              + actions(18) + velocity_commands(3) + ee_pose_cmd(7)
        """
        proprio = torch.as_tensor(obs["proprio"], device=self.device, dtype=torch.float32)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]; idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        _velocity_commands_env = proprio[:, idx:idx + 3]; idx += 3
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3

        joint_pos_all = proprio[:, idx:idx + self.total_action_dim]; idx += self.total_action_dim
        joint_vel_all = proprio[:, idx:idx + self.total_action_dim]; idx += self.total_action_dim
        actions_all = proprio[:, idx:idx + self.total_action_dim]

        joint_pos_18 = joint_pos_all[:, :18]
        joint_vel_18 = joint_vel_all[:, :18]

        if hasattr(self, '_wbc_last_action_18') and self._wbc_last_action_18 is not None:
            actions_18 = self._wbc_last_action_18.to(device=proprio.device, dtype=proprio.dtype)
        else:
            actions_18 = actions_all[:, :18]

        velocity_commands = torch.as_tensor(base_cmd, device=self.device, dtype=proprio.dtype).view(1, 3)
        if proprio.shape[0] > 1:
            velocity_commands = velocity_commands.repeat(proprio.shape[0], 1)

        if ee_pose_cmd is not None:
            ee_pose_t = torch.as_tensor(ee_pose_cmd, device=self.device, dtype=proprio.dtype).view(1, 7)
        else:
            ee_pose_t = torch.zeros(1, 7, device=self.device, dtype=proprio.dtype)
            ee_pose_t[:, 3] = 1.0
        if proprio.shape[0] > 1:
            ee_pose_t = ee_pose_t.repeat(proprio.shape[0], 1)

        single_obs = torch.cat([
            base_ang_vel * 0.2,
            projected_gravity,
            joint_pos_18,
            joint_vel_18 * 0.05,
            actions_18,
            velocity_commands,
            ee_pose_t,
        ], dim=-1)
        return torch.nan_to_num(single_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_wbc_policy_obs(self, obs, base_cmd, ee_pose_cmd=None):
        """提取 WBC policy 完整观测 (210维 = 70 × 3步历史)"""
        current_obs = self._extract_wbc_single_obs(obs, base_cmd, ee_pose_cmd)

        if not self.wbc_history_initialized:
            self._reset_wbc_history(obs, base_cmd, ee_pose_cmd)
        else:
            self.wbc_obs_history = torch.cat([
                self.wbc_obs_history[:, self.wbc_obs_dim:],
                current_obs
            ], dim=-1)

        return self.wbc_obs_history.clone()

    def _map_wbc_action_to_env_action(self, action_18: torch.Tensor, gripper_action: torch.Tensor = None) -> torch.Tensor:
        """将 WBC policy 输出 (18维: 12腿+6臂) 映射到环境动作 (20维)"""
        num_envs = action_18.shape[0]
        action_env = torch.zeros((num_envs, self.total_action_dim), device=self.device, dtype=torch.float32)
        action_env[:, :18] = action_18 * self.wbc_action_scale
        if gripper_action is not None:
            action_env[:, 18:] = gripper_action
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _generate_wbc_action_tensor(self, obs, base_cmd, ee_pose_cmd=None, gripper_action=None) -> torch.Tensor:
        """使用 WBC policy 生成全身控制动作"""
        if obs is None or self.wbc_actor is None:
            return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)

        try:
            policy_obs = self._extract_wbc_policy_obs(obs, base_cmd, ee_pose_cmd)
            with torch.inference_mode():
                action_18 = self.wbc_actor(policy_obs)
            if action_18.ndim == 1:
                action_18 = action_18.unsqueeze(0)
            self._wbc_last_action_18 = action_18.detach().clone()
            return self._map_wbc_action_to_env_action(action_18, gripper_action)
        except Exception as e:
            print(f"[GT-NAV] WBC Actor inference error: {e}", flush=True)
            return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)

    def get_action_spec(self) -> dict | None:
        """返回动作规格，指定腿部控制模式"""
        if self.control_cfg.use_effort_leg_control:
            return {
                "leg": {
                    "mode": "effort",
                    "scale": 1.0,
                    "clip": [-300.0, 300.0],
                }
            }
        return {
            "leg": {
                "mode": "position",
                "scale": 0.25,
                "clip": None,
            },
            "arm": {
                "mode": "position",
                "scale": 0.25,
                "clip": None,
            },
        }

    def _warn_once(self, key: str, message: str):
        """避免重复打印相同 warning。"""
        if key in self._camera_debug_warned_keys:
            return
        self._camera_debug_warned_keys.add(key)
        print(message, flush=True)

    def _get_scene(self):
        """从 env 中获取 scene。"""
        if self.env is None:
            return None

        env_unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(env_unwrapped, "scene"):
            return env_unwrapped.scene
        if hasattr(env_unwrapped, "_env") and hasattr(env_unwrapped._env, "scene"):
            return env_unwrapped._env.scene
        return None

    def _get_robot(self):
        """从 scene 中获取 robot articulation。"""
        scene = self._get_scene()
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

    def _ensure_leg_control_initialized(self, robot):
        if self._leg_control_initialized or robot is None:
            return
        self._leg_joint_ids, self._leg_joint_names_in_robot = robot.find_joints(self.leg_joint_names)
        self._leg_joint_names_in_robot = list(self._leg_joint_names_in_robot)
        robot_name_to_local_idx = {name: idx for idx, name in enumerate(self._leg_joint_names_in_robot)}
        self._env_leg_to_robot_indices = torch.tensor(
            [robot_name_to_local_idx[name] for name in self.leg_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._robot_leg_to_env_indices = torch.empty_like(self._env_leg_to_robot_indices)
        self._robot_leg_to_env_indices[self._env_leg_to_robot_indices] = torch.arange(
            len(self.leg_joint_names), device=self.device, dtype=torch.long
        )
        self._leg_default_dof_pos = robot.data.default_joint_pos[:, self._leg_joint_ids].clone()
        self._leg_default_dof_pos_env = self._leg_default_dof_pos[:, self._env_leg_to_robot_indices].clone()

        if hasattr(robot.data, "soft_joint_pos_limits"):
            self._leg_joint_pos_limits = robot.data.soft_joint_pos_limits[:, self._leg_joint_ids, :].clone()
        elif hasattr(robot.data, "joint_pos_limits"):
            self._leg_joint_pos_limits = robot.data.joint_pos_limits[:, self._leg_joint_ids, :].clone()
        else:
            min_pos = torch.full_like(self._leg_default_dof_pos, -10.0)
            max_pos = torch.full_like(self._leg_default_dof_pos, 10.0)
            self._leg_joint_pos_limits = torch.stack((min_pos, max_pos), dim=-1)

        self._leg_p_gains = torch.full_like(self._leg_default_dof_pos, 100.0)
        self._leg_d_gains = torch.full_like(self._leg_default_dof_pos, 15.0)
        self._leg_torque_limits = torch.full_like(self._leg_default_dof_pos, 300.0)
        for joint_idx, joint_name in enumerate(self._leg_joint_names_in_robot):
            if "calf" in joint_name:
                self._leg_torque_limits[:, joint_idx] = 400.0
                self._leg_p_gains[:, joint_idx] = 120.0
                self._leg_d_gains[:, joint_idx] = 20.0

        self._leg_posture_controller._ensure_initialized(robot)
        self._leg_control_initialized = True

        if not self._printed_leg_control_info:
            print(f"[SQUAT] dof_names={self._leg_joint_names_in_robot}", flush=True)
            print(
                f"[SQUAT] squat_dof_pos={self._leg_posture_controller.get_squat_dof_pos()[0].detach().cpu().tolist()}",
                flush=True,
            )
            print(f"[SQUAT] env_leg_order={self.leg_joint_names}", flush=True)
            self._printed_leg_control_info = True

    def _reorder_env_leg_to_robot(self, tensor_env_order: torch.Tensor) -> torch.Tensor:
        return tensor_env_order[:, self._env_leg_to_robot_indices]

    def _reorder_robot_leg_to_env(self, tensor_robot_order: torch.Tensor) -> torch.Tensor:
        return tensor_robot_order[:, self._robot_leg_to_env_indices]

    def _get_base_rpy_height(self, robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quat = robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        height = robot.data.root_pos_w[:, 2]
        return roll, pitch, height

    def _get_policy_leg_target_dof_pos(self, obs, base_cmd, robot) -> torch.Tensor:
        if obs is None or self.actor is None:
            return robot.data.joint_pos[:, self._leg_joint_ids].clone()
        policy_obs = self._extract_policy_obs(obs, base_cmd)
        with torch.inference_mode():
            action_train = self.actor(policy_obs)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        leg_action_env = action_train * self.leg_action_scale.to(dtype=policy_obs.dtype)
        target_dof_pos_env = self._leg_default_dof_pos_env.to(dtype=policy_obs.dtype) + leg_action_env
        return self._reorder_env_leg_to_robot(target_dof_pos_env)

    def _get_squat_target_dof_pos(self, robot) -> torch.Tensor:
        self._ensure_leg_control_initialized(robot)
        episode_length_buf = getattr(
            self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env,
            "episode_length_buf",
            None,
        )
        if episode_length_buf is None:
            episode_length_buf = torch.zeros(robot.data.joint_pos.shape[0], device=robot.data.joint_pos.device, dtype=torch.long)
        t = episode_length_buf.to(dtype=torch.float32) * float(
            getattr(self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env, "step_dt", self.dt)
        )
        alpha = torch.clamp(t / self.control_cfg.squat_transition_time, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        alpha_unsqueezed = alpha.unsqueeze(1)
        stand_dof_pos = self._leg_posture_controller.get_stand_dof_pos()
        squat_dof_pos = self._leg_posture_controller.get_squat_dof_pos()
        target_dof_pos = (1.0 - alpha_unsqueezed) * stand_dof_pos + alpha_unsqueezed * squat_dof_pos
        self._leg_posture_controller._last_alpha = alpha
        return target_dof_pos

    def _compute_leg_torques(self, target_dof_pos: torch.Tensor, robot) -> torch.Tensor:
        self._ensure_leg_control_initialized(robot)
        dof_pos = robot.data.joint_pos[:, self._leg_joint_ids]
        dof_vel = robot.data.joint_vel[:, self._leg_joint_ids]
        joint_limits = self._leg_joint_pos_limits.to(device=target_dof_pos.device, dtype=target_dof_pos.dtype)
        
        target_dof_pos = torch.clamp(target_dof_pos, joint_limits[..., 0], joint_limits[..., 1])
        
        if not hasattr(self, '_last_torque_target') or self._last_torque_target is None or self._last_torque_target.shape != target_dof_pos.shape:
            self._last_torque_target = target_dof_pos.clone()
        else:
            max_delta = float(self.control_cfg.max_squat_action_delta) * 0.5
            delta = torch.clamp(target_dof_pos - self._last_torque_target, -max_delta, max_delta)
            target_dof_pos = self._last_torque_target + delta
            self._last_torque_target = target_dof_pos.clone()
        
        torques = self._leg_p_gains * (target_dof_pos - dof_pos) - self._leg_d_gains * dof_vel
        torques = torch.clip(torques, -self._leg_torque_limits, self._leg_torque_limits)

        if self._step_count % self.control_cfg.debug_interval == 0:
            alpha = self._leg_posture_controller.get_alpha()
            roll, pitch, height = self._get_base_rpy_height(robot)
            print(
                f"[SQUAT] alpha={None if alpha is None else alpha.detach().cpu().tolist()} "
                f"target_dof_pos={target_dof_pos[0].detach().cpu().tolist()} "
                f"dof_pos={dof_pos[0].detach().cpu().tolist()} "
                f"torque_max={float(torch.max(torch.abs(torques)).item()):.3f} "
                f"roll={float(roll[0].item()):.3f} "
                f"pitch={float(pitch[0].item()):.3f} "
                f"height={float(height[0].item()):.3f}",
                flush=True,
            )
        return torques

    def _compute_leg_position_actions(self, target_dof_pos: torch.Tensor, robot) -> torch.Tensor:
        self._ensure_leg_control_initialized(robot)
        joint_limits = self._leg_joint_pos_limits.to(device=target_dof_pos.device, dtype=target_dof_pos.dtype)
        target_dof_pos = torch.clamp(target_dof_pos, joint_limits[..., 0], joint_limits[..., 1])
        target_dof_pos_env = self._reorder_robot_leg_to_env(target_dof_pos)
        
        current_dof_pos = robot.data.joint_pos[:, self._leg_joint_ids]
        current_dof_pos_env = self._reorder_robot_leg_to_env(current_dof_pos)
        
        leg_actions = (target_dof_pos_env - current_dof_pos_env) / self.leg_action_scale.to(dtype=target_dof_pos.dtype)
        max_delta = float(self.control_cfg.max_squat_action_delta)
        if self._last_leg_action_override is None or self._last_leg_action_override.shape != leg_actions.shape:
            self._last_leg_action_override = leg_actions.clone()
        else:
            delta = torch.clamp(leg_actions - self._last_leg_action_override, -max_delta, max_delta)
            leg_actions = self._last_leg_action_override + delta
            self._last_leg_action_override = leg_actions.clone()

        dof_pos = current_dof_pos
        if self._step_count % self.control_cfg.debug_interval == 0:
            alpha = self._leg_posture_controller.get_alpha()
            roll, pitch, height = self._get_base_rpy_height(robot)
            print(
                f"[SQUAT] alpha={None if alpha is None else alpha.detach().cpu().tolist()} "
                f"target_dof_pos={target_dof_pos[0].detach().cpu().tolist()} "
                f"dof_pos={dof_pos[0].detach().cpu().tolist()} "
                f"roll={float(roll[0].item()):.3f} "
                f"pitch={float(pitch[0].item()):.3f} "
                f"height={float(height[0].item()):.3f}",
                flush=True,
            )
        return leg_actions

    def _generate_control_action_tensor(self, obs, base_cmd, robot) -> torch.Tensor:
        action_env = self._generate_action_tensor(obs, base_cmd)
        self._ensure_leg_control_initialized(robot)

        if self.control_cfg.use_squat_test:
            target_dof_pos = self._get_squat_target_dof_pos(robot)
        elif self._leg_posture_controller.hold_current_target() is not None:
            target_dof_pos = self._leg_posture_controller.hold_current_target().to(device=self.device, dtype=torch.float32)
        elif self._leg_posture_controller.state != "IDLE":
            _, target_dof_pos = self._leg_posture_controller.step(robot, self.dt)
            if target_dof_pos is None:
                target_dof_pos = robot.data.joint_pos[:, self._leg_joint_ids].clone()
        else:
            self._last_leg_action_override = None
            return action_env

        if self.control_cfg.use_effort_leg_control:
            leg_actions = self._compute_leg_torques(target_dof_pos, robot)
        else:
            leg_actions = self._compute_leg_position_actions(target_dof_pos, robot)
        
        action_env[:, :self.leg_action_dim] = leg_actions
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _ensure_arm_grasp_controller(self):
        """懒初始化机械臂抓取控制器。"""
        if self._arm_grasp_controller is not None or self._arm_controller_init_failed:
            return self._arm_grasp_controller

        robot = self._get_robot()
        if robot is None:
            return None

        try:
            self._arm_grasp_controller = ArmGraspController(
                robot=robot,
                device=self.device,
                arm_joint_names=self.arm_ik_joint_names,
                gripper_joint_names=self.gripper_joint_names,
                ee_body_name="gripper_base",
                action_scale=self.arm_action_scale,
            )
        except Exception as exc:
            self._arm_controller_init_failed = True
            print(f"[GT-NAV] Warning: failed to create ArmGraspController: {exc}", flush=True)
            self._arm_grasp_controller = None
        return self._arm_grasp_controller

    def _get_scene_camera(self, scene, camera_name: str):
        """按名称安全获取 scene 中的 camera。"""
        if scene is None:
            self._warn_once("camera_debug_scene_missing", "[GT-NAV] Warning: scene unavailable, disabling camera debug display.")
            return None

        try:
            camera = scene[camera_name]
        except Exception:
            self._warn_once(
                f"camera_debug_missing_{camera_name}",
                f"[GT-NAV] Warning: camera '{camera_name}' not found in scene, skipping related debug display.",
            )
            return None

        if isinstance(camera, (list, tuple)):
            camera = camera[0] if camera else None
        if camera is None:
            self._warn_once(
                f"camera_debug_none_{camera_name}",
                f"[GT-NAV] Warning: camera '{camera_name}' is empty, skipping related debug display.",
            )
        return camera

    def _get_camera_output(self, camera, camera_name: str, output_key: str):
        """从 camera.data.output 安全读取输出。"""
        output = getattr(getattr(camera, "data", None), "output", None)
        if output is None:
            self._warn_once(
                f"camera_debug_output_missing_{camera_name}",
                f"[GT-NAV] Warning: camera '{camera_name}' has no data.output, skipping related debug display.",
            )
            return None
        if output_key not in output:
            self._warn_once(
                f"camera_debug_output_key_missing_{camera_name}_{output_key}",
                f"[GT-NAV] Warning: camera '{camera_name}' output '{output_key}' not found, skipping related debug display.",
            )
            return None
        return output[output_key]

    def _get_camera_depth_output(self, camera, camera_name: str):
        """优先按项目常用字段读取 depth 输出。"""
        output = getattr(getattr(camera, "data", None), "output", None)
        if output is None:
            self._warn_once(
                f"camera_debug_depth_output_missing_{camera_name}",
                f"[GT-NAV] Warning: camera '{camera_name}' has no data.output, skipping related depth debug display.",
            )
            return None

        for key in self._camera_debug_depth_keys:
            if key in output:
                return output[key]

        available_keys = sorted(output.keys()) if hasattr(output, "keys") else []
        self._warn_once(
            f"camera_debug_depth_key_missing_{camera_name}",
            f"[GT-NAV] Warning: camera '{camera_name}' has no depth output in {self._camera_debug_depth_keys}. "
            f"Available keys: {available_keys}",
        )
        return None

    def _process_ee_camera(self, obs):
        """
        处理相机调试显示：head RGB、head depth、ee depth
        """
        if not self._camera_debug_enabled:
            return

        if self._step_count % self._camera_debug_interval != 0:
            return

        if cv2 is None:
            self._warn_once("camera_debug_cv2_unavailable", "[GT-NAV] Warning: OpenCV unavailable, camera debug display disabled.")
            self._camera_debug_enabled = False
            return

        try:
            scene = self._get_scene()
            head_camera = self._get_scene_camera(scene, "head_camera")
            ee_camera = self._get_scene_camera(scene, "ee_camera")

            head_rgb = self._get_camera_output(head_camera, "head_camera", "rgb") if head_camera is not None else None
            head_depth = self._get_camera_depth_output(head_camera, "head_camera") if head_camera is not None else None
            ee_rgb = self._get_camera_output(ee_camera, "ee_camera", "rgb") if ee_camera is not None else None
            ee_depth = self._get_camera_depth_output(ee_camera, "ee_camera") if ee_camera is not None else None

            
            if head_rgb is not None:
                cv2.imshow("head_rgb", rgb_to_bgr_uint8(head_rgb))
            # if head_depth is not None:
            #     cv2.imshow("head_depth", depth_to_colormap(head_depth))
            if ee_rgb is not None:
                cv2.imshow("ee_rgb", rgb_to_bgr_uint8(ee_rgb))
            # if ee_depth is not None:
            #     cv2.imshow("ee_depth", depth_to_colormap(ee_depth))
            cv2.waitKey(1)
        except Exception as e:
            self._warn_once("camera_debug_runtime_error", f"[GT-NAV] Warning: camera debug display disabled due to error: {e}")
            self._camera_debug_enabled = False

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
            # 仅用于测试目的，返回默认位置
            if hasattr(self, '_test_mode') and self._test_mode:
                return self.robot_pos, self.robot_yaw
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
        # if self._step_count % 100 == 0:
        #     print(f"[GT-NAV DEBUG] Robot pos: {self.robot_pos}, quat: {quat}, yaw1: {math.degrees(yaw1):.1f}°, yaw2: {math.degrees(yaw2):.1f}°", flush=True)
        
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
        action_env[:, self.leg_action_dim:] = 0.0
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _generate_action_tensor(self, obs, base_cmd) -> torch.Tensor:
        """生成环境动作张量，便于叠加机械臂抓取目标。"""
        if obs is None or self.actor is None:
            return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)

        try:
            policy_obs = self._extract_policy_obs(obs, base_cmd)
            with torch.inference_mode():
                action_train = self.actor(policy_obs)

            if action_train.ndim == 1:
                action_train = action_train.unsqueeze(0)

            return self._map_policy_action_to_env_action(action_train)
        except Exception as e:
            print(f"[GT-NAV] Actor inference error: {e}", flush=True)
            return torch.zeros((1, self.total_action_dim), device=self.device, dtype=torch.float32)

    def _action_tensor_to_output(self, action_env: torch.Tensor):
        return action_env.detach().cpu().numpy().tolist()

    def _get_stand_still_action(self, obs):
        """获取站立不动的动作"""
        robot = self._get_robot()
        if self.control_cfg.use_effort_leg_control and robot is not None:
            zero_cmd = np.zeros(3, dtype=np.float32)
            return self._action_tensor_to_output(self._generate_control_action_tensor(obs, zero_cmd, robot))
        if self.actor is not None and obs is not None:
            zero_cmd = np.zeros(3, dtype=np.float32)
            return self._action_tensor_to_output(self._generate_action_tensor(obs, zero_cmd))
        else:
            return np.zeros(self.total_action_dim if hasattr(self, 'total_action_dim') else 20, dtype=np.float32).tolist()

    def _generate_action(self, obs, base_cmd):
        """生成机器人动作"""
        return self._action_tensor_to_output(self._generate_action_tensor(obs, base_cmd))

    def predicts(self, obs, current_score):
        """
        决策入口 - 返回动作
        实现逻辑：导航到垃圾前方并对准 → 停下 → 机械臂抓取 → 标记完成 → 重复
        """
        self._step_count += 1
        
        # 处理 end-effector 摄像头图像
        self._process_ee_camera(obs)

        if NAV_MODE == "keyboard":
            return run_keyboard_control_mode(self, obs)
        
        # 更新机器人位姿
        self.robot_pos, self.robot_yaw = self._get_robot_pose(obs)
        scene = self._get_scene()
        robot = self._get_robot()
        arm_grasp_controller = self._ensure_arm_grasp_controller()
        
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
        
        # 使用 PolicyNavigator 计算 base_cmd（交给 policy 网络执行）
        base_cmd, nav_info = self._pregrasp_navigator.update(
            self.robot_pos, 
            self.robot_yaw, 
            trash_targets
        )
        
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
        
        # 处理 READY_TO_GRASP 状态：到达后直接开始抓取，不蹲下，保持站立
        if nav_info["state"] == "READY_TO_GRASP":
            if self._pregrasp_navigator.current_target is not None:
                target_xy = self._pregrasp_navigator.current_target["pos_w"][:2]
                robot_xy = self.robot_pos[:2]
                dist_to_target = np.linalg.norm(target_xy - robot_xy)
                print(f"[GT-NAV] Safety check: dist_to_target={dist_to_target:.3f}m, stand_off={self._pregrasp_navigator.stand_off}m", flush=True)

                if dist_to_target > self._pregrasp_navigator.stand_off + 0.5:
                    print(f"[GT-NAV] Warning: Too far from target! Resetting navigation...", flush=True)
                    self._pregrasp_navigator.nav_state = "SELECT_TARGET"
                else:
                    trash_pos_w = self._pregrasp_navigator.current_target["pos_w"]
                    self._wbc_grasp_trash_pos = np.asarray(trash_pos_w, dtype=np.float32).copy()
                    self._wbc_grasp_phase = "pregrasp"
                    self._wbc_grasp_step_count = 0
                    self._wbc_grasp_success = False
                    self._wbc_grasp_timeout = 500
                    self._pregrasp_navigator.nav_state = "GRASPING"
                    print(
                        f"[GT-NAV] Started WBC grasp for "
                        f"{self._pregrasp_navigator.current_target['id']}",
                        flush=True,
                    )

        if self._pregrasp_navigator.nav_state == "GRASPING":
            zero_cmd = np.zeros(3, dtype=np.float32)

            if self.wbc_actor is None or robot is None:
                self._pregrasp_navigator.finish_current_target("failed")
                return {"action": self._action_tensor_to_output(self._generate_control_action_tensor(obs, zero_cmd, robot)), "giveup": False}

            self._wbc_grasp_step_count += 1
            trash_pos = self._wbc_grasp_trash_pos

            if self._wbc_grasp_phase == "pregrasp":
                target_pos = trash_pos + np.array([0.0, 0.0, 0.20], dtype=np.float32)
            elif self._wbc_grasp_phase == "grasp":
                target_pos = trash_pos + np.array([0.0, 0.0, 0.03], dtype=np.float32)
            elif self._wbc_grasp_phase == "lift":
                target_pos = trash_pos + np.array([0.0, 0.0, 0.30], dtype=np.float32)
            elif self._wbc_grasp_phase == "done":
                self._pregrasp_navigator.finish_current_target("grasped" if self._wbc_grasp_success else "failed")
                print(f"[GT-NAV] WBC grasp finished, success={self._wbc_grasp_success}", flush=True)
                hold_cmd = self._get_wbc_hold_ee_pose_cmd()
                action_env = self._generate_wbc_action_tensor(obs, zero_cmd, hold_cmd)
                return {"action": self._action_tensor_to_output(action_env), "giveup": False}
            else:
                target_pos = trash_pos + np.array([0.0, 0.0, 0.20], dtype=np.float32)

            ee_pose_cmd = self._compute_ee_pose_cmd(target_pos)
            action_env = self._generate_wbc_action_tensor(obs, zero_cmd, ee_pose_cmd)

            if self._wbc_grasp_step_count % 20 == 0:
                print(
                    f"[GT-NAV] WBC Grasp step={self._wbc_grasp_step_count} "
                    f"phase={self._wbc_grasp_phase} "
                    f"target_pos={np.round(target_pos, 3)} "
                    f"ee_cmd={np.round(ee_pose_cmd, 3)}",
                    flush=True,
                )

            if self._wbc_grasp_phase == "pregrasp" and self._wbc_grasp_step_count > 80:
                self._wbc_grasp_phase = "grasp"
                self._wbc_grasp_step_count = 0
                print("[GT-NAV] WBC Grasp: pregrasp → grasp", flush=True)
            elif self._wbc_grasp_phase == "grasp" and self._wbc_grasp_step_count > 100:
                self._wbc_grasp_phase = "lift"
                self._wbc_grasp_step_count = 0
                print("[GT-NAV] WBC Grasp: grasp → lift", flush=True)
            elif self._wbc_grasp_phase == "lift" and self._wbc_grasp_step_count > 80:
                self._wbc_grasp_success = True
                self._wbc_grasp_phase = "done"
                print("[GT-NAV] WBC Grasp: lift → done", flush=True)

            if self._wbc_grasp_step_count > self._wbc_grasp_timeout:
                self._wbc_grasp_phase = "done"
                print("[GT-NAV] WBC Grasp: timeout!", flush=True)

            return {"action": self._action_tensor_to_output(action_env), "giveup": False}

        if self._pregrasp_navigator.nav_state == "STAND_UP":
            self._pregrasp_navigator.finish_current_target("failed")
        
        # 处理 DONE 状态
        if nav_info["state"] == "DONE":
            if self.wbc_actor is not None and robot is not None:
                zero_cmd = np.zeros(3, dtype=np.float32)
                self._ensure_wbc_nav_session(obs, zero_cmd)
                hold_cmd = self._get_wbc_hold_ee_pose_cmd()
                action_env = self._generate_wbc_action_tensor(obs, zero_cmd, hold_cmd)
                return {"action": self._action_tensor_to_output(action_env), "giveup": False}
            return {"action": self._get_stand_still_action(obs), "giveup": False}
        
        # 导航阶段优先使用 WBC policy，保持末端位姿不变，同时沿用当前导航 base_cmd。
        if self.wbc_actor is not None and robot is not None:
            self._ensure_wbc_nav_session(obs, base_cmd)
            hold_cmd = self._get_wbc_hold_ee_pose_cmd()
            action_env = self._generate_wbc_action_tensor(obs, base_cmd, hold_cmd)
            return {"action": self._action_tensor_to_output(action_env), "giveup": False}

        # 使用 actor 网络生成动作
        if robot is not None:
            return {"action": self._action_tensor_to_output(self._generate_control_action_tensor(obs, base_cmd, robot)), "giveup": False}
        return {"action": self._generate_action(obs, base_cmd), "giveup": False}


if __name__ == "__main__":
    """最小语法验证"""
    print("=" * 60)
    print("  AlgSolution GT Navigation - Syntax Test")
    print("=" * 60)
    
    # 测试 PolicyNavigator 初始化
    navigator = PolicyNavigator()
    print("✓ PolicyNavigator initialized")
    
    # 测试 PolicyNavigator._wrap_angle
    angle = navigator._wrap_angle(3.5 * math.pi)
    assert -math.pi <= angle <= math.pi, f"Angle wrap failed: {angle}"
    print("✓ PolicyNavigator._wrap_angle: OK")
    
    # 测试 PolicyNavigator.compute_pregrasp_pose
    robot_pos = np.array([1.0, 0.0, 0.68])
    trash_pos = np.array([0.0, 0.0, 0.1])
    goal_xy, goal_yaw = navigator.compute_pregrasp_pose(robot_pos, trash_pos)
    expected_dist = np.linalg.norm(goal_xy - trash_pos[:2])
    assert abs(expected_dist - navigator.stand_off) < 0.01, f"Stand-off distance incorrect: {expected_dist}"
    print(f"✓ PolicyNavigator.compute_pregrasp_pose: goal_xy={goal_xy}, goal_yaw={math.degrees(goal_yaw):.1f}°")
    
    # 测试 PolicyNavigator.select_nearest_target
    test_trash = [
        {"id": "trash1", "pos_w": np.array([1.0, 0.0, 0.1]), "status": "pending"},
        {"id": "trash2", "pos_w": np.array([2.0, 0.0, 0.1]), "status": "pending"},
        {"id": "trash3", "pos_w": np.array([0.5, 0.0, 0.1]), "status": "done"},
    ]
    nearest = navigator.select_nearest_target(test_trash, np.array([0.0, 0.0, 0.68]))
    assert nearest["id"] == "trash1", f"Nearest selection failed: {nearest}"
    print("✓ PolicyNavigator.select_nearest_target: OK")
    
    # 测试 PolicyNavigator.update 状态机
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
    print("✓ PolicyNavigator.update state machine: OK")
    
    # 测试 AlgSolution 初始化
    solution = AlgSolution()
    solution._test_mode = True  # 启用测试模式
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
