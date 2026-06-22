#!/usr/bin/env python3
"""
EE 相机俯拍采集（YOLO-Seg）。臂前伸到狗前方看地面，窗口为狗上方跟拍。

  python taskb_perception/scripts/manual_collect_ee.py \\
    --task ATEC-TaskB-B2Piper --enable_cameras \\
    --out taskb_perception/dataset_ee_seg --resume

按键:
  W/S/A/D  移动狗   I/K 臂抬高/放低   J/L 前伸   U/O 左右偏
  P 存 ee_rgb   R reset   Q 退出
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SCRIPT_VERSION = "manual_collect_ee_v4"

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

if ".local/share/Trash" in str(REPO_ROOT) or "/Trash/" in str(REPO_ROOT):
    print("[ERROR] 不要在 Trash 回收站里运行")
    sys.exit(1)

sys.path.insert(0, str(ROOT))


def _resolve_out(out_arg: str) -> Path:
    p = Path(out_arg)
    if p.is_absolute():
        return p.resolve()
    if p.parts and p.parts[0] == "taskb_perception":
        return (REPO_ROOT / p).resolve()
    return (ROOT / p).resolve()


def _init_output_dirs(out_root: Path) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "depth/train", "depth/val"):
        os.makedirs(out_root / sub, exist_ok=True)


def _write_dataset_yaml(out_root: Path) -> None:
    (out_root / "dataset.yaml").write_text(
        f"path: {out_root}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: sugar\n  1: mustard\n  2: banana\n",
        encoding="utf-8",
    )


def _next_image_index(out_root: Path, split: str) -> int:
    img_dir = out_root / "images" / split
    max_idx = -1
    for p in img_dir.glob("*.jpg"):
        m = re.match(r"^(\d+)$", p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


parser = argparse.ArgumentParser(description="EE top-down 90deg dataset collection for YOLO-Seg.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--out", type=str, default="dataset_ee_seg")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
parser.add_argument("--policy", type=str, default="demo/policy.pt")
parser.add_argument("--vx", type=float, default=0.45)
parser.add_argument("--wz", type=float, default=0.60)
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--start-index", type=int, default=None)
parser.add_argument("--images-only", action="store_true", default=True)
parser.add_argument("--save-depth", action="store_true", help="同时存 depth/*.npy (米)")
parser.add_argument(
    "--arm-mode",
    type=str,
    default="preset",
    choices=["preset", "ik"],
    help="preset=关节前伸预设(默认,不趴狗背); ik=笛卡尔IK",
)
parser.add_argument("--enforce-topdown", action="store_true")
parser.add_argument("--down-tol-deg", type=float, default=15.0)
parser.add_argument("--ee-x", type=float, default=0.62, help="仅 --arm-mode ik")
parser.add_argument("--ee-y", type=float, default=0.0, help="仅 --arm-mode ik")
parser.add_argument("--ee-z", type=float, default=-0.30, help="仅 --arm-mode ik，负=低于背部往前")
parser.add_argument("--view-height", type=float, default=4.0)
parser.add_argument("--verify-only", action="store_true")

_pre, _ = parser.parse_known_args()
if _pre.verify_only:
    p = Path(__file__).resolve()
    print(f"version={SCRIPT_VERSION} path={p}")
    sys.exit(0)

OUT_ROOT = _resolve_out(_pre.out)
_init_output_dirs(OUT_ROOT)
_write_dataset_yaml(OUT_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from taskb_perception.collector_nav import B2LocomotionDriver, resolve_policy_path  # noqa: E402
from taskb_perception.ee_collect_utils import (  # noqa: E402
    EEIkArmController,
    EEPresetArmController,
    EETopdownTarget,
    ee_camera_down_angle_deg,
    ee_camera_is_topdown,
)
from taskb_perception.obs_utils import parse_depth, parse_rgb  # noqa: E402

Key = carb.input.KeyboardInput


def _viewport_follow_robot_top(env, height: float) -> None:
    """窗口 = 机器人正上方俯视跟拍（不是 ee_camera 画面）。"""
    unwrapped = env.unwrapped
    vpc = getattr(unwrapped, "viewport_camera_controller", None)
    if vpc is None:
        return
    robot = unwrapped.scene["robot"]
    rpos = robot.data.root_pos_w[0].detach().cpu().numpy()
    eye = rpos.copy()
    eye[2] = float(rpos[2]) + height
    lookat = rpos.copy()
    lookat[2] = max(float(rpos[2]), 0.05)
    vpc.update_view_location(eye=eye, lookat=lookat)


def _make_action(driver, arm_ctrl, obs, vel_cmd):
    leg_action = driver.compute_action(obs, vel_cmd)
    arm_action = arm_ctrl.compute_arm_action()
    action = leg_action.clone()
    action[:, arm_ctrl.ARM_IDX] = arm_action.unsqueeze(0)
    return action


def _settle_arm(env, driver, arm_ctrl, obs, steps: int = 80):
    zero = [0.0, 0.0, 0.0]
    for _ in range(steps):
        obs, _, _, _, _ = env.step(_make_action(driver, arm_ctrl, obs, zero))
    return obs


def _save_rgb(out_root: Path, split: str, stem: str, rgb: np.ndarray) -> Path:
    import cv2

    path = out_root / "images" / split / f"{stem}.jpg"
    if path.exists():
        raise FileExistsError(path)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def _save_depth(out_root: Path, split: str, stem: str, depth: np.ndarray) -> Path:
    path = out_root / "depth" / split / f"{stem}.npy"
    if path.exists():
        raise FileExistsError(path)
    np.save(path, depth.astype(np.float32))
    return path


class KeyboardController:
    def __init__(self):
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._inp = carb.input.acquire_input_interface()
        self._prev = {k: False for k in ("p", "r", "i", "k", "j", "l", "u", "o")}
        self.quit = self.reset = self.photo = False
        self.nudge = np.zeros(3, dtype=np.float32)

    def _edge(self, key, name: str) -> bool:
        now = self._inp.get_keyboard_value(self._kb, key) > 0
        fired = now and not self._prev[name]
        self._prev[name] = now
        return fired

    def _down(self, key) -> bool:
        return self._inp.get_keyboard_value(self._kb, key) > 0

    def update(self):
        self.reset = self.photo = False
        self.nudge[:] = 0.0
        if self._down(Key.Q) or self._down(Key.ESCAPE):
            self.quit = True
            return
        if self._edge(Key.P, "p"):
            self.photo = True
        if self._edge(Key.R, "r"):
            self.reset = True
        step = 0.03
        if self._down(Key.I):
            self.nudge[0] += step  # j2 抬高
        if self._down(Key.K):
            self.nudge[0] -= step
        if self._down(Key.J):
            self.nudge[1] -= step  # 少前伸
        if self._down(Key.L):
            self.nudge[1] += step  # 多前伸
        if self._down(Key.U):
            self.nudge[2] += step  # j1 左转
        if self._down(Key.O):
            self.nudge[2] -= step

    def velocity_command(self):
        vx = wz = 0.0
        if self._down(Key.W):
            vx += args_cli.vx
        if self._down(Key.S):
            vx -= args_cli.vx * 0.6
        if self._down(Key.A):
            wz += args_cli.wz
        if self._down(Key.D):
            wz -= args_cli.wz
        return [vx, 0.0, wz]


def main():
    policy_path = resolve_policy_path(args_cli.policy, REPO_ROOT)
    if policy_path is None:
        print(f"[ERROR] 找不到 policy: {REPO_ROOT / 'demo' / 'policy.pt'}")
        return

    next_idx = (
        args_cli.start_index
        if args_cli.start_index is not None
        else (_next_image_index(OUT_ROOT, args_cli.split) if args_cli.resume else 0)
    )
    print(f"[{SCRIPT_VERSION}] 从编号 {next_idx:06d} 开始  out={OUT_ROOT}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    # 使用官方 ee_camera 外参，不再 patch

    driver = B2LocomotionDriver(device=args_cli.device, policy_path=policy_path)
    kb = KeyboardController()
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    ee_cam = unwrapped.scene.sensors["ee_camera"]

    if args_cli.arm_mode == "preset":
        arm_ctrl = EEPresetArmController(robot, args_cli.device)
    else:
        ee_target = EETopdownTarget(
            pos_b=np.array([args_cli.ee_x, args_cli.ee_y, args_cli.ee_z], dtype=np.float32)
        )
        arm_ctrl = EEIkArmController(robot, args_cli.device, ee_target)

    obs, _ = env.reset()
    if args_cli.arm_mode == "ik":
        arm_ctrl.ik.reset()
    obs = _settle_arm(env, driver, arm_ctrl, obs, steps=120)
    session_saved = 0

    angle0 = ee_camera_down_angle_deg(ee_cam)
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    ee_w = robot.data.body_pos_w[0, ee_body_id].detach().cpu().numpy()
    print(f"[view] 狗上方俯视跟拍 {args_cli.view_height}m（窗口≠ee 画面）")
    print(f"[arm] mode={args_cli.arm_mode}  EE世界={ee_w}  俯角={angle0:.1f}°")
    if args_cli.arm_mode == "preset":
        print(f"[arm] 关节预设: {arm_ctrl.target_joints}")
    else:
        print(f"[arm] IK base 目标: {arm_ctrl.target.pos_b}")
    _viewport_follow_robot_top(env, args_cli.view_height)

    print("=" * 60)
    print(f"  {SCRIPT_VERSION}  EE 俯拍采集")
    print("  臂伸到狗前方看地面；I/K 抬臂  J/L 前伸")
    print("  W/S/A/D 移动   P 存 ee_rgb")
    print("=" * 60)

    while simulation_app.is_running():
        kb.update()
        if kb.quit:
            break
        if kb.reset:
            obs, _ = env.reset()
            if args_cli.arm_mode == "ik":
                arm_ctrl.ik.reset()
            obs = _settle_arm(env, driver, arm_ctrl, obs, steps=100)
            print("[reset]")

        if np.any(kb.nudge != 0):
            if args_cli.arm_mode == "preset":
                arm_ctrl.nudge(
                    d_j2=kb.nudge[0],
                    d_j3=-kb.nudge[1] * 0.4,
                    d_j1=kb.nudge[2],
                )
            else:
                arm_ctrl.nudge_target(kb.nudge[1], kb.nudge[2], kb.nudge[0])

        obs, _, term, trunc, _ = env.step(_make_action(driver, arm_ctrl, obs, kb.velocity_command()))
        _viewport_follow_robot_top(env, args_cli.view_height)

        angle = ee_camera_down_angle_deg(ee_cam)
        if kb.photo:
            rgb = parse_rgb(obs.get("image", {}), "ee_rgb")
            if rgb is None:
                print("[P] 无 ee_rgb")
                continue
            if args_cli.enforce_topdown:
                ok, _ = ee_camera_is_topdown(ee_cam, tol_deg=args_cli.down_tol_deg)
                if not ok:
                    print(f"[P] 跳过：俯角 {angle:.1f}° > {args_cli.down_tol_deg}°（去掉 --enforce-topdown 可强制存）")
                    continue
            stem = f"{next_idx:06d}"
            try:
                img_p = _save_rgb(OUT_ROOT, args_cli.split, stem, rgb)
                msg = f"[P] {stem}.jpg  down_angle={angle:.1f}°"
                if args_cli.arm_mode == "preset":
                    msg += f"  joints={arm_ctrl.target_joints[:3]}..."
                else:
                    msg += f"  ee={arm_ctrl.target.pos_b}"
                if args_cli.save_depth:
                    depth = parse_depth(obs.get("image", {}), "ee_depth")
                    if depth is not None:
                        _save_depth(OUT_ROOT, args_cli.split, stem, depth)
                        msg += " +depth"
                print(msg)
                next_idx += 1
                session_saved += 1
            except FileExistsError as exc:
                print(f"[P] {exc}")
                next_idx += 1

        if term.item() or trunc.item():
            obs, _ = env.reset()
            if args_cli.arm_mode == "ik":
                arm_ctrl.ik.reset()
            obs = _settle_arm(env, driver, arm_ctrl, obs, steps=80)

    env.close()
    simulation_app.close()
    print(f"本次 {session_saved} 张，下一编号 {next_idx:06d}")
    print(f"训练 Seg: python taskb_perception/scripts/train_yolo_seg.py --data {OUT_ROOT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
