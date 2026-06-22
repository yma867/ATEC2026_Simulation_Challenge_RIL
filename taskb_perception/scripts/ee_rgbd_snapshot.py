#!/usr/bin/env python3
"""
EE 相机 RGBD 快照：默认臂 **平收在狗身上**（stowed），视野朝前更开阔；非抓取俯视。

  cd ATEC2026_Simulation_Challenge-main
  python taskb_perception/scripts/ee_rgbd_snapshot.py \\
    --task ATEC-TaskB-B2Piper --enable_cameras

  # 抓取俯视（旧行为，看地面）
  python taskb_perception/scripts/ee_rgbd_snapshot.py \\
    --task ATEC-TaskB-B2Piper --enable_cameras --arm-pose overhead

Isaac 窗口按键（先点一下仿真窗口）:
  P     保存 ee_rgb + ee_depth.npy + 可视化 + depth_meta.json
  F     EE detect 检测并保存带框/置信度/深度图
  C     打印图像中心深度 D（米）
  T     打印探针像素深度
  I/K/J/L/U/O  微调 joint2/3/5
  Z/X  joint1   [ ]  joint4   ; '  joint6
  B    细调     Y    打印并保存关节角   H  帮助
  W/S/A/D  移动狗
  R     reset（默认回到 arm-pose；--no-reset-arm-on-r 则保留当前角度）
  Q/Esc 退出

  # 完全手调（不施加 stowed preset）
  python .../ee_rgbd_snapshot.py --manual-arm

  # 加载上次 Y 保存的角度
  python .../ee_rgbd_snapshot.py --arm-joints snapshots/ee_rgbd/arm_joints_tuned.json

默认 OpenCV 实时显示 ee-det 框 + conf + 真实 depth；--no-show-det 可关闭
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "ee_rgbd_snapshot_v5"

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="EE 相机 RGBD 快照（默认臂收在狗身上）")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--out", type=str, default="snapshots/ee_rgbd")
parser.add_argument("--policy", type=str, default="demo/policy.pt")
parser.add_argument(
    "--arm-pose",
    type=str,
    default="stowed",
    choices=["stowed", "overhead", "zero"],
    help="stowed=臂平收狗身(默认,视野开阔) | overhead=抓取俯视看地面 | zero=全0",
)
parser.add_argument(
    "--arm-joints",
    type=str,
    default=None,
    help="自定义 8 关节(JSON/npy/逗号分隔)，覆盖 --arm-pose",
)
parser.add_argument(
    "--manual-arm",
    action="store_true",
    help="不施加 preset，从 reset 后姿态开始，仅用键盘调",
)
parser.add_argument("--jstep", type=float, default=0.04, help="臂微调步长(rad)")
parser.add_argument(
    "--reset-arm-on-r",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="按 R reset 时是否回到 arm-pose",
)
parser.add_argument(
    "--show-det",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="OpenCV 实时显示 ee-det 框+置信度+depth（默认开）",
)
parser.add_argument("--opencv", action="store_true", help="同 --show-det，兼容旧参数")
parser.add_argument("--probe-u", type=int, default=320, help="探针像素 u")
parser.add_argument("--probe-v", type=int, default=240, help="探针像素 v")
parser.add_argument("--verify-only", action="store_true")

_pre, _ = parser.parse_known_args()
if _pre.verify_only:
    print(f"version={SCRIPT_VERSION}")
    sys.exit(0)


def _resolve_out(out_arg: str) -> Path:
    p = Path(out_arg)
    if p.is_absolute():
        return p.resolve()
    if p.parts and p.parts[0] == "taskb_perception":
        return (REPO_ROOT / p).resolve()
    return (ROOT / p).resolve()


OUT_ROOT = _resolve_out(_pre.out)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input  # noqa: E402
import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from taskb_perception.collector_nav import B2LocomotionDriver, resolve_policy_path  # noqa: E402
from taskb_perception.config import PerceptionConfig  # noqa: E402
from taskb_perception.ee_det3d import EEDetector, draw_ee_detections  # noqa: E402
from taskb_perception.ee_collect_utils import (  # noqa: E402
    ARM_POSE_OVERHEAD,
    ARM_TUNE_HELP,
    EEPresetArmController,
    apply_arm_keyboard_nudge,
    collect_arm_nudge_from_keyboard,
    format_ee_camera_view,
    init_arm_from_cli,
    print_arm_tune_report,
    try_bind_viewport_to_ee_camera,
    verify_arm_pose_view,
)
from taskb_perception.obs_utils import parse_depth, parse_rgb, sample_depth_median  # noqa: E402

Key = carb.input.KeyboardInput
CX, CY = 320, 240


def _depth_stats(depth: np.ndarray) -> dict:
    valid = depth[(depth > 0.05) & (depth < 50.0) & np.isfinite(depth)]
    if valid.size == 0:
        return {"valid_pixels": 0, "min_m": 0.0, "max_m": 0.0, "mean_m": 0.0}
    return {
        "valid_pixels": int(valid.size),
        "min_m": float(valid.min()),
        "max_m": float(valid.max()),
        "mean_m": float(valid.mean()),
    }


def _depth_at(depth: np.ndarray, u: int, v: int, radius: int = 2) -> float:
    return sample_depth_median(depth, u, v, radius=radius)


def _depth_colormap(depth: np.ndarray) -> np.ndarray:
    valid = depth[(depth > 0.05) & np.isfinite(depth)]
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.size == 0:
        return vis
    lo, hi = float(np.percentile(valid, 5)), float(np.percentile(valid, 95))
    if hi <= lo:
        hi = lo + 0.1
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    vis[depth <= 0.05] = (0, 0, 0)
    return vis


def _draw_probe(rgb: np.ndarray, depth: np.ndarray, u: int, v: int) -> np.ndarray:
    vis = rgb.copy()
    h, w = vis.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    d_center = _depth_at(depth, CX, CY)
    d_probe = _depth_at(depth, u, v)
    cv2.drawMarker(vis, (CX, CY), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.circle(vis, (u, v), 7, (255, 80, 0), 2)
    cv2.putText(
        vis,
        f"center({CX},{CY}) D={d_center:.3f}m",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"probe({u},{v}) D={d_probe:.3f}m",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        1,
        cv2.LINE_AA,
    )
    return vis


def _save_snapshot(
    out_root: Path,
    stem: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    probe_u: int,
    probe_v: int,
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    rgb_path = out_root / f"{stem}_ee_rgb.jpg"
    depth_path = out_root / f"{stem}_ee_depth.npy"
    vis_path = out_root / f"{stem}_depth_vis.jpg"
    meta_path = out_root / f"{stem}_depth_meta.json"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(str(depth_path), depth.astype(np.float32))

    overlay = _draw_probe(rgb, depth, probe_u, probe_v)
    depth_vis = _depth_colormap(depth)
    combo = np.hstack([overlay, depth_vis])
    cv2.imwrite(str(vis_path), cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "depth_unit": "meter",
        "center_uv": [CX, CY],
        "center_depth_m": _depth_at(depth, CX, CY),
        "probe_uv": [int(probe_u), int(probe_v)],
        "probe_depth_m": _depth_at(depth, probe_u, probe_v),
        **_depth_stats(depth),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[P] rgb   -> {rgb_path}")
    print(f"[P] depth -> {depth_path}  (load: np.load(...)[v,u] 单位米)")
    print(f"[P] vis   -> {vis_path}")
    print(f"[P] meta  -> center D={meta['center_depth_m']:.4f}m  probe D={meta['probe_depth_m']:.4f}m")
    return out_root


class Keyboard:
    def __init__(self):
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._inp = carb.input.acquire_input_interface()
        self._prev: dict[str, bool] = {}
        self.quit = False
        self.photo = self.reset = self.print_center = self.print_probe = False
        self.run_detect = False
        self.save_det_vis = False
        self.print_tune = False
        self.print_help = False
        self.fine_tune = False
        self.du = self.dv = 0

    def _edge(self, key, name: str) -> bool:
        now = self._inp.get_keyboard_value(self._kb, key) > 0
        prev = self._prev.get(name, False)
        self._prev[name] = now
        return now and not prev

    def _down(self, key) -> bool:
        return self._inp.get_keyboard_value(self._kb, key) > 0

    def update(self):
        self.photo = self.reset = self.print_center = self.print_probe = self.run_detect = False
        self.save_det_vis = False
        self.print_tune = False
        self.print_help = False
        self.du = self.dv = 0
        if self._down(Key.Q) or self._down(Key.ESCAPE):
            self.quit = True
            return
        if self._edge(Key.P, "p"):
            self.photo = True
        if self._edge(Key.F, "f"):
            self.run_detect = True
            self.save_det_vis = True
        if self._edge(Key.R, "r"):
            self.reset = True
        if self._edge(Key.C, "c"):
            self.print_center = True
        if self._edge(Key.T, "t"):
            self.print_probe = True
        if self._edge(Key.Y, "y"):
            self.print_tune = True
        if self._edge(Key.H, "h"):
            self.print_help = True
        if self._edge(Key.B, "b"):
            self.fine_tune = not self.fine_tune
        # 探针：方向键
        step = 5
        if self._down(Key.UP):
            self.dv -= step
        if self._down(Key.DOWN):
            self.dv += step
        if self._down(Key.LEFT):
            self.du -= step
        if self._down(Key.RIGHT):
            self.du += step

    def velocity(self):
        vx = wz = 0.0
        if self._down(Key.W):
            vx = 0.35
        if self._down(Key.S):
            vx = -0.2
        if self._down(Key.A):
            wz = 0.5
        if self._down(Key.D):
            wz = -0.5
        return [vx, 0.0, wz]


def _make_action(driver, arm_ctrl, obs, vel):
    leg = driver.compute_action(obs, vel)
    arm = arm_ctrl.compute_arm_action()
    action = leg.clone()
    action[:, arm_ctrl.ARM_IDX] = arm.unsqueeze(0)
    return action


def _settle(env, driver, arm_ctrl, obs, steps=200):
    z = [0.0, 0.0, 0.0]
    for _ in range(steps):
        obs, _, _, _, _ = env.step(_make_action(driver, arm_ctrl, obs, z))
    return obs


def _apply_arm_pose(arm_ctrl: EEPresetArmController, robot, pose: str, arm_joints: str | None) -> np.ndarray:
    if arm_joints:
        return init_arm_from_cli(robot, arm_ctrl, arm_pose=pose, arm_joints=arm_joints, manual_arm=False)
    return init_arm_from_cli(robot, arm_ctrl, arm_pose=pose, manual_arm=False)


def _print_detections(dets) -> None:
    if not dets:
        print("[det] 无检测目标")
        return
    for i, d in enumerate(dets):
        print(
            f"[det][{i}] {d.obj_class} conf={d.confidence:.2f} "
            f"uv=({d.u:.0f},{d.v:.0f}) depth={d.depth_m:.3f}m "
            f"pos_b=({d.pos_b[0]:.2f},{d.pos_b[1]:.2f},{d.pos_b[2]:.2f})"
        )


def _save_det_vis(out_root: Path, stem: str, rgb: np.ndarray, dets) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    vis = draw_ee_detections(rgb, dets, show_pos_b=True)
    path = out_root / f"{stem}_ee_det.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[det] 可视化 -> {path}")
    return path


def main():
    policy_path = resolve_policy_path(args_cli.policy, REPO_ROOT)
    if policy_path is None:
        print(f"[ERROR] 找不到 policy: {REPO_ROOT / 'demo' / 'policy.pt'}")
        return

    print(f"[{SCRIPT_VERSION}] out={OUT_ROOT}  arm_pose={args_cli.arm_pose}")

    driver = B2LocomotionDriver(device=args_cli.device, policy_path=policy_path)
    kb = Keyboard()
    env = gym.make(args_cli.task, cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1))
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    ee_cam = unwrapped.scene.sensors["ee_camera"]

    obs, _ = env.reset()
    arm_ctrl = EEPresetArmController(robot, args_cli.device)
    init_arm_from_cli(
        robot,
        arm_ctrl,
        arm_pose=args_cli.arm_pose,
        arm_joints=args_cli.arm_joints,
        manual_arm=args_cli.manual_arm,
    )
    ee_nav = EEDetector(PerceptionConfig())
    obs = _settle(env, driver, arm_ctrl, obs, steps=200)

    arm_tune_path = OUT_ROOT / "arm_joints_tuned.json"

    prim = try_bind_viewport_to_ee_camera(ee_cam)
    print(f"[view] Isaac 窗口 = ee_camera 视角" + (f" ({prim})" if prim else ""))
    print(f"[arm] pose={args_cli.arm_pose}  target={arm_ctrl.target_joints}")
    verify_arm_pose_view(ee_cam, args_cli.arm_pose)
    print(f"[arm] {format_ee_camera_view(ee_cam)}")
    if args_cli.arm_pose == ARM_POSE_OVERHEAD:
        print("[hint] overhead=抓取俯视地面；要开阔视野请用默认 --arm-pose stowed")
    else:
        print("[hint] 用 I/K/J/L/U/O/Z/X/[ ]/;' 调 EE 视角，满意后按 Y 保存关节角")
    print(ARM_TUNE_HELP.strip())

    probe_u, probe_v = args_cli.probe_u, args_cli.probe_v
    snap_idx = 0
    det_idx = 0
    mouse_uv = [probe_u, probe_v]
    mouse_d = 0.0
    show_det = args_cli.show_det or args_cli.opencv

    if show_det:
        cv2.namedWindow("ee_det_rgbd", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ee_det_rgbd", 1280, 480)

        def _on_mouse(event, x, y, _flags, _param):
            nonlocal mouse_uv, mouse_d
            if event == cv2.EVENT_MOUSEMOVE and x < 640:
                mouse_uv = [x, y]

        cv2.setMouseCallback("ee_det_rgbd", _on_mouse)

    print("=" * 60)
    print(f"  EE RGBD  arm_pose={args_cli.arm_pose}")
    print("  P=拍照  C=中心D  T=探针D  F=detect+存图  方向键=移探针")
    print("  I/K/J/L/U/O/Z/X/[ ]/;' 调臂  B=细调  Y=保存角度  H=帮助")
    print("  W/S/A/D 移狗  R=reset  Q=退出")
    if show_det:
        print("  OpenCV：左=ee-det框+conf+D  右=depth伪彩  鼠标看D")
    print("=" * 60)

    while simulation_app.is_running():
        kb.update()
        if kb.quit:
            break

        if kb.reset:
            obs, _ = env.reset()
            if args_cli.reset_arm_on_r and not args_cli.manual_arm:
                _apply_arm_pose(arm_ctrl, robot, args_cli.arm_pose, args_cli.arm_joints)
            elif args_cli.arm_joints:
                init_arm_from_cli(
                    robot, arm_ctrl, arm_pose=args_cli.arm_pose, arm_joints=args_cli.arm_joints
                )
            obs = _settle(env, driver, arm_ctrl, obs, steps=200)
            print(f"[reset] 臂 target={arm_ctrl.target_joints}")

        if kb.print_help:
            print(ARM_TUNE_HELP.strip())
        if kb.print_tune:
            print_arm_tune_report(arm_ctrl, ee_cam, save_path=arm_tune_path)

        arm_deltas = collect_arm_nudge_from_keyboard(
            kb._inp,
            kb._kb,
            Key,
            step=args_cli.jstep,
            fine=kb.fine_tune,
        )
        apply_arm_keyboard_nudge(
            arm_ctrl,
            arm_deltas,
            overhead=args_cli.arm_pose == ARM_POSE_OVERHEAD,
        )

        probe_u = int(np.clip(probe_u + kb.du, 0, 639))
        probe_v = int(np.clip(probe_v + kb.dv, 0, 479))

        obs, _, term, trunc, _ = env.step(_make_action(driver, arm_ctrl, obs, kb.velocity()))

        image_obs = obs.get("image", {})
        rgb = parse_rgb(image_obs, "ee_rgb")
        depth = parse_depth(image_obs, "ee_depth")

        if rgb is not None and depth is not None:
            dets_live: list = []
            if show_det and ee_nav.ready:
                dets_live = ee_nav.detect_all(rgb, depth)

            if kb.print_center:
                d = _depth_at(depth, CX, CY)
                print(f"[C] center ({CX},{CY})  D = {d:.4f} m  (raw depth[{CY},{CX}]={depth[CY, CX]:.4f})")
            if kb.print_probe:
                d = _depth_at(depth, probe_u, probe_v)
                print(
                    f"[T] probe ({probe_u},{probe_v})  D = {d:.4f} m  "
                    f"(raw depth[{probe_v},{probe_u}]={depth[probe_v, probe_u]:.4f})"
                )
            if kb.run_detect and ee_nav.ready:
                dets = ee_nav.detect_all(rgb, depth)
                _print_detections(dets)
                if kb.save_det_vis:
                    stem = f"{det_idx:04d}"
                    _save_det_vis(OUT_ROOT, stem, rgb, dets)
                    det_idx += 1
            elif kb.run_detect:
                print("[det] ee_det 权重未加载")
            if kb.photo:
                stem = f"{snap_idx:04d}"
                _save_snapshot(OUT_ROOT, stem, rgb, depth, probe_u, probe_v)
                snap_idx += 1

            if show_det:
                mx, my = mouse_uv
                mouse_d = _depth_at(depth, mx, my)
                if ee_nav.ready:
                    left = draw_ee_detections(rgb, dets_live, show_pos_b=False)
                else:
                    left = _draw_probe(rgb, depth, probe_u, probe_v)
                panel = np.hstack([left, _depth_colormap(depth)])
                cv2.putText(
                    panel,
                    f"arm fine={kb.fine_tune}  {format_ee_camera_view(ee_cam)}",
                    (8, panel.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (200, 255, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    f"mouse({mx},{my}) D={mouse_d:.3f}m",
                    (8, panel.shape[0] - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("ee_det_rgbd", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        if term.item() or trunc.item():
            obs, _ = env.reset()
            if args_cli.reset_arm_on_r and not args_cli.manual_arm:
                init_arm_from_cli(
                    robot,
                    arm_ctrl,
                    arm_pose=args_cli.arm_pose,
                    arm_joints=args_cli.arm_joints,
                    manual_arm=False,
                )
            obs = _settle(env, driver, arm_ctrl, obs, steps=200)

    if show_det:
        cv2.destroyAllWindows()
    env.close()
    simulation_app.close()
    print(f"完成。快照目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
