#!/usr/bin/env python3
"""
EE/Head 远距离导航 — 默认光轴中线对准：2D 框中心 → u=cx → 直行（不反算 base 3D）

  python taskb_perception/scripts/ee_det_nav.py \\
    --task ATEC-TaskB-B2Piper --enable_cameras

  # 旧方式：depth 反算 pos_b
  python .../ee_det_nav.py --nav-mode pos_b_3d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_VERSION = "ee_det_nav_v6"

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="EE detect → base 导航（方式一）")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--policy", type=str, default="demo/policy.pt")
parser.add_argument(
    "--arm-pose",
    type=str,
    default="stowed",
    choices=["stowed", "overhead", "zero"],
)
parser.add_argument(
    "--print-every",
    type=int,
    default=20,
    help="每 N 帧打印一次全部 pos_b（0=仅按 F）",
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
    help="不施加 preset，从 reset 后姿态开始手调",
)
parser.add_argument("--jstep", type=float, default=0.04, help="臂微调步长(rad)")
parser.add_argument(
    "--settle-steps",
    type=int,
    default=200,
    help="reset 后臂到位 settle 步数",
)
parser.add_argument(
    "--nav-mode",
    type=str,
    default="axis_align",
    choices=["axis_align", "pos_b_3d"],
    help="axis_align=光轴中线对准+直行 | pos_b_3d=depth反算base(旧)",
)
parser.add_argument(
    "--gt-eval",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="pos_b_3d 模式下与 GT 对比（axis 模式默认关）",
)
parser.add_argument("--verify-only", action="store_true")

_pre, _ = parser.parse_known_args()
if _pre.verify_only:
    print(SCRIPT_VERSION)
    sys.exit(0)

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

from taskb_perception.collector_nav import (  # noqa: E402
    AxisNavController,
    B2LocomotionDriver,
    draw_axis_nav_on_rgb,
    resolve_policy_path,
)
from taskb_perception.config import PerceptionConfig  # noqa: E402
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
from taskb_perception.ee_det3d import EEDetector, draw_ee_detections  # noqa: E402
from taskb_perception.gt_eval import (  # noqa: E402
    RunningEvalStats,
    draw_gt_eval_on_rgb,
    format_eval_report,
    match_detections_to_gt,
    read_objects_gt_base,
    reevaluate_with_live_extrinsic,
)
from taskb_perception.obs_utils import parse_depth, parse_rgb  # noqa: E402

Key = carb.input.KeyboardInput


def print_axis_targets(
    all_targets: list,
    nav_target,
    axis_nav: AxisNavController | None = None,
    prefix: str = "[axis-nav]",
) -> None:
    ee_targets = [t for t in all_targets if t.source == "ee"]
    print(f"{prefix} YOLO 检出 {len(all_targets)} 个 (ee={len(ee_targets)})")
    if axis_nav is not None:
        lock = axis_nav.nav_lock
        if lock is not None:
            print(
                f"{prefix} LOCK#{lock.lock_id} {lock.obj_class} "
                f"conf={lock.lock_confidence:.2f} missed={lock.missed_frames} "
                f"done={axis_nav.completed_count}"
            )
        elif axis_nav.completed_count:
            print(f"{prefix} 无当前锁，已完成 {axis_nav.completed_count} 个")
    if not all_targets:
        print(f"{prefix} 无目标（视野外 / conf 过低 / 权重未加载）")
        return
    for i, t in enumerate(all_targets):
        mark = " ← NAV" if nav_target is not None and t is nav_target else ""
        lock_tag = " [LOCK]" if getattr(t, "locked", False) else ""
        print(
            f"{prefix}[{i}] {t.obj_class} conf={t.confidence:.2f} src={t.source} "
            f"uv=({t.u:.0f},{t.v:.0f}) err_u={t.err_u:+.1f}px{lock_tag}{mark}"
        )


def print_detections_base(dets, prefix: str = "[ee-det]") -> None:
    if not dets:
        print(f"{prefix} 无目标")
        return
    for i, d in enumerate(dets):
        print(
            f"{prefix}[{i}] {d.obj_class} conf={d.confidence:.2f} "
            f"bbox={d.bbox} depth={d.depth_m:.3f}m "
            f"pos_b=({d.pos_b[0]:+.2f}, {d.pos_b[1]:+.2f}, {d.pos_b[2]:+.2f}) "
            f"dist_xy={d.distance_xy():.2f}m"
        )


def goto_vel_from_pos_b(pos_b: np.ndarray, arrive: float = 1.3) -> list[float]:
    """pos_b → [vx, vy, wz] body 系（给 B2 policy）。"""
    x, y = float(pos_b[0]), float(pos_b[1])
    dist = np.hypot(x, y)
    if dist < arrive:
        return [0.0, 0.0, 0.0]
    angle = np.arctan2(y, x)
    max_vx, max_wz = 0.55, 0.75
    wz = float(np.clip(angle * 1.2, -max_wz, max_wz))
    vx = 0.15 if abs(angle) > 0.8 else float(np.clip(0.25 + 0.35 * dist, 0.2, max_vx))
    return [vx, 0.0, wz]


class Keyboard:
    def __init__(self):
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._inp = carb.input.acquire_input_interface()
        self._prev: dict[str, bool] = {}
        self.quit = False
        self.print_now = False
        self.toggle_nav = False
        self.complete_target = False
        self.print_tune = False
        self.print_help = False
        self.fine_tune = False

    def _edge(self, key, name: str) -> bool:
        now = self._inp.get_keyboard_value(self._kb, key) > 0
        prev = self._prev.get(name, False)
        self._prev[name] = now
        return now and not prev

    def update(self):
        self.print_now = False
        self.toggle_nav = False
        self.complete_target = False
        self.print_tune = False
        self.print_help = False
        if self._inp.get_keyboard_value(self._kb, Key.Q) > 0:
            self.quit = True
            return
        if self._edge(Key.F, "f"):
            self.print_now = True
        if self._edge(Key.N, "n"):
            self.toggle_nav = True
        if self._edge(Key.C, "c"):
            self.complete_target = True
        if self._edge(Key.Y, "y"):
            self.print_tune = True
        if self._edge(Key.H, "h"):
            self.print_help = True
        if self._edge(Key.B, "b"):
            self.fine_tune = not self.fine_tune

    def velocity(self) -> list[float]:
        vx = wz = 0.0
        if self._inp.get_keyboard_value(self._kb, Key.W) > 0:
            vx = 0.45
        if self._inp.get_keyboard_value(self._kb, Key.S) > 0:
            vx = -0.25
        if self._inp.get_keyboard_value(self._kb, Key.A) > 0:
            wz = 0.55
        if self._inp.get_keyboard_value(self._kb, Key.D) > 0:
            wz = -0.55
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


def main():
    policy_path = resolve_policy_path(args_cli.policy, REPO_ROOT)
    if policy_path is None:
        print("[ERROR] 找不到 demo/policy.pt")
        return

    cfg = PerceptionConfig()
    cfg.nav_method = args_cli.nav_mode
    use_axis = args_cli.nav_mode == "axis_align"
    axis_nav = AxisNavController(cfg)
    ee_detector = EEDetector(cfg) if not use_axis else None
    if use_axis:
        print(f"[{SCRIPT_VERSION}] 光轴中线导航 axis_align (ee 2D → 锁定 → u=cx → 直行)")
    else:
        if ee_detector is None or not ee_detector.ready:
            print(f"[ERROR] ee_det 未加载: {cfg.yolo_ee_det_weights}")
            return
        print(f"[{SCRIPT_VERSION}] pos_b_3d 导航 (depth 反算 base)")
    print(f"[INFO] nav_source={cfg.nav_detect_source} lock={cfg.nav_lock_enabled}")

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
    obs = _settle(env, driver, arm_ctrl, obs, steps=args_cli.settle_steps)

    arm_tune_path = ROOT / "snapshots" / "ee_rgbd" / "arm_joints_tuned.json"

    prim = try_bind_viewport_to_ee_camera(ee_cam)
    print(f"[view] Isaac 窗口 = ee_camera" + (f" ({prim})" if prim else ""))
    print(f"[arm] pose={args_cli.arm_pose}  target={arm_ctrl.target_joints}")
    verify_arm_pose_view(ee_cam, args_cli.arm_pose)
    print(f"[arm] {format_ee_camera_view(ee_cam)}")

    cv2.namedWindow("ee_det_nav", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ee_det_nav", 960, 480)

    frame = 0
    auto_nav = False
    last_dets = []
    nav_target = None
    gt_stats = RunningEvalStats()
    last_eval = None

    nav_hint = "光轴对准+直行" if use_axis else "pos_b 转向"
    print("=" * 60)
    print(f"  F=打印导航状态  N=自动导航({nav_hint})  C=完成当前目标/下一个  W/S/A/D=移狗  Q=退出")
    print("  I/K/J/L…=调臂  Y=保存角度  B=细调  H=帮助")
    if args_cli.gt_eval and not use_axis:
        print("  GT=品红圈=GT投影")
    print("=" * 60)

    while simulation_app.is_running():
        kb.update()
        if kb.quit:
            break
        if kb.toggle_nav:
            auto_nav = not auto_nav
            print(f"[nav] auto_nav={auto_nav} mode={args_cli.nav_mode}")
            if auto_nav and use_axis:
                axis_nav.detect_targets(obs, manage_lock=False)
                if axis_nav.nav_lock is None:
                    axis_nav.acquire_lock(axis_nav.last_targets)

        if kb.complete_target and use_axis:
            axis_nav.complete_current()
            print(f"[nav] 手动完成当前目标，已完成 {axis_nav.completed_count} 个")
            if auto_nav:
                axis_nav.acquire_lock(axis_nav.last_targets)

        if kb.print_help:
            print(ARM_TUNE_HELP.strip())
        if kb.print_tune:
            print_arm_tune_report(arm_ctrl, ee_cam, save_path=arm_tune_path)

        apply_arm_keyboard_nudge(
            arm_ctrl,
            collect_arm_nudge_from_keyboard(
                kb._inp, kb._kb, Key, step=args_cli.jstep, fine=kb.fine_tune
            ),
            overhead=args_cli.arm_pose == ARM_POSE_OVERHEAD,
        )

        vel = kb.velocity()
        nav_vel = None
        if auto_nav:
            if use_axis:
                nav_vel = axis_nav.compute_velocity(obs)
                vel = [float(nav_vel[0]), float(nav_vel[1]), float(nav_vel[2])]
            elif last_dets:
                vel = goto_vel_from_pos_b(last_dets[0].pos_b)

        obs, _, term, trunc, _ = env.step(_make_action(driver, arm_ctrl, obs, vel))

        image_obs = obs.get("image", {})
        ee_rgb = parse_rgb(image_obs, "ee_rgb")
        ee_depth = parse_depth(image_obs, "ee_depth")
        head_rgb = parse_rgb(image_obs, "head_rgb")
        last_dets = []
        last_eval = None
        gt_all: list = []
        all_axis = []

        if use_axis:
            if auto_nav:
                nav_target = axis_nav.nav_target
            else:
                axis_nav.detect_targets(obs, manage_lock=False)
                nav_target = axis_nav.nav_target
            all_axis = axis_nav.last_targets
            if kb.print_now or (args_cli.print_every > 0 and frame % args_cli.print_every == 0):
                print_axis_targets(all_axis, nav_target, axis_nav=axis_nav)
                if auto_nav and nav_target and nav_vel is not None:
                    print(f"[vel] vx={nav_vel[0]:.2f} wz={nav_vel[2]:.2f}")
        elif ee_rgb is not None and ee_depth is not None and ee_detector is not None:
            last_dets = ee_detector.detect_all(ee_rgb, ee_depth)
            if args_cli.gt_eval:
                gt_all = read_objects_gt_base(unwrapped)
                last_eval = match_detections_to_gt(last_dets, gt_all)
                gt_stats.update(last_eval)
            if kb.print_now or (args_cli.print_every > 0 and frame % args_cli.print_every == 0):
                print_detections_base(last_dets)
                if args_cli.gt_eval and last_eval is not None:
                    live_eval = reevaluate_with_live_extrinsic(last_dets, unwrapped) if last_dets else None
                    print(format_eval_report(last_eval, live_summary=live_eval))
                    print(gt_stats.format())

        vis_rgb = ee_rgb
        vis_src = "ee"
        if use_axis:
            if nav_target is not None:
                vis_src = nav_target.source
            elif cfg.nav_detect_source.lower() == "head":
                vis_src = "head"
            vis_rgb = head_rgb if vis_src == "head" and head_rgb is not None else ee_rgb
        if vis_rgb is not None:
            if use_axis:
                vis = draw_axis_nav_on_rgb(
                    vis_rgb,
                    nav_target,
                    all_targets=all_axis if use_axis else None,
                    source=vis_src,
                )
            elif args_cli.gt_eval and last_eval is not None:
                vis = draw_gt_eval_on_rgb(vis_rgb, last_dets, last_eval, gt_all, ee_cam)
            else:
                vis = draw_ee_detections(vis_rgb, last_dets, show_pos_b=True)
            if not use_axis and last_dets:
                t = last_dets[0]
                cv2.putText(
                    vis,
                    f"pos_b=({t.pos_b[0]:.2f},{t.pos_b[1]:.2f}) d={t.distance_xy():.2f}m",
                    (8, vis.shape[0] - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                vis,
                f"fine={kb.fine_tune}  {format_ee_camera_view(ee_cam)}",
                (8, vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("ee_det_nav", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        frame += 1
        if term.item() or trunc.item():
            obs, _ = env.reset()
            axis_nav.reset()
            if not args_cli.manual_arm:
                init_arm_from_cli(
                    robot,
                    arm_ctrl,
                    arm_pose=args_cli.arm_pose,
                    arm_joints=args_cli.arm_joints,
                    manual_arm=False,
                )
            obs = _settle(env, driver, arm_ctrl, obs, steps=args_cli.settle_steps)

    if args_cli.gt_eval:
        print(gt_stats.format())
    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
