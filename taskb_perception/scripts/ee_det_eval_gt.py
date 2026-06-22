#!/usr/bin/env python3
"""
EE detect 反算 pos_b 与仿真 GT 对比（无键盘交互，适合批量验证）。

  cd ATEC2026_Simulation_Challenge-main
  python taskb_perception/scripts/ee_det_eval_gt.py \\
    --task ATEC-TaskB-B2Piper --enable_cameras --headless

输出：
  - config 外参（PerceptionConfig.robot.ee_cam）下的 est vs GT 误差
  - live 外参（仿真 ee_camera 实时位姿）下的误差，用于区分外参/深度问题
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_VERSION = "ee_det_eval_gt_v2"

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="EE detect vs GT base 误差评估")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--policy", type=str, default="demo/policy.pt")
parser.add_argument(
    "--arm-pose",
    type=str,
    default="stowed",
    choices=["stowed", "overhead", "zero"],
)
parser.add_argument("--frames", type=int, default=50, help="评估帧数（settle 之后）")
parser.add_argument("--settle-steps", type=int, default=200)
parser.add_argument("--print-every", type=int, default=10, help="每 N 帧打印一次明细（0=仅汇总）")
parser.add_argument("--verify-only", action="store_true")

_pre, _ = parser.parse_known_args()
if _pre.verify_only:
    print(SCRIPT_VERSION)
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if not hasattr(args_cli, "headless") or args_cli.headless is None:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from taskb_perception.collector_nav import B2LocomotionDriver, resolve_policy_path  # noqa: E402
from taskb_perception.config import PerceptionConfig  # noqa: E402
from taskb_perception.ee_collect_utils import (  # noqa: E402
    EEPresetArmController,
    apply_arm_pose_preset,
    format_ee_camera_view,
    try_bind_viewport_to_ee_camera,
    verify_arm_pose_view,
)
from taskb_perception.ee_det3d import EEDetector  # noqa: E402
from taskb_perception.gt_eval import (  # noqa: E402
    RunningEvalStats,
    format_eval_report,
    match_detections_to_gt,
    read_ee_cam_extrinsic_in_base,
    read_objects_gt_base,
    reevaluate_with_live_extrinsic,
)
from taskb_perception.obs_utils import parse_depth, parse_rgb  # noqa: E402


def _make_action(driver, arm_ctrl, obs, vel):
    leg = driver.compute_action(obs, vel)
    arm = arm_ctrl.compute_arm_action()
    action = leg.clone()
    action[:, arm_ctrl.ARM_IDX] = arm.unsqueeze(0)
    return action


def _settle(env, driver, arm_ctrl, obs, steps):
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
    detector = EEDetector(cfg)
    if not detector.ready:
        print(f"[ERROR] ee_det 未加载: {cfg.yolo_ee_det_weights}")
        return

    print(f"[{SCRIPT_VERSION}] frames={args_cli.frames} arm_pose={args_cli.arm_pose}")
    print(f"[INFO] ee_det: {cfg.yolo_ee_det_weights}")

    driver = B2LocomotionDriver(device=args_cli.device, policy_path=policy_path)
    env = gym.make(args_cli.task, cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1))
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    ee_cam = unwrapped.scene.sensors["ee_camera"]

    obs, _ = env.reset()
    arm_ctrl = EEPresetArmController(robot, args_cli.device)
    apply_arm_pose_preset(robot, arm_ctrl, args_cli.arm_pose)
    obs = _settle(env, driver, arm_ctrl, obs, args_cli.settle_steps)

    try_bind_viewport_to_ee_camera(ee_cam)
    print(f"[arm] pose={args_cli.arm_pose}  target={arm_ctrl.target_joints}")
    verify_arm_pose_view(ee_cam, args_cli.arm_pose)
    print(f"[arm] {format_ee_camera_view(ee_cam)}")

    cam_pos_b, cam_quat_b = read_ee_cam_extrinsic_in_base(unwrapped)
    cfg_pos = cfg.robot.ee_cam.pos_b if cfg.robot.ee_cam else None
    print(
        f"[ee_cam] config pos_b={cfg_pos} | live pos_b="
        f"({cam_pos_b[0]:+.3f},{cam_pos_b[1]:+.3f},{cam_pos_b[2]:+.3f})"
    )
    print(
        f"[ee_cam] config quat={cfg.robot.ee_cam.quat_b if cfg.robot.ee_cam else None} | "
        f"live quat=({cam_quat_b[0]:+.3f},{cam_quat_b[1]:+.3f},{cam_quat_b[2]:+.3f},{cam_quat_b[3]:+.3f})"
    )

    stats_cfg = RunningEvalStats()
    stats_live = RunningEvalStats()
    z = [0.0, 0.0, 0.0]

    for frame in range(args_cli.frames):
        obs, _, term, trunc, _ = env.step(_make_action(driver, arm_ctrl, obs, z))
        ee_rgb = parse_rgb(obs.get("image", {}), "ee_rgb")
        ee_depth = parse_depth(obs.get("image", {}), "ee_depth")
        if ee_rgb is None or ee_depth is None:
            continue

        dets = detector.detect_all(ee_rgb, ee_depth)
        eval_cfg = match_detections_to_gt(dets, read_objects_gt_base(unwrapped))
        eval_live = reevaluate_with_live_extrinsic(dets, unwrapped)
        stats_cfg.update(eval_cfg)
        stats_live.update(eval_live)

        if args_cli.print_every > 0 and frame % args_cli.print_every == 0:
            print(f"\n--- frame {frame} det={len(dets)} ---")
            print(format_eval_report(eval_cfg, live_summary=eval_live))

        if term.item() or trunc.item():
            obs, _ = env.reset()
            apply_arm_pose_preset(robot, arm_ctrl, args_cli.arm_pose)
            obs = _settle(env, driver, arm_ctrl, obs, args_cli.settle_steps)

    print("\n========== 汇总 ==========")
    print("[config ee_cam extrinsic]")
    print(stats_cfg.format())
    print("[live ee_camera extrinsic]")
    print(stats_live.format())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
