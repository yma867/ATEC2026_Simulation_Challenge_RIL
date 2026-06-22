#!/usr/bin/env python3
"""
手动采集：WASD 移动，P 拍照。

  # 自己用 LabelImg 标注（只存图，不覆盖旧图，断点续编号）
  python taskb_perception/scripts/manual_collect.py \\
    --task ATEC-TaskB-B2Piper --enable_cameras \\
    --view head --images-only --resume \\
    --out taskb_perception/dataset_head

按键: W/S 前后  A/D 转向  P 拍照  R reset  Q/Esc 退出
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SCRIPT_VERSION = "manual_collect_v5"

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

if ".local/share/Trash" in str(REPO_ROOT) or "/Trash/" in str(REPO_ROOT):
    print("[ERROR] 不要在 Trash 回收站里运行，请移到 ~/ATEC2026_Simulation_Challenge")
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
    out_root = out_root.resolve()
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(out_root / sub, exist_ok=True)


def _write_dataset_yaml(out_root: Path) -> None:
    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_root}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: sugar\n  1: mustard\n  2: banana\n",
        encoding="utf-8",
    )


def _next_image_index(out_root: Path, split: str) -> int:
    """扫描已有编号，续接而不覆盖。"""
    img_dir = out_root / "images" / split
    if not img_dir.is_dir():
        return 0
    max_idx = -1
    for p in img_dir.glob("*.jpg"):
        if "_debug" in p.stem or "_preview" in p.stem:
            continue
        m = re.match(r"^(\d+)$", p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


parser = argparse.ArgumentParser(description="Manual WASD collect for Task B.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--out", type=str, default="dataset_head")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
parser.add_argument("--policy", type=str, default="demo/policy.pt")
parser.add_argument("--vx", type=float, default=0.50)
parser.add_argument("--wz", type=float, default=0.65)
parser.add_argument(
    "--resume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="默认开启：接着已有最大编号继续存，不覆盖（默认 True）",
)
parser.add_argument(
    "--start-index",
    type=int,
    default=None,
    help="手动指定起始编号（会覆盖 --resume 的自动检测）",
)
parser.add_argument(
    "--images-only",
    action="store_true",
    help="只存 jpg，不生成 txt（自己用 LabelImg 标注时用）",
)
parser.add_argument(
    "--view",
    type=str,
    default="head",
    choices=["head", "follow", "free"],
    help="head=head 相机朝前看(推荐); follow=狗后方跟拍; free=不改窗口",
)
parser.add_argument("--verify-only", action="store_true")

_pre, _ = parser.parse_known_args()
if _pre.verify_only:
    p = Path(__file__).resolve()
    t = p.read_text(encoding="utf-8")
    print(f"version={SCRIPT_VERSION}  lines={len(t.splitlines())}  path={p}")
    sys.exit(0 if SCRIPT_VERSION in t else 1)

OUT_ROOT = _resolve_out(_pre.out)
_init_output_dirs(OUT_ROOT)
_write_dataset_yaml(OUT_ROOT)
print(f"[{SCRIPT_VERSION}] out={OUT_ROOT}  resume={_pre.resume}  images_only={_pre.images_only}")

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
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from taskb_perception.collector_nav import B2LocomotionDriver, resolve_policy_path  # noqa: E402
from taskb_perception.obs_utils import parse_rgb  # noqa: E402

if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
try:
    from rl_utils import camera_follow  # noqa: E402
except ImportError:
    camera_follow = None

Key = carb.input.KeyboardInput


def _head_cam_pose_world(robot, cam, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """用机器人实时位姿算 head 相机 eye + 光轴（每帧更新，不依赖 10Hz 的 cam.data）。"""
    rpos = robot.data.root_pos_w[0]
    rquat = robot.data.root_quat_w[0]
    off = cam.cfg.offset
    off_pos = torch.tensor(off.pos, dtype=torch.float32, device=device)
    eye = math_utils.transform_points(off_pos.unsqueeze(0), rpos.unsqueeze(0), rquat.unsqueeze(0)).squeeze(0)
    off_rot = torch.tensor(off.rot, dtype=torch.float32, device=device)
    cam_quat = math_utils.quat_mul(rquat.unsqueeze(0), off_rot.unsqueeze(0)).squeeze(0)
    # Isaac Camera 用 ROS 约定：光轴 +Z
    forward = quat_apply(
        cam_quat.unsqueeze(0),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device),
    ).squeeze(0)
    return eye, forward


def _try_bind_viewport_to_camera_prim(cam) -> str | None:
    """把窗口直接绑到 head_camera USD prim，会随机器人自动移动。"""
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            return None
        if not hasattr(cam, "_sensor_prims") or len(cam._sensor_prims) == 0:
            return None
        prim_path = cam._sensor_prims[0].GetPath().pathString
        if hasattr(viewport, "set_active_camera"):
            viewport.set_active_camera(prim_path)
        else:
            viewport.camera_path = prim_path
        return prim_path
    except Exception as exc:
        print(f"[view] 绑定 head_camera prim 失败: {exc}")
        return None


def _sync_viewport_head_cam(env, robot, cam, look_ahead: float = 5.0) -> None:
    """回退方案：每帧用机器人位姿同步窗口（与 camera_follow 一样传世界坐标）。"""
    unwrapped = env.unwrapped
    vpc = getattr(unwrapped, "viewport_camera_controller", None)
    if vpc is None:
        return
    eye, forward = _head_cam_pose_world(robot, cam, unwrapped.device)
    lookat = eye + forward * look_ahead
    lookat[2] = torch.clamp(lookat[2], min=0.02)
    vpc.update_view_location(eye=eye.detach().cpu().numpy(), lookat=lookat.detach().cpu().numpy())


def _save_image_only(out_root: Path, split: str, stem: str, rgb: np.ndarray) -> Path:
    import cv2

    img_dir = out_root / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{stem}.jpg"
    if path.exists():
        raise FileExistsError(f"已存在 {path}，编号冲突；请检查 --resume / --start-index")
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


class KeyboardController:
    def __init__(self):
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._inp = carb.input.acquire_input_interface()
        self._prev_p = self._prev_r = False
        self.quit = self.reset = self.photo = False

    def _down(self, key):
        return self._inp.get_keyboard_value(self._kb, key) > 0

    def update(self):
        self.reset = self.photo = False
        if self._down(Key.Q) or self._down(Key.ESCAPE):
            self.quit = True
            return
        p = self._down(Key.P)
        if p and not self._prev_p:
            self.photo = True
        self._prev_p = p
        r = self._down(Key.R)
        if r and not self._prev_r:
            self.reset = True
        self._prev_r = r

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
        print(f"[ERROR] 找不到 {REPO_ROOT / 'demo' / 'policy.pt'}")
        return

    if args_cli.start_index is not None:
        next_idx = args_cli.start_index
    elif args_cli.resume:
        next_idx = _next_image_index(OUT_ROOT, args_cli.split)
    else:
        next_idx = 0

    print(f"[{SCRIPT_VERSION}] 本次从编号 {next_idx:06d} 开始（不会覆盖已有文件）")

    driver = B2LocomotionDriver(device=args_cli.device, policy_path=policy_path)
    kb = KeyboardController()
    env = gym.make(args_cli.task, cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1))
    unwrapped = env.unwrapped
    cam = unwrapped.scene.sensors["head_camera"]
    robot = unwrapped.scene["robot"]
    obs, _ = env.reset()
    session_saved = 0
    head_view_bound_prim = False

    if args_cli.view == "head":
        prim_path = _try_bind_viewport_to_camera_prim(cam)
        if prim_path:
            head_view_bound_prim = True
            print(f"[view] 已绑定 head_camera: {prim_path}")
        else:
            _sync_viewport_head_cam(env, robot, cam)
            print("[view] 回退：每帧同步 head 相机位姿")
    elif args_cli.view == "follow" and camera_follow:
        camera_follow(env)

    print("=" * 60)
    print(f"  {SCRIPT_VERSION}  |  先点 Isaac 窗口再按键")
    print("  P=拍照  R=reset  Q=退出")
    print(f"  视角: {args_cli.view}  (head=广角第一人称，随机器人移动)")
    print(f"  模式: {'只存图(手动标注)' if args_cli.images_only else '自动txt'}")
    print(f"  目录: {OUT_ROOT / 'images' / args_cli.split}")
    print("=" * 60)

    while simulation_app.is_running():
        kb.update()
        if kb.quit:
            break
        if kb.reset:
            obs, _ = env.reset()
            print("[reset]")

        obs, _, term, trunc, _ = env.step(driver.compute_action(obs, kb.velocity_command()))

        if args_cli.view == "head" and not head_view_bound_prim:
            _sync_viewport_head_cam(env, robot, cam)
        elif args_cli.view == "follow" and camera_follow:
            camera_follow(env)

        if kb.photo:
            rgb = parse_rgb(obs.get("image", {}), "head_rgb")
            if rgb is None:
                print("[P] 无 head_rgb")
                continue
            stem = f"{next_idx:06d}"
            try:
                if args_cli.images_only:
                    path = _save_image_only(OUT_ROOT, args_cli.split, stem, rgb)
                    print(f"[P] 已存 {path.name}  (累计编号到 {next_idx})")
                else:
                    from taskb_perception.dataset_io import build_labels_from_env, save_snapshot

                    labels = build_labels_from_env(unwrapped, cam, debug=True)
                    r = save_snapshot(OUT_ROOT, stem, rgb, labels, split=args_cli.split, allow_empty=False)
                    if r is None:
                        print("[P] 跳过：无有效标签")
                        continue
                    print(f"[P] {stem}  labels={len(labels)}  -> {r[0].name}")
                next_idx += 1
                session_saved += 1
            except FileExistsError as e:
                print(f"[P] {e}")
                next_idx += 1

        if term.item() or trunc.item():
            obs, _ = env.reset()

    env.close()
    simulation_app.close()
    print(f"本次新拍 {session_saved} 张，下一编号 {next_idx:06d}")
    print(f"数据目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
