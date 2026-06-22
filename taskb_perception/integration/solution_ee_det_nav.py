"""
demo/solution.py 参考：远距离导航 — 光轴中线对准 + EE 目标锁

导航队友读:
  self.nav_target.err_u / on_axis / locked / lock_id
  self.compute_nav_velocity(obs) → [vx, vy, wz]
  self.axis_nav.complete_current()  # 抓取完成后释放锁，处理下一个

旧 pos_b_3d: PerceptionConfig(nav_method="pos_b_3d") + EEDetector
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "taskb_perception") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "taskb_perception"))

from taskb_perception import AxisNavController, PerceptionConfig


class AlgSolution:
    """2D detect → 光轴中线对准 → 直行。"""

    def __init__(self):
        self.cfg = PerceptionConfig()  # nav_method 默认 axis_align
        self.axis_nav = AxisNavController(self.cfg)
        self.nav_target = None

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return None

    def reset(self):
        self.axis_nav.reset()
        self.nav_target = None

    def compute_nav_velocity(self, obs) -> list[float]:
        vel = self.axis_nav.compute_velocity(obs)
        self.nav_target = self.axis_nav.nav_target
        return [float(vel[0]), float(vel[1]), float(vel[2])]

    def predicts(self, obs, current_score):
        if not hasattr(self, "_inited"):
            self._inited = True
            self.reset()

        vel = self.compute_nav_velocity(obs)

        proprio = obs["proprio"]
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        action = [0.0] * action_dim

        # TODO: B2LocomotionDriver.compute_action(obs, vel) 替换占位
        _ = vel

        return {"action": action, "giveup": False}
