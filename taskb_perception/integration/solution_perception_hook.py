"""
把本文件夹拷到 ATEC 仓库根目录后，在 demo/solution.py 里参考此写法对接。

导航队友读:  perception_out.next_target.pos_b
抓取队友读:  perception.lock_target(id)  -> track.grasp_pos_b / grasp_quat_b
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# 确保能 import taskb_perception（文件夹在仓库根目录）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "taskb_perception") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "taskb_perception"))

from taskb_perception import PerceptionConfig, PerceptionModule


class AlgSolution:
    """示例：只接感知层，腿/臂 action 仍为 0。队友替换 navigation / grasp 部分即可。"""

    def __init__(self):
        self.perception = PerceptionModule(
            PerceptionConfig(
                yolo_weights="taskb_perception/weights/taskb_yolo.pt",
                use_color_fallback=True,  # 无权重文件时自动回退 HSV
            )
        )
        self._phase = "SEARCH"
        self._locked_id: int | None = None

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return None

    def reset(self):
        self.perception.reset()
        self._phase = "SEARCH"
        self._locked_id = None

    def predicts(self, obs, current_score):
        # obs 结构与 play_atec_task.py 一致
        if not hasattr(self, "_inited"):
            self._inited = True
            self.reset()

        pout = self.perception.update(obs)

        if self._phase == "SEARCH" and pout.next_target is not None:
            self._phase = "APPROACH"

        # TODO 导航队友: 根据 pout.next_target.pos_b 算 leg action
        # TODO 抓取队友: GRASP 阶段调用 self.perception.lock_target(id)

        if self._phase == "GRASP" and self._locked_id is None and pout.next_target:
            locked = self.perception.lock_target(pout.next_target.track_id)
            self._locked_id = pout.next_target.track_id
            # locked.grasp_pos_b, locked.grasp_quat_b -> IK

        proprio = obs["proprio"]
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        action = [0.0] * action_dim
        return {"action": action, "giveup": False}
