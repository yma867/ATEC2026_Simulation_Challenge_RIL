"""
ATEC 2026 Task B 比赛入口 — AlgSolution
将此文件复制到 demo/solution.py 即可参赛

感知层 (PerceptionPipeline) → 操作层 (TODO: 你的队友填入) → 20维动作
"""

import numpy as np
import torch

from perception_pipeline import PerceptionPipeline


class AlgSolution:
    """
    比赛要求的接口类。
    模拟器每步调用 predicts() 一次。
    """

    def __init__(self):
        print("[AlgSolution] Initializing...")

        # ---- 加载感知 pipeline ----
        self.perception = PerceptionPipeline(device='cuda:0')
        print("[AlgSolution] Perception pipeline ready.")

        # ---- 机器人动作维度 ----
        self.action_dim = 20  # B2-Piper: 12 leg + 8 arm

        # ---- 控制周期 (Isaac Lab 默认 50Hz) ----
        self.dt = 0.02

        # ---- 初始动作（安全站立姿态）----
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        # ---- Episode 计数 ----
        self.episode_count = 0

    def reset(self):
        """新 episode 开始时重置感知状态"""
        self.perception.reset()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.episode_count += 1
        print(f"[AlgSolution] Episode {self.episode_count} started.")

    def predicts(self, obs: dict, current_score: float):
        """
        比赛核心接口 — 每帧被调用一次

        Args:
            obs: dict
                {
                    'proprio':  torch.Tensor (1, 72),
                    'extero':   torch.Tensor (1, ...),   # LiDAR
                    'image': {
                        'head_rgb':   torch.Tensor (1, 480, 640, 3) uint8,
                        'head_depth': torch.Tensor (1, 480, 640, 1) float32,
                        'ee_rgb':     torch.Tensor (1, 480, 640, 3) uint8,
                        'ee_depth':   torch.Tensor (1, 480, 640, 1) float32,
                    }
                }
            current_score: float, 当前累计得分

        Returns:
            dict: {'action': [[a0, a1, ..., a19]], 'giveup': False}
        """
        # ---- 感知层: 分析当前帧 ----
        perception_output = self.perception.process(obs, dt=self.dt)

        # ═══════════════════════════════════════════════════════
        #  操作层 (TODO: 你的队友填入)
        #
        #  输入: perception_output (见 config.py 注释)
        #  输出: action (20维 numpy array)
        #
        #  现在用零动作占位 — 机器人会站在原地不动
        # ═══════════════════════════════════════════════════════

        action = self._placeholder_action(perception_output)

        # ---- 包成比赛格式返回 ----
        self.last_action = action

        return {
            'action': [action.tolist()],
            'giveup': False,
        }

    # ==========================================================================
    #  占位操作层 (等你队友替换)
    # ==========================================================================
    def _placeholder_action(self, perception_output):
        """占位: 返回零动作（机器人站立不动）"""
        # 你可以在这里打印感知输出调试:
        p = perception_output
        if p['target'] is not None:
            pass  # print(f"Frame: target={p['target']['class']} dist={p['target']['dist_to_robot']:.2f}")

        return np.zeros(self.action_dim, dtype=np.float32)


# ==========================================================================
#  本地测试
# ==========================================================================
if __name__ == "__main__":
    """
    本地测试: 用仿真数据验证 pipeline 能正常运行
    
    用法:
        cd taskb_perception
        python solution.py
    
    需要先将仿真器生成的 sensor 数据保存为 numpy 文件，
    或直接连接到仿真器。
    """
    print("=" * 60)
    print("ATEC 2026 Task B — Perception Layer Self-Test")
    print("=" * 60)

    # 创建 AlgSolution 实例
    solution = AlgSolution()

    # 生成模拟观测数据 (全零 / 随机)
    # 实际测试时替换为仿真器数据
    dummy_obs = {
        'proprio': torch.zeros(1, 72, dtype=torch.float32),
        'image': {
            'head_rgb': torch.randint(0, 255, (1, 480, 640, 3), dtype=torch.uint8),
            'head_depth': torch.ones(1, 480, 640, 1, dtype=torch.float32) * 3.0,
        },
    }

    # 跑一帧测试
    result = solution.predicts(dummy_obs, current_score=0.0)
    print(f"\nAction shape: {len(result['action'][0])} (expect 20)")
    print(f"Giveup: {result['giveup']}")
    print("Test OK — pipeline runs without crash.")