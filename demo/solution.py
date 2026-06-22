"""评测入口 — 加载同目录下的 solution_rl.AlgSolution"""
import os
import sys

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from solution_rl import AlgSolution  # noqa: F401
