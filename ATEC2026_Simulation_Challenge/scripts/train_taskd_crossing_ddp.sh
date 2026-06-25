#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

torchrun --standalone --nproc_per_node=2 ATEC2026_Simulation_Challenge/scripts/parkour_taskd/train.py \
  --task ATEC-TaskD-Crossing-B2Piper \
  --agent rsl_rl_cfg_entry_point \
  --distributed \
  --device cuda \
  --headless \
  --num_envs 2048 \
  --max_iterations 50000
