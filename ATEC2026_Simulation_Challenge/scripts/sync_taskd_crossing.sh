#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-user@server}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/junyi/ATEC}"

rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$REPO_ROOT/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/" \
  "$REMOTE_HOST:$REMOTE_ROOT/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/"

rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$REPO_ROOT/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/parkour_taskd/" \
  "$REMOTE_HOST:$REMOTE_ROOT/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/parkour_taskd/"

rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$REPO_ROOT/ATEC2026_Simulation_Challenge/scripts/parkour_taskd/" \
  "$REMOTE_HOST:$REMOTE_ROOT/ATEC2026_Simulation_Challenge/scripts/parkour_taskd/"

rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$REPO_ROOT/ATEC2026_Simulation_Challenge/scripts/view_task_d_crossing.py" \
  "$REMOTE_HOST:$REMOTE_ROOT/ATEC2026_Simulation_Challenge/scripts/view_task_d_crossing.py"

rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$REPO_ROOT/ATEC2026_Simulation_Challenge/scripts/smoke_taskd_crossing.py" \
  "$REMOTE_HOST:$REMOTE_ROOT/ATEC2026_Simulation_Challenge/scripts/smoke_taskd_crossing.py"

rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$REPO_ROOT/ATEC2026_Simulation_Challenge/scripts/train_taskd_crossing_ddp.sh" \
  "$REMOTE_HOST:$REMOTE_ROOT/ATEC2026_Simulation_Challenge/scripts/train_taskd_crossing_ddp.sh"
