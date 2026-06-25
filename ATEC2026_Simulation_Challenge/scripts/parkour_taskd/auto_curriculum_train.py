"""Automatically train TaskD crossing curriculum stages by success rate.

This supervisor keeps the policy/deployment contract unchanged.  It switches
between separately-generated TaskD crossing terrain stages by stopping the
current training process, resuming from its latest checkpoint, and launching the
next stage task.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


STAGES = (
    ("easy", "ATEC-TaskD-Crossing-B2Piper-Easy"),
    ("mid", "ATEC-TaskD-Crossing-B2Piper-Mid"),
    ("hard", "ATEC-TaskD-Crossing-B2Piper-Hard"),
    ("final", "ATEC-TaskD-Crossing-B2Piper"),
)

ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
SUCCESS_RE = re.compile(r"Episode_Termination/x_reached:\s+([-+0-9.eE]+)")
FALL_RE = re.compile(r"Episode_Termination/fall:\s+([-+0-9.eE]+)")


def _latest_checkpoint(log_root: Path, run_name_hint: str | None = None) -> tuple[str, str]:
    """Return (run_dir_name, checkpoint_name) for the latest checkpoint."""
    if not log_root.exists():
        raise FileNotFoundError(f"Log root does not exist: {log_root}")

    run_dirs = [p for p in log_root.iterdir() if p.is_dir()]
    if run_name_hint is not None:
        hinted = [p for p in run_dirs if p.name == run_name_hint or p.name.endswith(f"_{run_name_hint}")]
        if hinted:
            run_dirs = hinted
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {log_root}")

    def run_key(path: Path) -> float:
        return path.stat().st_mtime

    for run_dir in sorted(run_dirs, key=run_key, reverse=True):
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            return run_dir.name, checkpoints[-1].name

    raise FileNotFoundError(f"No model_*.pt checkpoints found under: {log_root}")


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float = 90.0) -> int:
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(signal.SIGINT)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(1.0)
    proc.terminate()
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(1.0)
    proc.kill()
    return proc.wait()


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-curriculum trainer for TaskD crossing.")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default="taskd_crossing_parkour")
    parser.add_argument("--run_name_prefix", type=str, default="auto_curriculum")
    parser.add_argument("--stage_max_iterations", type=str, default="12000,12000,16000,50000")
    parser.add_argument("--stage_min_iterations", type=str, default="3000,3000,4000,8000")
    parser.add_argument("--success_thresholds", type=str, default="0.25,0.20,0.12")
    parser.add_argument("--fall_max", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=3, help="Consecutive metric prints required before switching.")
    parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    stage_max = [int(x) for x in _parse_csv_floats(args.stage_max_iterations)]
    stage_min = [int(x) for x in _parse_csv_floats(args.stage_min_iterations)]
    thresholds = _parse_csv_floats(args.success_thresholds)
    if len(stage_max) != len(STAGES) or len(stage_min) != len(STAGES):
        raise ValueError("--stage_max_iterations and --stage_min_iterations must have 4 comma-separated values.")
    if len(thresholds) != len(STAGES) - 1:
        raise ValueError("--success_thresholds must have 3 comma-separated values for easy/mid/hard.")

    repo_root = Path.cwd()
    log_root = repo_root / "logs" / "rsl_rl" / args.experiment_name
    previous_run: str | None = None
    previous_checkpoint: str | None = None

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for stage_idx, (stage_name, task_name) in enumerate(STAGES):
        run_name = f"{args.run_name_prefix}_{stage_idx}_{stage_name}"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node",
            str(args.nproc_per_node),
            "scripts/parkour_taskd/train.py",
            "--task",
            task_name,
            "--agent",
            "rsl_rl_cfg_entry_point",
            "--headless",
            "--distributed",
            "--device",
            args.device,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(stage_max[stage_idx]),
            "--experiment_name",
            args.experiment_name,
            "--run_name",
            run_name,
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if previous_run is not None and previous_checkpoint is not None:
            cmd += ["--resume", "--load_run", previous_run, "--checkpoint", previous_checkpoint]
        cmd += args.extra_train_args

        print(f"\n[AUTO-CURRICULUM] Starting stage {stage_idx + 1}/{len(STAGES)}: {stage_name} -> {task_name}")
        if previous_run is not None:
            print(f"[AUTO-CURRICULUM] Resuming from {previous_run}/{previous_checkpoint}")
        print("[AUTO-CURRICULUM] Command:", " ".join(cmd), flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        current_iter = 0
        latest_success = 0.0
        latest_fall = 0.0
        pass_count = 0
        switched = False

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            m_iter = ITER_RE.search(line)
            if m_iter:
                current_iter = int(m_iter.group(1))
            m_success = SUCCESS_RE.search(line)
            if m_success:
                latest_success = float(m_success.group(1))
            m_fall = FALL_RE.search(line)
            if m_fall:
                latest_fall = float(m_fall.group(1))

            if stage_idx < len(STAGES) - 1 and current_iter >= stage_min[stage_idx]:
                ok = latest_success >= thresholds[stage_idx] and latest_fall <= args.fall_max
                pass_count = pass_count + 1 if ok else 0
                if ok:
                    print(
                        f"[AUTO-CURRICULUM] stage={stage_name} pass_count={pass_count}/{args.patience} "
                        f"success={latest_success:.4f} fall={latest_fall:.4f} iter={current_iter}",
                        flush=True,
                    )
                if pass_count >= args.patience:
                    print(f"[AUTO-CURRICULUM] Switching from {stage_name} to next stage.", flush=True)
                    _terminate_process(proc)
                    switched = True
                    break

        if not switched:
            return_code = proc.wait()
            if return_code != 0:
                print(f"[AUTO-CURRICULUM] Stage {stage_name} failed with return code {return_code}.", file=sys.stderr)
                return return_code

        # Give rank-0 save a moment to flush after graceful interruption.
        time.sleep(3.0)
        previous_run, previous_checkpoint = _latest_checkpoint(log_root, run_name)
        print(f"[AUTO-CURRICULUM] Latest checkpoint: {previous_run}/{previous_checkpoint}", flush=True)

    print("[AUTO-CURRICULUM] Finished all stages.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
