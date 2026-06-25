"""Export TaskD crossing Parkour-style checkpoints for deployment."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

THIS_DIR = os.path.dirname(__file__)
ATEC_SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "rsl_rl"))
PARKOUR_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "Isaaclab_Parkour"))
if ATEC_SCRIPTS_DIR not in sys.path:
    sys.path.append(ATEC_SCRIPTS_DIR)
if PARKOUR_ROOT not in sys.path:
    sys.path.append(PARKOUR_ROOT)

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export a TaskD crossing Parkour checkpoint.")
parser.add_argument("--task", type=str, default="ATEC-TaskD-Crossing-B2Piper", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create for export.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.tasks  # noqa: F401
import atec_rl_lab.train  # noqa: F401
from scripts.rsl_rl.exporter import (
    export_deploy_policy_as_jit,
    export_deploy_policy_as_onnx,
    export_teacher_policy_as_jit,
    export_teacher_policy_as_onnx,
)
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from vecenv_wrapper import TaskDParkourRslRlVecEnvWrapper


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = TaskDParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = get_checkpoint_path(env_cfg.log_dir, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    estimator = runner.get_estimator_inference_policy(device=env.device)
    if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        depth_encoder = runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, estimator, depth_encoder, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_deploy_policy_as_onnx(policy_nn, estimator, depth_encoder, agent_cfg, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")
    else:
        policy_nn = runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(policy_nn, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
