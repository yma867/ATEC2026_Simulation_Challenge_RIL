import gymnasium as gym

from . import agents

gym.register(
    id="ATEC-Isaac-Stand-Height-Flat-Unitree-B2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2PiperStandHeightFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperStandHeightFlatPPORunnerCfg",
    },
)
