import gymnasium as gym

from . import agents

##
# Register Gym environments for Unitree B2 + Piper (train locomotion only)
##

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperFlatPPORunnerCfg"
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Unitree-B2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2PiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperRoughPPORunnerCfg"
    },
)