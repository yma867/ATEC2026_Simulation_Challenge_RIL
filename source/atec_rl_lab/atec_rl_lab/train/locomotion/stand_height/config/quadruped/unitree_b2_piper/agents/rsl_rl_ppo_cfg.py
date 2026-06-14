from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.agents.rsl_rl_ppo_cfg import (
    UnitreeB2PiperFlatPPORunnerCfg,
)


@configclass
class UnitreeB2PiperStandHeightFlatPPORunnerCfg(UnitreeB2PiperFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_piper_stand_height_flat"
        self.max_iterations = 5000
