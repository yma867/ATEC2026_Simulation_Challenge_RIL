from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import (
    UnitreeGo2ParkourTeacherPPORunnerCfg,
)


@configclass
class UnitreeB2ParkourTeacherPPORunnerCfg(UnitreeGo2ParkourTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_parkour"

