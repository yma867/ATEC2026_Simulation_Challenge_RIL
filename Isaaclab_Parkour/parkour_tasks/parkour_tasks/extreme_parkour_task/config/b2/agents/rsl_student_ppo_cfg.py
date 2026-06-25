from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_ppo_cfg import (
    UnitreeGo2ParkourStudentPPORunnerCfg,
)


@configclass
class UnitreeB2ParkourStudentPPORunnerCfg(UnitreeGo2ParkourStudentPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_parkour"

