from __future__ import annotations

from isaaclab.utils import configclass

from .parkour_rl_cfg import ParkourRslRlDepthEncoderCfg, ParkourRslRlDistillationAlgorithmCfg
from .rsl_teacher_ppo_cfg import TaskDCrossingParkourTeacherPPORunnerCfg


@configclass
class TaskDCrossingParkourStudentPPORunnerCfg(TaskDCrossingParkourTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "taskd_crossing_parkour"
        self.depth_encoder = ParkourRslRlDepthEncoderCfg(
            hidden_dims=512,
            learning_rate=1.0e-3,
            num_steps_per_env=24 * 5,
            depth_shape=(58, 87),
        )
        self.algorithm = ParkourRslRlDistillationAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=2.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
