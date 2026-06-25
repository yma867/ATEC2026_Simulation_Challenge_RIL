import gymnasium as gym

from .terrain import TASK_D_TERRAIN_CFG
from .env_cfg import TaskDEnvCfg, TaskDEnvB2Cfg, TaskDEnvTron2ALeggedCfg, TaskDEnvTron2AWheelCfg
from .crossing_env_cfg import TaskDCrossingEasyEnvCfg, TaskDCrossingEnvCfg, TaskDCrossingHardEnvCfg, TaskDCrossingMidEnvCfg


gym.register(
    id = "ATEC-TaskD-G1",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvG1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskD-Tron1Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskD-Tron2ALegged",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron2ALeggedCfg"
    },
)

gym.register(
    id = "ATEC-TaskD-Tron2AWheel",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron2AWheelCfg"
    },
)

gym.register(
    id = "ATEC-TaskD-B2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2Cfg"
    },
)

gym.register(
    id="ATEC-TaskD-Crossing-B2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crossing_env_cfg:TaskDCrossingEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_teacher_ppo_cfg:"
            "TaskDCrossingParkourTeacherPPORunnerCfg"
        ),
        "rsl_rl_student_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_student_ppo_cfg:"
            "TaskDCrossingParkourStudentPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-TaskD-Crossing-B2Piper-Easy",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crossing_env_cfg:TaskDCrossingEasyEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_teacher_ppo_cfg:"
            "TaskDCrossingParkourTeacherPPORunnerCfg"
        ),
        "rsl_rl_student_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_student_ppo_cfg:"
            "TaskDCrossingParkourStudentPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-TaskD-Crossing-B2Piper-Mid",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crossing_env_cfg:TaskDCrossingMidEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_teacher_ppo_cfg:"
            "TaskDCrossingParkourTeacherPPORunnerCfg"
        ),
        "rsl_rl_student_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_student_ppo_cfg:"
            "TaskDCrossingParkourStudentPPORunnerCfg"
        ),
    },
)

gym.register(
    id="ATEC-TaskD-Crossing-B2Piper-Hard",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crossing_env_cfg:TaskDCrossingHardEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_teacher_ppo_cfg:"
            "TaskDCrossingParkourTeacherPPORunnerCfg"
        ),
        "rsl_rl_student_cfg_entry_point": (
            "atec_rl_lab.train.parkour_taskd.agents.rsl_student_ppo_cfg:"
            "TaskDCrossingParkourStudentPPORunnerCfg"
        ),
    },
)

gym.register(
    id = "ATEC-TaskD-B2wPiper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2WCfg"
    },
)

__all__ = [
    'TaskDEnvCfg',
    'TaskDEnvB2Cfg',
    'TaskDEnvTron2ALeggedCfg',
    'TaskDEnvTron2AWheelCfg',
    'TaskDCrossingEnvCfg',
    'TaskDCrossingEasyEnvCfg',
    'TaskDCrossingMidEnvCfg',
    'TaskDCrossingHardEnvCfg',
]
