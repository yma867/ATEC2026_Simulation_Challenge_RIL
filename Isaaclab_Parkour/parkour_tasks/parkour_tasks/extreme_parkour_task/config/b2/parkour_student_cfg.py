from copy import deepcopy

from isaaclab.utils import configclass

from atec_rl_lab.assets.robots import UNITREE_B2_CFG
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from parkour_tasks.default_cfg import CAMERA_CFG, VIEWER
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import (
    UnitreeGo2StudentParkourEnvCfg_EVAL,
    UnitreeGo2StudentParkourEnvCfg_PLAY,
)

from .parkour_mdp_cfg import (
    ActionsCfg,
    B2EventCfg,
    B2StudentObservationsCfg,
    B2StudentRewardsCfg,
    B2TeacherRewardsCfg,
    CommandsCfg,
    ParkourEventsCfg,
    TerminationsCfg,
)
from .parkour_teacher_cfg import B2ParkourTeacherSceneCfg


B2_CAMERA_CFG = deepcopy(CAMERA_CFG)
B2_CAMERA_CFG.prim_path = "{ENV_REGEX_NS}/Robot/base_link"


def _materialize_b2_student_observations(obs_cfg: B2StudentObservationsCfg):
    """Make nested observation terms visible to IsaacLab's ObservationManager."""
    obs_cfg.policy = B2StudentObservationsCfg.PolicyCfg()
    obs_cfg.policy.extreme_parkour_observations = deepcopy(obs_cfg.policy.extreme_parkour_observations)

    obs_cfg.depth_camera = B2StudentObservationsCfg.DepthCameraPolicyCfg()
    obs_cfg.depth_camera.depth_cam = deepcopy(obs_cfg.depth_camera.depth_cam)
    obs_cfg.depth_camera.depth_cam.params["debug_vis"] = False

    obs_cfg.delta_yaw_ok = B2StudentObservationsCfg.DeltaYawOkPolicyCfg()
    obs_cfg.delta_yaw_ok.deta_yaw_ok = deepcopy(obs_cfg.delta_yaw_ok.deta_yaw_ok)
    return obs_cfg


@configclass
class B2ParkourStudentSceneCfg(B2ParkourTeacherSceneCfg):
    depth_camera = B2_CAMERA_CFG
    depth_camera_usd = None

    def __post_init__(self):
        super().__post_init__()
        self.robot = deepcopy(UNITREE_B2_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.1
        for key, sub_terrain in self.terrain.terrain_generator.sub_terrains.items():
            sub_terrain.use_simplified = True
            sub_terrain.horizontal_scale = 0.1
            if key == "parkour_demo":
                sub_terrain.proportion = 0.15
            elif key == "parkour_flat":
                sub_terrain.proportion = 0.05
            else:
                sub_terrain.proportion = 0.2
                if key != "parkour":
                    sub_terrain.y_range = (-0.1, 0.1)


@configclass
class UnitreeB2StudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: B2ParkourStudentSceneCfg = B2ParkourStudentSceneCfg(num_envs=192, env_spacing=1.0)
    observations: B2StudentObservationsCfg = B2StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: B2StudentRewardsCfg = B2StudentRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: B2EventCfg = B2EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8
        # Replace the entire observation config with a clean B2-only object.
        # ObservationManager iterates over cfg.__dict__; replacing the object
        # avoids stale inherited empty groups from the Go2 student config.
        self.observations = _materialize_b2_student_observations(B2StudentObservationsCfg())


@configclass
class UnitreeB2StudentParkourEnvCfg_EVAL(UnitreeB2StudentParkourEnvCfg):
    viewer = VIEWER
    rewards: B2TeacherRewardsCfg = B2TeacherRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.0
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        self.observations.depth_camera.depth_cam.params["debug_vis"] = False
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)
        self.events.random_camera_position.params["rot_noise_range"] = {"pitch": (0, 1)}
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key in ["parkour_flat", "parkour_demo"]:
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.25
                sub_terrain.noise_range = (0.02, 0.02)


@configclass
class UnitreeB2StudentParkourEnvCfg_PLAY(UnitreeB2StudentParkourEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.events.push_by_setting_velocity = None
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key == "parkour_flat":
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.25
                sub_terrain.noise_range = (0.02, 0.02)
