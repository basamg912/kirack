import math
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp

# -----------------------------------------util
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen

# -----------------------------------------assets
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
)

# -----------------------------------------env
from isaaclab.envs import ManagerBasedRLEnvCfg

# -----------------------------------------manager
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# -----------------------------------------scene
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import (
    HfPyramidSlopedTerrainCfg,  # 피라미드 경사 terrain (difficulty 반영)
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,  # 그리드 박스 terrain (difficulty 반영)
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniNoise

# -----------------------------------------config
import kirack.tasks.standing.mdp as mdp
from kirack.tasks.environment.kapex.observations.observation import (
    cache_feet_description,
    cache_joint_description,
    get_feet_description,
    get_feet_observation,
    get_global_observation,
    get_joint_description,
    get_joint_observation,
)
from kirack.tasks.standing.robot.kapex import KAPEX0_CFG

PLAY_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(3, 3),  # sub terrain 크기
    num_rows=9,  # 9 level
    num_cols=21,
    horizontal_scale=0.05,
    slope_threshold=0.0,
    curriculum=True,  # 난이도 별 커리큘럼
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # difficulty 가 실제로 반영되는 sub-terrain 으로 구성 (slope/boxes 는 row 별 interpolation)
        "slope": HfPyramidSlopedTerrainCfg(
            proportion=0.4, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.0, 0.05), platform_width=2.0
        ),
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
    },
)

# MultiAsset : https://github.com/isaac-sim/IsaacLab/discussions/2863
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveSceneCfg
# !!! https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html


@configclass
class KapexStandingSceneCfg(InteractiveSceneCfg):
    """Scene for the Kapex standing task: ground + robot + contact sensor + light."""

    ground_plane = AssetBaseCfg(
        prim_path="/World/default",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_generator=PLAY_TERRAIN_CFG,
        max_init_terrain_level=0,  # 초기 학습 안정성 위해 lowest terrain 에 스폰
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = KAPEX0_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/WL3",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint position control over all joints with default-position offset."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Five flat top-level observation groups consumed by the attention policy.

    Each group wraps a single ObsTerm so that the env returns a *flat*
    TensorDict ``{joint_desc: T, joint_obs: T, feet_desc: T, feet_obs: T,
    global_obs: T}`` rather than nested groups. rsl_rl's rollout storage
    requires a flat top-level structure to allocate its buffers correctly.

        joint_desc   (E, J, 16)   static per-joint morphology (cached)
        joint_obs    (E, J, 3)    [pos_rel, vel_rel, last_action] per joint
        feet_desc    (E, F, 3)    static per-foot position (cached)
        feet_obs     (E, F, 2)    [foot_z_in_root, foot_vz_in_root] kinematic
        global_obs   (E, 15)      base/command/com context

    Actor and critic share the same groups via :attr:`PPORunnerCfg.obs_groups`.
    """

    @configclass
    class JointDescCfg(ObsGroup):
        joint_desc = ObsTerm(func=get_joint_description)

        def __post_init__(self):
            self.concatenate_terms = True  # single term -> identity concat
            self.history_length = 0

    @configclass
    class JointObsCfg(ObsGroup):
        joint_obs = ObsTerm(func=get_joint_observation)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class FeetDescCfg(ObsGroup):
        feet_desc = ObsTerm(func=get_feet_description)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class FeetObsCfg(ObsGroup):
        feet_obs = ObsTerm(func=get_feet_observation)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class GlobalObsCfg(ObsGroup):
        global_obs = ObsTerm(func=get_global_observation)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    joint_desc: JointDescCfg = JointDescCfg()
    joint_obs: JointObsCfg = JointObsCfg()
    feet_desc: FeetDescCfg = FeetDescCfg()
    feet_obs: FeetObsCfg = FeetObsCfg()
    global_obs: GlobalObsCfg = GlobalObsCfg()


@configclass
class EventCfg:
    """Domain randomization & resets."""

    # --- Attention-model description caches ---------------------------------
    # Run on the first reset, before any pose-altering randomization, so that
    # ``body_pos_w`` reflects the default pose when the description is captured.
    # The cache function is guarded -- subsequent resets are no-ops.
    cache_joint_desc = EventTerm(
        func=cache_joint_description,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "root_body": "WL3",
        },
    )
    cache_feet_desc = EventTerm(
        func=cache_feet_description,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "root_body": "WL3",
            "foot_bodies": ("LL7", "RL7"),
        },
    )

    feet_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        min_step_count_between_reset=1000,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["LL[67]", "RL[67]"],
            ),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # ! callable object not confine function,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="WL3"),  # BODY TORSO LINK
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "distribution": "gaussian",
            "recompute_inertia": False,  # ! 4/6 : 학습 안정을 위해 우선 False
        },
    )
    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        min_step_count_between_reset=15000,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="WL3"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    # ------------------------reset

    scale_all_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    # ------------------------interval
    add_joint_noise = EventTerm(
        func=mdp.joint_position_noise,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "noise_range": (-0.1, 0.1),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    # ------------------------


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    feet_to_close = RewTerm(
        func=mdp.feet_to_close,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL6", "RL6"]),
            "threshold": 0.1,
        },
    )

    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["LA.*", "RA.*"],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "WL.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LLJ[12]", "RLJ[12]"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.91})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LL6", "RL6"]),
        },
    )
    feet_phase = RewTerm(
        func=mdp.feet_phase,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "swing_height": 0.1,
            "tracking_sigma": 0.25,
            "threshold": 0.55,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL6", "RL6"]),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL[67]", "RL[67]"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LL[67]", "RL[67]"]),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.05,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.3,
            "body_vel_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL[67]", "RL[67]"]),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                # ! 1) find LL6,7 or RL6,7 2) matching 되면 제외, 조건 통과하면 .*
                "contact_forces",
                body_names=["(?!.*(LL[67]|RL[67]).*).*"],
            ),
        },
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        rel_standing_envs=0.05,  # 20% stand
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # ! start low level command
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.2, 0.2),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.2, 0.2),
        ),
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum_relative, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


##
# Env
##


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_survival)
    lin_vel_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    joint_armature_levels = CurrTerm(func=mdp.actuator_armature_range_levels)
    stiff_damping_levels = CurrTerm(func=mdp.actuator_gain_range_levels)
    push_robot_levels = CurrTerm(
        func=mdp.perturb_levels,
        params={
            "delta": 0.05,
            "limit_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
            },
        },
    )
    torso_com_levels = CurrTerm(
        func=mdp.torso_com_levels,
        params={
            "delta": 0.01,
            "threshold": 0.5,
            "limit_range": {
                "x": (-0.3, 0.3),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
            },
        },
    )


@configclass
class KapexStandingEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL config for the Kapex standing task."""

    scene: KapexStandingSceneCfg = KapexStandingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (5.0, -5.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1.0 / 200.0  # 5 ms -> physics dt : 500 Hz, policy dt 20ms : 50Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        from kirack.utils import exp_loader

        exp = exp_loader.get_exp_cfg()
        if exp is not None:
            exp_loader.apply_exp_cfg(self, exp)


PLAY_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(3, 3),
    num_rows=3,  # play 용으로 작게
    num_cols=3,
    horizontal_scale=0.05,
    slope_threshold=0.0,
    curriculum=False,  # play 에서는 난이도 변경 없이 평지 유지
    difficulty_range=(0.0, 0.0),
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class KapexStandingPlayEnvCfg(KapexStandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # ---- Camera: third-person, robot-following ----
        # asset_root: viewer 가 robot 의 root body 를 origin 삼아 따라감
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        # 로봇 약간 뒤+옆+위에서 바라보는 cinematic 3인칭 시점
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.6)
        self.viewer.resolution = (1920, 1080)

        # ---- Terrain: flat-only, 작은 그리드, curriculum off ----
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_generator=PLAY_FLAT_TERRAIN_CFG,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        # play 시 terrain curriculum 비활성화 (KapexStandingEnvCfg.__post_init__ 가 curriculum
        # 존재 여부로 분기하므로, 명시적으로 끄기 위해 generator 의 curriculum 도 False 로 설정됨)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
