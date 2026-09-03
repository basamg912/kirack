"""Kapex + G1 combined standing task config (both robots spawned per env).

Architecture:
- Scene spawns BOTH ``robot`` (Kapex) and ``robot_g1`` (G1) in every env.
- Kapex is offset to ``x=-1.5``, G1 to ``x=+1.5`` → 3 m gap → no physical contact
  during normal episodes (foot prints + perturbations stay < ±1 m).
- ``env_spacing`` bumped to ``5.0`` so adjacent envs' robot pairs don't collide.
- Observations: per-joint / per-foot / global tensors are concatenated
  ``[Kapex, G1]`` so a single URMA attention policy sees both morphologies in
  one batch (44 joints, 4 feet, 30-dim global). URMA's attention pool is
  morphology-agnostic so no model change is needed.
- Actions: ``ActionManager`` runs two ``JointPositionActionCfg`` terms back to
  back. Policy output (44,) = first 21 → Kapex, last 23 → G1.
- Rewards: shared / robot-agnostic terms once; robot-specific terms duplicated
  with different ``SceneEntityCfg`` (kapex / robot_g1) and joint-name patterns.
- Terminations: either-robot failure terminates the env (standard).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import (
    HfPyramidSlopedTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import kirack.tasks.standing.mdp as mdp
from kirack.tasks.standing.observations_combined import (
    cache_feet_desc_combined,
    cache_joint_desc_combined,
    combined_feet_description,
    combined_feet_observation,
    combined_global_observation,
    combined_joint_description,
    combined_joint_observation,
)
from kirack.tasks.standing.robot.g1 import G1_23DOF_CFG
from kirack.tasks.standing.robot.kapex import KAPEX0_CFG

# -- Spatial layout (within each env) -----------------------------------------
KAPEX_XY_OFFSET = (-1.5, 0.0)
G1_XY_OFFSET = (1.5, 0.0)
ENV_SPACING = 5.0  # tile size; pair occupies x in [-2, +2] so 5 m is safe

# -- G1 morphology shorthands -------------------------------------------------
G1_ROOT_BODY = "pelvis"
G1_TORSO_BODY = "torso_link"
G1_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_ANKLE_BODIES_RE = "(left|right)_ankle_(pitch|roll)_link"
G1_NOT_ANKLE_RE = "(?!.*(left|right)_ankle_(pitch|roll)_link).*"
G1_ARM_JOINT_RE = [".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
G1_WAIST_JOINT_RE = ["waist_yaw_joint"]
G1_LEG_HIP_JOINT_RE = [".*_hip_pitch_joint", ".*_hip_roll_joint"]
G1_STANDING_HEIGHT = 0.74


# -- Articulation cfgs with intra-env spatial offset --------------------------

def _offset_init_state(base_cfg: ArticulationCfg, xy: tuple[float, float]) -> ArticulationCfg.InitialStateCfg:
    """Return base_cfg.init_state with pos.xy overridden, pos.z kept."""
    z = base_cfg.init_state.pos[2]
    return base_cfg.init_state.replace(pos=(xy[0], xy[1], z))


KAPEX_COMBINED_CFG = KAPEX0_CFG.replace(init_state=_offset_init_state(KAPEX0_CFG, KAPEX_XY_OFFSET))
G1_COMBINED_CFG = G1_23DOF_CFG.replace(init_state=_offset_init_state(G1_23DOF_CFG, G1_XY_OFFSET))


# -- Terrain (shared) ---------------------------------------------------------
COMBINED_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(3, 3),
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.05,
    slope_threshold=0.0,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "slope": HfPyramidSlopedTerrainCfg(
            proportion=0.4, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.0, 0.05), platform_width=2.0
        ),
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
    },
)


@configclass
class CombinedSceneCfg(InteractiveSceneCfg):
    """Scene: terrain + Kapex + G1 + contact sensors + lights + height scanner."""

    ground_plane = AssetBaseCfg(
        prim_path="/World/default",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_generator=COMBINED_TERRAIN_CFG,
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

    # Two robots per env, spatially separated to avoid contact.
    robot: ArticulationCfg = KAPEX_COMBINED_CFG.replace(prim_path="{ENV_REGEX_NS}/Kapex")
    robot_g1: ArticulationCfg = G1_COMBINED_CFG.replace(prim_path="{ENV_REGEX_NS}/G1")

    # Per-robot contact sensors. Distinct prefixes keep the body-name regexes simple.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Kapex/.*",
        history_length=3,
        track_air_time=True,
    )
    contact_forces_g1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/G1/.*",
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Height scanner attached to Kapex's torso for terrain reference.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Kapex/WL3",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ActionsCfg:
    """Two action terms in fixed order: Kapex (21) then G1 (23). Total 44 dims."""

    kapex_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )
    g1_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot_g1",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Five flat groups, each combining Kapex + G1 along the element axis."""

    @configclass
    class JointDescCfg(ObsGroup):
        joint_desc = ObsTerm(func=combined_joint_description)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class JointObsCfg(ObsGroup):
        joint_obs = ObsTerm(func=combined_joint_observation)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class FeetDescCfg(ObsGroup):
        feet_desc = ObsTerm(func=combined_feet_description)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class FeetObsCfg(ObsGroup):
        feet_obs = ObsTerm(func=combined_feet_observation)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class GlobalObsCfg(ObsGroup):
        global_obs = ObsTerm(func=combined_global_observation)

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
    """Domain randomization & resets for both robots."""

    # -- Description caches (combined: caches both robots on reset) ---------
    cache_joint_desc = EventTerm(func=cache_joint_desc_combined, mode="reset")
    cache_feet_desc = EventTerm(func=cache_feet_desc_combined, mode="reset")

    # -- Kapex randomization (mirrors kirack_env_cfg.py) -------------
    feet_material_kapex = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        min_step_count_between_reset=1000,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL[67]", "RL[67]"]),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    torso_mass_kapex = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="WL3"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "distribution": "gaussian",
            "recompute_inertia": False,
        },
    )
    scale_all_link_masses_kapex = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        min_step_count_between_reset=15000,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    torso_com_kapex = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="WL3"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    scale_joint_armature_kapex = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_stiffness_damping_kapex = EventTerm(
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
    reset_base_kapex = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_joints_kapex = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    push_robot_kapex = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    # -- G1 randomization ---------------------------------------------------
    feet_material_g1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        min_step_count_between_reset=1000,
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", body_names=[G1_ANKLE_BODIES_RE]),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    torso_mass_g1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", body_names=G1_TORSO_BODY),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "distribution": "gaussian",
            "recompute_inertia": False,
        },
    )
    scale_all_link_masses_g1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        min_step_count_between_reset=15000,
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    torso_com_g1 = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", body_names=G1_TORSO_BODY),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    scale_joint_armature_g1 = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", joint_names=[".*"]),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_stiffness_damping_g1 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    reset_base_g1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_g1"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_joints_g1 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_g1"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    push_robot_g1 = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot_g1"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    # -- Joint position noise (interval, both robots via default asset_cfg) --
    add_joint_noise_kapex = EventTerm(
        func=mdp.joint_position_noise,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={"asset_cfg": SceneEntityCfg("robot"), "noise_range": (-0.1, 0.1)},
    )
    add_joint_noise_g1 = EventTerm(
        func=mdp.joint_position_noise,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={"asset_cfg": SceneEntityCfg("robot_g1"), "noise_range": (-0.1, 0.1)},
    )


@configclass
class RewardsCfg:
    """Per-robot rewards. Generic env-level (alive) once."""

    # -- env-level / shared ---------------------------------------------------
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # -- Kapex rewards --------------------------------------------------------
    feet_to_close_kapex = RewTerm(
        func=mdp.feet_to_close, weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL6", "RL6"]),
            "threshold": 0.1,
        },
    )
    base_lin_vel_kapex = RewTerm(
        func=mdp.lin_vel_z_l2, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_ang_vel_kapex = RewTerm(
        func=mdp.ang_vel_xy_l2, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel_kapex = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc_kapex = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_pos_limits_kapex = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    energy_kapex = RewTerm(
        func=mdp.energy, weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    flat_orientation_kapex = RewTerm(
        func=mdp.flat_orientation_l2, weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height_kapex = RewTerm(
        func=mdp.base_height_l2, weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.91},
    )
    joint_deviation_arms_kapex = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LA.*", "RA.*"])},
    )
    joint_deviation_waists_kapex = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["WL.*"])},
    )
    joint_deviation_legs_kapex = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LLJ[12]", "RLJ[12]"])},
    )
    feet_slide_kapex = RewTerm(
        func=mdp.feet_slide, weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL[67]", "RL[67]"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LL[67]", "RL[67]"]),
        },
    )
    feet_clearance_kapex = RewTerm(
        func=mdp.foot_clearance_reward, weight=0.05,
        params={
            "std": 0.05, "tanh_mult": 2.0, "target_height": 0.3, "body_vel_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=["LL[67]", "RL[67]"]),
        },
    )
    undesired_contacts_kapex = RewTerm(
        func=mdp.undesired_contacts, weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*(LL[67]|RL[67]).*).*"]),
        },
    )

    # -- G1 rewards -----------------------------------------------------------
    feet_to_close_g1 = RewTerm(
        func=mdp.feet_to_close, weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", body_names=list(G1_FOOT_BODIES)),
            "threshold": 0.1,
        },
    )
    base_lin_vel_g1 = RewTerm(
        func=mdp.lin_vel_z_l2, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot_g1")},
    )
    base_ang_vel_g1 = RewTerm(
        func=mdp.ang_vel_xy_l2, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot_g1")},
    )
    joint_vel_g1 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot_g1")},
    )
    joint_acc_g1 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-6,
        params={"asset_cfg": SceneEntityCfg("robot_g1")},
    )
    dof_pos_limits_g1 = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot_g1")},
    )
    energy_g1 = RewTerm(
        func=mdp.energy, weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot_g1")},
    )
    flat_orientation_g1 = RewTerm(
        func=mdp.flat_orientation_l2, weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot_g1")},
    )
    base_height_g1 = RewTerm(
        func=mdp.base_height_l2, weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot_g1"), "target_height": G1_STANDING_HEIGHT},
    )
    joint_deviation_arms_g1 = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot_g1", joint_names=G1_ARM_JOINT_RE)},
    )
    joint_deviation_waists_g1 = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot_g1", joint_names=G1_WAIST_JOINT_RE)},
    )
    joint_deviation_legs_g1 = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot_g1", joint_names=G1_LEG_HIP_JOINT_RE)},
    )
    feet_slide_g1 = RewTerm(
        func=mdp.feet_slide, weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot_g1", body_names=[G1_ANKLE_BODIES_RE]),
            "sensor_cfg": SceneEntityCfg("contact_forces_g1", body_names=[G1_ANKLE_BODIES_RE]),
        },
    )
    feet_clearance_g1 = RewTerm(
        func=mdp.foot_clearance_reward, weight=0.05,
        params={
            "std": 0.05, "tanh_mult": 2.0, "target_height": 0.3, "body_vel_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot_g1", body_names=[G1_ANKLE_BODIES_RE]),
        },
    )
    undesired_contacts_g1 = RewTerm(
        func=mdp.undesired_contacts, weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces_g1", body_names=[G1_NOT_ANKLE_RE]),
        },
    )


@configclass
class CommandsCfg:
    """Single shared zero-velocity command (both robots are standing)."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        rel_standing_envs=0.05,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
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
    """Env resets if EITHER robot fails."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height_kapex = DoneTerm(
        func=mdp.root_height_below_minimum_relative,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.5},
    )
    bad_orientation_kapex = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},
    )
    base_height_g1 = DoneTerm(
        func=mdp.root_height_below_minimum_relative,
        params={"asset_cfg": SceneEntityCfg("robot_g1"), "minimum_height": 0.4},
    )
    bad_orientation_g1 = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot_g1"), "limit_angle": 0.8},
    )


@configclass
class CurriculumCfg:
    """Use only env-level curricula (terrain). Asset-specific curricula are
    Kapex-only in this repo, so we leave them off for the combined task."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_survival)


@configclass
class CombinedStandingEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL config: Kapex + G1 trained jointly per env."""

    scene: CombinedSceneCfg = CombinedSceneCfg(num_envs=4096, env_spacing=ENV_SPACING)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, -8.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.7)
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15  # doubled for 2 robots

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces_g1.update_period = self.sim.dt
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
    num_rows=3,
    num_cols=3,
    horizontal_scale=0.05,
    slope_threshold=0.0,
    curriculum=False,
    difficulty_range=(0.0, 0.0),
    sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class CombinedStandingPlayEnvCfg(CombinedStandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.viewer.origin_type = "world"
        self.viewer.eye = (6.0, 6.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.7)
        self.viewer.resolution = (1920, 1080)

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
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
