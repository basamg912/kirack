# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
#   Author : Sangyun Bae - Research Intern, KIST ARC LAB
#   https://tkdyun.xyz
#
# SPDX-License-Identifier: BSD-3-Clause

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
    HfRandomUniformTerrainCfg,  # 랜덤 ruggedly terrain
    HfWaveTerrainCfg,  # 파도모양 terrain
    HfPyramidSlopedTerrainCfg,
    MeshPlaneTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
    MeshRandomGridTerrainCfg
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniNoise

# -----------------------------------------config
import kirack.tasks.locomotion.kapex0_vel_track.config.mdp as mdp
from kirack.assets.robot import KAPEX0_CFG

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


@configclass
class KirackSceneCfg(InteractiveSceneCfg):
    default_plane = AssetBaseCfg(
        prim_path="/World/default",
        spawn = sim_utils.GroundPlaneCfg(),
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
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/WL3",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class EventCfg:
    # ----- debug
    # ! check
    # debug_com = EventTerm(
    #     func = mdp.print_com,
    #     mode="reset",
    #     params={},
    # )
    # ------------------------startup
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
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
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
        interval_range_s=(3.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            }
        },
    )
    # ------------------------


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        rel_standing_envs=0.3,  # 30% stand — locomotion+standing 통합 학습
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # ! start low level command
            lin_vel_x=(-0.1, 0.5),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.2, 0.2),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.2),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.2, 0.2),
        ),
        resampling_time_range=(4.0, 5.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func = mdp.base_lin_vel, noise =UniNoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=UniNoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniNoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=UniNoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=UniNoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        dif_torso_com = ObsTerm(func=mdp.torso_com, params={"asset_cfg": SceneEntityCfg("robot", body_names="WL3")})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # ! critic network 는 noise X
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        dif_torso_com = ObsTerm(func=mdp.torso_com, params={"asset_cfg": SceneEntityCfg("robot", body_names="WL3")})
        # ! flat terrain 이라서 비활성
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        obs_feet_air_time = ObsTerm(
            func=mdp.obs_feet_air_time,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LL[67]", "RL[67]"]),
            },
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # cmd-aware tracking: cmd=0 → 정지 보상, cmd>0 → 명령 추종 보상
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # alive = RewTerm(func=mdp.is_alive, weight=0.25)
    # 507 alive -> terminated 변경
    terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)  # ! 4/10 -0.001 -> 0.005
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)  # ! 4/10 -2.5e-7 -> -2.5e-6
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)  # ! 4/10 -0.05 -> 0.15
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
    # joint_deviation_legs를 cmd-conditional 버전으로 교체.
    # 보행 시: baseline weight, 정지 시: stand_still_scale × weight 로 강화.
    joint_position_penalty = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["LLJ[12]", "RLJ[12]"]),
            "stand_still_scale": 10.0,
            "velocity_threshold": 0.1,
        },
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
            "offset": [0.0, 0.05, 0.0, 0.05],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LL6", "LL7", "RL6", "RL7"]),
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

    # -- standing 전용 (cmd≈0 일 때만 발화)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LL[67]", "RL[67]"]),
            "command_name": "base_velocity",
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
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum_relative, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_survival)
    lin_vel_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    joint_armature_levels = CurrTerm(
        func=mdp.actuator_armature_range_levels,
        params={
            "threshold" : 0.5,
        },
    )
    stiff_damping_levels = CurrTerm(
        func=mdp.actuator_gain_range_levels,
        params={
            "threshold" : 0.5,
        },
    )
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
class Kapex0VelTrackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: KirackSceneCfg = KirackSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # ! 1개 policy step -> 4개 동일 action
        # ! delay : action 이 적용되기까지 걸리는 physical step
        # 0.005s -> delay=4 면 0.02s => 50Hz
        # control dt = 0.02s -> 50Hz
        # 물리엔진 dt = 0.005 -> 200Hz
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.005  # ! 200Hz , 5ms
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
