"""*******************************************************************************
* HumARConoid-KAPEX
*
* Advanced Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid-KAPEX.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
*******************************************************************************"""

"* Authors: Sol Choi *"

import os

from kirack import kirack_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import UrdfFileCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ARMATURE_RO80 = 0.084934656  # Hip Roll, Pitch & Ankle Pitch, Roll & Shoulder Pitch, Roll
ARMATURE_RO100 = 0.2808152064  # Hip Pitch, Knee
ARMATURE_RI60 = 0.001754431488  # Toe
ARMATURE_AK8064 = 0.2312192  # Waist RYP
ARMATURE_AK1093 = 0.0081162  # Shoulder Yaw, Elbow
ARMATURE_AK709 = 0.00288  # Wrist Yaw
ARMATURE_WRIST_RP = 0.00003232  # Wrist Roll & Pitch
ARMATURE_AK7010 = 0.00414  # Head Pitch
ARMATURE_AK606 = 0.0008766  # Head Yaw

NATURAL_FREQ = 5 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_RO80 = ARMATURE_RO80 * NATURAL_FREQ**2  # 83.8271454615
STIFFNESS_RO100 = ARMATURE_RO100 * NATURAL_FREQ**2  # 277.153499682
STIFFNESS_RI60 = ARMATURE_RI60 * NATURAL_FREQ**2  # 1.73155447344
STIFFNESS_AK8064 = ARMATURE_AK8064 * NATURAL_FREQ**2  # 228.204203381
STIFFNESS_AK1093 = ARMATURE_AK1093 * NATURAL_FREQ**2  # 8.01036832355
STIFFNESS_AK709 = ARMATURE_AK709 * NATURAL_FREQ**2  # 2.84244606735
STIFFNESS_WRIST_RP = ARMATURE_WRIST_RP * NATURAL_FREQ**2  # 0.0318985614225
STIFFNESS_AK7010 = ARMATURE_AK7010 * NATURAL_FREQ**2  # 4.08601622182
STIFFNESS_AK606 = ARMATURE_AK606 * NATURAL_FREQ**2  # 0.86516952175
DAMPING_RO80 = 2.0 * DAMPING_RATIO * ARMATURE_RO80 * NATURAL_FREQ  # 10.6732036527
DAMPING_RO100 = 2.0 * DAMPING_RATIO * ARMATURE_RO100 * NATURAL_FREQ  # 35.2882795767
DAMPING_RI60 = 2.0 * DAMPING_RATIO * ARMATURE_RI60 * NATURAL_FREQ  # 0.220468362951
DAMPING_AK8064 = 2.0 * DAMPING_RATIO * ARMATURE_AK8064 * NATURAL_FREQ  # 29.0558616027
DAMPING_AK1093 = 2.0 * DAMPING_RATIO * ARMATURE_AK1093 * NATURAL_FREQ  # 1.01991177177
DAMPING_AK709 = 2.0 * DAMPING_RATIO * ARMATURE_AK709 * NATURAL_FREQ  # 0.361911473683
DAMPING_WRIST_RP = 2.0 * DAMPING_RATIO * ARMATURE_WRIST_RP * NATURAL_FREQ  # 0.00406145098244
DAMPING_AK7010 = 2.0 * DAMPING_RATIO * ARMATURE_AK7010 * NATURAL_FREQ  # 0.52024774342
DAMPING_AK606 = 2.0 * DAMPING_RATIO * ARMATURE_AK606 * NATURAL_FREQ  # 0.110156804802

KAPEX_G1_CFG = ArticulationCfg(
    spawn=sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            # ---- KAPEX (URDF) ----
            sim_utils.UrdfFileCfg(
                asset_path=os.path.join(kirack_DATA_DIR, "KAPEX_wo_hand_head.urdf"),
                fix_base=False,
                merge_fixed_joints=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                ),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            # ---- G1 (USD) ----
            sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,  # G1 권장 설정
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
        ],
        random_choice=True,
    ),
    # ---- 공용 init_state: 두 로봇의 joint 이름이 안 겹치므로 한 dict에 합침 ----
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),  # 평균값. 로봇별로 안 맞으면 spawn 후 약간 떨어지지만 reset이 잡아줌
        joint_pos={
            # KAPEX
            "LLJ3": -0.035,
            "RLJ3": 0.035,
            "LLJ4": 0.38,
            "RLJ4": -0.38,
            "LLJ5": -0.33,
            "RLJ5": 0.33,
            "LAJ1": 0.2,
            "RAJ1": -0.2,
            "LAJ2": 0.2,
            "RAJ2": -0.2,
            "LAJ3": 0.18,
            "RAJ3": -0.18,
            "LAJ4": -0.35,
            "RAJ4": 0.35,
            # G1
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # ---- 공용 actuators: KAPEX/G1 joint regex 가 겹치지 않으면 둘 다 둬도 됨 ----
    actuators={
        # ===== KAPEX actuators (기존 그대로) =====
        "kapex_legs": DelayedPDActuatorCfg(
            joint_names_expr=["(L|R)LJ[1-4]"],
            effort_limit_sim={"(L|R)LJ[1-2]": 70, "(L|R)LJ[3-4]": 180},
            velocity_limit_sim={"(L|R)LJ[1-2]": 12, "(L|R)LJ[3-4]": 12},
            stiffness={"(L|R)LJ[1-2]": STIFFNESS_RO80, "(L|R)LJ[3-4]": STIFFNESS_RO100},
            damping={"(L|R)LJ[1-2]": DAMPING_RO80, "(L|R)LJ[3-4]": 20},
            armature={"(L|R)LJ[1-2]": ARMATURE_RO80, "(L|R)LJ[3-4]": ARMATURE_RO100},
            min_delay=0,
            max_delay=1,
        ),
        "kapex_feet": DelayedPDActuatorCfg(
            joint_names_expr=["(L|R)LJ[5-7]"],
            effort_limit_sim={"(L|R)LJ[5-6]": 60, "(L|R)LJ7": 20},
            velocity_limit_sim={"(L|R)LJ[5-6]": 10, "(L|R)LJ7": 5},
            stiffness={"(L|R)LJ[5-6]": 40, "(L|R)LJ7": 5},
            damping={"(L|R)LJ[5-6]": 2, "(L|R)LJ7": 0.05},
            armature={"(L|R)LJ[5-6]": 0.5 * ARMATURE_RO80, "(L|R)LJ7": 3.0 * ARMATURE_RI60},
            min_delay=0,
            max_delay=1,
        ),
        "kapex_torso_roll": DelayedPDActuatorCfg(
            joint_names_expr=["WLJ1"],
            effort_limit_sim=100,
            velocity_limit_sim=15,
            stiffness=STIFFNESS_AK8064,
            damping=DAMPING_AK8064,
            armature=ARMATURE_AK8064,
            min_delay=0,
            max_delay=1,
        ),
        "kapex_torso_yaw_pitch": DelayedPDActuatorCfg(
            joint_names_expr=["WLJ[2-3]"],
            effort_limit_sim=200,
            velocity_limit_sim=15,
            stiffness=2.0 * STIFFNESS_AK8064,
            damping=2.0 * DAMPING_AK8064,
            armature=2.0 * ARMATURE_AK8064,
            min_delay=0,
            max_delay=1,
        ),
        "kapex_arm": DelayedPDActuatorCfg(
            joint_names_expr=[".*AJ[1-7]"],
            effort_limit_sim={
                ".*AJ[1-2]": 70,
                ".*AJ[3-4]": 30,
                ".*AJ5": 20,
                ".*AJ[6-7]": 10,
            },
            velocity_limit_sim={".*AJ.*": 7},
            stiffness={
                ".*AJ[1-2]": STIFFNESS_RO80,
                ".*AJ[3-4]": STIFFNESS_AK1093,
                ".*AJ5": STIFFNESS_AK709,
                ".*AJ[6-7]": 5,
            },
            damping={
                ".*AJ[1-2]": DAMPING_RO80,
                ".*AJ[3-4]": DAMPING_AK1093,
                ".*AJ5": DAMPING_AK709,
                ".*AJ[6-7]": 0.05,
            },
            armature={
                ".*AJ[1-2]": ARMATURE_RO80,
                ".*AJ[3-4]": ARMATURE_AK1093,
                ".*AJ5": ARMATURE_AK709,
                ".*AJ[6-7]": 2 * ARMATURE_WRIST_RP,
            },
            min_delay=0,
            max_delay=1,
        ),
        # ===== G1 actuators (G1_CFG에서 복사) =====
        "g1_legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={".*_hip_.*": 0.01, ".*_knee_joint": 0.01, "torso_joint": 0.01},
        ),
        "g1_feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=20,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "g1_arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit_sim=300,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)
KAPEX0_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=os.path.join(kirack_DATA_DIR, "KAPEX_wo_hand_head.urdf"),
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.91),
        joint_pos={
            # ".*": 0.0,
            ## LEGS (14)
            "LLJ3": -0.035,
            "RLJ3": 0.035,
            "LLJ4": 0.38,
            "RLJ4": -0.38,
            "LLJ5": -0.33,
            "RLJ5": 0.33,
            ## ARMS (14)
            "LAJ1": 0.2,
            "RAJ1": -0.2,
            "LAJ2": 0.2,
            "RAJ2": -0.2,
            "LAJ3": 0.18,
            "RAJ3": -0.18,
            "LAJ4": -0.35,
            "RAJ4": 0.35,
            ## HANDS (40)
            # ".*HJ_index2": 0.2,
            # ".*HJ_index3": 0.2,
            # ".*HJ_little1": 0.2,
            # ".*HJ_little2": 0.2,
            # ".*HJ_middle2": 0.2,
            # ".*HJ_middle3": 0.2,
            # ".*HJ_ring2": 0.2,
            # ".*HJ_ring3": 0.2,
            # "LHJ_thumb0": -0.45,
            # "RHJ_thumb0": 0.45,
            # ".*HJ_thumb3": 0.3,
            # ".*HJ_thumb4": 0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["(L|R)LJ[1-4]"],
            effort_limit_sim={
                "(L|R)LJ[1-2]": 70,
                "(L|R)LJ[3-4]": 180,
            },
            velocity_limit_sim={
                "(L|R)LJ[1-2]": 12,
                "(L|R)LJ[3-4]": 12,
            },
            stiffness={
                "(L|R)LJ[1-2]": STIFFNESS_RO80,
                "(L|R)LJ[3-4]": STIFFNESS_RO100,
            },
            damping={
                "(L|R)LJ[1-2]": DAMPING_RO80,
                "(L|R)LJ[3-4]": 20,  # DAMPING_RO100,
            },
            armature={
                "(L|R)LJ[1-2]": ARMATURE_RO80,
                "(L|R)LJ[3-4]": ARMATURE_RO100,
            },
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=1,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=["(L|R)LJ[5-7]"],
            effort_limit_sim={
                "(L|R)LJ[5-6]": 60,
                "(L|R)LJ7": 20,
            },
            velocity_limit_sim={
                "(L|R)LJ[5-6]": 10,
                "(L|R)LJ7": 5,
            },
            stiffness={
                "(L|R)LJ[5-6]": 40,
                "(L|R)LJ7": 5,
            },
            damping={
                "(L|R)LJ[5-6]": 2,
                "(L|R)LJ7": 0.05,
            },
            armature={
                "(L|R)LJ[5-6]": 0.5 * ARMATURE_RO80,
                "(L|R)LJ7": 3.0 * ARMATURE_RI60,
            },
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=1,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "torso_roll": DelayedPDActuatorCfg(
            joint_names_expr=["WLJ1"],
            effort_limit_sim=100,
            velocity_limit_sim=15,
            stiffness=STIFFNESS_AK8064,
            damping=DAMPING_AK8064,
            armature=ARMATURE_AK8064,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=1,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "torso_yaw_pitch": DelayedPDActuatorCfg(
            joint_names_expr=["WLJ[2-3]"],
            effort_limit_sim=200,
            velocity_limit_sim=15,
            stiffness=2.0 * STIFFNESS_AK8064,
            damping=2.0 * DAMPING_AK8064,
            armature=2.0 * ARMATURE_AK8064,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=1,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        # "head": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*HLJ[1-2]"],
        #     effort_limit_sim=75,
        #     velocity_limit_sim=7,
        #     stiffness={
        #         "HLJ1": STIFFNESS_AK7010,
        #         "HLJ2": STIFFNESS_AK606,
        #     },
        #     damping={
        #         "HLJ1": DAMPING_AK7010,
        #         "HLJ2": DAMPING_AK606,
        #     },
        #     armature={
        #         "HLJ1": ARMATURE_AK7010,
        #         "HLJ2": ARMATURE_AK606,
        #     },
        #     min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
        #     max_delay=1,  # physics time steps (max: 2.0*4=8.0ms)
        # ),
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=[".*AJ[1-7]"],
            effort_limit_sim={
                ".*AJ[1-2]": 70,
                ".*AJ[3-4]": 30,
                ".*AJ5": 20,
                ".*AJ[6-7]": 10,
            },
            velocity_limit_sim={
                ".*AJ.*": 7,
            },
            stiffness={
                ".*AJ[1-2]": STIFFNESS_RO80,
                ".*AJ[3-4]": STIFFNESS_AK1093,
                ".*AJ5": STIFFNESS_AK709,
                ".*AJ[6-7]": 5,  # 2 * STIFFNESS_WRIST_RP,
            },
            damping={
                ".*AJ[1-2]": DAMPING_RO80,
                ".*AJ[3-4]": DAMPING_AK1093,
                ".*AJ5": DAMPING_AK709,
                ".*AJ[6-7]": 0.05,  # 2 * DAMPING_WRIST_RP,
            },
            armature={
                ".*AJ[1-2]": ARMATURE_RO80,
                ".*AJ[3-4]": ARMATURE_AK1093,
                ".*AJ5": ARMATURE_AK709,
                ".*AJ[6-7]": 2 * ARMATURE_WRIST_RP,
            },
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=1,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        # "hand": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*HJ_(thumb|index|middle|ring|little)[0-4]"],
        #     effort_limit_sim=10,
        #     velocity_limit_sim=2,
        #     stiffness=10,
        #     damping=0.1,
        #     armature=0.001,
        #     min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
        #     max_delay=1,  # physics time steps (max: 2.0*4=8.0ms)
        # ),
    },
)
"""Configuration for the KAPEX_GEN2_WHOLEBODY_CFG robot."""

# """Joint names of KAPEX Gen2 Wholebody robot."""
# 0  'LLJ1'
# 1  'RLJ1'
# 2  'WLJ1'
# 3  'LLJ2'
# 4  'RLJ2'
# 5  'WLJ2'
# 6  'LLJ3'
# 7  'RLJ3'
# 8  'WLJ3'
# 9  'LLJ4'
# 10 'RLJ4'
# 11 'HLJ1'
# 12 'LAJ1'
# 13 'RAJ1'
# 14 'LLJ5'
# 15 'RLJ5'
# 16 'HLJ2'
# 17 'LAJ2'
# 18 'RAJ2'
# 19 'LLJ6'
# 20 'RLJ6'
# 21 'LAJ3'
# 22 'RAJ3'
# 23 'LLJ7'
# 24 'RLJ7'
# 25 'LAJ4'
# 26 'RAJ4'
# 27 'LAJ5'
# 28 'RAJ5'
# 29 'LAJ6'
# 30 'RAJ6'
# 31 'LAJ7'
# 32 'RAJ7'
# 33 'LHJ_index0'
# 34 'LHJ_little0'
# 35 'LHJ_middle0'
# 36 'LHJ_ring0'
# 37 'LHJ_thumb0'
# 38 'RHJ_index0'
# 39 'RHJ_little0'
# 40 'RHJ_middle0'
# 41 'RHJ_ring0'
# 42 'RHJ_thumb0'
# 43 'LHJ_index1'
# 44 'LHJ_little1'
# 45 'LHJ_middle1'
# 46 'LHJ_ring1'
# 47 'LHJ_thumb1'
# 48 'RHJ_index1'
# 49 'RHJ_little1'
# 50 'RHJ_middle1'
# 51 'RHJ_ring1'
# 52 'RHJ_thumb1'
# 53 'LHJ_index2'
# 54 'LHJ_little2'
# 55 'LHJ_middle2'
# 56 'LHJ_ring2'
# 57 'LHJ_thumb2'
# 58 'RHJ_index2'
# 59 'RHJ_little2'
# 60 'RHJ_middle2'
# 61 'RHJ_ring2'
# 62 'RHJ_thumb2'
# 63 'LHJ_index3'
# 64 'LHJ_middle3'
# 65 'LHJ_ring3'
# 66 'LHJ_thumb3'
# 67 'RHJ_index3'
# 68 'RHJ_middle3'
# 69 'RHJ_ring3'
# 70 'RHJ_thumb3'
# 71 'LHJ_thumb4'
# 72 'RHJ_thumb4'
