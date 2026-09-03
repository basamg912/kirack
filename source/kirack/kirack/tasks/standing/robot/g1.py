"""Unitree G1 23DOF ArticulationCfg for the standing task.

Mirrors the Kapex articulation pattern (DelayedPDActuator-based) so the same
standing pipeline can train G1 with the same URMA-style attention policy.
Actuator params follow the Unitree G1 spec used in ``utils/unitree.py``.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import UrdfFileCfg
from isaaclab.sim.converters import UrdfConverterCfg

from kirack import kirack_EXT_DIR

G1_23DOF_DATA_DIR = os.path.join(
    kirack_EXT_DIR, "tasks", "environment", "g1", "robots", "g1_description"
)
G1_URDF_PATH = os.path.join(G1_23DOF_DATA_DIR, "g1_23dof.urdf")

G1_23DOF_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=G1_URDF_PATH,
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch_yaw_waist": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_pitch_joint": 100.0,
                ".*_hip_yaw_joint": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 2.0,
                ".*_hip_yaw_joint": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
        "hip_roll_knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_roll_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
        "shoulder_elbow_wrist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
        "ankle": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
    },
)
"""Unitree G1 23DOF whole-body configuration matching the standing task style."""
