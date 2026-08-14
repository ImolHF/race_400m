"""Unitree G1 articulation configuration used by the RL environment."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[4]
G1_USD_PATH = (
    PROJECT_ROOT
    / "unitree_model"
    / "G1"
    / "29dof"
    / "usd"
    / "g1_29dof_rev_1_0"
    / "g1_29dof_rev_1_0.usd"
).as_posix()


G1_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        # Required for the GPU contact sensor used by gait-shaping rewards.
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # A real G1 cannot pass one leg through the other.  Keeping this
            # enabled closes the reward-exploitation loophole that produced
            # knee/leg mesh penetration in the previous policies.
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            "left_hip_pitch_joint": -0.20,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.42,
            "left_ankle_pitch_joint": -0.23,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.20,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.42,
            "right_ankle_pitch_joint": -0.23,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
    ),
    actuators={
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["left_hip.*", "left_knee.*", "left_ankle.*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=50.0,
            stiffness={
                "left_hip_yaw_joint": 280.0,
                "left_hip_roll_joint": 280.0,
                "left_hip_pitch_joint": 360.0,
                "left_knee_joint": 360.0,
                "left_ankle_pitch_joint": 220.0,
                "left_ankle_roll_joint": 220.0,
            },
            damping={
                "left_hip_yaw_joint": 14.0,
                "left_hip_roll_joint": 14.0,
                "left_hip_pitch_joint": 20.0,
                "left_knee_joint": 20.0,
                "left_ankle_pitch_joint": 12.0,
                "left_ankle_roll_joint": 12.0,
            },
        ),
        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["right_hip.*", "right_knee.*", "right_ankle.*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=50.0,
            stiffness={
                "right_hip_yaw_joint": 280.0,
                "right_hip_roll_joint": 280.0,
                "right_hip_pitch_joint": 360.0,
                "right_knee_joint": 360.0,
                "right_ankle_pitch_joint": 220.0,
                "right_ankle_roll_joint": 220.0,
            },
            damping={
                "right_hip_yaw_joint": 14.0,
                "right_hip_roll_joint": 14.0,
                "right_hip_pitch_joint": 20.0,
                "right_knee_joint": 20.0,
                "right_ankle_pitch_joint": 12.0,
                "right_ankle_roll_joint": 12.0,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist.*"],
            effort_limit_sim=150.0,
            velocity_limit_sim=50.0,
            stiffness=250.0,
            damping=25.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder.*", "left_elbow.*", "left_wrist.*"],
            effort_limit_sim=50.0,
            velocity_limit_sim=50.0,
            stiffness=120.0,
            damping=12.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder.*", "right_elbow.*", "right_wrist.*"],
            effort_limit_sim=50.0,
            velocity_limit_sim=50.0,
            stiffness=120.0,
            damping=12.0,
        ),
    },
)
