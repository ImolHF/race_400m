"""Manager-based RL configuration for G1 following a fixed 400 m path."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp

from .mdp.rewards import (
    alignment_reward as _alignment_reward,
    alive_reward as _alive_reward,
    deviation_penalty as _deviation_penalty,
    alternating_foot_gait as _alternating_foot_gait,
    crossed_feet_penalty as _crossed_feet_penalty,
    course_heading_alignment as _course_heading_alignment,
    feet_air_time_positive_biped as _feet_air_time_positive_biped,
    forward_course_speed as _forward_course_speed,
    forward_foot_landing as _forward_foot_landing,
    compact_stride_landing_penalty as _compact_stride_landing_penalty,
    contact_synchronized_arm_swing as _contact_synchronized_arm_swing,
    contact_foot_velocity_penalty as _contact_foot_velocity_penalty,
    contact_foot_yaw_error as _contact_foot_yaw_error,
    excess_swing_time_penalty as _excess_swing_time_penalty,
    arm_counter_swing_penalty as _arm_counter_swing_penalty,
    lateral_velocity_penalty as _lateral_velocity_penalty,
    progress_reward as _progress_reward,
    reached_checkpoint as _reached_checkpoint,
    reset_path_progress as _reset_path_progress,
    target_direction as _target_direction,
    target_distance as _target_distance,
    swing_foot_clearance as _swing_foot_clearance,
    swing_foot_forward as _swing_foot_forward,
    rear_swing_foot_penalty as _rear_swing_foot_penalty,
)
from .mdp.terminations import is_completed as _is_completed
from .mdp.terminations import robot_fallen as _robot_fallen
from .track_scene_cfg import TrackSceneCfg, TrackpointCfg


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception plus the next precomputed target; no sensors are used."""

        joint_pos = ObsTerm(func="isaaclab.envs.mdp:joint_pos_rel", params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func="isaaclab.envs.mdp:joint_vel_rel", params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func="isaaclab.envs.mdp:base_lin_vel", params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func="isaaclab.envs.mdp:base_ang_vel", params={"asset_cfg": SceneEntityCfg("robot")}, scale=0.25)
        projected_gravity = ObsTerm(func="isaaclab.envs.mdp:projected_gravity", params={"asset_cfg": SceneEntityCfg("robot")})
        previous_action = ObsTerm(func="isaaclab.envs.mdp:last_action")
        target_direction = ObsTerm(func=_target_direction)
        target_distance = ObsTerm(func=_target_distance)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Leg actions plus sagittal-plane shoulder swing with locked elbows."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        # The only arm DoFs exposed to PPO are left/right shoulder pitch.
        # Elbows, shoulder roll, shoulder yaw, and wrists keep their default
        # position targets through their dedicated PD actuators.
        joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", ".*_shoulder_pitch_joint"],
        scale={
            ".*_hip_.*": 0.25,
            ".*_knee_joint": 0.25,
            ".*_ankle_.*": 0.20,
            ".*_shoulder_pitch_joint": 0.18,
        },
        use_default_offset=True,
    )


@configclass
class LegOnlyActionsCfg:
    """Original 12-DoF leg controller; arms remain at their safe defaults."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
        scale={
            ".*_hip_.*": 0.25,
            ".*_knee_joint": 0.25,
            ".*_ankle_.*": 0.20,
        },
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    # The order is important. Progress must be evaluated before checkpoint
    # advancement resets the potential for the next target.
    progress = RewTerm(func=_progress_reward, weight=8.0)
    reached_checkpoint = RewTerm(func=_reached_checkpoint, weight=5.0)
    # Retained only as a weak recovery signal when off the centre line.  The
    # terms immediately below define the intended forward-running solution.
    alignment = RewTerm(func=_alignment_reward, weight=0.20)
    course_heading = RewTerm(func=_course_heading_alignment, weight=1.25)
    forward_course_speed = RewTerm(func=_forward_course_speed, weight=1.50)
    lateral_velocity = RewTerm(func=_lateral_velocity_penalty, weight=-1.00)
    deviation = RewTerm(func=_deviation_penalty, weight=0.5)
    alive = RewTerm(func=_alive_reward, weight=0.02)
    # Stability and smoothness terms are essential when learning bipedal
    # locomotion from scratch.  The task terms above alone reward any brief
    # forward motion, including a forward fall.
    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-100.0)
    flat_orientation = RewTerm(func=isaac_mdp.flat_orientation_l2, weight=-2.0)
    # A modestly lower trunk target gives the leg-only policy a small knee bend
    # while moving.  This reduces the center of mass and improves recovery
    # margin without forcing a deep squat that would harm forward progress.
    base_height = RewTerm(func=isaac_mdp.base_height_l2, weight=-12.0, params={"target_height": 0.70})
    lin_vel_z = RewTerm(func=isaac_mdp.lin_vel_z_l2, weight=-4.0)
    ang_vel_xy = RewTerm(func=isaac_mdp.ang_vel_xy_l2, weight=-0.1)
    joint_vel = RewTerm(func=isaac_mdp.joint_vel_l2, weight=-2.0e-4)
    action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(func=isaac_mdp.joint_pos_limits, weight=-2.0)
    # The official G1 task applies an extra ankle-limit cost because ankle
    # saturation is a common precursor to unstable landings and inward knees.
    ankle_pos_limits = RewTerm(
        func=isaac_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])} ,
    )

    # Imported from Isaac Lab's official G1 velocity task.  These shape the
    # gait only; all waypoint, progress, and completion rewards stay intact.
    hip_deviation = RewTerm(
        func=isaac_mdp.joint_deviation_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_acc = RewTerm(
        func=isaac_mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    joint_torques = RewTerm(
        func=isaac_mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    feet_air_time = RewTerm(
        func=_feet_air_time_positive_biped,
        weight=0.10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.22,
        },
    )
    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    arm_posture = RewTerm(
        func=isaac_mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"]
            )
        },
    )
    shoulder_pitch_deviation = RewTerm(
        func=isaac_mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint")},
    )
    # Unitree RL Gym's contact_no_vel: a loaded foot should not translate or
    # bounce.  This reduces stance-leg wobble without prescribing a stride.
    contact_no_vel = RewTerm(
        func=_contact_foot_velocity_penalty,
        weight=-0.04,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # Keep each loaded foot parallel to the pelvis/track heading.  This is a
    # direct toe-in/toe-out cost and is deliberately contact-gated so swing
    # clearance and foot placement are not constrained.
    foot_yaw = RewTerm(
        func=_contact_foot_yaw_error,
        weight=-1.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )
    # Standard Isaac Lab undesired-contact term.  With self-collision enabled,
    # a knee or hip link striking the floor or another body is explicitly
    # costly instead of being an exploitable way to stabilize the policy.
    undesired_leg_contacts = RewTerm(
        func=isaac_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_hip_.*_link", ".*_knee_link"]
            ),
        },
    )
    # Unitree RL Lab and Unitree RL Mjlab both use an explicit alternating
    # contact schedule and swing-foot clearance for G1.  The terms below are
    # adapted to this task's waypoint-driven movement gate.
    alternating_gait = RewTerm(
        func=_alternating_foot_gait,
        weight=0.15,
        params={
            # 0.50 s is a controlled increase from the previous 0.55 s gait
            # period.  Step placement remains constrained by the compact-stride
            # terms below, so speed comes from cadence rather than overstriding.
            "period": 0.50,
            "offset": (0.0, 0.5),
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )
    swing_clearance = RewTerm(
        func=_swing_foot_clearance,
        weight=0.08,
        params={
            "target_height": 0.06,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    # This directly addresses the observed cross-step while preserving the
    # source projects' preference for symmetric alternating contacts.
    crossed_feet = RewTerm(
        func=_crossed_feet_penalty,
        weight=-0.5,
        params={
            "min_half_width": 0.04,
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    # Forward-swing and forward-landing rewards use the existing GPU contact
    # sensor.  They apply only to a foot in flight/at touchdown, never to both
    # grounded feet, avoiding a two-foot hopping solution.
    swing_forward = RewTerm(
        func=_swing_foot_forward,
        weight=0.12,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    forward_landing = RewTerm(
        func=_forward_foot_landing,
        weight=0.40,
        params={
            "min_forward": 0.03,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    # Compact running: discourage long airborne phases and large touchdown
    # reach.  Together with the faster alternating-contact period, these steer
    # the policy toward short, quick steps instead of hopping.
    excess_swing_time = RewTerm(
        func=_excess_swing_time_penalty,
        weight=-0.70,
        params={
            "max_air_time": 0.18,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )
    compact_stride = RewTerm(
        func=_compact_stride_landing_penalty,
        weight=-5.0,
        params={
            "max_forward": 0.17,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    rear_swing_foot = RewTerm(
        func=_rear_swing_foot_penalty,
        weight=-4.0,
        params={
            "max_rearward": 0.16,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    arm_swing = RewTerm(
        func=_contact_synchronized_arm_swing,
        weight=0.25,
        params={
            "max_speed": 0.8,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_elbow_link", "right_elbow_link"]),
        },
    )
    arm_counter_swing = RewTerm(
        func=_arm_counter_swing_penalty,
        weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_elbow_link", "right_elbow_link"]),
        },
    )


@configclass
class LegOnlyRewardsCfg(RewardsCfg):
    """Compact, safe gait shaping without arm-control objectives."""

    arm_posture = None
    shoulder_pitch_deviation = None
    arm_swing = None
    arm_counter_swing = None

    # Return to the more forgiving cadence that previously completed the lap.
    # Compact-stride and low-air-time limits still prevent hopping.
    alternating_gait = RewTerm(
        func=_alternating_foot_gait,
        weight=0.15,
        params={
            "period": 0.55,
            "offset": (0.0, 0.5),
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )


@configclass
class EventsCfg:
    """Keep physical and navigation state synchronized on every reset."""

    reset_scene = EventTerm(func=isaac_mdp.reset_scene_to_default, mode="reset")
    reset_path_progress = EventTerm(func=_reset_path_progress, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func="isaaclab.envs.mdp:time_out", time_out=True)
    robot_fallen = DoneTerm(func=_robot_fallen, params={"asset_cfg": SceneEntityCfg("robot")})
    completed = DoneTerm(func=_is_completed)


@configclass
class RaceEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed target-point navigation task for the G1 robot."""

    # This is the number of environments *per GPU*.  The two-GPU command
    # collects 2 * 4096 = 8,192 environments per PPO update.  Concentrating
    # environments on two 72-GB GPUs raises per-device PhysX/CUDA occupancy
    # when a many-GPU run is bottlenecked by host-side simulation work.
    scene: TrackSceneCfg = TrackSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    decimation: int = 4
    episode_length_s: float = 120.0

    path_points: list[tuple[float, float]] | None = None

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # PhysX GPU buffers sized for 4,096 G1 articulations on one device.
        # These avoid buffer growth and broad-phase capacity stalls at scale.
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**22
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        self.sim.physx.gpu_max_num_partitions = 32
        # Keep contact history in sync with the physics step for gait rewards.
        self.scene.contact_forces.update_period = self.sim.dt

        track_cfg = TrackpointCfg()
        self.path_points = track_cfg.path_points
        print(f"[INFO] Fixed navigation track: {len(self.path_points)} targets, 0 m to 400 m.")


@configclass
class LegOnlyRaceEnvCfg(RaceEnvCfg):
    """Recovery task compatible with the original 12-action race policy."""

    actions: LegOnlyActionsCfg = LegOnlyActionsCfg()
    rewards: LegOnlyRewardsCfg = LegOnlyRewardsCfg()
