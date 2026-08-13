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
    feet_air_time_positive_biped as _feet_air_time_positive_biped,
    progress_reward as _progress_reward,
    reached_checkpoint as _reached_checkpoint,
    reset_path_progress as _reset_path_progress,
    target_direction as _target_direction,
    target_distance as _target_distance,
    swing_foot_clearance as _swing_foot_clearance,
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
    """Leg-only residual actions about G1's configured default standing pose.

    Keeping the torso and arms at their default targets makes the initial
    locomotion problem tractable: the policy first learns balance and stepping
    instead of exploiting large upper-body motions.
    """

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    # The order is important. Progress must be evaluated before checkpoint
    # advancement resets the potential for the next target.
    progress = RewTerm(func=_progress_reward, weight=8.0)
    reached_checkpoint = RewTerm(func=_reached_checkpoint, weight=5.0)
    alignment = RewTerm(func=_alignment_reward, weight=0.75)
    deviation = RewTerm(func=_deviation_penalty, weight=0.5)
    alive = RewTerm(func=_alive_reward, weight=0.02)
    # Stability and smoothness terms are essential when learning bipedal
    # locomotion from scratch.  The task terms above alone reward any brief
    # forward motion, including a forward fall.
    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-100.0)
    flat_orientation = RewTerm(func=isaac_mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(func=isaac_mdp.base_height_l2, weight=-8.0, params={"target_height": 0.74})
    lin_vel_z = RewTerm(func=isaac_mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=isaac_mdp.ang_vel_xy_l2, weight=-0.1)
    joint_vel = RewTerm(func=isaac_mdp.joint_vel_l2, weight=-2.0e-4)
    action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(func=isaac_mdp.joint_pos_limits, weight=-2.0)

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
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
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
    # Unitree RL Lab and Unitree RL Mjlab both use an explicit alternating
    # contact schedule and swing-foot clearance for G1.  The terms below are
    # adapted to this task's waypoint-driven movement gate.
    alternating_gait = RewTerm(
        func=_alternating_foot_gait,
        weight=0.15,
        params={
            "period": 0.8,
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
            "target_height": 0.10,
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
