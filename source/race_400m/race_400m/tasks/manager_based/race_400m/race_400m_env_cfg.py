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

from .mdp.rewards import (
    alignment_reward as _alignment_reward,
    alive_reward as _alive_reward,
    deviation_penalty as _deviation_penalty,
    progress_reward as _progress_reward,
    reached_checkpoint as _reached_checkpoint,
    reset_path_progress as _reset_path_progress,
    target_direction as _target_direction,
    target_distance as _target_distance,
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
        target_direction = ObsTerm(func=_target_direction)
        target_distance = ObsTerm(func=_target_distance)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Joint-position residual actions about G1's configured default pose."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_joint"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    reached_checkpoint = RewTerm(func=_reached_checkpoint, weight=10.0)
    progress = RewTerm(func=_progress_reward, weight=5.0)
    alignment = RewTerm(func=_alignment_reward, weight=0.5)
    deviation = RewTerm(func=_deviation_penalty, weight=0.5)
    alive = RewTerm(func=_alive_reward, weight=0.02)


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

    scene: TrackSceneCfg = TrackSceneCfg(num_envs=16, env_spacing=2.5)
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

        track_cfg = TrackpointCfg()
        self.path_points = track_cfg.path_points
        print(f"[INFO] Fixed navigation track: {len(self.path_points)} targets, 0 m to 400 m.")
