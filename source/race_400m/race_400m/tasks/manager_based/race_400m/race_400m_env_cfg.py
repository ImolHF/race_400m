# source/race_400m/race_400m/tasks/manager_based/race_400m/race_env_cfg.py

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.utils import configclass

from .track_scene_cfg import TrackSceneCfg, TrackpointCfg

# 导入自定义 MDP 函数
from .mdp.rewards import (
    reached_checkpoint as _reached_checkpoint,
    progress_reward as _progress_reward,
    deviation_penalty as _deviation_penalty,
    backward_penalty as _backward_penalty,
    alive_reward as _alive_reward,
    move_reward as _move_reward,
)
from .mdp.terminations import (
    robot_fallen as _robot_fallen,
    is_completed as _is_completed,
)


# ============================================================
# 1. 观测配置
# ============================================================
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        joint_pos = ObsTerm(func="isaaclab.envs.mdp:joint_pos_rel", params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func="isaaclab.envs.mdp:joint_vel_rel", params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func="isaaclab.envs.mdp:base_lin_vel", params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func="isaaclab.envs.mdp:base_ang_vel", params={"asset_cfg": SceneEntityCfg("robot")}, scale=0.25)
        base_quat = ObsTerm(func="isaaclab.envs.mdp:base_quat", params={"asset_cfg": SceneEntityCfg("robot")})
        base_height = ObsTerm(func="isaaclab.envs.mdp:base_pos_z", params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================
# 2. 动作配置
# ============================================================
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_joint"],
        scale=0.5,
    )


# ============================================================
# 3. 奖励配置
# ============================================================
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reached_checkpoint = RewTerm(func=_reached_checkpoint, weight=10.0)
    progress = RewTerm(func=_progress_reward, weight=0.1)
    deviation = RewTerm(func=_deviation_penalty, weight=1.0)
    backward = RewTerm(func=_backward_penalty, weight=0.5)
    alive = RewTerm(func=_alive_reward, weight=0.01)
    move = RewTerm(func=_move_reward, weight=0.01)


# ============================================================
# 4. 终止配置
# ============================================================
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func="isaaclab.envs.mdp:time_out", time_out=True)
    robot_fallen = DoneTerm(func=_robot_fallen, params={"asset_cfg": SceneEntityCfg("robot")})
    completed = DoneTerm(func=_is_completed)


# ============================================================
# 5. 环境配置
# ============================================================
@configclass
class RaceEnvCfg(ManagerBasedRLEnvCfg):
    """400米竞速环境配置 (Manager-based)"""

    # 场景配置
    scene: TrackSceneCfg = TrackSceneCfg(num_envs=1, env_spacing=2.0)

    # MDP 配置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 仿真参数
    decimation: int = 4
    episode_length_s: float = 60.0

    # 路径点数据
    path_points: list = None
    path_points_tensor: torch.Tensor = None

    def __post_init__(self):
        super().__post_init__()

        # 仿真步长
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        # 生成路径点
        track_cfg = TrackpointCfg()
        self.path_points = track_cfg.path_points

        # 转换为 torch 张量
        points_array = np.array(self.path_points)
        self.path_points_tensor = torch.tensor(points_array, dtype=torch.float32)

        # 注入场景
        self.scene.path_points = self.path_points

        # 设置机器人初始位置在起点
        start_x, start_z = self.path_points[0]
        self.scene.robot.init_state.pos = (start_x, 0.05, start_z)

        print(f"[INFO] 环境配置加载完成")
        print(f"[INFO] 路径点数量: {len(self.path_points)}")
        print(f"[INFO] 起点: ({self.path_points[0][0]:.2f}, {self.path_points[0][1]:.2f})")
        print(f"[INFO] 终点: ({self.path_points[-1][0]:.2f}, {self.path_points[-1][1]:.2f})")