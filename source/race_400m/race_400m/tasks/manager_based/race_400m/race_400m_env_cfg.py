# source/race_400m/race_400m/tasks/manager_based/race_400m/race_env_cfg.py

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationManagerCfg,
    RewardManagerCfg,
    TerminationManagerCfg,
    ActionManagerCfg,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from .field_scene_cfg import TrackSceneCfg
from .trackpoint_cfg import TrackpointCfg


# ============================================================
# 自定义奖励函数（在 mdp/rewards.py 中定义）
# ============================================================
# 这里先占位，稍后我们会创建 mdp/rewards.py 文件
# 现在先定义一个占位函数，让配置能通过


def reached_checkpoint(env):
    """判断是否到达下一个路径点"""
    # 这个函数会在 mdp/rewards.py 中实现
    # 现在返回 0 作为占位
    return 0.0


def progress_reward(env):
    """计算向目标点前进的奖励"""
    return 0.0


def is_completed(env):
    """判断是否跑完全程"""
    return False


def robot_fallen(env):
    """判断机器人是否摔倒"""
    return False


# ============================================================
# 环境配置
# ============================================================
@configclass
class RaceEnvCfg(ManagerBasedRLEnvCfg):
    """400米竞速环境配置 (Manager-based)"""

    # ============================================================
    # 1. 场景配置
    # ============================================================
    scene: TrackSceneCfg = TrackSceneCfg(num_envs=1, env_spacing=2.0)

    # ============================================================
    # 2. 观测管理器
    # ============================================================
    observations: ObservationManagerCfg = ObservationManagerCfg(
        policy={
            "joint_pos": {
                "func": "isaaclab.managers.ObservationManagerCfg.compute_joint_pos",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
                "scale": 1.0,
            },
            "joint_vel": {
                "func": "isaaclab.managers.ObservationManagerCfg.compute_joint_vel",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
                "scale": 1.0,
            },
            "base_pos": {
                "func": "isaaclab.managers.ObservationManagerCfg.compute_base_pos",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
                "scale": 1.0,
            },
            "base_quat": {
                "func": "isaaclab.managers.ObservationManagerCfg.compute_base_quat",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
                "scale": 1.0,
            },
            "base_lin_vel": {
                "func": "isaaclab.managers.ObservationManagerCfg.compute_base_lin_vel",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
                "scale": 1.0,
            },
            "base_ang_vel": {
                "func": "isaaclab.managers.ObservationManagerCfg.compute_base_ang_vel",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
                "scale": 1.0,
            },
        }
    )

    # ============================================================
    # 3. 动作管理器
    # ============================================================
    actions: ActionManagerCfg = ActionManagerCfg(
        joint_pos={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale": 0.5,
            "offset": 0.0,
            "joint_names": [".*_joint"],
        }
    )

    # ============================================================
    # 4. 奖励管理器
    # ============================================================
    rewards: RewardManagerCfg = RewardManagerCfg(
        terms={
            # 到达路径点奖励
            "reached_checkpoint": {
                "func": "race_400m.tasks.manager_based.race_400m.mdp.rewards.reached_checkpoint",
                "weight": 10.0,
            },
            # 向目标点前进奖励
            "progress": {
                "func": "race_400m.tasks.manager_based.race_400m.mdp.rewards.progress_reward",
                "weight": 0.1,
            },
            #偏离路径点惩罚
            "deviation":{
               "func":"race_400m.tasks.manager_based.race_400m.mdp.rewards.deviation_penalty",
                "weight":1.0
            },#后退惩罚
            "backward":{
                "func":"race_400m.tasks.manager_based.race_400m.mdp.rewards.backward_penalty",
                "weight":0.5,
            },
            # 存活奖励
            "alive": {
                "func": "race_400m.tasks.manager_based.race_400m.mdp.rewards.alive_reward",
                "weight": 0.01,
            },
            #运动奖励
            "move": {
                "func": "race_400m.tasks.manager_based.race_400m.mdp.rewards.move_reward",
                "weight": 0.01,
            },
        }
    )

    # ============================================================
    # 5. 终止管理器
    # ============================================================
    terminations: TerminationManagerCfg = TerminationManagerCfg(
        terms={
            "timeout": {
                "func": "isaaclab.managers.TerminationManagerCfg.timeout",
                "params": {"timeout": 60.0},
            },
            "robot_fallen": {
                "func": "race_400m.tasks.manager_based.race_400m.mdp.terminations.robot_fallen",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
            },
            "completed": {
                "func": "race_400m.tasks.manager_based.race_400m.mdp.terminations.is_completed",
                "params": {"asset_cfg": SceneEntityCfg("robot")},
            },
        }
    )

    # ============================================================
    # 6. 仿真参数
    # ============================================================
    sim = {
        "dt": 0.01,
        "render_interval": 1,
        "gravity": (0.0, 0.0, -9.81),
    }

    max_episode_length_s: float = 60.0
    decimation: int = 4

    viewer = {
        "eye": (5.0, 5.0, 5.0),
        "lookat": (0.0, 0.0, 0.0),
    }

    # ============================================================
    # 7. 路径点数据
    # ============================================================
    path_points: list = None
    path_points_tensor: torch.Tensor = None

    # ============================================================
    # 8. 初始化方法
    # ============================================================
    def __post_init__(self):
        super().__post_init__()

        # 生成路径点
        track_cfg = TrackpointCfg()
        self.path_points = track_cfg.path_points

        # 转换为 torch 张量，方便后续计算
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