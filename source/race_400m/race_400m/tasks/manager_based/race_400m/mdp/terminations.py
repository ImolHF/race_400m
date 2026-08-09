"""Vectorized termination conditions for the 400 m navigation task."""

import torch

from .rewards import _ensure_navigation_state


def robot_fallen(env, asset_cfg=None, minimum_height: float = 0.35, maximum_tilt: float = 0.8) -> torch.Tensor:
    robot = env.scene["robot"]
    base_height = robot.data.root_pos_w[:, 2]
    tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    return (base_height < minimum_height) | (tilt > maximum_tilt)


def is_completed(env, asset_cfg=None) -> torch.Tensor:
    _ensure_navigation_state(env)
    return env._next_target_idx >= len(env.cfg.path_points)
