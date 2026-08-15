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


def is_completed_after_stop(
    env,
    asset_cfg=None,
    hold_time_s: float = 1.5,
    speed_threshold: float = 0.18,
    tilt_threshold: float = 0.20,
) -> torch.Tensor:
    """Complete only after the final target and a sustained stable stand."""
    _ensure_navigation_state(env)
    robot = env.scene["robot"]
    finished = env._next_target_idx >= len(env.cfg.path_points)
    planar_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    stable = finished & (planar_speed < speed_threshold) & (tilt < tilt_threshold)
    env._finish_stable_time = torch.where(
        stable,
        env._finish_stable_time + env.step_dt,
        torch.zeros_like(env._finish_stable_time),
    )
    return env._finish_stable_time >= hold_time_s
