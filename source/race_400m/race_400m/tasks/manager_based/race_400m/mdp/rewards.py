"""Vectorized navigation observations and rewards for the fixed 400 m track."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def _ensure_navigation_state(env) -> None:
    """Create device-local, per-environment navigation state on first use."""
    if not hasattr(env, "_track_points"):
        env._track_points = torch.as_tensor(env.cfg.path_points, dtype=torch.float32, device=env.device)
        env._next_target_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._previous_target_distance = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def reset_path_progress(env, env_ids: torch.Tensor | None) -> None:
    """Reset only the selected environments to the first target point."""
    _ensure_navigation_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env._next_target_idx[env_ids] = 0
    env._previous_target_distance[env_ids] = 0.0


def _target_data(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local position, current target and target displacement for every environment."""
    _ensure_navigation_state(env)
    robot = env.scene["robot"]
    position_local = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target_indices = env._next_target_idx.clamp(max=len(env.cfg.path_points) - 1)
    targets = env._track_points[target_indices]
    return position_local, targets, targets - position_local


def target_direction(env) -> torch.Tensor:
    """Next target displacement in the robot yaw frame, scaled to a stable range."""
    _, _, displacement_xy = _target_data(env)
    displacement_world = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    displacement_world[:, :2] = displacement_xy
    robot = env.scene["robot"]
    displacement_body = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), displacement_world)
    return torch.clamp(displacement_body[:, :2] * 0.25, -5.0, 5.0)


def target_distance(env) -> torch.Tensor:
    """Distance to the active target, supplied as a one-value policy observation."""
    _, _, displacement_xy = _target_data(env)
    return torch.linalg.vector_norm(displacement_xy, dim=-1, keepdim=True) * 0.1


def reached_checkpoint(env, threshold: float = 1.0) -> torch.Tensor:
    """Advance exactly one target when the active target is reached."""
    position_local, targets, displacement_xy = _target_data(env)
    distance = torch.linalg.vector_norm(displacement_xy, dim=-1)
    active = env._next_target_idx < len(env.cfg.path_points)
    reached = active & (distance < threshold)
    env._next_target_idx = torch.where(reached, env._next_target_idx + 1, env._next_target_idx)

    # A new target has a larger distance.  Reset its progress baseline so the
    # progress reward does not punish a successful checkpoint transition.
    new_indices = env._next_target_idx.clamp(max=len(env.cfg.path_points) - 1)
    new_targets = env._track_points[new_indices]
    env._previous_target_distance = torch.linalg.vector_norm(new_targets - position_local, dim=-1)
    return reached.float()


def progress_reward(env) -> torch.Tensor:
    """Reward only a reduction of distance to the currently active target."""
    _, _, displacement_xy = _target_data(env)
    distance = torch.linalg.vector_norm(displacement_xy, dim=-1)
    progress = (env._previous_target_distance - distance).clamp(min=0.0, max=0.5)
    env._previous_target_distance = distance
    return progress


def alignment_reward(env) -> torch.Tensor:
    """Reward planar velocity that points towards the current target."""
    _, _, displacement_xy = _target_data(env)
    distance = torch.linalg.vector_norm(displacement_xy, dim=-1).clamp_min(1.0e-6)
    target_direction_xy = displacement_xy / distance.unsqueeze(-1)
    velocity_xy = env.scene["robot"].data.root_lin_vel_w[:, :2]
    forward_speed = (velocity_xy * target_direction_xy).sum(dim=-1)
    return forward_speed.clamp(min=-1.0, max=2.0)


def deviation_penalty(env) -> torch.Tensor:
    """Penalize moving more than one metre from any generated track point."""
    position_local, _, _ = _target_data(env)
    points = env._track_points.unsqueeze(0).expand(env.num_envs, -1, -1)
    nearest_distance = torch.cdist(position_local.unsqueeze(1), points).squeeze(1).min(dim=1).values
    return -(nearest_distance - 1.0).clamp_min(0.0)


def alive_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
