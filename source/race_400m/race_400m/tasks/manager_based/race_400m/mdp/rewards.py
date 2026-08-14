"""Vectorized navigation observations and rewards for the fixed 400 m track."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat


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
    # Target zero is the robot's reset position.  Starting at target one avoids
    # granting a checkpoint reward before the policy has taken an action.
    env._next_target_idx[env_ids] = 1
    robot = env.scene["robot"]
    position_local = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    target = env._track_points[1]
    # Initialize the potential with the actual reset distance.  A zero
    # baseline suppresses all positive progress reward until a checkpoint.
    env._previous_target_distance[env_ids] = torch.linalg.vector_norm(target - position_local, dim=-1)


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
    """Advance one target after reaching or safely passing it.

    The pass test prevents a running robot from getting stuck when its root
    crosses a 2 m-spaced target between two control updates.
    """
    position_local, targets, displacement_xy = _target_data(env)
    distance = torch.linalg.vector_norm(displacement_xy, dim=-1)
    active = env._next_target_idx < len(env.cfg.path_points)
    previous_indices = (env._next_target_idx - 1).clamp_min(0)
    previous_targets = env._track_points[previous_indices]
    segment = targets - previous_targets
    segment_length = torch.linalg.vector_norm(segment, dim=-1).clamp_min(1.0e-6)
    passed_distance = ((position_local - targets) * segment).sum(dim=-1) / segment_length
    lateral_error = torch.abs((position_local - targets)[:, 0] * segment[:, 1] - (position_local - targets)[:, 1] * segment[:, 0]) / segment_length
    passed = (passed_distance > 0.0) & (lateral_error < 1.25)
    reached = active & ((distance < threshold) | passed)
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


def _course_tangent(env) -> torch.Tensor:
    """Return the unit forward tangent of the active ordered path segment."""
    _ensure_navigation_state(env)
    current = env._next_target_idx.clamp(max=len(env.cfg.path_points) - 1)
    previous = (current - 1).clamp_min(0)
    tangent = env._track_points[current] - env._track_points[previous]
    return tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1.0e-6)


def course_heading_alignment(env) -> torch.Tensor:
    """Reward the robot's facing direction matching the track tangent.

    This closes the previous reward loophole where a robot could retain a
    sideways yaw and travel laterally to the next checkpoint.
    """
    robot = env.scene["robot"]
    forward_b = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(yaw_quat(robot.data.root_quat_w), forward_b)[:, :2]
    return (forward_w * _course_tangent(env)).sum(dim=-1).clamp(min=-1.0, max=1.0)


def forward_course_speed(env) -> torch.Tensor:
    """Reward track progress produced by forward, rather than lateral, motion."""
    robot = env.scene["robot"]
    velocity_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    heading = course_heading_alignment(env).clamp_min(0.0)
    return velocity_b[:, 0].clamp(min=-1.0, max=2.0) * heading


def lateral_velocity_penalty(env) -> torch.Tensor:
    """Penalize root sideways speed in the robot yaw frame (crab walking)."""
    robot = env.scene["robot"]
    velocity_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    return torch.square(velocity_b[:, 1])


def deviation_penalty(env) -> torch.Tensor:
    """Penalize moving more than one metre from any generated track point."""
    position_local, _, _ = _target_data(env)
    points = env._track_points.unsqueeze(0).expand(env.num_envs, -1, -1)
    nearest_distance = torch.cdist(position_local.unsqueeze(1), points).squeeze(1).min(dim=1).values
    return -(nearest_distance - 1.0).clamp_min(0.0)


def alive_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)


def feet_air_time_positive_biped(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Official G1 biped air-time shaping adapted to direct waypoint navigation.

    The velocity task gates this reward using its velocity command.  This task
    has no command manager, so actual planar root speed is used instead.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    moving = torch.linalg.vector_norm(env.scene["robot"].data.root_lin_vel_w[:, :2], dim=-1) > 0.1
    return reward * moving


def alternating_foot_gait(
    env, period: float, offset: tuple[float, float], threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward an alternating left/right contact schedule while the robot moves.

    Adapted from the phase-based G1 gait reward in Unitree RL Lab and the
    equivalent Mjlab implementation.  The phase is local to each environment,
    so parallel resets do not couple robots to one another.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.tensor(offset, dtype=phase.dtype, device=env.device).view(1, -1)
    stance_expected = ((phase + offsets) % 1.0) < threshold
    match = (stance_expected == in_contact).float().mean(dim=1)
    moving = torch.linalg.vector_norm(env.scene["robot"].data.root_lin_vel_w[:, :2], dim=-1) > 0.1
    return match * moving


def swing_foot_clearance(
    env, target_height: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward airborne ankle-roll links reaching a modest, repeatable clearance.

    This is the contact-gated form of the foot-clearance shaping used in
    Unitree RL Lab/Mjlab.  Grounded feet are excluded so the policy does not
    learn to pull both feet upward together.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    in_air = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    planar_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    height_error = torch.square(height - target_height)
    reward = torch.exp(-height_error / 0.05**2) * torch.tanh(2.0 * planar_speed) * in_air
    moving = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1) > 0.1
    return reward.sum(dim=1) * moving


def swing_foot_forward(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward only airborne feet moving forward in the robot yaw frame."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    in_air = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    relative_velocity_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids] - robot.data.root_lin_vel_w.unsqueeze(1)
    yaw = yaw_quat(robot.data.root_quat_w)
    relative_velocity_b = quat_apply_inverse(
        yaw.unsqueeze(1).expand(-1, relative_velocity_w.shape[1], -1).reshape(-1, 4),
        relative_velocity_w.reshape(-1, 3),
    ).reshape(env.num_envs, -1, 3)
    moving = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1) > 0.1
    return (relative_velocity_b[:, :, 0].clamp(min=0.0, max=2.0) * in_air).sum(dim=1) * moving


def forward_foot_landing(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, min_forward: float) -> torch.Tensor:
    """Reward an airborne foot landing ahead of the pelvis, not to its side."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    relative_position_w = robot.data.body_pos_w[:, asset_cfg.body_ids] - robot.data.root_pos_w.unsqueeze(1)
    yaw = yaw_quat(robot.data.root_quat_w)
    relative_position_b = quat_apply_inverse(
        yaw.unsqueeze(1).expand(-1, relative_position_w.shape[1], -1).reshape(-1, 4),
        relative_position_w.reshape(-1, 3),
    ).reshape(env.num_envs, -1, 3)
    moving = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1) > 0.1
    return ((relative_position_b[:, :, 0] - min_forward).clamp(min=0.0, max=0.4) * first_contact).sum(dim=1) * moving


def crossed_feet_penalty(env, min_half_width: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet entering the wrong side of the robot's yaw frame.

    The left ankle must remain left of the sagittal plane and the right ankle
    right of it.  It targets cross-stepping without imposing a fixed stride
    length or changing the ordered waypoint objective.
    """
    robot = env.scene[asset_cfg.name]
    foot_offset_w = robot.data.body_pos_w[:, asset_cfg.body_ids] - robot.data.root_pos_w.unsqueeze(1)
    yaw = yaw_quat(robot.data.root_quat_w)
    foot_offset_b = quat_apply_inverse(yaw.unsqueeze(1).expand(-1, foot_offset_w.shape[1], -1).reshape(-1, 4), foot_offset_w.reshape(-1, 3))
    foot_y_b = foot_offset_b.reshape(env.num_envs, -1, 3)[:, :, 1]
    # body_names are configured left then right below.
    return (min_half_width - foot_y_b[:, 0]).clamp_min(0.0) + (min_half_width + foot_y_b[:, 1]).clamp_min(0.0)
