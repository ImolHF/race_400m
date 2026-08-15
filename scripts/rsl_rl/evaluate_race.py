# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation-only script that compares checkpoints without modifying them.

This script loads a policy checkpoint, rolls it out in headless simulation and
writes per-episode metrics, a JSON summary and a Markdown report.  It never
trains, never exports a new policy, never records video and never mutates the
checkpoint, reward terms, task config or environment.

Two checkpoints with *different* action/observation interfaces are expected to
be evaluated with their own matching ``--task``:

* ``Template-Race-400m-LegOnly-HighCadence`` -- 12 actions / 82 observations
* ``Template-Race-400m`` (shoulder-swing, locked elbows) -- 14 actions / 84 obs

Never cross-load these checkpoints.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint without training.")
parser.add_argument("--task", type=str, required=True, help="Gym task id matching the checkpoint.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory for episodes.csv/summary.json/report.md.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required (absolute path to the checkpoint .pt file).")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim has booted."""

import time  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import race_400m.tasks  # noqa: F401
from race_400m.tasks.manager_based.race_400m.mdp.rewards import (  # noqa: E402
    _course_tangent,
    _ensure_navigation_state,
)


def _foot_body_ids(env):
    """Resolve the left/right ankle-roll body indices for the robot asset and contact sensor."""
    asset_cfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
    asset_cfg.resolve(env.unwrapped.scene)
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
    sensor_cfg.resolve(env.unwrapped.scene)
    return asset_cfg.body_ids, sensor_cfg.body_ids


def _leg_contact_ids(env):
    """Resolve hip/knee body indices for the contact sensor (undesired-contact metric)."""
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=[".*_hip_.*_link", ".*_knee_link"])
    sensor_cfg.resolve(env.unwrapped.scene)
    return sensor_cfg.body_ids


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run the evaluation rollout."""
    # --- resolve configs -------------------------------------------------
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # checkpoint must be an absolute, existing path
    checkpoint = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # --- create environment ----------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped

    # --- load policy ------------------------------------------------------
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=unwrapped.device)

    # --- resolve entity indices ------------------------------------------
    robot = unwrapped.scene["robot"]
    contact = unwrapped.scene.sensors["contact_forces"]
    foot_asset_ids, foot_sensor_ids = _foot_body_ids(env)
    leg_contact_ids = _leg_contact_ids(env)
    _ensure_navigation_state(unwrapped)
    num_points = len(unwrapped.cfg.path_points)

    num_envs = unwrapped.num_envs
    dt = unwrapped.step_dt
    device = unwrapped.device

    # --- running per-environment accumulators -----------------------------
    def _zeros(*shape):
        return torch.zeros(*shape, device=device, dtype=torch.float32)

    step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
    # sums (for means)
    sum_planar = _zeros(num_envs)
    sum_lateral = _zeros(num_envs)
    sum_tilt = _zeros(num_envs)
    sum_heading_err = _zeros(num_envs)
    sum_joint_vel = _zeros(num_envs)
    sum_action_rate = _zeros(num_envs)
    sum_air_time = _zeros(num_envs)
    # maxima / minima
    max_planar = _zeros(num_envs)
    max_lateral = _zeros(num_envs)
    max_tilt = _zeros(num_envs)
    max_heading_err = _zeros(num_envs)
    max_joint_vel = _zeros(num_envs)
    max_air_time = _zeros(num_envs)
    min_foot_spacing = torch.full((num_envs,), 1e9, device=device)
    max_rear_swing = _zeros(num_envs)
    # event counters
    crossed_events = torch.zeros(num_envs, device=device, dtype=torch.long)
    slide_events = torch.zeros(num_envs, device=device, dtype=torch.long)
    undesired_events = torch.zeros(num_envs, device=device, dtype=torch.long)

    # --- bookkeeping -------------------------------------------------------
    episodes = []
    prev_action = torch.zeros((num_envs, unwrapped.action_space.shape[-1]), device=device)

    obs, _ = env.get_observations()
    obs_dim = obs.shape[-1]
    act_dim = unwrapped.action_space.shape[-1]
    print(f"[INFO] Starting evaluation: obs_dim={obs_dim}, act_dim={act_dim}", flush=True)

    step_total = 0
    while len(episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)

        step_total += 1
        if step_total % 100 == 0:
            print(f"[INFO] step={step_total}, episodes={len(episodes)}/{args_cli.num_episodes}", flush=True)

        dones_b = dones.bool()
        time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
        terminated = dones_b & ~time_outs
        truncated = time_outs

        # ---- instantaneous metrics (all envs) ---------------------------
        root_pos = robot.data.root_pos_w
        root_lin_vel = robot.data.root_lin_vel_w
        root_quat = robot.data.root_quat_w
        proj_gravity = robot.data.projected_gravity_b
        joint_vel = robot.data.joint_vel

        planar = torch.norm(root_lin_vel[:, :2], dim=-1)
        yaw = yaw_quat(root_quat)
        vel_b = quat_apply_inverse(yaw, root_lin_vel)
        lateral = torch.abs(vel_b[:, 1])
        tilt = torch.norm(proj_gravity[:, :2], dim=-1)

        # heading error relative to the course tangent
        tangent = _course_tangent(unwrapped)
        forward_b = torch.zeros((num_envs, 3), device=device)
        forward_b[:, 0] = 1.0
        forward_w = quat_apply(yaw, forward_b)
        heading_cos = (forward_w[:, :2] * tangent).sum(dim=-1).clamp(-1.0, 1.0)
        heading_err = torch.acos(heading_cos)

        jv = torch.norm(joint_vel, dim=-1)
        act_rate = torch.norm(actions - prev_action, dim=-1)
        prev_action = actions.clone()

        # foot spacing / crossing in the robot yaw frame
        foot_off_w = robot.data.body_pos_w[:, foot_asset_ids] - root_pos.unsqueeze(1)
        foot_off_b = quat_apply_inverse(
            yaw.unsqueeze(1).expand(-1, foot_off_w.shape[1], -1).reshape(-1, 4),
            foot_off_w.reshape(-1, 3),
        ).reshape(num_envs, -1, 3)
        foot_y = foot_off_b[:, :, 1]  # (N, 2) left, right
        foot_spacing = torch.abs(foot_y[:, 0] - foot_y[:, 1])
        # left foot should stay y>0, right foot y<0; crossing the midline is a cross-step
        crossed = (foot_y[:, 0] < 0.0) | (foot_y[:, 1] > 0.0)

        # air time (contact-gated foot air time)
        air = contact.data.current_air_time[:, foot_sensor_ids].max(dim=1).values

        # rear swing magnitude (foot behind pelvis in yaw frame)
        rear = (-foot_off_b[:, :, 0]).clamp_min(0.0).max(dim=1).values

        # foot slide: loaded foot moving while in contact
        in_contact = contact.data.current_contact_time[:, foot_sensor_ids] > 0.0
        foot_vel = robot.data.body_lin_vel_w[:, foot_asset_ids]
        foot_slide = (torch.norm(foot_vel, dim=-1) * in_contact).max(dim=1).values
        sliding = foot_slide > 0.05

        # undesired contacts: hip/knee link contacting with force
        leg_force = contact.data.net_forces_w[:, leg_contact_ids]
        undesired = (torch.norm(leg_force, dim=-1).max(dim=1).values) > 1.0

        # ---- accumulate -------------------------------------------------
        alive = ~dones_b  # accumulate only while this env is still in the episode
        mask = alive.float()
        step_count += alive.long()
        sum_planar += planar * mask
        sum_lateral += lateral * mask
        sum_tilt += tilt * mask
        sum_heading_err += heading_err * mask
        sum_joint_vel += jv * mask
        sum_action_rate += act_rate * mask
        sum_air_time += air * mask
        max_planar = torch.maximum(max_planar, planar * mask)
        max_lateral = torch.maximum(max_lateral, lateral * mask)
        max_tilt = torch.maximum(max_tilt, tilt * mask)
        max_heading_err = torch.maximum(max_heading_err, heading_err * mask)
        max_joint_vel = torch.maximum(max_joint_vel, jv * mask)
        max_air_time = torch.maximum(max_air_time, air * mask)
        min_foot_spacing = torch.minimum(min_foot_spacing, torch.where(dones_b, 1e9, foot_spacing))
        max_rear_swing = torch.maximum(max_rear_swing, rear * mask)
        crossed_events += (crossed & alive).long()
        slide_events += (sliding & alive).long()
        undesired_events += (undesired & alive).long()

        # ---- finalize finished episodes ----------------------------------
        if dones_b.any():
            done_ids = dones_b.nonzero(as_tuple=False).flatten().cpu().numpy()
            for eid in done_ids:
                if len(episodes) >= args_cli.num_episodes:
                    break
                n = int(step_count[eid].item()) + 1
                if n == 0:
                    continue
                completed = bool(terminated[eid] and (unwrapped._next_target_idx[eid] >= num_points))
                fallen = bool(terminated[eid] and not completed)
                timed_out = bool(truncated[eid])
                waypoint = int(unwrapped._next_target_idx[eid].item())
                episodes.append(
                    {
                        "env_id": int(eid),
                        "completed": completed,
                        "fallen": fallen,
                        "timeout": timed_out,
                        "max_waypoint": waypoint,
                        "duration_s": float(n * dt),
                        "mean_planar_speed": float(sum_planar[eid] / n),
                        "max_planar_speed": float(max_planar[eid]),
                        "mean_lateral_speed": float(sum_lateral[eid] / n),
                        "max_lateral_speed": float(max_lateral[eid]),
                        "mean_trunk_tilt_rad": float(sum_tilt[eid] / n),
                        "max_trunk_tilt_rad": float(max_tilt[eid]),
                        "mean_heading_err_rad": float(sum_heading_err[eid] / n),
                        "max_heading_err_rad": float(max_heading_err[eid]),
                        "mean_joint_vel": float(sum_joint_vel[eid] / n),
                        "max_joint_vel": float(max_joint_vel[eid]),
                        "mean_action_rate": float(sum_action_rate[eid] / n),
                        "min_foot_spacing": float(min_foot_spacing[eid]),
                        "crossed_feet_events": int(crossed_events[eid]),
                        "mean_air_time": float(sum_air_time[eid] / n),
                        "max_air_time": float(max_air_time[eid]),
                        "max_rear_swing": float(max_rear_swing[eid]),
                        "foot_slide_events": int(slide_events[eid]),
                        "undesired_contact_events": int(undesired_events[eid]),
                    }
                )
                # reset accumulators for this env
                step_count[eid] = 0
                sum_planar[eid] = 0; sum_lateral[eid] = 0; sum_tilt[eid] = 0
                sum_heading_err[eid] = 0; sum_joint_vel[eid] = 0; sum_action_rate[eid] = 0
                sum_air_time[eid] = 0
                max_planar[eid] = 0; max_lateral[eid] = 0; max_tilt[eid] = 0
                max_heading_err[eid] = 0; max_joint_vel[eid] = 0; max_air_time[eid] = 0
                min_foot_spacing[eid] = 1e9; max_rear_swing[eid] = 0
                crossed_events[eid] = 0; slide_events[eid] = 0; undesired_events[eid] = 0

    env.close()

    # --- write outputs ----------------------------------------------------
    os.makedirs(args_cli.output_dir, exist_ok=True)
    _write_outputs(args_cli, checkpoint, episodes, num_points, obs_dim, act_dim)

    # close sim app
    simulation_app.close()


def _write_outputs(args_cli, checkpoint, episodes, num_points, obs_dim, act_dim):
    """Write episodes.csv, summary.json and report.md."""
    import statistics
    import subprocess

    out = args_cli.output_dir
    num_episodes = len(episodes)

    # episodes.csv
    csv_path = os.path.join(out, "episodes.csv")
    if episodes:
        keys = list(episodes[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for e in episodes:
                f.write(",".join(str(e[k]) for k in keys) + "\n")

    # summary.json
    def _pct(cond):
        return round(100.0 * sum(1 for e in episodes if cond(e)) / num_episodes, 2) if num_episodes else 0.0

    def _mean(key):
        return round(statistics.mean(e[key] for e in episodes), 4) if num_episodes else 0.0

    def _median(key):
        return round(statistics.median(e[key] for e in episodes), 4) if num_episodes else 0.0

    def _p90(key):
        vals = sorted(e[key] for e in episodes)
        idx = min(len(vals) - 1, int(0.9 * len(vals)))
        return round(vals[idx], 4) if vals else 0.0

    finish_times = [e["duration_s"] for e in episodes if e["completed"]]

    summary = {
        "checkpoint": os.path.abspath(checkpoint),
        "task": args_cli.task,
        "action_dim": int(act_dim),
        "observation_dim": int(obs_dim),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "num_episodes": num_episodes,
        "num_path_points": num_points,
        "completion_rate_pct": _pct(lambda e: e["completed"]),
        "fall_rate_pct": _pct(lambda e: e["fallen"]),
        "timeout_rate_pct": _pct(lambda e: e["timeout"]),
        "mean_finish_time_s": round(statistics.mean(finish_times), 4) if finish_times else None,
        "median_finish_time_s": round(statistics.median(finish_times), 4) if finish_times else None,
        "p90_finish_time_s": _p90_finish(finish_times),
        "mean_planar_speed": _mean("mean_planar_speed"),
        "max_planar_speed": _mean("max_planar_speed"),
        "mean_lateral_speed": _mean("mean_lateral_speed"),
        "mean_trunk_tilt_rad": _mean("mean_trunk_tilt_rad"),
        "max_trunk_tilt_rad": _mean("max_trunk_tilt_rad"),
        "mean_heading_err_rad": _mean("mean_heading_err_rad"),
        "max_heading_err_rad": _mean("max_heading_err_rad"),
        "mean_joint_vel": _mean("mean_joint_vel"),
        "max_joint_vel": _mean("max_joint_vel"),
        "mean_action_rate": _mean("mean_action_rate"),
        "mean_min_foot_spacing": _mean("min_foot_spacing"),
        "mean_crossed_feet_events": _mean("crossed_feet_events"),
        "mean_air_time": _mean("mean_air_time"),
        "max_air_time": _mean("max_air_time"),
        "mean_max_rear_swing": _mean("max_rear_swing"),
        "mean_foot_slide_events": _mean("foot_slide_events"),
        "mean_undesired_contact_events": _mean("undesired_contact_events"),
    }

    # git commit
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__) or ".")
        summary["git_commit"] = git_commit.decode().strip()
    except Exception:
        summary["git_commit"] = "unknown"

    json_path = os.path.join(out, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # report.md
    md_path = os.path.join(out, "report.md")
    with open(md_path, "w") as f:
        f.write(f"# Race 400m Evaluation Report\n\n")
        f.write(f"- **Checkpoint**: `{os.path.abspath(checkpoint)}`\n")
        f.write(f"- **Task**: `{args_cli.task}`\n")
        f.write(f"- **Action dim**: `{act_dim}`\n")
        f.write(f"- **Observation dim**: `{summary['observation_dim']}`\n")
        f.write(f"- **Git commit**: `{summary.get('git_commit', 'unknown')}`\n")
        f.write(f"- **Seed**: `{args_cli.seed}`\n")
        f.write(f"- **num_envs**: `{args_cli.num_envs}`\n")
        f.write(f"- **num_episodes**: `{num_episodes}`\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for k, v in summary.items():
            if k not in ("checkpoint", "task", "action_dim", "observation_dim", "seed", "num_envs", "num_episodes", "num_path_points", "git_commit"):
                f.write(f"| {k} | {v} |\n")
        f.write("\n> These 100 episodes measure repeatability and base stability only; "
                "they are **not** a sim2real safety certificate.\n")

    print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {md_path}")


def _p90_finish(vals):
    vals = sorted(vals)
    if not vals:
        return None
    idx = min(len(vals) - 1, int(0.9 * len(vals)))
    return round(vals[idx], 4)


if __name__ == "__main__":
    main()
    simulation_app.close()
