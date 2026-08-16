"""Headless MuJoCo sim2sim smoke test for the 14-action locked-elbow policy.

This is deliberately an evaluation harness, never a Unitree SDK controller.
It uses the official Unitree G1 29-DoF MJCF, recreates the Isaac Lab policy
observation contract, and applies a conservative PD torque controller at
MuJoCo's 500 Hz physics rate while the policy runs at 25 Hz.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch


JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
POLICY_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 22])
ACTION_SCALE = np.array([0.25] * 4 + [0.20, 0.20] + [0.25] * 4 + [0.20, 0.20] + [0.18, 0.18], dtype=np.float32)
DEFAULT_Q = np.array([-0.20, 0, 0, 0.42, -0.23, 0, -0.20, 0, 0, 0.42, -0.23, 0, 0, 0, 0, 0, 0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0.95, 0, 0, 0], dtype=np.float32)
KP = np.array([420, 320, 320, 420, 260, 260] * 2 + [250] * 3 + [100, 160, 160, 320, 160, 160, 160, 100, 160, 160, 320, 160, 160, 160], dtype=np.float32)
KD = np.array([24, 16, 16, 24, 14, 14] * 2 + [25] * 3 + [10, 16, 16, 32, 16, 16, 16, 10, 16, 16, 32, 16, 16, 16], dtype=np.float32)


def track() -> np.ndarray:
    points = []
    straight, radius, start = 110.43, 23.24, 32.5
    half_curve = math.pi * radius
    for i in range(201):
        d = 2.0 * i
        if d < start:
            p = (d, 0.0)
        elif d < start + half_curve:
            t = (d - start) / radius; p = (start + radius * math.sin(t), radius * (1 - math.cos(t)))
        elif d < start + half_curve + straight:
            p = (start - (d - start - half_curve), 2 * radius)
        elif d < start + 2 * half_curve + straight:
            t = (d - start - half_curve - straight) / radius; p = (-77.93 - radius * math.sin(t), 2 * radius - radius * (1 - math.cos(t)))
        else:
            p = (-77.0 + d - start - 2 * half_curve - straight, 0.0)
        points.append(p)
    return np.asarray(points, dtype=np.float32)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/sim2sim/locked_elbow_nominal.json"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    if model.nu != 29 or model.nq != 36:
        raise RuntimeError(f"Expected official G1 29-DoF model; got nq={model.nq}, nu={model.nu}")
    actual = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    if actual != JOINT_NAMES:
        raise RuntimeError("MuJoCo joint order differs from the verified G1 29-DoF contract.")
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    data.qpos[:3] = (0, 0, 0.74); data.qpos[3:7] = (1, 0, 0, 0); data.qpos[7:] = DEFAULT_Q
    mujoco.mj_forward(model, data)
    points, target_idx, previous_action = track(), 1, np.zeros(14, np.float32)
    target_q, fallen, max_tilt, policy_steps = DEFAULT_Q.copy(), False, 0.0, 0
    physics_per_policy = round(0.04 / model.opt.timestep)
    total_steps = round(args.seconds / model.opt.timestep)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for step in range(total_steps):
        if step % physics_per_policy == 0:
            q, dq, quat = data.qpos[7:].copy(), data.qvel[6:].copy(), data.qpos[3:7].copy()
            R = quat_to_mat(quat)
            lin_b, ang_b = R.T @ data.qvel[:3], R.T @ data.qvel[3:6]
            delta_w = points[target_idx] - data.qpos[:2]
            delta_b = R[:2, :2].T @ delta_w
            obs = np.concatenate((q-DEFAULT_Q, dq, lin_b, 0.25*ang_b, R.T @ np.array([0,0,-1], np.float32), previous_action, 0.25*delta_b, [0.1*np.linalg.norm(delta_w)])).astype(np.float32)
            if obs.shape != (84,): raise RuntimeError(f"Observation shape {obs.shape}, expected 84")
            with torch.inference_mode(): action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            action = np.clip(action, -1, 1).astype(np.float32)
            target_q[POLICY_IDS] = DEFAULT_Q[POLICY_IDS] + ACTION_SCALE * action
            previous_action, policy_steps = action, policy_steps + 1
            if np.linalg.norm(delta_w) < 1.0 and target_idx < len(points)-1: target_idx += 1
        # MuJoCo XML exposes torque motors, so reproduce Isaac's implicit PD law explicitly.
        torque = KP * (target_q - data.qpos[7:]) - KD * data.qvel[6:]
        data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        tilt = float(np.linalg.norm((quat_to_mat(data.qpos[3:7]).T @ np.array([0,0,-1]))[:2]))
        max_tilt = max(max_tilt, tilt)
        if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.35 or tilt > 0.65:
            fallen = True; break
    result = {"policy": str(args.policy), "xml": str(args.xml), "seconds_requested": args.seconds, "seconds_simulated": float(data.time), "policy_steps": policy_steps, "fallen": fallen, "max_tilt_rad_proxy": max_tilt, "max_waypoint": int(target_idx), "progress_m_proxy": float(2*target_idx), "root_xy": data.qpos[:2].round(4).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
