"""Offline-only validation for the G1 U2 deployment runtime.

This script never imports the Unitree SDK and never opens a network socket.
It feeds a nominal standing state into ``SafeRaceRuntime`` to validate the
84-value observation, TorchScript policy, 14-action mapping, action limits,
and 29-motor command construction.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from g1_u2_safe_runtime import DEFAULT_Q, G1U2Adapter, G1U2State, MotorCommand, RuntimeConfig, SafeRaceRuntime


def generate_training_track() -> list[list[float]]:
    """Reproduce the fixed 201-point track used by the main training task."""

    straight, radius, start, gap = 110.43, 23.24, 32.5, 2.0
    half_curve = math.pi * radius
    points: list[list[float]] = []
    for index in range(201):
        distance = index * gap
        if distance < start:
            point = (distance, 0.0)
        elif distance < start + half_curve:
            theta = (distance - start) / radius
            point = (start + radius * math.sin(theta), radius * (1.0 - math.cos(theta)))
        elif distance < start + half_curve + straight:
            point = (start - (distance - start - half_curve), 2.0 * radius)
        elif distance < start + 2.0 * half_curve + straight:
            theta = (distance - start - half_curve - straight) / radius
            point = (-77.93 - radius * math.sin(theta), 2.0 * radius - radius * (1.0 - math.cos(theta)))
        else:
            point = (-77.0 + distance - start - 2.0 * half_curve - straight, 0.0)
        points.append([float(point[0]), float(point[1])])
    return points


class NominalStandingAdapter(G1U2Adapter):
    """No-hardware adapter that records, but never transmits, commands."""

    def __init__(self) -> None:
        self.sent_count = 0
        self.hold_count = 0
        self.last_command: MotorCommand | None = None

    def read_state(self) -> G1U2State:
        return G1U2State(
            q=DEFAULT_Q.copy(),
            dq=np.zeros(29, dtype=np.float32),
            imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            position_w_xy=np.zeros(2, dtype=np.float32),
            stamp_s=0.0,
        )

    def send(self, command: MotorCommand) -> None:
        self.sent_count += 1
        self.last_command = command
        raise AssertionError("dry-run adapter must never receive a hardware command")

    def emergency_hold(self) -> None:
        self.hold_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline-only G1 U2 policy dry-run.")
    parser.add_argument("--policy", type=Path, required=True, help="Exported TorchScript policy.pt, not an RSL-RL model_*.pt checkpoint.")
    parser.add_argument("--steps", type=int, default=100, help="Number of 40 ms policy cycles to validate.")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    if not args.policy.is_file():
        raise FileNotFoundError(args.policy)

    adapter = NominalStandingAdapter()
    # The selected main model was trained without a command-delay action term.
    config = RuntimeConfig(policy_path=args.policy, waypoints_xy=generate_training_track(), action_delay_steps=0)
    runtime = SafeRaceRuntime(config, adapter, dry_run=True)
    runtime.enable_dry_run()

    max_action = 0.0
    max_delta = 0.0
    for _ in range(args.steps):
        command = runtime.step()
        if command.q_target.shape != (29,):
            raise RuntimeError(f"Invalid command shape: {command.q_target.shape}")
        max_action = max(max_action, float(np.max(np.abs(runtime.previous_action))))
        max_delta = max(max_delta, float(np.max(np.abs(command.q_target - DEFAULT_Q))))

    if adapter.sent_count != 0:
        raise RuntimeError("Dry-run safety violation: a hardware send was attempted.")
    print("DRY-RUN PASS")
    print(f"policy={args.policy}")
    print(f"cycles={args.steps}, control_dt_s={config.control_dt_s}, observation_dim=84, action_dim=14")
    print(f"hardware_sends={adapter.sent_count}, emergency_holds={adapter.hold_count}")
    print(f"max_abs_policy_action={max_action:.4f}, max_abs_target_offset_rad={max_delta:.4f}")


if __name__ == "__main__":
    main()
