"""Fail-safe, hardware-agnostic runtime for the G1 U2 race policy.

This module is deliberately *not* a ready-to-run motor driver.  It assembles
the policy's 84 observations, maps its 14 actions to the G1 U2's 29 motors,
and enforces a conservative safety state machine.  A lab-specific SDK2/DDS
adapter must implement :class:`G1U2Adapter` and pass the explicit arming gate
before any command can reach hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


G1_NUM_MOTORS = 29
# Official SDK2 low-level ordering for the 29-DoF G1. Verify this against the
# robot firmware's reported joint names before enabling hardware output.
POLICY_MOTOR_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 22])
ACTION_SCALE = np.array([0.25] * 4 + [0.20, 0.20] + [0.25] * 4 + [0.20, 0.20] + [0.18, 0.18])
DEFAULT_Q = np.array(
    [-0.20, 0.0, 0.0, 0.42, -0.23, 0.0, -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
     0.0, 0.0, 0.0], dtype=np.float32,
)


class Mode(str, Enum):
    DISABLED = "disabled"
    STAND = "stand"
    POLICY = "policy"
    FAULT = "fault"


@dataclass
class G1U2State:
    """All quantities must use SI units; quaternion is world-frame ``wxyz``."""

    q: np.ndarray
    dq: np.ndarray
    imu_quat_wxyz: np.ndarray
    base_lin_vel_b: np.ndarray
    base_ang_vel_b: np.ndarray
    position_w_xy: np.ndarray
    stamp_s: float


@dataclass
class MotorCommand:
    q_target: np.ndarray
    dq_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray


class G1U2Adapter(ABC):
    """Implement with SDK2/DDS. Keep ``send`` disabled until bench validation."""

    @abstractmethod
    def read_state(self) -> G1U2State: ...

    @abstractmethod
    def send(self, command: MotorCommand) -> None: ...

    @abstractmethod
    def emergency_hold(self) -> None: ...


@dataclass
class RuntimeConfig:
    policy_path: Path
    waypoints_xy: Sequence[Sequence[float]]
    control_dt_s: float = 0.04
    waypoint_reach_radius_m: float = 1.0
    # The selected locked-elbow main policy was trained without command delay.
    action_delay_steps: int = 0
    # These first-day limits are intentionally much tighter than the training
    # action scale.  They are for suspended/slow tests only; a successful gait
    # test must deliberately raise max_target_step_rad towards 0.20--0.25.
    max_action_abs: float = 0.5
    max_target_step_rad: float = 0.03
    max_tilt_rad: float = 0.30
    command_timeout_s: float = 0.08
    dry_run: bool = True
    kp: np.ndarray = field(default_factory=lambda: np.array([60, 60, 60, 100, 40, 40] * 2 + [40] * 17, dtype=np.float32))
    kd: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 2, 1, 1] * 2 + [1] * 17, dtype=np.float32))

    @classmethod
    def from_json(cls, config_path: Path, *, policy_path: Path | None = None) -> "RuntimeConfig":
        """Load the one authoritative deployment configuration.

        Relative policy/waypoint paths are resolved from the repository root
        (the parent of ``deploy``), not from the caller's working directory.
        ``policy_path`` is an explicit CLI-only override for the placeholder
        in the checked-in template.
        """
        config_path = config_path.resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        required = {
            "policy_path", "waypoints_path", "control_dt_s", "waypoint_reach_radius_m",
            "action_delay_steps", "max_action_abs", "max_target_step_rad", "max_tilt_rad",
            "command_timeout_s", "dry_run",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"Missing deployment configuration fields: {sorted(missing)}")
        repo_root = config_path.parents[2]
        chosen_policy = policy_path or Path(raw["policy_path"])
        if not chosen_policy.is_absolute():
            chosen_policy = repo_root / chosen_policy
        waypoint_path = Path(raw["waypoints_path"])
        if not waypoint_path.is_absolute():
            waypoint_path = repo_root / waypoint_path
        points = json.loads(waypoint_path.read_text(encoding="utf-8"))
        if len(points) != 201 or any(not isinstance(point, list) or len(point) != 2 for point in points):
            raise ValueError("Expected exactly 201 [x, y] waypoints.")
        config = cls(policy_path=chosen_policy, waypoints_xy=points, **{key: raw[key] for key in required - {"policy_path", "waypoints_path"}})
        if config.control_dt_s <= 0 or config.action_delay_steps < 0:
            raise ValueError("control_dt_s must be positive and action_delay_steps non-negative.")
        return config


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _projected_gravity(quat: np.ndarray) -> np.ndarray:
    """World down direction expressed in the IMU/body frame."""
    w, x, y, z = quat / np.linalg.norm(quat)
    rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
    return (rot.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)


class SafeRaceRuntime:
    """Policy invocation, 200-point navigation and command safety guard."""

    def __init__(self, config: RuntimeConfig, adapter: G1U2Adapter, dry_run: bool | None = None):
        self.cfg, self.adapter = config, adapter
        self.dry_run = config.dry_run if dry_run is None else dry_run
        self.policy = torch.jit.load(str(config.policy_path), map_location="cpu").eval()
        self.waypoints = np.asarray(config.waypoints_xy, dtype=np.float32)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2:
            raise ValueError("waypoints_xy must have shape [N, 2].")
        self.mode, self.target_index = Mode.DISABLED, 1
        self.previous_action = np.zeros(14, dtype=np.float32)
        self.target_fifo = deque([DEFAULT_Q[POLICY_MOTOR_IDS].copy()] * (config.action_delay_steps + 1), maxlen=config.action_delay_steps + 1)
        self.last_target = DEFAULT_Q.copy()

    def arm_policy(self, physical_enable: bool) -> None:
        """Require an explicit external/human gate; dry-run can never enable motors."""
        if not physical_enable or self.dry_run:
            raise RuntimeError("Refusing to arm: set physical_enable only after hanging and E-stop checks; dry_run must be False.")
        self.mode = Mode.STAND

    def enable_dry_run(self) -> None:
        """Enable policy inference without ever permitting a hardware send.

        This is intentionally separate from :meth:`arm_policy`: it is useful
        for validating the observation/action contract on a development PC,
        but it cannot be reused to arm a physical robot.
        """

        if not self.dry_run:
            raise RuntimeError("enable_dry_run is only available with dry_run=True.")
        self.mode = Mode.STAND

    def _observation(self, state: G1U2State) -> np.ndarray:
        if state.q.shape != (29,) or state.dq.shape != (29,):
            raise ValueError("Expected 29 G1 U2 joint positions and velocities in official SDK2 order.")
        yaw = _yaw(state.imu_quat_wxyz)
        delta_w = self.waypoints[min(self.target_index, len(self.waypoints) - 1)] - state.position_w_xy
        c, s = np.cos(yaw), np.sin(yaw)
        delta_b = np.array([c * delta_w[0] + s * delta_w[1], -s * delta_w[0] + c * delta_w[1]], dtype=np.float32)
        obs = np.concatenate((state.q - DEFAULT_Q, state.dq, state.base_lin_vel_b, 0.25 * state.base_ang_vel_b,
                              _projected_gravity(state.imu_quat_wxyz), self.previous_action,
                              0.25 * delta_b, [0.1 * np.linalg.norm(delta_w)]), dtype=np.float32)
        if obs.shape != (84,):
            raise RuntimeError(f"Observation contract violated: expected 84, got {obs.size}.")
        return obs

    def _fault_if_unsafe(self, state: G1U2State) -> None:
        tilt = float(np.linalg.norm(_projected_gravity(state.imu_quat_wxyz)[:2]))
        if not np.isfinite(state.q).all() or not np.isfinite(state.dq).all() or tilt > self.cfg.max_tilt_rad:
            self.mode = Mode.FAULT
            self.adapter.emergency_hold()
            raise RuntimeError(f"Safety fault: tilt={tilt:.3f} rad or invalid state.")

    def step(self) -> MotorCommand:
        state = self.adapter.read_state()
        self._fault_if_unsafe(state)
        if self.mode == Mode.DISABLED:
            raise RuntimeError("Runtime is disabled; do not send motor commands.")
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(self._observation(state)).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -self.cfg.max_action_abs, self.cfg.max_action_abs).astype(np.float32)
        self.target_fifo.append(DEFAULT_Q[POLICY_MOTOR_IDS] + ACTION_SCALE * action)
        delayed = self.target_fifo[0]
        target = DEFAULT_Q.copy()
        target[POLICY_MOTOR_IDS] = delayed
        target = np.clip(target, self.last_target - self.cfg.max_target_step_rad, self.last_target + self.cfg.max_target_step_rad)
        self.last_target, self.previous_action = target, action
        if (np.linalg.norm(self.waypoints[self.target_index] - state.position_w_xy) < self.cfg.waypoint_reach_radius_m
                and self.target_index < len(self.waypoints) - 1):
            self.target_index += 1
        command = MotorCommand(target, np.zeros(29, np.float32), self.cfg.kp, self.cfg.kd, np.zeros(29, np.float32))
        if not self.dry_run:
            self.adapter.send(command)
        self.mode = Mode.POLICY
        return command
