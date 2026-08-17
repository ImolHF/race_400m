"""Scene and deterministic 400 m track configuration.

This module is deliberately import-safe.  Training imports it to build the
environment, so it must not parse command-line arguments or launch Isaac Sim.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from race_400m.config.g1_robotcfg import G1_CONFIG, G1_ROBUST_CONFIG


@configclass
class TrackSceneCfg(InteractiveSceneCfg):
    """Ground, lighting, and one G1 robot per parallel environment."""

    num_envs: int = 16
    env_spacing: float = 2.5

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = G1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # GPU contact history is required by the imported G1 gait rewards.  It is
    # not a perception sensor and is never exposed as a policy observation.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,
    )


@configclass
class RobustTrackSceneCfg(TrackSceneCfg):
    """Track scene with MuJoCo/real-actuator-aligned G1 parameters."""

    robot: ArticulationCfg = G1_ROBUST_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class TrackpointCfg:
    """Generate local ``(x, y)`` navigation targets for one 400 m lap.

    Isaac Lab uses the X-Y plane as the ground plane and Z as height.  The
    targets therefore never encode a height value.
    """

    straight_length: float = 110.43
    radius: float = 23.24
    point_gap: float = 2.0
    total_distance: float = 400.0
    start_offset: float = 32.5
    path_points: list[tuple[float, float]] | None = None

    def __post_init__(self):
        self.path_points = self._generate_path_points()

    def _get_track_point(self, distance: float) -> tuple[float, float]:
        """Return the local ground-plane coordinate at arc length ``distance``."""
        straight = self.straight_length
        radius = self.radius
        start = self.start_offset
        half_curve = math.pi * radius

        if distance < start:
            return distance, 0.0
        if distance < start + half_curve:
            theta = (distance - start) / radius
            return start + radius * math.sin(theta), radius * (1.0 - math.cos(theta))
        if distance < start + half_curve + straight:
            distance_on_straight = distance - (start + half_curve)
            return start - distance_on_straight, 2.0 * radius
        if distance < start + 2.0 * half_curve + straight:
            theta = (distance - (start + half_curve + straight)) / radius
            return -77.93 - radius * math.sin(theta), 2.0 * radius - radius * (1.0 - math.cos(theta))

        distance_on_finish = distance - (start + 2.0 * half_curve + straight)
        return -77.0 + distance_on_finish, 0.0

    def _generate_path_points(self) -> list[tuple[float, float]]:
        point_count = int(self.total_distance / self.point_gap)
        return [self._get_track_point(index * self.point_gap) for index in range(point_count + 1)]


@configclass
class StadiumLaneTrackpointCfg(TrackpointCfg):
    """Parameterized 400 m stadium lane from the five supplied lane layouts.

    The supplied survey equations are preserved without coordinate
    translation.  The environment configuration resets the robot at target
    zero, so each lane's physical start coordinate remains its survey value.
    """

    def _get_track_point(self, distance: float) -> tuple[float, float]:
        straight, radius, start = self.straight_length, self.radius, self.start_offset
        half_curve = math.pi * radius
        if distance < start:
            return -start + distance, 0.0
        if distance < start + half_curve:
            theta = (distance - start) / radius
            return radius * math.sin(theta), radius * (1.0 - math.cos(theta))
        if distance < start + half_curve + straight:
            return -(distance - (start + half_curve)), 2.0 * radius
        if distance < start + 2.0 * half_curve + straight:
            theta = (distance - (start + half_curve + straight)) / radius
            return -straight - radius * math.sin(theta), 2.0 * radius - radius * (1.0 - math.cos(theta))
        return -straight + distance - (start + 2.0 * half_curve + straight), 0.0

    def _generate_path_points(self) -> list[tuple[float, float]]:
        point_count = int(self.total_distance / self.point_gap)
        return [self._get_track_point(index * self.point_gap) for index in range(point_count + 1)]


@configclass
class Lane1TrackpointCfg(StadiumLaneTrackpointCfg):
    radius: float = 22.990
    start_offset: float = 34.690


@configclass
class Lane2TrackpointCfg(StadiumLaneTrackpointCfg):
    radius: float = 25.090
    start_offset: float = 21.495


@configclass
class Lane3TrackpointCfg(StadiumLaneTrackpointCfg):
    radius: float = 27.190
    start_offset: float = 8.300


@configclass
class Lane4TrackpointCfg(StadiumLaneTrackpointCfg):
    radius: float = 29.290
    start_offset: float = -4.895


@configclass
class Lane5TrackpointCfg(StadiumLaneTrackpointCfg):
    radius: float = 31.390
    start_offset: float = -18.089
