# Copyright (c) 2022-2025, The Isaac Lab Project Developers...
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# ✅ 修复1：去掉 source.
from source.race_400m.config.g1_robotcfg import G1_CONFIG


@configclass
class TrackSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: ArticulationCfg = G1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class TrackpointCfg(InteractiveSceneCfg):
    straight_length=110.43
    radius=23.24
    point_gap=2.0
    total_distance=400.0

    def __post_init__(self):
        super.__post_init__()
        self.path_points = self._generate_path_points()
        print(f"[INFO] 生成了 {len(self.path_points)} 个路径点")

    def _get_track_points(self,d:float):
        l=self.straight_length
        r=self.radius

    def _generate_path_points(self):
        points=[]
        d=0.0
        while d < self.total_distance:

            x,z=self._get_track_point(d)
            points.append((x,z))
            d += self.point_gap
        return points
        


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # ✅ 修复3：使用正确的 key
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")

        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # ✅ 修复2：使用正确的场景配置类
    scene_cfg = TrackSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()