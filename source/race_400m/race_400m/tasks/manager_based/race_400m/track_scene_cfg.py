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
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# 导入G1配置
from race_400m.config.g1_robotcfg import G1_CONFIG


# ============================================================
# 场景配置（地面 + 灯光 + 机器人）
# ============================================================
@configclass
class TrackSceneCfg(InteractiveSceneCfg):
    """场景配置：地面、灯光、机器人"""

    num_envs: int = 1
    env_spacing: float = 2.0

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: ArticulationCfg = G1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ============================================================
# 路径点配置（包含赛道几何计算）
# ============================================================
@configclass
class TrackpointCfg:
    """路径点配置：生成400米赛道的路径点"""

    # 赛道参数
    straight_length: float = 110.43
    radius: float = 23.24
    point_gap: float = 2.0
    total_distance: float = 400.0
    start_offset: float = 32.5
    path_points: list = None

    def __post_init__(self):
        """初始化后自动生成路径点"""
        self.path_points = self._generate_path_points()
        print(f"[INFO] 生成了 {len(self.path_points)} 个路径点")

    def _get_track_point(self, d: float):
        """
        输入：距离 d (米)，从原点开始算
        输出：(x, z) 坐标
        """
        L = self.straight_length
        R = self.radius
        START = self.start_offset

        if d < START:
            # 直道1：从 (0,0) 到 (32.5, 0)
            return (d, 0.0)

        elif d < START + np.pi * R:
            # 弯道1：从直道1末端到弯道1末端
            theta = (d - START) / R
            x = START + R * np.sin(theta)
            z = R * (1 - np.cos(theta))
            return (x, z)

        elif d < START + np.pi * R + L:
            # 直道2：从弯道1末端到直道2末端
            d2 = d - (START + np.pi * R)
            x = START - d2
            z = 2 * R
            return (x, z)

        elif d < START + 2 * np.pi * R + L:
            # 弯道2：从直道2末端回到原点
            theta = (d - (START + np.pi * R + L)) / R
            x = -77.93 - R * np.sin(theta)
            z = 2 * R - R * (1 - np.cos(theta))
            return (x, z)

        else:
            # 直道3：从 (-77,0) 开始继续向前延伸
            x = -77 + (d - START - 2 * np.pi * R - L)
            z = 0.0
            return (x, z)

    def _generate_path_points(self):
        """生成从起点到终点的所有路径点"""
        points = []
        end_d = self.total_distance
        d = 0.0
        while d <= end_d:
            x, z = self._get_track_point(d)
            points.append((x, z))
            d += self.point_gap
        return points
# ============================================================
# 仿真主循环
# ============================================================
def run_simulator(sim: SimulationContext, scene: InteractiveScene, path_points: list):
    """运行仿真循环，带路径追踪"""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0

    # ===== 路径追踪变量 =====
    next_target_idx = 0
    threshold = 1.0  # 判定"经过"的距离阈值
    total_points = len(path_points)

    print(f"[INFO] 总共 {total_points} 个路径点等待经过")
    print(f"[INFO] 起点: {path_points[0]}")
    print(f"[INFO] 终点: {path_points[-1]}")

    while simulation_app.is_running():
        # ---- 重置逻辑 ----
        if count % 500 == 0 and count > 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = robot.data.default_joint_pos.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, robot.data.default_joint_vel.clone())
            scene.reset()
            next_target_idx = 0  # 重置路径追踪
            print("[INFO]: Resetting robot state...")

        # ---- 路径追踪 ----
        if next_target_idx < total_points:
            # 获取机器人当前位置（第一个环境）
            robot_pos = robot.data.root_pos_w[0]
            pos_x, pos_z = robot_pos[0].item(), robot_pos[2].item()

            # 获取目标点
            target_x, target_z = path_points[next_target_idx]

            # 计算距离
            dist = np.sqrt((pos_x - target_x) ** 2 + (pos_z - target_z) ** 2)

            # 判断是否经过
            if dist < threshold:
                next_target_idx += 1
                if next_target_idx % 10 == 0 or next_target_idx == total_points:
                    print(f"[INFO] 经过了 {next_target_idx}/{total_points} 个路径点")

        # ---- 检查是否跑完全程 ----
        if next_target_idx >= total_points:
            print(f"[INFO] 🎉 跑完全程！共 {total_points} 个路径点")
            break

        # ---- 控制机器人 ----
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)

        # ---- 步进仿真 ----
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


# ============================================================
# 主函数
# ============================================================
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # 1. 创建场景（地面 + 灯光 + 机器人）
    scene_cfg = TrackSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 2. 生成路径点
    track_cfg = TrackpointCfg()
    path_points = track_cfg.path_points

    sim.reset()
    print("[INFO]: Setup complete...")
    print(f"[INFO]: 机器人数量: {scene['robot'].count}")
    print(f"[INFO]: 路径点数量: {len(path_points)}")

    run_simulator(sim, scene, path_points)


if __name__ == "__main__":
    main()
    simulation_app.close()