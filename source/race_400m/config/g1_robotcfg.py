# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

G1_USD_PATH="C:/Users/1molHF/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"

G1_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=G1_USD_PATH,rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
        )),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # 左腿 (6个)
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.1,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            # 右腿 (6个)
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.1,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            # 腰部 (3个)
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # 左臂 (7个)
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # 右臂 (7个)
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.8),  # 躯干高度约0.8米
    ),
    actuators={
        # 左腿
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=["left_hip.*", "left_knee.*", "left_ankle.*"],
            effort_limit_sim=150.0,
            velocity_limit_sim=50.0,
            stiffness=200.0,
            damping=20.0,
        ),
        # 右腿
        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=["right_hip.*", "right_knee.*", "right_ankle.*"],
            effort_limit_sim=150.0,
            velocity_limit_sim=50.0,
            stiffness=200.0,
            damping=20.0,
        ),
        # 腰部
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist.*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=50.0,
            stiffness=100.0,
            damping=10.0,
        ),
        # 左臂
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder.*", "left_elbow.*", "left_wrist.*"],
            effort_limit_sim=50.0,
            velocity_limit_sim=50.0,
            stiffness=100.0,
            damping=10.0,
        ),
        # 右臂
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder.*", "right_elbow.*", "right_wrist.*"],
            effort_limit_sim=50.0,
            velocity_limit_sim=50.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
)




class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    g1_bot = G1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/g1_bot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_jetbot_state = scene["g1_bot"].data.default_root_state.clone()
            root_jetbot_state[:, :3] += scene.env_origins


            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["g1_bot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["g1_bot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])


            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["g1_bot"].data.default_joint_pos.clone(),
                scene["g1_bot"].data.default_joint_vel.clone(),
            )
            scene["g1_bot"].write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting g1机器人 state...")

        # drive around
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])

        target_pos = scene["g1_bot"].data.default_joint_pos.clone()
        time_tensor = torch.full_like(target_pos, sim_time)
        target_pos += 0.1 * torch.sin(2 * np.pi * 0.5 * time_tensor)
        scene["g1_bot"].set_joint_position_target(target_pos)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()