# source/race_400m/race_400m/tasks/manager_based/race_400m/mdp/rewards.py

import torch
import numpy as np


def reached_checkpoint(env):
    """判断是否到达下一个路径点"""
    # 获取路径点数据
    path_points = env.cfg.path_points
    if path_points is None:
        return torch.tensor(0.0, device=env.device)

    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[0]
    pos_x, pos_z = robot_pos[0].item(), robot_pos[2].item()

    # 获取当前追踪索引
    if not hasattr(env, "_next_target_idx"):
        env._next_target_idx = 0

    threshold = 0.5
    if env._next_target_idx < len(path_points):
        target_x, target_z = path_points[env._next_target_idx]
        dist = np.sqrt((pos_x - target_x) ** 2 + (pos_z - target_z) ** 2)

        if dist < threshold:
            env._next_target_idx += 1
            if env._next_target_idx % 10 == 0 or env._next_target_idx == len(path_points):
                print(f"[INFO] 到达第 {env._next_target_idx}/{len(path_points)} 个路径点")
            return torch.tensor(1.0, device=env.device)

    return torch.tensor(0.0, device=env.device)

def progress_reward(env):
    """判断是否前进"""
    # 获取路径点数据
    path_points = env.cfg.path_points
    if path_points is None:
        return torch.tensor(0.0, device=env.device)
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[0]
    pos_x, pos_z = robot_pos[0].item(), robot_pos[2].item()

    if not hasattr(env, "_next_target_idx"):
        env._next_target_idx = 0
    if env._next_target_idx >= len(path_points):
        return torch.tensor(0.0, device=env.device)
    target_x,target_z=path_points[env._next_target_idx]
    dist = np.sqrt((pos_x - target_x) ** 2 + (pos_z - target_z) ** 2)
    reward = -dist*0.01
    return torch.tensor(reward, device=env.device)

def backward_penalty(env):
    path_points = env.cfg.path_points
    if path_points is None:
        return torch.tensor(0.0, device=env.device)
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[0]
    pos_x, pos_z = robot_pos[0].item(), robot_pos[2].item()

    if not hasattr(env,"_next_target_idx"):
        env._next_target_idx = 0
    if env._next_target_idx >= len(path_points):
        return torch.tensor(0.0, device=env.device)
    target_x,target_z=path_points[env._next_target_idx]
    lin_vel=robot.data.root_lin_vel_w[0]
    vel_x,vel_z=lin_vel[0].item(),lin_vel[2].item()

    dist=np.sqrt((target_x-pos_x)**2+(target_z-pos_z)**2)
    if dist < 0.1:
        return torch.tensor(0.0, device=env.device)
    speed=np.sqrt(vel_x**2+vel_z**2)
    if speed<0.05:
        return torch.tensor(0.0, device=env.device)
    vec_target = (target_x - pos_x, target_z - pos_z)
    vec_vel = (vel_x, vel_z)
    dot=np.dot(vec_target,vec_vel)
    cos=np.dot(vec_target,vec_vel)/speed*dist
    if cos<-0.3:
        penalty=-0.2
        return torch.tensor(penalty, device=env.device)
    else:
        return torch.tensor(0.0, device=env.device)

def deviation_penalty(env):
    path_points = env.cfg.path_points
    if path_points is None:
        return torch.tensor(0.0, device=env.device)
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[0]
    pos_x, pos_z = robot_pos[0].item(), robot_pos[2].item()

    if not hasattr(env, "_next_target_idx"):
        env._next_target_idx = 0
    if env._next_target_idx>=len(path_points):
        return torch.tensor(0.0, device=env.device)
    min_dist=float('inf')
    for x,z in path_points:
        dist=np.sqrt((pos_x-x)**2+(pos_z-z)**2)
        if dist<min_dist:
            min_dist=dist
    if min_dist<1.2:
        penalty=-0.1*(min_dist-1.2)
        return torch.tensor(penalty, device=env.device)
    return torch.tensor(0.0, device=env.device)

def alive_reward(env):
    return torch.tensor(0.005, device=env.device)
def move_reward(env):
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_w[0]
    speed = torch.norm(lin_vel)
    if speed>0.1:
        return torch.tensor(0.01, device=env.device)
    else:
        return torch.tensor(0.005, device=env.device)
