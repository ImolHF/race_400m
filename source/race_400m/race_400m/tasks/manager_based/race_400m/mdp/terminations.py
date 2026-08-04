import torch
import numpy as np

def robot_fallen(env,asset_cfg=None):
    robot = env.scene['robot']
    robot_pos=robot.data.root_pos_w[0]
    height=robot_pos[1]
    if height<0.2:
        print(f"[INFO]机器人摔倒，高度{height}")
        return torch.tensor(True, device=env.device)

    #姿态控制
    projected_gravity = robot.data.projected_gravity[0]
    tilt = torch.norm(projected_gravity[:2])  # 水平分量的长度

    if tilt > 0.5:
        print(f"[INFO] 机器人摔倒：倾斜度 {tilt:.2f} > 0.5")
        return torch.tensor(True, device=env.device)

    return torch.tensor(False, device=env.device)

def is_completed(env,asset_cfg=None):
    path_points = env.cfg.path_points
    if path_points is None or len(path_points)==0:
        return torch.tensor(False, device=env.device)
    if not hasattr(env, "_next_target_idx"):
        env._next_target_idx = 0
    if env._next_target_idx >= len(path_points):
        print(f"[INFO] 🎉 跑完全程！共 {len(path_points)} 个路径点")
        return torch.tensor(True, device=env.device)
    return torch.tensor(False, device=env.device)


def off_track(env, asset_cfg=None):
    """
    判断是否跑出赛道（可选）

    判断依据：偏离最近路径点超过 5 米
    """
    path_points = env.cfg.path_points
    if path_points is None or len(path_points) == 0:
        return torch.tensor(False, device=env.device)

    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[0]
    pos_x, pos_z = robot_pos[0].item(), robot_pos[2].item()

    # 找到最近的路径点
    min_dist = float('inf')
    for tx, tz in path_points:
        dist = np.sqrt((pos_x - tx) ** 2 + (pos_z - tz) ** 2)
        if dist < min_dist:
            min_dist = dist

    # 如果偏离超过 5 米，视为跑出赛道
    if min_dist > 5.0:
        print(f"[INFO] 机器人跑出赛道：偏离 {min_dist:.2f}m > 5m")
        return torch.tensor(True, device=env.device)

    return torch.tensor(False, device=env.device)