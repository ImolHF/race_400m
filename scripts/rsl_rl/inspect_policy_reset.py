"""Print an RSL-RL checkpoint's exact policy input/output at an Isaac Lab reset.

This is simulation-only.  It creates no robot SDK, ROS, or hardware client.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import race_400m.tasks  # noqa: F401, E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    if args_cli.checkpoint is None:
        parser.error("--checkpoint is required")
    checkpoint = os.path.abspath(args_cli.checkpoint)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()
    with torch.inference_mode():
        action = policy(obs)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    action_term.process_actions(action)
    target = action_term.processed_actions
    joint_ids = action_term._joint_ids
    joint_names = action_term._joint_names
    limits = env.unwrapped.scene["robot"].data.soft_joint_pos_limits[0, joint_ids]
    print("RESET_OBS_0=" + ",".join(f"{v:.9f}" for v in obs[0].tolist()), flush=True)
    print("RESET_ACTION_0=" + ",".join(f"{v:.9f}" for v in action[0].tolist()), flush=True)
    print(f"RESET_RANGE obs=[{obs.min().item():.6f},{obs.max().item():.6f}] action=[{action.min().item():.6f},{action.max().item():.6f}]", flush=True)
    print("ACTION_TARGETS (training conversion):", flush=True)
    for index, name in enumerate(joint_names):
        print(
            f"{index},{int(joint_ids[index])},{name},raw={action[0, index].item():.6f},"
            f"target={target[0, index].item():.6f},"
            f"limit=[{limits[index, 0].item():.6f},{limits[index, 1].item():.6f}]",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
