# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # With two GPUs and 4,096 environments per GPU, one PPO rollout contains
    # 196,608 transitions (2 * 4,096 * 24).
    num_steps_per_env = 24
    # This experiment intentionally starts from scratch: it uses 12 leg
    # actions plus two shoulder-pitch actions, not the old 16-action policy.
    max_iterations = 8000
    save_interval = 50
    experiment_name = "g1_track_400m"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        # Conservative exploration is important for position-controlled
        # humanoid joints; 1.0 produced full-scale random residuals at reset.
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SingleGpuPPORunnerCfg(PPORunnerCfg):
    """One-GPU batching for the unchanged locked-elbow main task.

    4096 environments x 48 rollout steps equals 196,608 samples per update,
    matching the established two-GPU setting (2 x 4096 x 24).
    """

    num_steps_per_env = 48


@configclass
class LegOnlyPPORunnerCfg(PPORunnerCfg):
    """Runner name retained so existing 12-action checkpoints can resume."""

    experiment_name = "g1_track_400m"


@configclass
class DelayedLockedElbowPPORunnerCfg(PPORunnerCfg):
    """From-scratch PPO run for the delayed 98-observation controller."""

    # The executed-action observation changes the actor input from 84 to 98,
    # so loading an old no-delay actor would be invalid.  Keep its logs in a
    # separate root and learn the timing-aware gait from scratch.
    experiment_name = "g1_track_400m_locked_elbow_delay"
    # Eight GPUs x 4096 envs x 12 steps keeps each PPO update at 393,216
    # transitions, matching the proven multi-GPU configuration.
    num_steps_per_env = 12
    max_iterations = 10000
    save_interval = 50


@configclass
class RandomDelayLockedElbowPPORunnerCfg(PPORunnerCfg):
    """Conservative fine-tuning of the proven 84-observation arm policy."""

    experiment_name = "g1_track_400m_random_delay"
    num_steps_per_env = 12
    max_iterations = 4000
    save_interval = 50
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.004,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-4, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class RobustPhysicsPPORunnerCfg(PPORunnerCfg):
    experiment_name = "g1_track_400m_robust_physics"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50


@configclass
class LegOnlyHighCadencePPORunnerCfg(LegOnlyPPORunnerCfg):
    """Same policy interface; the completed leg-only checkpoint can resume."""

    experiment_name = "g1_track_400m"


@configclass
class LegOnlyStartStopPPORunnerCfg(LegOnlyHighCadencePPORunnerCfg):
    """New 83-observation policy for phase-aware start and stop control."""

    experiment_name = "g1_track_400m_start_stop"


@configclass
class LockedElbowStartStopPPORunnerCfg(PPORunnerCfg):
    """Eight-GPU PPO configuration for the 14-action, 85-observation task."""

    # 8 * 4096 * 12 = 393,216 transitions per update, identical to the former
    # six-GPU (6 * 4096 * 16) batch. This preserves PPO update scale while
    # reducing per-rank rollout work instead of making each update 33% larger.
    num_steps_per_env = 12
    max_iterations = 10000
    save_interval = 50
    experiment_name = "g1_track_400m_locked_elbow_start_stop"
