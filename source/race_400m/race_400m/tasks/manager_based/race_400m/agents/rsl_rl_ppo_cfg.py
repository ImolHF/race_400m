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
class LegOnlyPPORunnerCfg(PPORunnerCfg):
    """Runner name retained so existing 12-action checkpoints can resume."""

    experiment_name = "g1_track_400m"


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
    """Six-GPU friendly PPO configuration for the 14-action, 85-observation task."""

    # 6 * 4096 * 16 = 393,216 transitions per update. Keeping 24 rollout
    # steps would create 589,824 transitions/update and increases PPO and
    # distributed synchronization time without improving simulator occupancy.
    num_steps_per_env = 16
    max_iterations = 10000
    save_interval = 50
    experiment_name = "g1_track_400m_locked_elbow_start_stop"
