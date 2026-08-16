"""Action terms used to make the training policy tolerant to control latency."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


def executed_action(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the position target that is actually being applied this step.

    ``last_action`` reports the policy's latest command.  With a transport
    delay that is not necessarily the command received by the actuators.
    Exposing the executed target makes the delay-augmented control process
    Markov to PPO and, importantly, reports the default standing target after
    every reset.
    """

    return env.action_manager.get_term(action_name).processed_actions


class DelayedJointPositionAction(JointPositionAction):
    """Apply position targets after a fixed number of environment control steps."""

    cfg: "DelayedJointPositionActionCfg"

    def __init__(self, cfg: "DelayedJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        if cfg.delay_steps < 1:
            raise ValueError("DelayedJointPositionAction requires delay_steps >= 1.")
        self._delay_steps = cfg.delay_steps
        # use_default_offset=True makes _offset the G1's default standing target.
        self._delayed_targets = self._offset.unsqueeze(0).repeat(self._delay_steps + 1, 1, 1).clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        # The policy still sees and records the action produced this step.
        super().process_actions(actions)
        latest_target = self._processed_actions.clone()
        self._delayed_targets = torch.roll(self._delayed_targets, shifts=-1, dims=0)
        self._delayed_targets[-1] = latest_target
        # The articulation receives a target from one control period ago.
        self._processed_actions = self._delayed_targets[0]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._delayed_targets[:] = self._offset
        else:
            self._delayed_targets[:, env_ids] = self._offset[env_ids].unsqueeze(0)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for a fixed position-target delay."""

    class_type: type = DelayedJointPositionAction
    delay_steps: int = 1
