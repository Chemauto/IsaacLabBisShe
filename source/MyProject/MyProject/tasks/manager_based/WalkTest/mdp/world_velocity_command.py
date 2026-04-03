# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame velocity command for climb training."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformWorldVelocityCommand(UniformVelocityCommand):
    """Sample a fixed world-frame planar velocity and expose its body-frame equivalent to the policy."""

    cfg: "UniformWorldVelocityCommandCfg"

    def __init__(self, cfg: "UniformWorldVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.vel_command_w = torch.zeros_like(self.vel_command_b)

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_w[:, :2] - self.robot.data.root_lin_vel_w[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_w[:, 2] - self.robot.data.root_ang_vel_w[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self._sync_body_command(env_ids)

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_w[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_w[standing_env_ids, :] = 0.0
        self._sync_body_command(slice(None))

    def _sync_body_command(self, env_ids: Sequence[int] | slice):
        command_w = torch.zeros((self.vel_command_w[env_ids].shape[0], 3), device=self.device)
        command_w[:, :2] = self.vel_command_w[env_ids, :2]
        command_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w[env_ids]),
            command_w,
        )
        self.vel_command_b[env_ids, :2] = command_b[:, :2]
        self.vel_command_b[env_ids, 2] = self.vel_command_w[env_ids, 2]


@configclass
class UniformWorldVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`UniformWorldVelocityCommand`."""

    class_type: type = UniformWorldVelocityCommand
