# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from .commands_cfg import AdvancedPose2dCommandCfg


class AdvancedPose2dCommand(UniformPose2dCommand):
    """Position command sampled in polar coordinates for final-position tasks.

    The target is sampled around each environment origin with:
    - radius in ``cfg.radius_range``
    - angle in ``[0, 2*pi]``
    """

    cfg: AdvancedPose2dCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]

        num_envs = len(env_ids)
        angles = torch.empty(num_envs, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        radii = torch.empty(num_envs, device=self.device).uniform_(*self.cfg.radius_range)

        self.pos_command_w[env_ids, 0] += radii * torch.cos(angles)
        self.pos_command_w[env_ids, 1] += radii * torch.sin(angles)
        self.pos_command_w[env_ids, 2] = (
            self.robot.data.default_root_state[env_ids, 2] + self.cfg.goal_height_offset
        )

        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_heading = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_heading = wrap_to_pi(target_heading + torch.pi)

            current_heading = self.robot.data.heading_w[env_ids]
            err_target = wrap_to_pi(target_heading - current_heading).abs()
            err_flipped = wrap_to_pi(flipped_heading - current_heading).abs()
            self.heading_command_w[env_ids] = torch.where(err_target < err_flipped, target_heading, flipped_heading)
        else:
            self.heading_command_w[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
                *self.cfg.ranges.heading
            )
