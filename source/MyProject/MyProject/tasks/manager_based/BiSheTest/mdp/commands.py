# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from .commands_cfg import AdvancedPose2dCommandCfg

logger = logging.getLogger(__name__)


class AdvancedPose2dCommand(UniformPose2dCommand):
    """Position command sampled in polar coordinates for final-position tasks.

    The target is sampled around each environment origin with:
    - radius in ``cfg.radius_range``
    - angle in ``[0, 2*pi]``
    """

    cfg: AdvancedPose2dCommandCfg

    def __init__(self, cfg: AdvancedPose2dCommandCfg, env):
        super().__init__(cfg, env)

        self.terrain: TerrainImporter | None = None
        self.valid_targets: torch.Tensor | None = None
        self._last_fallback_log_step = -1

        if self.cfg.use_valid_target_patches:
            try:
                self.terrain = env.scene["terrain"]
                self.valid_targets = self.terrain.flat_patches.get(self.cfg.target_patch_name)
                if self.valid_targets is None:
                    logger.warning(
                        "[BiSheTest] No flat patches found for key '%s'. Falling back to polar target sampling.",
                        self.cfg.target_patch_name,
                    )
            except Exception:
                logger.warning(
                    "[BiSheTest] Terrain asset is unavailable in command term. Falling back to polar target sampling."
                )

    def _to_env_ids_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _sample_polar_targets(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        num_envs = env_ids_t.numel()
        targets = torch.zeros(num_envs, 3, device=self.device)
        targets[:, :] = self._env.scene.env_origins[env_ids_t]

        angles = torch.empty(num_envs, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        radii = torch.empty(num_envs, device=self.device).uniform_(*self.cfg.radius_range)
        targets[:, 0] += radii * torch.cos(angles)
        targets[:, 1] += radii * torch.sin(angles)
        return targets

    def _sample_patch_targets(self, env_ids_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = env_ids_t.numel()
        targets = torch.zeros(num_envs, 3, device=self.device)
        has_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if self.valid_targets is None or self.terrain is None:
            return targets, has_valid

        terrain_levels = self.terrain.terrain_levels[env_ids_t]
        terrain_types = self.terrain.terrain_types[env_ids_t]
        env_origins = self._env.scene.env_origins[env_ids_t]
        candidate_targets = self.valid_targets[terrain_levels, terrain_types]

        distances = torch.norm(candidate_targets[:, :, :2] - env_origins[:, None, :2], dim=-1)
        valid_mask = (distances >= self.cfg.radius_range[0]) & (distances <= self.cfg.radius_range[1])
        if self.cfg.max_target_height_offset is not None:
            height_delta = torch.abs(candidate_targets[:, :, 2] - env_origins[:, None, 2])
            valid_mask &= height_delta <= self.cfg.max_target_height_offset

        valid_counts = valid_mask.sum(dim=1)
        has_valid = valid_counts > 0

        valid_env_local_ids = torch.nonzero(has_valid, as_tuple=False).squeeze(-1)
        for local_id in valid_env_local_ids.tolist():
            patch_ids = torch.nonzero(valid_mask[local_id], as_tuple=False).squeeze(-1)
            sampled_idx = patch_ids[torch.randint(0, patch_ids.numel(), (1,), device=self.device)[0]]
            targets[local_id] = candidate_targets[local_id, sampled_idx]

        return targets, has_valid

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = self._to_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return

        polar_targets = self._sample_polar_targets(env_ids_t)
        sampled_targets = polar_targets

        if self.cfg.use_valid_target_patches:
            patch_targets, has_valid = self._sample_patch_targets(env_ids_t)
            if has_valid.any():
                sampled_targets[has_valid] = patch_targets[has_valid]

            missing_count = int((~has_valid).sum().item())
            if missing_count > 0 and not self.cfg.fallback_to_polar_sampling:
                sampled_targets[~has_valid] = self._env.scene.env_origins[env_ids_t][~has_valid]
            if missing_count > 0 and self.cfg.log_patch_fallback and self._env.common_step_counter != self._last_fallback_log_step:
                logger.warning(
                    "[BiSheTest] Target patch filtering fallback for %d envs at step %d.",
                    missing_count,
                    int(self._env.common_step_counter),
                )
                self._last_fallback_log_step = int(self._env.common_step_counter)

        sampled_targets[:, 2] += self.robot.data.default_root_state[env_ids_t, 2]
        sampled_targets[:, 2] += self.cfg.goal_height_offset
        self.pos_command_w[env_ids_t] = sampled_targets

        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids_t] - self.robot.data.root_pos_w[env_ids_t]
            target_heading = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_heading = wrap_to_pi(target_heading + torch.pi)

            current_heading = self.robot.data.heading_w[env_ids_t]
            err_target = wrap_to_pi(target_heading - current_heading).abs()
            err_flipped = wrap_to_pi(flipped_heading - current_heading).abs()
            self.heading_command_w[env_ids_t] = torch.where(err_target < err_flipped, target_heading, flipped_heading)
        else:
            self.heading_command_w[env_ids_t] = torch.empty(env_ids_t.numel(), device=self.device).uniform_(
                *self.cfg.ranges.heading
            )
