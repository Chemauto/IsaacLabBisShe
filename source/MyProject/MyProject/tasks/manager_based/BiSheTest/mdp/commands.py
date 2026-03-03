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
        self._last_fallback_log_step = -200
        self._fallback_log_interval = 200

        if self.cfg.use_valid_target_patches:
            try:
                self.terrain = env.scene["terrain"]
                self.valid_targets = self.terrain.flat_patches.get(self.cfg.target_patch_name)
            except Exception as exc:
                raise RuntimeError(
                    "[BiSheTest] Terrain asset is unavailable for target validity filtering."
                ) from exc
            if self.valid_targets is None:
                raise RuntimeError(
                    f"[BiSheTest] No flat patches found for key '{self.cfg.target_patch_name}'."
                )

    def _to_env_ids_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _sample_polar_targets(self, centers: torch.Tensor) -> torch.Tensor:
        num_envs = centers.shape[0]
        targets = torch.zeros(num_envs, 3, device=self.device)
        targets[:, :] = centers

        angles = torch.empty(num_envs, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        radii = torch.empty(num_envs, device=self.device).uniform_(*self.cfg.radius_range)
        targets[:, 0] += radii * torch.cos(angles)
        targets[:, 1] += radii * torch.sin(angles)
        return targets

    def _get_candidate_targets(self, env_ids_t: torch.Tensor) -> torch.Tensor | None:
        if self.valid_targets is None or self.terrain is None:
            return None
        terrain_levels = self.terrain.terrain_levels[env_ids_t]
        terrain_types = self.terrain.terrain_types[env_ids_t]
        return self.valid_targets[terrain_levels, terrain_types]

    def _sample_from_valid_patches(
        self,
        local_centers: torch.Tensor,
        local_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample directly from valid patch targets satisfying radius/height constraints."""
        distances = torch.norm(local_candidates[:, :, :2] - local_centers[:, None, :2], dim=-1)
        valid_mask = (distances >= self.cfg.radius_range[0]) & (distances <= self.cfg.radius_range[1])
        if self.cfg.max_target_height_offset is not None:
            height_delta = torch.abs(local_candidates[:, :, 2] - local_centers[:, None, 2])
            valid_mask &= height_delta <= self.cfg.max_target_height_offset

        has_valid = valid_mask.any(dim=1)
        selected = torch.zeros(local_centers.shape[0], 3, device=self.device)
        valid_rows = torch.nonzero(has_valid, as_tuple=False).squeeze(-1)
        for row_id in valid_rows.tolist():
            patch_ids = torch.nonzero(valid_mask[row_id], as_tuple=False).squeeze(-1)
            sampled_idx = patch_ids[torch.randint(0, patch_ids.numel(), (1,), device=self.device)[0]]
            selected[row_id] = local_candidates[row_id, sampled_idx]

        return selected, has_valid

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = self._to_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return

        # Paper: sample in polar coordinates around initial robot position.
        centers = self.robot.data.root_pos_w[env_ids_t].clone()
        sampled_targets = torch.zeros(env_ids_t.numel(), 3, device=self.device)
        unresolved = torch.arange(env_ids_t.numel(), device=self.device, dtype=torch.long)
        candidate_targets = self._get_candidate_targets(env_ids_t) if self.cfg.use_valid_target_patches else None

        attempts = 0
        while unresolved.numel() > 0 and attempts < self.cfg.max_sampling_attempts:
            attempts += 1
            local_centers = centers[unresolved]
            local_samples = self._sample_polar_targets(local_centers)

            if candidate_targets is None:
                sampled_targets[unresolved] = local_samples
                unresolved = unresolved[:0]
                break

            local_candidates = candidate_targets[unresolved]
            dist_xy = torch.norm(local_candidates[:, :, :2] - local_samples[:, None, :2], dim=-1)
            min_dist, min_ids = torch.min(dist_xy, dim=1)
            nearest_z = local_candidates[torch.arange(local_candidates.shape[0], device=self.device), min_ids, 2]

            is_valid = min_dist <= self.cfg.patch_match_tolerance
            if self.cfg.max_target_height_offset is not None:
                is_valid &= torch.abs(nearest_z - local_centers[:, 2]) <= self.cfg.max_target_height_offset

            if is_valid.any():
                valid_local = unresolved[is_valid]
                sampled_targets[valid_local, :2] = local_samples[is_valid, :2]
                sampled_targets[valid_local, 2] = nearest_z[is_valid]

            unresolved = unresolved[~is_valid]

        if unresolved.numel() > 0:
            # Paper-style re-sampling until valid: if polar retries miss, sample directly from valid patches.
            if candidate_targets is not None:
                local_centers = centers[unresolved]
                local_candidates = candidate_targets[unresolved]
                direct_samples, has_valid = self._sample_from_valid_patches(local_centers, local_candidates)
                if has_valid.any():
                    valid_local = unresolved[has_valid]
                    sampled_targets[valid_local] = direct_samples[has_valid]
                unresolved = unresolved[~has_valid]

        if unresolved.numel() > 0:
            missing_count = int(unresolved.numel())
            if self.cfg.fallback_to_polar_sampling:
                local_centers = centers[unresolved]
                local_samples = self._sample_polar_targets(local_centers)
                sampled_targets[unresolved] = local_samples
                if (
                    self.cfg.log_patch_fallback
                    and (self._env.common_step_counter - self._last_fallback_log_step) >= self._fallback_log_interval
                ):
                    logger.warning(
                        "[BiSheTest] Target patch filtering fallback for %d envs at step %d.",
                        missing_count,
                        int(self._env.common_step_counter),
                    )
                    self._last_fallback_log_step = int(self._env.common_step_counter)
            else:
                raise RuntimeError(
                    "[BiSheTest] Failed to sample valid targets for "
                    f"{missing_count} envs after polar+patch retries. "
                    "Adjust terrain and patch coverage for radius_range=(1,5)."
                )

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
