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
        self._forward_only_terrain_type_mask = self._resolve_forward_only_terrain_type_mask()

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
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if self._forward_only_terrain_type_mask is None:
            self._resample_velocity(
                env_ids,
                self.cfg.ranges.lin_vel_x,
                self.cfg.ranges.lin_vel_y,
                self.cfg.ranges.ang_vel_z,
            )
        else:
            terrain_types = self._env.scene.terrain.terrain_types[env_ids]
            forward_only_mask = self._forward_only_terrain_type_mask[terrain_types]

            default_env_ids = env_ids[~forward_only_mask]
            if len(default_env_ids) > 0:
                self._resample_velocity(
                    default_env_ids,
                    self.cfg.ranges.lin_vel_x,
                    self.cfg.ranges.lin_vel_y,
                    self.cfg.ranges.ang_vel_z,
                )

            forward_only_env_ids = env_ids[forward_only_mask]
            if len(forward_only_env_ids) > 0:
                self._resample_velocity(
                    forward_only_env_ids,
                    self.cfg.forward_only_lin_vel_x,
                    self.cfg.forward_only_lin_vel_y,
                    self.cfg.forward_only_ang_vel_z,
                )

        r = torch.empty(len(env_ids), device=self.device)
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

    def _resolve_forward_only_terrain_type_mask(self) -> torch.Tensor | None:
        """Return a per-terrain-column mask for terrain types that should use forward-only commands."""
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_types"):
            return None

        terrain_generator_cfg = terrain.cfg.terrain_generator
        if terrain_generator_cfg is None or not terrain_generator_cfg.curriculum:
            return None

        target_terrain_names = set(self.cfg.forward_only_terrain_names)
        if not target_terrain_names:
            return None

        sub_terrain_names = list(terrain_generator_cfg.sub_terrains.keys())
        proportions = torch.tensor(
            [terrain_generator_cfg.sub_terrains[name].proportion for name in sub_terrain_names],
            device=self.device,
            dtype=torch.float32,
        )
        proportions = proportions / torch.sum(proportions)
        cumulative_proportions = torch.cumsum(proportions, dim=0)

        num_cols = terrain_generator_cfg.num_cols
        terrain_type_mask = torch.zeros(num_cols, device=self.device, dtype=torch.bool)
        for terrain_type in range(num_cols):
            ratio = terrain_type / num_cols + 0.001
            sub_terrain_index = torch.searchsorted(
                cumulative_proportions,
                torch.tensor(ratio, device=self.device, dtype=cumulative_proportions.dtype),
            )
            sub_terrain_index = torch.clamp(sub_terrain_index, max=len(sub_terrain_names) - 1)
            sub_terrain_name = sub_terrain_names[int(sub_terrain_index)]
            terrain_type_mask[terrain_type] = sub_terrain_name in target_terrain_names

        return terrain_type_mask

    def _resample_velocity(
        self,
        env_ids: torch.Tensor,
        lin_vel_x_range: tuple[float, float],
        lin_vel_y_range: tuple[float, float],
        ang_vel_z_range: tuple[float, float],
    ):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_w[env_ids, 0] = r.uniform_(*lin_vel_x_range)
        self.vel_command_w[env_ids, 1] = r.uniform_(*lin_vel_y_range)
        self.vel_command_w[env_ids, 2] = r.uniform_(*ang_vel_z_range)


@configclass
class UniformWorldVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`UniformWorldVelocityCommand`."""

    class_type: type = UniformWorldVelocityCommand

    forward_only_terrain_names: tuple[str, ...] = ()
    """Terrain names that should use forward-only velocity ranges."""

    forward_only_lin_vel_x: tuple[float, float] = (0.0, 1.0)
    """Linear x velocity range used on forward-only terrain types."""

    forward_only_lin_vel_y: tuple[float, float] = (0.0, 0.0)
    """Linear y velocity range used on forward-only terrain types."""

    forward_only_ang_vel_z: tuple[float, float] = (0.0, 0.0)
    """Yaw-rate range used on forward-only terrain types."""
