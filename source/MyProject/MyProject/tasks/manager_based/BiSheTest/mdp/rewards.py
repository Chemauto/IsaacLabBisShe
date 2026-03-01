# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def final_position_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    activate_s: float = 1.0,
    distance_scale: float = 4.0,
) -> torch.Tensor:
    """Reward only near episode end, based on target distance in base frame."""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    remaining_time_s = (env.max_episode_length - env.episode_length_buf).float() * env.step_dt
    active = remaining_time_s <= activate_s
    reward = 1.0 / (1.0 + distance_scale * torch.square(distance))
    return reward * active.float()


def velocity_towards_target_bias(
    env: ManagerBasedRLEnv,
    command_name: str,
    disable_after_steps: int = 1_000_000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Early-stage exploration reward: velocity projection towards target."""
    if env.common_step_counter >= disable_after_steps:
        return torch.zeros(env.num_envs, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_vec = command[:, :2]
    target_norm = torch.norm(target_vec, dim=1, keepdim=True).clamp(min=1e-6)
    target_dir = target_vec / target_norm

    vel_xy = asset.data.root_lin_vel_b[:, :2]
    vel_projection = torch.sum(vel_xy * target_dir, dim=1)

    valid = (target_norm.squeeze(1) > 0.05).float()
    return vel_projection * valid


def stalling_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_threshold: float = 0.1,
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty when far from target but moving too slowly."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    distance = torch.norm(command[:, :2], dim=1)
    is_stalling = (speed < speed_threshold) & (distance > distance_threshold)
    return is_stalling.float()


def feet_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet linear accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(torch.norm(foot_acc, dim=-1)), dim=1)
