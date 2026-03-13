# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _box_goal_delta(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    box: RigidObject = env.scene[box_cfg.name]
    goal_pos_e = env.command_manager.get_command(command_name)
    goal_pos_w = goal_pos_e + env.scene.env_origins
    delta = goal_pos_w[:, :2] - box.data.root_pos_w[:, :2]
    distance = torch.norm(delta, dim=1)
    return delta, distance


def box_goal_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward moving the box center close to the target point."""
    _, distance = _box_goal_delta(env, command_name, box_cfg)
    return 1.0 - torch.tanh(distance / std)


def box_goal_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward step-wise progress of the box towards the goal."""
    _, current_distance = _box_goal_delta(env, command_name, box_cfg)
    buffer_name = "_push_box_prev_goal_distance"
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, current_distance.clone())
    previous_distance = getattr(env, buffer_name)
    if hasattr(env, "episode_length_buf"):
        reset_ids = env.episode_length_buf == 0
        previous_distance = previous_distance.clone()
        previous_distance[reset_ids] = current_distance[reset_ids]
    progress = previous_distance - current_distance
    setattr(env, buffer_name, current_distance.clone())
    return progress


def box_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward box velocity aligned with the goal direction."""
    box: RigidObject = env.scene[box_cfg.name]
    delta, distance = _box_goal_delta(env, command_name, box_cfg)
    goal_dir = delta / (distance.unsqueeze(-1) + 1.0e-6)
    return torch.sum(box.data.root_lin_vel_w[:, :2] * goal_dir, dim=1) * (distance > 0.05).float()


def robot_box_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward the robot for staying close enough to interact with the box."""
    robot: RigidObject = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    distance = torch.norm(box.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    return 1.0 - torch.tanh(distance / std)


def box_goal_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.08,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Sparse bonus when the box reaches the placement zone."""
    _, distance = _box_goal_delta(env, command_name, box_cfg)
    return (distance < distance_threshold).float()


def orientation_l2(
    env: ManagerBasedRLEnv,
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for keeping the body aligned with the desired gravity direction.

    This follows the same idea as the reference unitree locomotion rewards, but uses
    desired gravity ``[0, 0, -1]`` for an upright Go2 on flat ground.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    desired_gravity_tensor = torch.tensor(
        desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype
    )
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity_tensor, dim=-1)
    normalized = 0.5 * cos_dist + 0.5
    return torch.square(normalized)
