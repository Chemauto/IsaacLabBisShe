# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()




########################################惩罚 Penalties########################################################

def feet_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """足端加速度惩罚，抑制高冲击落脚。"""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        foot_acc = asset.data.body_lin_acc_w[:, body_ids, :]
    else:
        if len(body_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        foot_acc = asset.data.body_lin_acc_w[:, body_ids, :]
    # 与 sum(||a_i||^2) 等价，但避免先开方再平方的数值与性能损耗。
    return torch.sum(torch.sum(torch.square(foot_acc), dim=-1), dim=1)


########################################惩罚 Penalties########################################################




########################################任务奖励 Rewards########################################################

def final_position_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    activate_s: float = 1.0,
    distance_scale: float = 4.0,
) -> torch.Tensor:
    """末端任务奖励：仅在回合最后 activate_s 秒激活。"""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :3], dim=1)
    remaining_time_s = (env.max_episode_length - env.episode_length_buf).float() * env.step_dt
    active = remaining_time_s < activate_s
    # 对齐论文形式：1/Tr * 1/(1 + k*||e||^2)
    return (1.0 / activate_s) * (1.0 / (1.0 + distance_scale * torch.square(distance))) * active.float()


def velocity_towards_target_bias(
    env: ManagerBasedRLEnv,
    command_name: str,
    remove_threshold: float = 0.5,
    speed_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """探索奖励（论文 Eq.(3)）：朝目标方向移动时给正奖励。"""
    if getattr(env, "_manual_disable_exploration_bias", False):
        return torch.zeros(env.num_envs, device=env.device)

    command = env.command_manager.get_command(command_name)
    vel_xy = env.scene["robot"].data.root_lin_vel_b[:, :2]
    target_vec = command[:, :2]
    vel_norm = torch.norm(vel_xy, dim=1)
    target_norm = torch.norm(target_vec, dim=1)
    denom = vel_norm * target_norm
    cosine = torch.sum(vel_xy * target_vec, dim=1) / torch.clamp(denom, min=speed_epsilon)
    cosine = torch.clamp(cosine, min=-1.0, max=1.0)

    # 当末端任务奖励达到最大值 50% 时关闭探索项（论文描述）。
    task_reward_mean = torch.mean(final_position_reward(env, command_name=command_name)).item()
    if task_reward_mean >= remove_threshold:
        env._manual_disable_exploration_bias = True
        return torch.zeros(env.num_envs, device=env.device)

    valid = (target_norm > 0.05) & (vel_norm > 0.05)
    return cosine * valid.float()


def stalling_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_threshold: float = 0.1,
    distance_threshold: float = 0.5,
) -> torch.Tensor:
    """远离目标但移动很慢时惩罚，减少“站桩”。"""
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
    distance = torch.norm(command[:, :2], dim=1)
    is_stalling = (speed < speed_threshold) & (distance > distance_threshold)
    return is_stalling.float()


########################################任务奖励 Rewards########################################################
