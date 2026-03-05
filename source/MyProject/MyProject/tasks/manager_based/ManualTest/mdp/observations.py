# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def normalized_time_to_go(env: ManagerBasedRLEnv) -> torch.Tensor:
    """归一化剩余回合时间，范围 [0, 1]。"""
    remaining_steps = (env.max_episode_length - env.episode_length_buf).float().unsqueeze(1)
    return torch.clamp(remaining_steps / float(env.max_episode_length), min=0.0, max=1.0)


def pose_command_position_b(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """提取目标在机体坐标系下的 3D 位置 (dx, dy, dz)。

    说明：
    - UniformPose2dCommand 的原始输出是 4 维: (dx, dy, dz, dheading)。
    - 论文描述中的命令观测只需要目标 3D 位置 + 剩余时间。
    """
    return env.command_manager.get_command(command_name)[:, :3]
