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

