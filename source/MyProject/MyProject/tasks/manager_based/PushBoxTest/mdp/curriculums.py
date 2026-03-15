# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_goal_progress_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "box_goal",
) -> float:
    """Curriculum based on how much the box-goal distance improves over one episode.

    The curriculum value A is defined as:
        A = clamp(1 - final_distance / initial_distance, 0, 1)

    The next sampled goal range is then centered at the command range midpoint, and scaled by A.
    """
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(list(env_ids), device=env.device, dtype=torch.long)

    command_term = env.command_manager.get_term(command_name)
    box: RigidObject = env.scene[command_term.cfg.asset_name]

    state_name = "_push_box_goal_curriculum_state"
    if not hasattr(env, state_name):
        setattr(
            env,
            state_name,
            {
                "base_pos_x": tuple(command_term.cfg.ranges.pos_x),
                "base_pos_y": tuple(command_term.cfg.ranges.pos_y),
                "value": 0.0,
            },
        )

    state = getattr(env, state_name)

    if env.common_step_counter > 0 and len(env_ids) > 0:
        initial_distance = command_term.initial_error_pos[env_ids].clamp_min(1e-6)
        final_distance = torch.norm(command_term.pos_command_w[env_ids, :2] - box.data.root_pos_w[env_ids, :2], dim=1)
        progress = torch.clamp(1.0 - final_distance / initial_distance, min=0.0, max=1.0)
        state["value"] = torch.mean(progress).item()

    base_pos_x = state["base_pos_x"]
    base_pos_y = state["base_pos_y"]
    center_x = 0.5 * (base_pos_x[0] + base_pos_x[1])
    center_y = 0.5 * (base_pos_y[0] + base_pos_y[1])
    half_range_x = 0.5 * (base_pos_x[1] - base_pos_x[0]) * state["value"]
    half_range_y = 0.5 * (base_pos_y[1] - base_pos_y[0]) * state["value"]

    command_term.cfg.ranges.pos_x = (center_x - half_range_x, center_x + half_range_x)
    command_term.cfg.ranges.pos_y = (center_y - half_range_y, center_y + half_range_y)
    env.cfg.commands.box_goal.ranges.pos_x = command_term.cfg.ranges.pos_x
    env.cfg.commands.box_goal.ranges.pos_y = command_term.cfg.ranges.pos_y

    return state["value"]
