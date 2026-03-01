# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_position(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "target_pose",
    success_threshold: float = 0.5,
    fail_threshold: float = 2.0,
) -> torch.Tensor:
    """Terrain curriculum based on final distance to target."""
    terrain: TerrainImporter = env.scene.terrain
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    command = env.command_manager.get_command(command_name)
    distance_to_target = torch.norm(command[:, :2], dim=1)

    move_up = distance_to_target[env_ids] < success_threshold
    move_down = distance_to_target[env_ids] > fail_threshold
    move_down = move_down & (~move_up)

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
