# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for NavigationTest tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_terrain_tile(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Move the specified environments onto a randomly selected pre-generated terrain tile."""

    terrain: TerrainImporter = env.scene.terrain
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if terrain.terrain_origins is None or env_ids.numel() == 0:
        return

    num_rows, num_cols = terrain.terrain_origins.shape[:2]

    terrain.terrain_levels[env_ids] = torch.randint(0, num_rows, (env_ids.numel(),), device=env.device)
    if num_cols > 1:
        terrain.terrain_types[env_ids] = torch.randint(0, num_cols, (env_ids.numel(),), device=env.device)
    else:
        terrain.terrain_types[env_ids] = 0

    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids],
        terrain.terrain_types[env_ids],
    ]
