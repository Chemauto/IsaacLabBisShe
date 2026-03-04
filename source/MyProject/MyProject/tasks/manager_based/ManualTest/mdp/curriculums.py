# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_ids_to_tensor(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """将 env_ids 统一转换为当前设备上的 long tensor。"""
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _compute_column_map(terrain_gen_cfg) -> dict[str, list[int]]:
    """按照 sub_terrain 的 proportion，计算每个地形对应的列索引集合。"""
    names = list(terrain_gen_cfg.sub_terrains.keys())
    proportions = [terrain_gen_cfg.sub_terrains[name].proportion for name in names]
    total = sum(proportions)
    if total <= 0.0:
        return {name: [] for name in names}

    cumulative = []
    running = 0.0
    for p in proportions:
        running += p / total
        cumulative.append(running)

    col_map = {name: [] for name in names}
    for col in range(terrain_gen_cfg.num_cols):
        value = col / float(terrain_gen_cfg.num_cols) + 1.0e-3
        idx = 0
        while idx < len(cumulative) - 1 and value >= cumulative[idx]:
            idx += 1
        col_map[names[idx]].append(col)
    return col_map


def _select_stage(current_iteration: int, iter_stage_boundaries: tuple[int, ...]) -> int:
    """根据当前迭代数选择课程阶段索引。"""
    stage = 0
    for idx, boundary in enumerate(iter_stage_boundaries):
        if current_iteration >= boundary:
            stage = idx
    return stage


def pit_terrain_by_iteration(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    iter_stage_boundaries: tuple[int, int, int, int] = (0, 400, 900, 1300),
    steps_per_iteration: int = 8,
    stage_weights: tuple[tuple[float, float, float], ...] = (
        (0.85, 0.14, 0.01),  # easy/medium/hard
        (0.65, 0.28, 0.07),
        (0.45, 0.35, 0.20),
        (0.25, 0.35, 0.40),
    ),
    stage_max_level_ratio: tuple[float, ...] = (0.35, 0.55, 0.75, 1.0),
) -> torch.Tensor:
    """仅基于训练迭代数的坑洞课程学习。

    前期：主要采样简单坑，并限制较低地形等级；
    后期：逐步提高中/困难坑占比，并放开更高地形等级。
    """
    terrain: TerrainImporter = env.scene.terrain
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    env_ids_t = _env_ids_to_tensor(env, env_ids)
    if env_ids_t.numel() == 0:
        return torch.tensor(0.0, device=env.device)

    terrain_gen_cfg = terrain.cfg.terrain_generator
    col_map = _compute_column_map(terrain_gen_cfg)

    terrain_keys = ("easy_pit", "medium_pit", "hard_pit")
    if any(len(col_map.get(key, [])) == 0 for key in terrain_keys):
        return torch.tensor(0.0, device=env.device)

    # Isaac Lab 中 common_step_counter 是全局 step，这里换算为训练迭代轮数。
    current_iteration = int(env.common_step_counter) // max(1, int(steps_per_iteration))
    stage = _select_stage(current_iteration, iter_stage_boundaries)
    stage = min(stage, len(stage_weights) - 1, len(stage_max_level_ratio) - 1)

    # 按当前阶段的 easy/medium/hard 权重采样地形类型。
    weights = torch.tensor(stage_weights[stage], device=env.device, dtype=torch.float32)
    probs = weights / torch.clamp(weights.sum(), min=1.0e-6)
    picked_keys = torch.multinomial(probs, num_samples=env_ids_t.numel(), replacement=True)

    sampled_types = torch.empty(env_ids_t.numel(), dtype=torch.long, device=env.device)
    for key_idx, key in enumerate(terrain_keys):
        mask = picked_keys == key_idx
        if not torch.any(mask):
            continue
        columns = torch.as_tensor(col_map[key], dtype=torch.long, device=env.device)
        sampled = torch.randint(0, columns.numel(), (int(mask.sum().item()),), device=env.device)
        sampled_types[mask] = columns[sampled]

    terrain.terrain_types[env_ids_t] = sampled_types

    # 按阶段限制最大 terrain level，形成“由易到难”的课程推进。
    max_level = max(
        0,
        min(
            terrain.max_terrain_level - 1,
            int(round((terrain.max_terrain_level - 1) * float(stage_max_level_ratio[stage]))),
        ),
    )
    terrain.terrain_levels[env_ids_t] = torch.clamp(terrain.terrain_levels[env_ids_t], min=0, max=max_level)
    terrain.env_origins[env_ids_t] = terrain.terrain_origins[terrain.terrain_levels[env_ids_t], terrain.terrain_types[env_ids_t]]

    return torch.tensor(float(stage), device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
