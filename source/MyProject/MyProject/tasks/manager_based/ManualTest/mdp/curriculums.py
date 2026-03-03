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


def pit_difficulty_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """基于训练进度的坑洞地形课程学习（纯进度驱动，无性能反馈）

    难度递进：
    - Stage 1 (0-25%): Level 0-2, 只用简单坑 (10-20cm)
    - Stage 2 (25-50%): Level 1-4, 逐渐引入中等坑 (20-27cm)
    - Stage 3 (50-75%): Level 3-6, 主要中等坑 (27-35cm)
    - Stage 4 (75-100%): Level 5-8, 挑战深坑 (35-50cm)

    Args:
        env: 学习环境
        env_ids: 环境ID
        asset_cfg: 机器人配置

    Returns:
        当前难度级别 (0.0 到 1.0)
    """
    # 获取地形对象
    terrain: TerrainImporter = env.scene.terrain

    # 步骤1: 获取训练进度 (0% → 100%)
    total_iterations = getattr(env, "common_step_counter", 0)
    max_iterations = 15000  # 根据训练计划调整
    training_progress = min(total_iterations / max_iterations, 1.0)

    # 步骤2: 根据训练进度确定目标难度级别
    if training_progress < 0.25:      # 前25%训练
        target_level_min = 0
        target_level_max = 2
    elif training_progress < 0.5:     # 25%-50%训练
        target_level_min = 1
        target_level_max = 4
    elif training_progress < 0.75:    # 50%-75%训练
        target_level_min = 3
        target_level_max = 6
    else:                             # 75%-100%训练
        target_level_min = 5
        target_level_max = 8

    # 计算目标级别（范围的中点）
    target_level = (target_level_min + target_level_max) / 2.0

    # 获取当前地形级别
    current_levels = terrain.terrain_levels[env_ids]

    # 步骤3: 平滑过渡到目标级别
    level_diff = target_level - current_levels.float()
    max_step = 0.05  # 每次最多变化0.05个级别（更平滑）
    level_diff = torch.clamp(level_diff, -max_step, max_step)

    # 确定哪些环境需要升级或降级
    move_up = level_diff > 0.01  # 大于0.01才升级
    move_down = level_diff < -0.01  # 小于-0.01才降级

    # 更新地形级别
    if torch.any(move_up) or torch.any(move_down):
        terrain.update_env_origins(env_ids, move_up, move_down)

    # 返回归一化的难度 (0.0 到 1.0)
    max_possible_level = terrain.cfg.max_init_terrain_level
    current_difficulty = torch.mean(current_levels.float()) / max(max_possible_level, 1)

    return current_difficulty


# def adaptive_pit_curriculum(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     command_threshold: float = 0.5,
# ) -> torch.Tensor:
#     """Adaptive curriculum that adjusts difficulty based on command tracking performance.
#
#     This term is specifically designed for navigation tasks where the robot must reach
#     target positions. It increases difficulty when the robot successfully reaches targets
#     and decreases difficulty when it fails consistently.
#
#     Args:
#         env: The learning environment.
#         env_ids: The environment IDs to update.
#         asset_cfg: The asset configuration.
#         command_threshold: Velocity threshold for considering movement as successful.
#
#     Returns:
#         The mean terrain level for the given environment ids.
#     """
#     # Extract used quantities
#     asset: Articulation = env.scene[asset_cfg.name]
#     terrain: TerrainImporter = env.scene.terrain
#
#     # Get position command (for navigation tasks)
#     try:
#         command = env.command_manager.get_command("pose_command")
#         command_pos = command[env_ids, :2]  # x, y position targets
#     except (KeyError, AttributeError):
#         # Fallback to velocity command
#         command = env.command_manager.get_command("base_velocity")
#         command_pos = command[env_ids, :2]
#
#     # Compute current position relative to environment origin
#     current_pos = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
#
#     # Compute distance to command
#     distance_to_command = torch.norm(current_pos - command_pos, dim=1)
#
#     # Robots that are close to their target (success) move to harder terrains
#     success_threshold = 0.5  # meters
#     move_up = distance_to_command < success_threshold
#
#     # Robots that are far from target and not moving (failure) move to easier terrains
#     failure_threshold = 2.0  # meters
#     move_down = distance_to_command > failure_threshold
#
#     # Get current velocity to detect stuck robots
#     if hasattr(asset, "data") and hasattr(asset.data, "root_vel_w"):
#         vel_mag = torch.norm(asset.data.root_vel_w[env_ids, :2], dim=1)
#         is_stuck = vel_mag < 0.1
#         move_down = move_down & is_stuck
#
#     # Ensure no environment moves both up and down
#     move_down = move_down & ~move_up
#
#     # Update terrain levels
#     if torch.any(move_up) or torch.any(move_down):
#         terrain.update_env_origins(env_ids, move_up, move_down)
#
#     # Return the mean terrain level
#     return torch.mean(terrain.terrain_levels.float())
