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
    min_episode_length: int = 500,
    difficulty_window: int = 100,
) -> torch.Tensor:
    """Custom curriculum term for pit terrain based on robot performance and training progress.

    This curriculum adjusts terrain difficulty based on:
    1. Training progress (total number of iterations)
    2. Robot performance (episode success rate and average velocity)
    3. Smooth transitions between difficulty levels

    The difficulty progression:
    - Stage 1 (0-25% training): 70% easy pits, 30% medium pits
    - Stage 2 (25-50% training): 40% easy pits, 50% medium pits, 10% hard pits
    - Stage 3 (50-75% training): 20% easy pits, 50% medium pits, 30% hard pits
    - Stage 4 (75-100% training): 10% easy pits, 40% medium pits, 50% hard pits

    Args:
        env: The learning environment.
        env_ids: The environment IDs to update.
        asset_cfg: The asset configuration.
        min_episode_length: Minimum episode length before considering advancement.
        difficulty_window: Window size for averaging performance metrics.

    Returns:
        The current difficulty level (0.0 to 1.0).
    """
    # Extract used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # Get training progress (normalized iteration count)
    total_iterations = getattr(env, "common_step_counter", 0)
    max_iterations = 15000  # Adjust based on your training plan
    training_progress = min(total_iterations / max_iterations, 1.0)

    # Determine difficulty stage based on training progress
    if training_progress < 0.25:
        target_level_min = 0
        target_level_max = 2
    elif training_progress < 0.5:
        target_level_min = 1
        target_level_max = 4
    elif training_progress < 0.75:
        target_level_min = 3
        target_level_max = 6
    else:
        target_level_min = 5
        target_level_max = 8

    # Get current terrain levels
    current_levels = terrain.terrain_levels[env_ids]

    # Check robot performance for additional adjustments
    # Compute average velocity magnitude
    if hasattr(asset, "data") and hasattr(asset.data, "root_vel_w"):
        vel_mag = torch.norm(asset.data.root_vel_w[env_ids, :2], dim=1)
        avg_vel = torch.mean(vel_mag)

        # Robots performing well can progress faster
        performance_multiplier = 1.0
        avg_vel = avg_vel.item() if torch.is_tensor(avg_vel) else avg_vel
        if avg_vel > 0.8:  # Robot is moving well
            performance_multiplier = 1.2
        elif avg_vel < 0.3:  # Robot is struggling
            performance_multiplier = 0.8
    else:
        performance_multiplier = 1.0

    # Compute target level with performance adjustment
    target_level = (target_level_min + target_level_max) / 2 * performance_multiplier
    target_level = min(max(target_level, 0), terrain.cfg.max_init_terrain_level)

    # Smooth transition: only move a small step toward target
    level_diff = target_level - current_levels.float()
    max_step = 0.1  # Maximum level change per update
    level_diff = torch.clamp(level_diff, -max_step, max_step)

    # Determine which environments should move up or down
    move_up = level_diff > 0.02
    move_down = level_diff < -0.02

    # Update terrain levels
    if torch.any(move_up) or torch.any(move_down):
        terrain.update_env_origins(env_ids, move_up, move_down)

    # Return normalized difficulty (0.0 to 1.0)
    max_possible_level = terrain.cfg.max_init_terrain_level
    current_difficulty = torch.mean(current_levels.float()) / max(max_possible_level, 1)

    return current_difficulty


def adaptive_pit_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Adaptive curriculum that adjusts difficulty based on command tracking performance.

    This term is specifically designed for navigation tasks where the robot must reach
    target positions. It increases difficulty when the robot successfully reaches targets
    and decreases difficulty when it fails consistently.

    Args:
        env: The learning environment.
        env_ids: The environment IDs to update.
        asset_cfg: The asset configuration.
        command_threshold: Velocity threshold for considering movement as successful.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # Extract used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # Get position command (for navigation tasks)
    try:
        command = env.command_manager.get_command("pose_command")
        command_pos = command[env_ids, :2]  # x, y position targets
    except (KeyError, AttributeError):
        # Fallback to velocity command
        command = env.command_manager.get_command("base_velocity")
        command_pos = command[env_ids, :2]

    # Compute current position relative to environment origin
    current_pos = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]

    # Compute distance to command
    distance_to_command = torch.norm(current_pos - command_pos, dim=1)

    # Robots that are close to their target (success) move to harder terrains
    success_threshold = 0.5  # meters
    move_up = distance_to_command < success_threshold

    # Robots that are far from target and not moving (failure) move to easier terrains
    failure_threshold = 2.0  # meters
    move_down = distance_to_command > failure_threshold

    # Get current velocity to detect stuck robots
    if hasattr(asset, "data") and hasattr(asset.data, "root_vel_w"):
        vel_mag = torch.norm(asset.data.root_vel_w[env_ids, :2], dim=1)
        is_stuck = vel_mag < 0.1
        move_down = move_down & is_stuck

    # Ensure no environment moves both up and down
    move_down = move_down & ~move_up

    # Update terrain levels
    if torch.any(move_up) or torch.any(move_down):
        terrain.update_env_origins(env_ids, move_up, move_down)

    # Return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
