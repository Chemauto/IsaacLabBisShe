# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Box position expressed in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, box.data.root_pos_w[:, :3])
    return box_pos_b


def goal_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Goal position expressed in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    goal_pos_e = env.command_manager.get_command(command_name)
    goal_pos_w = goal_pos_e + env.scene.env_origins
    goal_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_w)
    return goal_pos_b


def goal_position_in_box_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Goal position expressed in the current box frame."""
    box: RigidObject = env.scene[box_cfg.name]
    goal_pos_e = env.command_manager.get_command(command_name)
    goal_pos_w = goal_pos_e + env.scene.env_origins
    goal_pos_b, _ = subtract_frame_transforms(box.data.root_pos_w, box.data.root_quat_w, goal_pos_w)
    return goal_pos_b
