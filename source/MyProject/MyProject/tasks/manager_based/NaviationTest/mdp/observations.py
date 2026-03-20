# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def asset_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Return the asset root position expressed in the robot root frame."""

    robot: RigidObject = env.scene[robot_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    asset_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, asset.data.root_pos_w[:, :3])
    return asset_pos_b
