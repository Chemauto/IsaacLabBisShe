# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_distance: float = 3.5,
    min_height: float = 0.02,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Terminate if the box is pushed far away or falls through the ground."""
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_e = box.data.root_pos_w - env.scene.env_origins
    xy_distance = torch.norm(box_pos_e[:, :2], dim=1)
    return (xy_distance > max_distance) | (box.data.root_pos_w[:, 2] < min_height)
