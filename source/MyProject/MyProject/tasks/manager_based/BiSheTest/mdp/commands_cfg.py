# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.utils import configclass

from .commands import AdvancedPose2dCommand


@configclass
class AdvancedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration of polar-sampled 2D position commands."""

    class_type: type = AdvancedPose2dCommand

    simple_heading: bool = True
    """If True, heading points towards the sampled target."""

    radius_range: tuple[float, float] = (1.0, 5.0)
    """Target radius range in meters."""

    goal_height_offset: float = 0.0
    """Extra world-z offset for command position."""

    @configclass
    class Ranges(UniformPose2dCommandCfg.Ranges):
        # Kept for compatibility with parent class; x/y are not used directly.
        pos_x: tuple[float, float] = (-5.0, 5.0)
        pos_y: tuple[float, float] = (-5.0, 5.0)
        heading: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = Ranges()
