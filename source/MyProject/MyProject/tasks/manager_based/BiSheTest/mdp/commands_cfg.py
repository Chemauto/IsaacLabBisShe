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

    use_valid_target_patches: bool = True
    """If True, sample targets from terrain flat patches when available."""

    target_patch_name: str = "target"
    """Name of the flat-patch key used for valid target sampling."""

    max_target_height_offset: float | None = 0.6
    """Allowed |target_z - env_origin_z| in meters.

    This filters out deep pit bottoms and high obstacle tops.
    """

    fallback_to_polar_sampling: bool = False
    """Fallback to polar sampling if no valid patch is found for an environment."""

    log_patch_fallback: bool = True
    """Whether to log warnings when patch sampling falls back to polar sampling."""

    max_sampling_attempts: int = 50
    """Maximum attempts to sample a valid target for each environment."""

    patch_match_tolerance: float = 0.5
    """Max XY distance (m) from a sampled polar point to a valid terrain patch."""

    @configclass
    class Ranges(UniformPose2dCommandCfg.Ranges):
        # Kept for compatibility with parent class; x/y are not used directly.
        pos_x: tuple[float, float] = (-5.0, 5.0)
        pos_y: tuple[float, float] = (-5.0, 5.0)
        heading: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = Ranges()
