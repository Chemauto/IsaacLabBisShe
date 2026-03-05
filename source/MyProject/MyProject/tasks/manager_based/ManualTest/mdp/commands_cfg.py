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
    """极坐标采样 + 有效目标过滤的 2D 目标命令配置。"""

    class_type: type = AdvancedPose2dCommand

    simple_heading: bool = True

    radius_range: tuple[float, float] = (1.0, 5.0)
    goal_height_offset: float = 0.5

    use_valid_target_patches: bool = True
    target_patch_name: str = "target"
    max_target_height_offset: float | None = 0.6

    fallback_to_polar_sampling: bool = False
    log_patch_fallback: bool = True
    max_sampling_attempts: int = 50
    patch_match_tolerance: float = 0.5

    @configclass
    class Ranges(UniformPose2dCommandCfg.Ranges):
        # 与父类保持兼容：pos_x/pos_y 在该命令中不直接使用。
        pos_x: tuple[float, float] = (-5.0, 5.0)
        pos_y: tuple[float, float] = (-5.0, 5.0)
        heading: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = Ranges()

