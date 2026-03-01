# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .bishe_rough_env_cfg import BiSheGo2RoughEnvCfg


@configclass
class BiSheGo2FlatEnvCfg(BiSheGo2RoughEnvCfg):
    """Advanced-skills task on flat terrain for easier warm-up."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class BiSheGo2FlatEnvCfg_Play(BiSheGo2FlatEnvCfg):
    """Play config for flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
