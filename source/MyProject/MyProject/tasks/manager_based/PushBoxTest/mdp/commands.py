# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BoxGoalCommand(CommandTerm):
    """Sample a target position for the box in the local environment frame."""

    cfg: "BoxGoalCommandCfg"

    def __init__(self, cfg: "BoxGoalCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.box: RigidObject = env.scene[cfg.asset_name]
        self.pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_w = torch.zeros_like(self.pos_command_e)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        # 诊断指标：用于区分“整体都差一点”还是“只有大侧向目标很难”。
        self.metrics["error_pos_p50"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pos_p90"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pos_center"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pos_side"] = torch.zeros(self.num_envs, device=self.device)
        self.initial_error_pos = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pos_command_e

    def _update_metrics(self):
        error_pos = torch.norm(self.pos_command_w[:, :2] - self.box.data.root_pos_w[:, :2], dim=1)
        goal_abs_y = torch.abs(self.pos_command_e[:, 1])

        self.metrics["error_pos"] = error_pos

        # 分位数直接反映误差分布，避免只看均值掩盖难例。
        error_pos_p50 = torch.quantile(error_pos, 0.50)
        error_pos_p90 = torch.quantile(error_pos, 0.90)
        self.metrics["error_pos_p50"].fill_(error_pos_p50)
        self.metrics["error_pos_p90"].fill_(error_pos_p90)

        # 按目标点侧向偏移划分样本：
        # center 表示接近正前方的目标；side 表示大侧向目标。
        center_mask = goal_abs_y < 0.2
        side_mask = goal_abs_y > 0.6

        center_error = error_pos[center_mask].mean() if torch.any(center_mask) else torch.tensor(0.0, device=self.device)
        side_error = error_pos[side_mask].mean() if torch.any(side_mask) else torch.tensor(0.0, device=self.device)
        self.metrics["error_pos_center"].fill_(center_error)
        self.metrics["error_pos_side"].fill_(side_error)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_e[env_ids] = 0.0
        self.pos_command_e[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_e[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_e[env_ids, 2] = self.box.data.default_root_state[env_ids, 2]
        self.pos_command_w[env_ids] = self.pos_command_e[env_ids] + self._env.scene.env_origins[env_ids]
        self.initial_error_pos[env_ids] = torch.norm(
            self.pos_command_w[env_ids, :2] - self.box.data.root_pos_w[env_ids, :2], dim=1
        )

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
        marker_quat[:, 0] = 1.0
        self.goal_pose_visualizer.visualize(translations=self.pos_command_w, orientations=marker_quat)


@configclass
class BoxGoalCommandCfg(CommandTermCfg):
    """Configuration for the box target command."""

    class_type: type = BoxGoalCommand

    asset_name: str = MISSING

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/box_goal"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
