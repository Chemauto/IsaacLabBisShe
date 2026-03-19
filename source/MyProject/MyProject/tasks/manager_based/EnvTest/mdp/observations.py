# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from MyProject.tasks.manager_based.EnvTest.scene_layout import BOX_SIZE, HIGH_OBSTACLE_SIZE, LOW_OBSTACLE_SIZE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _runtime_buffer(env: "ManagerBasedEnv", attr_name: str, dim: int) -> torch.Tensor:
    """获取 EnvTest 运行时缓冲；若不存在则自动创建。

    这些缓冲用于把“外部控制器实时给出的指令”写进统一观测：
    - 低层速度命令 `velocity_commands`
    - 推箱子目标点 `goal_command`（xyz + yaw）
    - 推箱子高层上一步动作 `push_actions`
    """

    num_envs = getattr(env, "num_envs", env.cfg.scene.num_envs)
    device = getattr(env, "device", env.cfg.sim.device)
    buffer = getattr(env, attr_name, None)
    expected_shape = (num_envs, dim)

    if buffer is None or buffer.shape != expected_shape or str(buffer.device) != str(device):
        buffer = torch.zeros(expected_shape, dtype=torch.float32, device=device)
        setattr(env, attr_name, buffer)
    return buffer


class StructuredHeightScanHelper:
    """按 EnvTest 规则化障碍构造 17x11 的局部高度图。"""

    def __init__(self, device: torch.device | str):
        self.device = device
        self.offset = 0.5
        self.local_points = self._build_local_points()
        self.asset_specs = {
            "left_low_obstacle": LOW_OBSTACLE_SIZE,
            "right_low_obstacle": LOW_OBSTACLE_SIZE,
            "left_high_obstacle": HIGH_OBSTACLE_SIZE,
            "right_high_obstacle": HIGH_OBSTACLE_SIZE,
            "support_box": BOX_SIZE,
        }

    def _build_local_points(self) -> torch.Tensor:
        """构造与 walk/climb 训练时一致的 17x11 网格采样点。"""

        x = torch.arange(start=-0.8, end=0.8 + 1.0e-9, step=0.1, device=self.device)
        y = torch.arange(start=-0.5, end=0.5 + 1.0e-9, step=0.1, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        local_points = torch.zeros(grid_x.numel(), 3, device=self.device)
        local_points[:, 0] = grid_x.flatten()
        local_points[:, 1] = grid_y.flatten()
        return local_points

    def compute(self, env: "ManagerBasedEnv", robot_name: str = "robot") -> torch.Tensor:
        """根据当前机器人姿态和 EnvTest 障碍布局计算高度扫描。"""

        robot = env.scene[robot_name]
        num_envs = robot.data.root_pos_w.shape[0]
        num_points = self.local_points.shape[0]

        local_points = self.local_points.unsqueeze(0).expand(num_envs, -1, -1)
        quat_yaw = robot.data.root_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
        world_points = math_utils.quat_apply_yaw(quat_yaw, local_points.reshape(-1, 3)).view(num_envs, num_points, 3)
        world_points = world_points + robot.data.root_pos_w.unsqueeze(1)

        max_heights = torch.zeros(num_envs, num_points, device=self.device)

        for asset_name, size in self.asset_specs.items():
            try:
                asset = env.scene[asset_name]
            except KeyError:
                continue

            centers = asset.data.root_pos_w
            half_x = 0.5 * size[0]
            half_y = 0.5 * size[1]
            top_height = centers[:, 2].unsqueeze(1) + 0.5 * size[2]

            inside_x = torch.abs(world_points[..., 0] - centers[:, 0].unsqueeze(1)) <= half_x
            inside_y = torch.abs(world_points[..., 1] - centers[:, 1].unsqueeze(1)) <= half_y
            inside = inside_x & inside_y
            max_heights = torch.where(inside, torch.maximum(max_heights, top_height), max_heights)

        return robot.data.root_pos_w[:, 2].unsqueeze(1) - max_heights - self.offset


def structured_height_scan(env: "ManagerBasedEnv") -> torch.Tensor:
    """返回 walk/climb 低层策略需要的 187 维高度扫描。"""

    helper = getattr(env, "_envtest_height_scan_helper", None)
    device = getattr(env, "device", env.cfg.sim.device)
    if helper is None or str(helper.device) != str(device):
        helper = StructuredHeightScanHelper(device)
        setattr(env, "_envtest_height_scan_helper", helper)
    return helper.compute(env)


def velocity_commands(env: "ManagerBasedEnv") -> torch.Tensor:
    """低层策略速度命令槽位。"""

    return _runtime_buffer(env, "_envtest_velocity_commands", 3)


def push_goal_command(env: "ManagerBasedEnv") -> torch.Tensor:
    """推箱子高层目标点槽位。"""

    return _runtime_buffer(env, "_envtest_push_goal_command", 4)


def push_actions(env: "ManagerBasedEnv") -> torch.Tensor:
    """推箱子高层上一步裁剪后动作槽位。"""

    return _runtime_buffer(env, "_envtest_push_actions", 3)


def box_pose(env: "ManagerBasedEnv") -> torch.Tensor:
    """EnvTest 中 support_box 的位姿；若当前场景无箱子则返回全 0。"""

    try:
        box = env.scene["support_box"]
    except KeyError:
        num_envs = getattr(env, "num_envs", env.cfg.scene.num_envs)
        device = getattr(env, "device", env.cfg.sim.device)
        return torch.zeros((num_envs, 7), dtype=torch.float32, device=device)

    box_pos_e = box.data.root_pos_w[:, :3] - env.scene.env_origins
    box_quat_w = math_utils.quat_unique(box.data.root_quat_w)
    return torch.cat((box_pos_e, box_quat_w), dim=-1)


def robot_position(env: "ManagerBasedEnv") -> torch.Tensor:
    """机器人根节点在环境坐标系下的位置。"""

    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, :3] - env.scene.env_origins


def compute_push_goal_from_scene(env: "ManagerBasedEnv") -> torch.Tensor:
    """按当前障碍位置生成一个“把箱子推到障碍前”的目标位姿。

    该逻辑和 `envtest_model_use_player.py` 使用的推箱子目标保持一致。
    如果当前场景没有箱子或没有可选障碍，则抛出异常，让上层显式处理。
    返回格式为 `[x, y, z, yaw]`，当前默认目标 yaw 为 0。
    """

    scene = env.scene
    env_origins = scene.env_origins

    try:
        box = scene["support_box"]
    except KeyError as err:
        raise RuntimeError("当前 EnvTest 场景中没有 support_box，无法生成推箱子目标点。") from err

    box_pos_e = box.data.root_pos_w[:, :3] - env_origins

    candidates: list[torch.Tensor] = []
    candidate_sizes = (
        ("left_high_obstacle", HIGH_OBSTACLE_SIZE),
        ("right_high_obstacle", HIGH_OBSTACLE_SIZE),
        ("left_low_obstacle", LOW_OBSTACLE_SIZE),
        ("right_low_obstacle", LOW_OBSTACLE_SIZE),
    )
    for asset_name, size in candidate_sizes:
        try:
            asset = scene[asset_name]
        except KeyError:
            continue
        obstacle_pos_e = asset.data.root_pos_w[:, :3] - env_origins
        goal = torch.zeros((env.num_envs, 4), dtype=obstacle_pos_e.dtype, device=obstacle_pos_e.device)
        goal[:, 0] = obstacle_pos_e[:, 0] - 0.5 * size[0] - 0.5 * BOX_SIZE[0] - 0.02
        goal[:, 1] = obstacle_pos_e[:, 1]
        goal[:, 2] = 0.5 * BOX_SIZE[2]
        goal[:, 3] = 0.0
        candidates.append(goal)

    if not candidates:
        raise RuntimeError("当前场景中没有可供 push_box 使用的目标障碍物。")

    candidate_tensor = torch.stack(candidates, dim=1)
    distances = torch.linalg.norm(candidate_tensor[..., :2] - box_pos_e[:, None, :2], dim=-1)
    best_indices = torch.argmin(distances, dim=1)
    env_indices = torch.arange(env.num_envs, device=box_pos_e.device)
    return candidate_tensor[env_indices, best_indices]
