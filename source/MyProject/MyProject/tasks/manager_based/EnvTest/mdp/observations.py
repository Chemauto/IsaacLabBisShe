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


GOAL_FRONT_GAP = 0.03
GOAL_LATERAL_MARGIN = 0.03


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
        # x 方向：让箱子前表面尽量贴近障碍物前表面，但保留很小安全间隙。
        goal[:, 0] = obstacle_pos_e[:, 0] - 0.5 * size[0] - 0.5 * BOX_SIZE[0] - GOAL_FRONT_GAP
        # y 方向：不再强制对齐到障碍物中心，而是在“箱子仍能贴住障碍”的可行区间内，
        # 选取距离当前箱子横向位置最近的中心点，减少无意义侧移。
        lateral_half_range = 0.5 * (size[1] - BOX_SIZE[1]) - GOAL_LATERAL_MARGIN
        if lateral_half_range > 0.0:
            goal_y_min = obstacle_pos_e[:, 1] - lateral_half_range
            goal_y_max = obstacle_pos_e[:, 1] + lateral_half_range
            goal[:, 1] = torch.clamp(box_pos_e[:, 1], min=goal_y_min, max=goal_y_max)
        else:
            goal[:, 1] = obstacle_pos_e[:, 1]
        goal[:, 2] = 0.5 * BOX_SIZE[2]
        goal[:, 3] = 0.0
        candidates.append(goal)

    if not candidates:
        raise RuntimeError("当前场景中没有可供 push_box 使用的目标障碍物。")

    candidate_tensor = torch.stack(candidates, dim=1)
    # 选择“从当前箱子位姿移动量最小”的可行目标，而不是单纯最近障碍中心。
    distances = torch.linalg.norm(candidate_tensor[..., :2] - box_pos_e[:, None, :2], dim=-1)
    best_indices = torch.argmin(distances, dim=1)
    env_indices = torch.arange(env.num_envs, device=box_pos_e.device)
    return candidate_tensor[env_indices, best_indices]
