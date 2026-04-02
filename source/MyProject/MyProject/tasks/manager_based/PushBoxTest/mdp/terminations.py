# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .rewards import box_goal_settled_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_distance: float = 3.5,
    min_height: float = 0.02,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """箱子越界终止条件 / Terminate if the box is pushed far away or falls through the ground.

    当箱子被推得太远或掉落到地面以下时，终止回合。
    Terminates the episode when the box is pushed too far away or falls below the ground.

    逻辑 / Logic:
        1. 获取箱子对象 / Get box object
        2. 将箱子位置转换到环境坐标系 / Convert box position to environment frame
        3. 计算 xy 平面的距离（从原点）/ Compute xy-plane distance from origin
        4. 检查两个终止条件 / Check two termination conditions:
           - xy_distance > max_distance：箱子被推得太远 / Box pushed too far
           - z < min_height：箱子掉落或翻转 / Box fell or flipped over

    为什么要终止？/ Why terminate?
        - **防止无效探索**：箱子被推得太远，任务已经不可能完成 / Prevent invalid exploration: box too far, task impossible
        - **物理异常**：箱子穿透地面或翻转，说明物理状态异常 / Physics anomaly: box penetrated ground or flipped
        - **节省计算资源**：及早终止无望的回合，提高训练效率 / Save computation: early termination of hopeless episodes improves training efficiency

    作用 / Purpose:
        - 作为安全机制，防止策略学习"逃避"行为 / Safety mechanism to prevent strategy from learning "escape" behaviors
        - 箱子被推得太远应该被惩罚而非继续 / Box pushed too far should be penalized, not continued
        - 避免物理引擎的不稳定情况 / Avoid physics engine instability

    典型参数 / Typical Parameters:
        - max_distance: 3.5 米（根据场地大小调整）/ 3.5 meters (adjust based on arena size)
        - min_height: 0.02 米（2cm，允许轻微地面穿透）/ 0.02 meters (2cm, allows minor ground penetration)

    Args:
        env: 强化学习环境 / RL environment
        max_distance: 最大允许距离（米）/ Maximum allowed distance in meters, default 3.5
        min_height: 最小允许高度（米）/ Minimum allowed height in meters, default 0.02 (2cm)
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的布尔掩码 /
                     Shape (num_envs,), boolean mask
            - True: 箱子越界，应终止回合 / True: box out of bounds, should terminate episode
            - False: 箱子在有效范围内 / False: box within valid range
    """
    box: RigidObject = env.scene[box_cfg.name]
    # 将箱子位置转换到环境坐标系（减去环境原点）/ Convert box position to environment frame (subtract env origins)
    box_pos_e = box.data.root_pos_w - env.scene.env_origins
    # 计算 xy 平面的欧几里得距离（从环境原点）/ Compute xy-plane Euclidean distance from environment origin
    xy_distance = torch.norm(box_pos_e[:, :2], dim=1)
    # 两个条件满足任一即终止：距离太远 或 高度太低 / Terminate if either condition: too far OR too low
    return (xy_distance > max_distance) | (box.data.root_pos_w[:, 2] < min_height)


def goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.08,
    yaw_threshold: float = 0.2,
    box_speed_threshold: float = 0.50,
    robot_speed_threshold: float = 0.50,
    settle_steps: int = 12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """任务成功完成终止条件 / Terminate once the box reaches the goal and both the robot and box remain nearly still for a few steps.

    当箱子到达目标点且系统保持稳定一段时间后，终止回合。
    Terminates the episode when the box reaches the goal and both robot and box remain settled for a few consecutive steps.

    逻辑 / Logic:
        1. 调用 box_goal_settled_mask 检查当前帧是否满足稳定条件 / Call box_goal_settled_mask to check if current frame meets settle conditions
        2. 使用计数器追踪连续满足条件的步数 / Use counter to track consecutive settled steps
        3. 处理回合重置：重置计数器 / Handle episode reset: reset counter
        4. 更新计数器：满足则+1，不满足则清零 / Update counter: +1 if settled, reset to 0 if not
        5. 判断计数器是否达到阈值 / Check if counter reaches threshold

    为什么需要连续稳定？/ Why need consecutive settling?
        - **防止误判**：箱子可能"经过"目标点但未停止 / Prevent false positive: box may "pass through" goal but not stop
        - **确保完成**：需要确认任务真正完成，而非暂时状态 / Ensure completion: confirm task is actually done, not transient state
        - **避免过早终止**：给策略时间调整到最终姿态 / Avoid early termination: give policy time to adjust to final pose
        - **物理稳定**：确保物理系统已经静止 / Physical stability: ensure physics system has settled

    作用 / Purpose:
        - 作为任务成功的主要终止条件 / Primary termination condition for task success
        - 给予策略明确的成功信号 / Provide clear success signal to policy
        - 允许在成功后立即终止，节省计算 / Allow immediate termination after success, saving computation

    典型参数 / Typical Parameters:
        - settle_steps: 12 步（约 0.24 秒 @ 50 Hz）/ 12 steps (~0.24s @ 50Hz)
        - 这个值需要平衡：太短可能误判，太长浪费时间 / Must balance: too short may misclassify, too long wastes time

    Args:
        env: 强化学习环境 / RL environment
        command_name: 命令名称 / Command name
        distance_threshold: 距离阈值（米）/ Distance threshold in meters, default 0.08 (8cm)
        box_speed_threshold: 箱子速度阈值（米/秒）/ Box speed threshold in m/s, default 0.05
        robot_speed_threshold: 机器人速度阈值（米/秒）/ Robot speed threshold in m/s, default 0.08
        settle_steps: 需要连续保持稳定的步数 / Number of consecutive steps to remain settled, default 12
        robot_cfg: 机器人场景实体配置 / Robot scene entity configuration
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的布尔掩码 /
                     Shape (num_envs,), boolean mask
            - True: 任务成功完成，应终止回合 / True: task successfully completed, should terminate episode
            - False: 任务未完成 / False: task not completed
    """
    # 获取当前帧的稳定掩码 / Get settled mask for current frame
    settled = box_goal_settled_mask(
        env,
        command_name=command_name,
        distance_threshold=distance_threshold,
        yaw_threshold=yaw_threshold,
        box_speed_threshold=box_speed_threshold,
        robot_speed_threshold=robot_speed_threshold,
        robot_cfg=robot_cfg,
        box_cfg=box_cfg,
    )

    # 计数器名称 / Counter name
    counter_name = "_push_box_goal_reached_steps"

    # 如果计数器不存在，初始化为零 / If counter doesn't exist, initialize to zeros
    if not hasattr(env, counter_name):
        setattr(env, counter_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))

    # 获取当前计数器值 / Get current counter value
    settled_steps = getattr(env, counter_name)

    # 处理回合重置：重置的环境需要清零计数器 / Handle episode reset: reset counter for reset environments
    if hasattr(env, "episode_length_buf"):
        # 克隆避免修改原始数据 / Clone to avoid modifying original data
        settled_steps = settled_steps.clone()
        # 将重置环境的计数器清零 / Reset counter to 0 for reset environments
        settled_steps[env.episode_length_buf == 0] = 0

    # 更新计数器：
    # - 如果当前稳定：计数器 +1 / If currently settled: counter + 1
    # - 如果当前不稳定：计数器清零 / If currently not settled: reset counter to 0
    settled_steps = torch.where(settled, settled_steps + 1, torch.zeros_like(settled_steps))

    # 保存更新后的计数器 / Save updated counter
    setattr(env, counter_name, settled_steps)

    # 判断计数器是否达到阈值 / Check if counter reaches threshold
    # True 表示连续 settle_steps 帧都满足稳定条件 / True means settled for settle_steps consecutive frames
    return settled_steps >= settle_steps
