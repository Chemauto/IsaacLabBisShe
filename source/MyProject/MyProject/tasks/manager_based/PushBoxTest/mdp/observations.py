# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_pose(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """箱子位姿 / Box pose in the environment frame as position (xyz) and unique quaternion (wxyz).

    获取箱子在环境坐标系中的位姿，包括位置和姿态四元数。
    Gets the box pose in the environment frame, including position and orientation quaternion.

    逻辑 / Logic:
        1. 从场景中获取箱子对象 / Get box object from scene
        2. 将世界坐标位置转换为环境坐标（减去环境原点）/ Convert world position to environment frame (subtract env origins)
        3. 对四元数进行唯一化处理（消除四元数双值性）/ Apply unique quaternion transformation (eliminate quaternion double cover)
        4. 拼接位置和四元数返回 / Concatenate position and quaternion

    作用 / Purpose:
        - 提供箱子的完整位姿信息，让策略知道箱子的位置和朝向 / Provide complete box pose for policy to know box position and orientation
        - 用于判断箱子的姿态是否适合推动 / Used to determine if box orientation is suitable for pushing

    Args:
        env: 强化学习环境 / RL environment
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs, 7) 的张量，包含 [位置(x,y,z), 四元数(w,x,y,z)] /
                     Shape (num_envs, 7), containing [position(x,y,z), quaternion(w,x,y,z)]
    """
    box: RigidObject = env.scene[box_cfg.name]
    # 转换为环境坐标系（减去各环境的原点偏移）/ Convert to environment frame (subtract environment origin offsets)
    box_pos_e = box.data.root_pos_w[:, :3] - env.scene.env_origins
    # 四元数唯一化处理：确保四元数的实部为正，消除 q 和 -q 的等价性 / Unique quaternion: ensure real part is positive
    box_quat_w = math_utils.quat_unique(box.data.root_quat_w)
    # 拼接位置和姿态 / Concatenate position and orientation
    return torch.cat([box_pos_e, box_quat_w], dim=-1)


def robot_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机器人根节点位置 / Robot root position in the environment frame.

    获取机器人在环境坐标系中的根节点位置（不包含姿态信息）。
    Gets the robot's root position in the environment frame (without orientation).

    逻辑 / Logic:
        1. 从场景中获取机器人对象 / Get robot object from scene
        2. 将世界坐标位置转换为环境坐标（减去环境原点）/ Convert world position to environment frame (subtract env origins)

    作用 / Purpose:
        - 提供机器人在环境中的绝对位置 / Provide robot's absolute position in the environment
        - 与箱子位置配合，计算机器人与箱子的相对位置关系 / Work with box position to compute relative robot-box positioning
        - 用于判断机器人是否接近箱子或目标点 / Used to determine if robot is approaching box or goal

    Args:
        env: 强化学习环境 / RL environment
        robot_cfg: 机器人场景实体配置 / Robot scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs, 3) 的张量，包含位置 [x, y, z] /
                     Shape (num_envs, 3), containing position [x, y, z]
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    # 转换为环境坐标系（减去各环境的原点偏移）/ Convert to environment frame (subtract environment origin offsets)
    return robot.data.root_pos_w[:, :3] - env.scene.env_origins


def box_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """机器人坐标系下的箱子位置 / Box position expressed in the robot root frame.

    获取箱子在机器人局部坐标系中的位置（以机器人的视角）。
    Gets the box position in the robot's local coordinate frame (from robot's perspective).

    逻辑 / Logic:
        1. 从场景中获取机器人和箱子对象 / Get robot and box objects from scene
        2. 使用坐标变换函数，将箱子的世界坐标转换到机器人坐标系 / Use frame transform to convert box world position to robot frame
        3. 变换考虑了机器人的位置和姿态（前进方向为 x 轴） / Transform accounts for robot's position and orientation (forward direction is x-axis)

    作用 / Purpose:
        - 让机器人以"自我为中心"的视角感知箱子位置 / Provide egocentric perception of box position
        - 便于策略理解"箱子在我的前方/后方/左侧/右侧" / Helps policy understand "box is in front/behind/left/right of me"
        - 在推箱子任务中至关重要，决定从哪个方向接近箱子 / Critical for push-box task: determines approach direction

    注意 / Note:
        - 在机器人坐标系中，x 轴通常指向机器人前进方向 / In robot frame, x-axis typically points to robot's forward direction
        - y 轴指向左侧，z 轴向上 / y-axis points left, z-axis points up

    Args:
        env: 强化学习环境 / RL environment
        robot_cfg: 机器人场景实体配置 / Robot scene entity configuration
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs, 3) 的张量，包含相对位置 [x, y, z] /
                     Shape (num_envs, 3), containing relative position [x, y, z]
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    # 将箱子位置从世界坐标系转换到机器人局部坐标系 / Transform box position from world frame to robot's local frame
    box_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, box.data.root_pos_w[:, :3])
    return box_pos_b


def goal_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机器人坐标系下的目标点位置 / Goal position expressed in the robot root frame.

    获取目标点在机器人局部坐标系中的位置（以机器人的视角）。
    Gets the goal position in the robot's local coordinate frame (from robot's perspective).

    逻辑 / Logic:
        1. 从场景中获取机器人对象 / Get robot object from scene
        2. 从命令管理器获取目标点的环境坐标 / Get goal position in environment frame from command manager
        3. 将目标点环境坐标转换为世界坐标（加上环境原点）/ Convert goal position from environment to world frame (add env origins)
        4. 使用坐标变换函数，将目标点的世界坐标转换到机器人坐标系 / Transform goal world position to robot's local frame

    作用 / Purpose:
        - 让机器人以"自我为中心"的视角感知目标位置 / Provide egocentric perception of goal position
        - 便于策略理解"目标在我的前方/后方/左侧/右侧" / Helps policy understand "goal is in front/behind/left/right of me"
        - 在导航任务中至关重要，决定运动方向 / Critical for navigation: determines movement direction

    注意 / Note:
        - 与 box_position_in_robot_frame 配合使用，可以判断"是否应该先去推箱子再去目标" /
          Used with box_position_in_robot_frame to determine "should I push box first or go to goal"
        - 目标点通常是箱子的最终目的地 / Goal position is typically the final destination for the box

    Args:
        env: 强化学习环境 / RL environment
        command_name: 命令名称（用于从命令管理器获取目标点） / Command name (to get goal from command manager)
        robot_cfg: 机器人场景实体配置 / Robot scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs, 3) 的张量，包含相对位置 [x, y, z] /
                     Shape (num_envs, 3), containing relative position [x, y, z]
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    # 从命令管理器获取目标点的环境坐标 / Get goal position in environment frame from command manager
    goal_pos_e = env.command_manager.get_command(command_name)
    # 转换为世界坐标系（加上各环境的原点偏移）/ Convert to world frame (add environment origin offsets)
    goal_pos_w = goal_pos_e + env.scene.env_origins
    # 将目标点位置从世界坐标系转换到机器人局部坐标系 / Transform goal position from world frame to robot's local frame
    goal_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_w)
    return goal_pos_b


def goal_position_in_box_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """箱子坐标系下的目标点位置 / Goal position expressed in the current box frame.

    获取目标点在箱子局部坐标系中的位置（以箱子的视角）。
    Gets the goal position in the box's local coordinate frame (from box's perspective).

    逻辑 / Logic:
        1. 从场景中获取箱子对象 / Get box object from scene
        2. 从命令管理器获取目标点的环境坐标 / Get goal position in environment frame from command manager
        3. 将目标点环境坐标转换为世界坐标（加上环境原点）/ Convert goal position from environment to world frame (add env origins)
        4. 使用坐标变换函数，将目标点的世界坐标转换到箱子坐标系 / Transform goal world position to box's local frame

    作用 / Purpose:
        - 让策略知道目标点相对于箱子的位置和方向 / Provide policy with goal position relative to box
        - **这是推箱子任务最关键的观测**：决定应该从哪个方向推箱子 /
          **This is the MOST critical observation for push-box task**: determines which direction to push the box
        - 例如：如果目标在箱子的"前方"，策略应该推箱子的"后方" /
          Example: if goal is in "front" of box, strategy should push from "behind" the box
        - 结合箱子的当前姿态，规划最优的推动路径 / Combined with box orientation, plan optimal pushing path

    注意 / Note:
        - 这是实现"将箱子推到目标点"任务的核心观测 / This is the core observation for "push box to goal" task
        - 与 box_pose 配合使用，可以计算推动向量 / Used with box_pose to compute pushing vector
        - 箱子的坐标系方向取决于其当前朝向（可能已经旋转）/ Box frame direction depends on its current orientation (may be rotated)

    Args:
        env: 强化学习环境 / RL environment
        command_name: 命令名称（用于从命令管理器获取目标点） / Command name (to get goal from command manager)
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs, 3) 的张量，包含相对位置 [x, y, z] /
                     Shape (num_envs, 3), containing relative position [x, y, z]
    """
    box: RigidObject = env.scene[box_cfg.name]
    # 从命令管理器获取目标点的环境坐标 / Get goal position in environment frame from command manager
    goal_pos_e = env.command_manager.get_command(command_name)
    # 转换为世界坐标系（加上各环境的原点偏移）/ Convert to world frame (add environment origin offsets)
    goal_pos_w = goal_pos_e + env.scene.env_origins
    # 将目标点位置从世界坐标系转换到箱子局部坐标系 / Transform goal position from world frame to box's local frame
    goal_pos_b, _ = subtract_frame_transforms(box.data.root_pos_w, box.data.root_quat_w, goal_pos_w)
    return goal_pos_b
