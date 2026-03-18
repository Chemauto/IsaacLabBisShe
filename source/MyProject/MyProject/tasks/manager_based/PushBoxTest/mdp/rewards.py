# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from .goal_pose import quat_to_yaw, split_box_goal_command, yaw_error_abs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def processed_action_rate_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "pre_trained_policy_action",
) -> torch.Tensor:
    """惩罚裁剪后高层动作的变化率 / Penalize the rate of change of processed high-level actions.

    这里使用的是缩放并裁剪后的动作，而不是 PPO actor 的原始输出。
    这样可以让正则项约束真实执行到低层 walking policy 的速度命令，避免原始动作数值爆炸时把训练直接拖崩。
    """
    action_term = env.action_manager.get_term(action_name)
    current_action = action_term.processed_actions

    buffer_name = f"_push_box_prev_processed_action_{action_name}"
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, current_action.clone())

    previous_action = getattr(env, buffer_name)

    if hasattr(env, "episode_length_buf"):
        reset_ids = env.episode_length_buf == 0
        previous_action = previous_action.clone()
        previous_action[reset_ids] = current_action[reset_ids]

    action_delta = current_action - previous_action
    setattr(env, buffer_name, current_action.clone())
    return torch.sum(torch.square(action_delta), dim=1)


def _box_goal_delta(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算箱子到目标点的差向量和距离 / Compute delta vector and distance from box to goal.

    这是一个内部辅助函数，用于计算箱子与目标点之间的位置差和距离。
    This is an internal helper function to compute position delta and distance between box and goal.

    逻辑 / Logic:
        1. 获取箱子对象 / Get box object
        2. 从命令管理器获取目标点的环境坐标 / Get goal position in environment frame from command manager
        3. 将目标点转换到世界坐标系 / Convert goal position to world frame
        4. 计算差向量（仅考虑 xy 平面，忽略高度） / Compute delta vector (xy-plane only, ignore height)
        5. 计算欧几里得距离 / Compute Euclidean distance

    注意 / Note:
        - 仅考虑 xy 平面，因为推箱子任务通常在平地上进行 / Only considers xy-plane as push-box is typically on flat ground
        - 返回的 delta 是从箱子指向目标点的向量 / Returned delta points from box to goal

    Args:
        env: 强化学习环境 / RL environment
        command_name: 命令名称 / Command name
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (差向量 delta, 距离 distance) / (delta vector, distance)
            - delta: 形状 (num_envs, 2)，从箱子到目标点的向量 / Shape (num_envs, 2), vector from box to goal
            - distance: 形状 (num_envs,)，距离标量 / Shape (num_envs,), distance scalar
    """
    box: RigidObject = env.scene[box_cfg.name]
    # 从命令管理器获取目标点的环境坐标 / Get goal position in environment frame from command manager
    goal_command = env.command_manager.get_command(command_name)
    goal_pos_e, _ = split_box_goal_command(goal_command)
    # 转换到世界坐标系 / Convert to world frame
    goal_pos_w = goal_pos_e + env.scene.env_origins
    # 计算 xy 平面的差向量（目标点 - 箱子）/ Compute xy-plane delta (goal - box)
    delta = goal_pos_w[:, :2] - box.data.root_pos_w[:, :2]
    # 计算欧几里得距离 / Compute Euclidean distance
    distance = torch.norm(delta, dim=1)
    return delta, distance


def box_goal_yaw_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Compute the wrapped absolute yaw error between the box and its target yaw."""
    box: RigidObject = env.scene[box_cfg.name]
    goal_command = env.command_manager.get_command(command_name)
    _, goal_yaw = split_box_goal_command(goal_command)
    box_yaw = quat_to_yaw(box.data.root_quat_w)
    return yaw_error_abs(box_yaw, goal_yaw)


def box_goal_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """箱子到目标点距离奖励（tanh 核函数）/ Reward moving the box center close to the target point.

    使用 tanh 核函数奖励箱子接近目标点。距离越近，奖励越高。
    Uses tanh kernel function to reward box getting closer to goal. Closer distance = higher reward.

    逻辑 / Logic:
        1. 获取箱子到目标点的距离 / Get box-to-goal distance
        2. 使用 tanh 函数将距离映射到 [0, 1] 范围 / Use tanh to map distance to [0, 1] range
        3. 公式：reward = 1 - tanh(distance / std) / Formula: reward = 1 - tanh(distance / std)

    数学特性 / Mathematical Properties:
        - 当 distance = 0 时，reward = 1（最大奖励）/ When distance = 0, reward = 1 (max reward)
        - 当 distance >> std 时，reward → 0（奖励趋于 0）/ When distance >> std, reward → 0
        - std 控制奖励的衰减速度 / std controls reward decay rate
        - tanh 函数提供平滑的非线性映射 / tanh provides smooth non-linear mapping

    作用 / Purpose:
        - 鼓励机器人将箱子推向目标点 / Encourage robot to push box towards goal
        - 提供密集的引导信号 / Provide dense guidance signal
        - 避免奖励稀疏问题（即使离得很远也有小奖励）/ Avoid sparse reward (small reward even when far)

    Args:
        env: 强化学习环境 / RL environment
        std: tanh 函数的标准差参数，控制奖励衰减速度 / Standard deviation parameter for tanh, controls reward decay rate
        command_name: 命令名称 / Command name
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的奖励张量，范围 [0, 1] /
                     Shape (num_envs,), reward tensor in range [0, 1]
    """
    # 获取箱子到目标点的距离 / Get box-to-goal distance
    _, distance = _box_goal_delta(env, command_name, box_cfg)
    # 使用 tanh 核函数计算奖励：距离越小，奖励越高 / Use tanh kernel: smaller distance → higher reward
    return 1.0 - torch.tanh(distance / std)


def box_goal_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """箱子向目标点进展奖励（增量奖励）/ Reward step-wise progress of the box towards the goal.

    奖励箱子在每一步中向目标点接近的程度，而不是绝对距离。
    Rewards how much the box gets closer to the goal at each step, rather than absolute distance.

    逻辑 / Logic:
        1. 获取当前箱子到目标点的距离 / Get current box-to-goal distance
        2. 从环境缓存中读取上一步的距离 / Get previous distance from environment buffer
        3. 如果是第一次调用，初始化缓存 / If first call, initialize buffer
        4. 检测回合重置（episode_length_buf == 0），重置对应环境的缓存 / Detect episode reset, reset corresponding buffers
        5. 计算进展 = 上一步距离 - 当前距离 / Compute progress = previous_distance - current_distance
        6. 更新缓存为当前距离 / Update buffer to current distance

    关键特点 / Key Features:
        - 增量奖励：只奖励"改善"的部分，而非绝对状态 / Incremental reward: only rewards "improvement", not absolute state
        - 正值：箱子接近目标点 / Positive: box gets closer to goal
        - 负值：箱子远离目标点 / Negative: box moves away from goal
        - 零值：箱子距离不变 / Zero: distance unchanged

    作用 / Purpose:
        - 鼓励持续进展，而非停留在好的位置 / Encourage continuous progress, not staying at good position
        - 避免局部最优：即使已经很接近，仍需要继续改进 / Avoid local optimum: even when close, keep improving
        - 提供细粒度的学习信号 / Provide fine-grained learning signal

    注意 / Note:
        - 使用环境缓存存储历史状态（前一步距离）/ Uses environment buffer to store history (previous distance)
        - 需要正确处理回合重置，避免跨回合的数据污染 / Must handle episode reset properly to avoid cross-episode contamination

    Args:
        env: 强化学习环境 / RL environment
        command_name: 命令名称 / Command name
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的进展奖励张量 /
                     Shape (num_envs,), progress reward tensor
            - 正值表示箱子接近目标点 / Positive: box getting closer
            - 负值表示箱子远离目标点 / Negative: box moving away
    """
    # 获取当前箱子到目标点的距离 / Get current box-to-goal distance
    _, current_distance = _box_goal_delta(env, command_name, box_cfg)

    # 缓存名称 / Buffer name
    buffer_name = "_push_box_prev_goal_distance"

    # 如果缓存不存在，初始化为当前距离 / If buffer doesn't exist, initialize with current distance
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, current_distance.clone())

    # 获取上一步的距离 / Get previous distance
    previous_distance = getattr(env, buffer_name)

    # 处理回合重置：重置的环境需要更新缓存 / Handle episode reset: update buffer for reset environments
    if hasattr(env, "episode_length_buf"):
        # 找出所有重置的环境 / Find all reset environments
        reset_ids = env.episode_length_buf == 0
        # 克隆避免修改原始数据 / Clone to avoid modifying original data
        previous_distance = previous_distance.clone()
        # 将重置环境的上一步距离设为当前距离 / Set previous distance to current for reset environments
        previous_distance[reset_ids] = current_distance[reset_ids]

    # 计算进展：上一步距离 - 当前距离 / Compute progress: previous - current
    # 如果距离减小（接近目标），progress 为正 / If distance decreases (closer), progress is positive
    # 如果距离增加（远离目标），progress 为负 / If distance increases (away), progress is negative
    progress = previous_distance - current_distance

    # 更新缓存为当前距离，供下一步使用 / Update buffer to current distance for next step
    setattr(env, buffer_name, current_distance.clone())

    return progress


def box_goal_yaw_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward the box for aligning its yaw with the target yaw."""
    error_yaw = box_goal_yaw_error(env, command_name, box_cfg)
    return 1.0 - torch.tanh(error_yaw / std)


def box_goal_settled_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.08,
    yaw_threshold: float = 0.2,
    box_speed_threshold: float = 0.05,
    robot_speed_threshold: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """判断箱子是否已到达目标点且稳定（静止）/ Check whether the box is at the goal and both the box and robot have nearly stopped.

    返回一个布尔掩码，表示哪些环境满足"箱子到达目标且系统稳定"的条件。
    Returns a boolean mask indicating which environments satisfy "box at goal AND system settled".

    逻辑 / Logic:
        1. 获取机器人和箱子对象 / Get robot and box objects
        2. 计算箱子到目标点的距离 / Compute box-to-goal distance
        3. 计算箱子的线速度（xy 平面）/ Compute box linear velocity (xy-plane)
        4. 计算机器人的线速度（xy 平面）/ Compute robot linear velocity (xy-plane)
        5. 检查三个条件是否同时满足 / Check if all three conditions are met:
           - 距离 < distance_threshold（箱子足够接近目标）/ Distance < threshold (box close enough to goal)
           - 箱子速度 < box_speed_threshold（箱子几乎静止）/ Box speed < threshold (box nearly stationary)
           - 机器人速度 < robot_speed_threshold（机器人几乎静止）/ Robot speed < threshold (robot nearly stationary)

    为什么需要检查速度？/ Why check speeds?
        - 仅检查距离可能误判：箱子"经过"目标点但未停止 / Only checking distance may misclassify: box "passes" goal but doesn't stop
        - 需要确保任务真正完成，而不是暂时经过 / Need to ensure task is actually done, not just passing through
        - 机器人也需要停止，避免"推完就跑"的行为 / Robot also needs to stop, avoid "push and run" behavior

    作用 / Purpose:
        - 用于成功判定 / Used for success determination
        - 可作为回合终止条件 / Can be used as episode termination condition
        - 可用于触发额外奖励 / Can be used to trigger bonus rewards

    Args:
        env: 强化学习环境 / RL environment
        command_name: 命令名称 / Command name
        distance_threshold: 距离阈值（米）/ Distance threshold in meters, default 0.08 (8cm)
        box_speed_threshold: 箱子速度阈值（米/秒）/ Box speed threshold in m/s, default 0.05
        robot_speed_threshold: 机器人速度阈值（米/秒）/ Robot speed threshold in m/s, default 0.08
        robot_cfg: 机器人场景实体配置 / Robot scene entity configuration
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的布尔掩码 /
                     Shape (num_envs,), boolean mask
            - True: 箱子到达目标且系统已稳定 / True: box at goal and system settled
            - False: 条件不满足 / False: conditions not met
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    # 获取箱子到目标点的距离 / Get box-to-goal distance
    _, distance = _box_goal_delta(env, command_name, box_cfg)
    error_yaw = box_goal_yaw_error(env, command_name, box_cfg)

    # 计算箱子速度（xy 平面的线速度）/ Compute box speed (xy-plane linear velocity)
    box_speed = torch.norm(box.data.root_lin_vel_w[:, :2], dim=1)

    # 计算机器人速度（机器人坐标系的 xy 平面线速度）/ Compute robot speed (robot frame xy-plane linear velocity)
    robot_speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)

    # 三个条件同时满足才返回 True / All three conditions must be True
    return (
        (distance < distance_threshold)
        & (error_yaw < yaw_threshold)
        & (box_speed < box_speed_threshold)
        & (robot_speed < robot_speed_threshold)
    )


def robot_box_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """机器人与箱子距离奖励（鼓励保持接触）/ Reward the robot for staying close enough to interact with the box.

    使用 tanh 核函数奖励机器人接近箱子。距离越近，奖励越高。
    Uses tanh kernel function to reward robot getting closer to box. Closer distance = higher reward.

    逻辑 / Logic:
        1. 获取机器人和箱子对象 / Get robot and box objects
        2. 计算机器人到箱子的 xy 平面距离 / Compute robot-to-box distance in xy-plane
        3. 使用 tanh 函数将距离映射到 [0, 1] 范围 / Use tanh to map distance to [0, 1] range
        4. 公式：reward = 1 - tanh(distance / std) / Formula: reward = 1 - tanh(distance / std)

    为什么需要这个奖励？/ Why need this reward?
        - 机器人必须靠近箱子才能推动它 / Robot must be close to box to push it
        - 防止机器人"远程操控"或放弃任务 / Prevent robot from "remote control" or giving up
        - 确保机器人保持与箱子的交互 / Ensure robot maintains interaction with box

    作用 / Purpose:
        - 鼓励机器人接近并保持与箱子的接触 / Encourage robot to approach and maintain contact with box
        - 作为辅助奖励，避免机器人远离箱子 / Auxiliary reward to prevent robot from staying far from box
        - 与其他奖励配合，平衡接近箱子与推向目标 / Works with other rewards to balance approaching vs pushing

    Args:
        env: 强化学习环境 / RL environment
        std: tanh 函数的标准差参数，控制奖励衰减速度 / Standard deviation parameter for tanh, controls reward decay rate
        robot_cfg: 机器人场景实体配置 / Robot scene entity configuration
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的奖励张量，范围 [0, 1] /
                     Shape (num_envs,), reward tensor in range [0, 1]
            - 1.0: 机器人与箱子重合（理论最大值）/ 1.0: robot overlaps box (theoretical max)
            - 0.0: 距离远大于 std / 0.0: distance much larger than std
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    # 计算机器人到箱子的 xy 平面距离 / Compute robot-to-box distance in xy-plane
    distance = torch.norm(box.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)

    # 使用 tanh 核函数：距离越近，奖励越高 / Use tanh kernel: closer distance → higher reward
    return 1.0 - torch.tanh(distance / std)


def head_point_in_box_penalty(
    env: ManagerBasedRLEnv,
    head_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    footprint_margin: float = 0.02,
    top_surface_margin: float = 0.0,
    head_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Head_.*"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Penalize the robot when a head proxy point projects inside the box top footprint.

    几何规则 / Geometry:
        1. 在机器人的头部刚体上取一个点（默认取 `Head_.*` 刚体原点） /
           Take a point on the robot head rigid body (defaults to the `Head_.*` body origin)
        2. 将该点转换到世界坐标，再投到箱子局部坐标系 / Transform the point to world, then into the box frame
        3. 只有当该点的 XY 投影落入箱子的矩形 footprint，且 Z 严格高于箱子顶面，才返回 1 /
           Return 1 only when the XY projection falls inside the box footprint and Z is strictly above the box top surface

    Note:
        - 如果后续发现刚体原点不在你想要的位置，可以再加 `head_local_offset` 微调 / `head_local_offset` can be used later if the head body origin is not ideal
        - 这正对应“头不要到箱子上方”的约束 / This directly matches the requirement that the head must not move above the box
    """
    robot: RigidObject = env.scene[head_body_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    head_body_id = head_body_cfg.body_ids[0]
    head_body_pos_w = robot.data.body_pos_w[:, head_body_id, :]
    head_body_quat_w = robot.data.body_quat_w[:, head_body_id, :]
    head_offset_b = torch.tensor(head_local_offset, device=env.device, dtype=head_body_pos_w.dtype).unsqueeze(0)
    head_offset_b = head_offset_b.expand(env.num_envs, -1)
    head_point_w = head_body_pos_w + quat_apply(head_body_quat_w, head_offset_b)

    head_point_b, _ = subtract_frame_transforms(
        box.data.root_pos_w[:, :3],
        box.data.root_quat_w,
        head_point_w,
    )

    box_scene_cfg = getattr(env.cfg.scene, box_cfg.name)
    box_size = box_scene_cfg.spawn.size
    half_size_x = 0.5 * box_size[0] + footprint_margin
    half_size_y = 0.5 * box_size[1] + footprint_margin
    top_surface_z = 0.5 * box_size[2] + top_surface_margin
    inside_x = torch.abs(head_point_b[:, 0]) <= half_size_x
    inside_y = torch.abs(head_point_b[:, 1]) <= half_size_y
    above_top_surface = head_point_b[:, 2] > top_surface_z
    return (inside_x & inside_y & above_top_surface).float()


def box_goal_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.08,
    yaw_threshold: float = 0.2,
    box_speed_threshold: float = 0.05,
    robot_speed_threshold: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """任务成功完成奖励（额外奖励）/ Bonus when the box reaches the placement zone and the system has settled.

    当箱子成功到达目标点且系统稳定时，给予额外的完成奖励。
    Gives an additional completion bonus when the box successfully reaches the goal and system settles.

    逻辑 / Logic:
        1. 调用 box_goal_settled_mask 获取成功掩码 / Call box_goal_settled_mask to get success mask
        2. 将布尔掩码转换为浮点数 / Convert boolean mask to float
        3. 成功的环境返回 1.0，其他返回 0.0 / Successful environments return 1.0, others return 0.0

    为什么需要额外奖励？/ Why need bonus?
        - 稀疏奖励：任务完成是一个重要里程碑 / Sparse reward: task completion is an important milestone
        - 引导学习：明确告诉策略"这是成功" / Guide learning: clearly tell policy "this is success"
        - 加速收敛：提供强化的完成信号 / Accelerate convergence: provide strong completion signal
        - 解决奖励工程：即使其他奖励设计不佳，成功奖励仍能引导策略 / Reward engineering: even if other rewards are poorly designed, success bonus guides policy

    作用 / Purpose:
        - 作为任务完成的主要奖励信号 / Primary reward signal for task completion
        - 通常设置较大的权重（例如 10.0~100.0）/ Typically given large weight (e.g., 10.0~100.0)
        - 与其他持续奖励配合，平衡"过程"与"结果"/ Works with ongoing rewards to balance "process" vs "outcome"

    注意 / Note:
        - 这是一个"二元"奖励：要么成功要么失败 / This is a "binary" reward: either success or failure
        - 仅在箱子到达目标且稳定时触发 / Only triggered when box reaches goal and settles
        - 需要与其他密集奖励配合使用，否则训练初期很难学习 / Must be used with other dense rewards, otherwise hard to learn early in training

    Args:
        env: 强化学习环境 / RL environment
        command_name: 命令名称 / Command name
        distance_threshold: 距离阈值（米）/ Distance threshold in meters, default 0.08 (8cm)
        box_speed_threshold: 箱子速度阈值（米/秒）/ Box speed threshold in m/s, default 0.05
        robot_speed_threshold: 机器人速度阈值（米/秒）/ Robot speed threshold in m/s, default 0.08
        robot_cfg: 机器人场景实体配置 / Robot scene entity configuration
        box_cfg: 箱子场景实体配置 / Box scene entity configuration

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的奖励张量 /
                     Shape (num_envs,), reward tensor
            - 1.0: 箱子成功到达目标且系统稳定 / 1.0: box successfully reached goal and system settled
            - 0.0: 任务未完成 / 0.0: task not completed
    """
    # 获取成功掩码并转换为浮点数 / Get success mask and convert to float
    return box_goal_settled_mask(
        env,
        command_name=command_name,
        distance_threshold=distance_threshold,
        yaw_threshold=yaw_threshold,
        box_speed_threshold=box_speed_threshold,
        robot_speed_threshold=robot_speed_threshold,
        robot_cfg=robot_cfg,
        box_cfg=box_cfg,
    ).float()


def orientation_l2(
    env: ManagerBasedRLEnv,
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机器人姿态奖励（保持直立）/ Reward the robot for keeping the body aligned with the desired gravity direction.

    奖励机器人保持期望的姿态方向，通常用于鼓励四足机器人保持直立。
    Rewards robot for maintaining desired orientation direction, typically used to encourage quadruped to stay upright.

    逻辑 / Logic:
        1. 获取资产对象（通常是机器人）/ Get asset object (typically robot)
        2. 将期望重力向量转换为张量 / Convert desired gravity vector to tensor
        3. 计算当前投影重力与期望重力的余弦相似度 / Compute cosine similarity between current projected gravity and desired
        4. 归一化到 [0, 1] 范围：normalized = 0.5 * cos_dist + 0.5 / Normalize to [0, 1]: normalized = 0.5 * cos_dist + 0.5
        5. 应用平方函数增强对比度 / Apply square function to enhance contrast

    数学原理 / Mathematical Principle:
        - projected_gravity_b: 重力在机器人坐标系中的投影 / Gravity projection in robot frame
        - 对于直立机器人，projected_gravity_b ≈ [0, 0, -1] / For upright robot, projected_gravity_b ≈ [0, 0, -1]
        - cos_dist = dot(projected_gravity_b, desired_gravity) / Cosine distance
        - 当完全对齐时，cos_dist = 1，normalized = 1，reward = 1 / When aligned: cos_dist=1, normalized=1, reward=1
        - 当完全相反时，cos_dist = -1，normalized = 0，reward = 0 / When opposite: cos_dist=-1, normalized=0, reward=0

    为什么使用平方？/ Why square?
        - 增强对齐与不对齐的对比 / Enhance contrast between aligned and misaligned
        - 对接近正确的姿态给予更高奖励 / Give higher reward for nearly correct orientation
        - 惩罚偏离更严厉 / Penalize deviations more severely

    作用 / Purpose:
        - 鼓励四足机器人保持直立姿态 / Encourage quadruped to maintain upright posture
        - 防止翻倒 / Prevent falling over
        - 这是标准的运动控制奖励 / This is a standard locomotion reward

    参考来源 / Reference:
        遵循 Unitree 运动控制奖励的设计思想。
        Follows the same idea as reference Unitree locomotion rewards.
        对于平地上的 Go2 机器人，期望重力为 [0, 0, -1]。
        Uses desired gravity [0, 0, -1] for an upright Go2 on flat ground.

    Args:
        env: 强化学习环境 / RL environment
        desired_gravity: 期望的重力方向向量 / Desired gravity direction vector, typically [0, 0, -1] for upright
        asset_cfg: 资产场景实体配置（通常是机器人）/ Asset scene entity configuration (typically robot)

    Returns:
        torch.Tensor: 形状为 (num_envs,) 的奖励张量，范围 [0, 1] /
                     Shape (num_envs,), reward tensor in range [0, 1]
            - 1.0: 姿态完全对齐（直立）/ 1.0: orientation perfectly aligned (upright)
            - 0.0: 姿态完全相反（倒立）/ 0.0: orientation completely opposite (inverted)
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 将期望重力向量转换为张量 / Convert desired gravity vector to tensor
    desired_gravity_tensor = torch.tensor(
        desired_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype
    )

    # 计算余弦相似度（点积）/ Compute cosine similarity (dot product)
    # projected_gravity_b 是重力在机器人坐标系中的投影 / projected_gravity_b is gravity projection in robot frame
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity_tensor, dim=-1)

    # 归一化到 [0, 1] 范围 / Normalize to [0, 1] range
    # cos_dist ∈ [-1, 1] → normalized ∈ [0, 1] / cos_dist in [-1, 1] → normalized in [0, 1]
    normalized = 0.5 * cos_dist + 0.5

    # 应用平方函数增强对比度 / Apply square to enhance contrast
    # 接近 1 的值会更大，接近 0 的值会更小 / Values near 1 become larger, values near 0 become smaller
    return torch.square(normalized)
