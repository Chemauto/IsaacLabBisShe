# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def lateral_deviation_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral drift once the robot leaves the central corridor.

    The penalty is computed in the local env frame, so terrain tiling does not affect it.
    """
    asset = env.scene[asset_cfg.name]
    root_pos_e = asset.data.root_pos_w - env.scene.env_origins
    lateral_error = torch.abs(root_pos_e[:, 1]) - threshold
    return torch.square(torch.clamp(lateral_error, min=0.0))


def velocity_toward_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward velocity aligned with target direction."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_dir = command[:, :2]

    vel_b = asset.data.root_lin_vel_b[:, :2]
    target_norm = torch.norm(target_dir, dim=1, keepdim=True)
    vel_norm = torch.norm(vel_b, dim=1, keepdim=True)

    cos_sim = (vel_b * target_dir).sum(dim=1) / (vel_norm.squeeze() * target_norm.squeeze() + 1e-6)

    return cos_sim * (target_norm.squeeze() > 0.1).float()




def linear_velocity_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands using exponential kernel.

    Based on the paper's velocity tracking reward formulation:
    r_t = exp(-||v_cmd - v_actual||^2 / std^2)

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        std: Standard deviation for exponential kernel (default: 0.5 m/s).
        asset_cfg: Asset configuration.

    Returns:
        Reward tensor with values in range [0, 1], where 1 means perfect tracking.
    """
    # Extract used quantities
    asset = env.scene[asset_cfg.name]
    # Get commanded velocity
    commanded_vel = env.command_manager.get_command(command_name)[:, :2]  # vx, vy
    # Get actual velocity in body frame
    actual_vel = asset.data.root_lin_vel_b[:, :2]

    # Compute L2 squared error
    vel_error = torch.sum(torch.square(commanded_vel - actual_vel), dim=1)

    # Exponential kernel reward
    return torch.exp(-vel_error / (std ** 2))


def angular_velocity_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity command using exponential kernel.

    Based on the paper's angular velocity tracking reward formulation:
    r_t = exp(-(ω_cmd - ω_actual)^2 / std^2)

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        std: Standard deviation for exponential kernel (default: 0.5 rad/s).
        asset_cfg: Asset configuration.

    Returns:
        Reward tensor with values in range [0, 1], where 1 means perfect tracking.
    """
    # Extract used quantities
    asset = env.scene[asset_cfg.name]
    # Get commanded angular velocity (yaw)
    commanded_ang_vel = env.command_manager.get_command(command_name)[:, 2]  # ωz
    # Get actual angular velocity in body frame
    actual_ang_vel = asset.data.root_ang_vel_b[:, 2]

    # Compute squared error
    ang_vel_error = torch.square(commanded_ang_vel - actual_ang_vel)

    # Exponential kernel reward
    return torch.exp(-ang_vel_error / (std ** 2))


"""
Penalty terms for stair climbing (based on paper)
"""


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint position deviation from default position.

    This encourages the robot to stay close to its default joint configuration.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration specifying which joints to penalize.

    Returns:
        Penalty tensor (L2 norm of deviation).
    """
    from isaaclab.envs import mdp as base_mdp
    return base_mdp.joint_deviation_l1(env, asset_cfg)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities.

    This encourages smooth motion and prevents excessive joint speeds.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration specifying which joints to penalize.

    Returns:
        Penalty tensor (L2 norm of joint velocities).
    """
    from isaaclab.envs import mdp as base_mdp
    return base_mdp.joint_vel_l2(env, asset_cfg)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of actions.

    This encourages smooth control by penalizing large changes between consecutive actions.

    Returns:
        Penalty tensor (L2 norm of action difference).
    """
    from isaaclab.envs import mdp as base_mdp
    return base_mdp.action_rate_l2(env)


def torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize applied joint torques.

    This encourages energy-efficient motion by penalizing large torques.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration specifying which joints to penalize.

    Returns:
        Penalty tensor (L2 norm of applied torques).
    """
    from isaaclab.envs import mdp as base_mdp
    return base_mdp.joint_torques_l2(env, asset_cfg)


def body_linear_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the linear acceleration of robot bodies.

    This encourages smooth motion and prevents jerky movements.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration with body_ids to penalize.

    Returns:
        Penalty tensor (sum of L2 norms of body linear accelerations).
    """
    from isaaclab.envs import mdp as base_mdp
    return base_mdp.body_lin_acc_l2(env, asset_cfg)


def orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat orientation (pitch and roll).

    This encourages the robot to maintain an upright posture.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration.

    Returns:
        Penalty tensor (L2 norm of xy components of projected gravity vector).
    """
    from isaaclab.envs import mdp as base_mdp
    return base_mdp.flat_orientation_l2(env, asset_cfg)


def vertical_lin_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical linear velocity.

    This discourages excessive jumping and encourages stable ground contact.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration.

    Returns:
        Penalty tensor (squared vertical velocity).
    """
    from isaaclab.envs import mdp as base_mdp
    return base_mdp.lin_vel_z_l2(env, asset_cfg)


