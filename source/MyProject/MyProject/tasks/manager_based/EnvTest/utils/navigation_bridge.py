from __future__ import annotations

import math

import torch


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""

    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _yaw_from_quat_w(quat_w: torch.Tensor) -> torch.Tensor:
    """Extract yaw from Isaac Lab `(w, x, y, z)` quaternions."""

    w = quat_w[:, 0]
    x = quat_w[:, 1]
    y = quat_w[:, 2]
    z = quat_w[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def align_navigation_goal_height(
    goal_command: torch.Tensor,
    default_root_height: torch.Tensor,
) -> torch.Tensor:
    """Align navigation goal z to the training-time default root height."""

    if goal_command.shape[-1] != 4:
        raise ValueError(f"goal_command must have shape (*, 4), got {tuple(goal_command.shape)}")

    if default_root_height.ndim == 1:
        default_root_height = default_root_height.unsqueeze(-1)
    if default_root_height.ndim != 2 or default_root_height.shape[-1] != 1:
        raise ValueError(
            f"default_root_height must have shape (*,) or (*, 1), got {tuple(default_root_height.shape)}"
        )

    aligned_goal = goal_command.clone()
    aligned_goal[:, 2:3] = default_root_height.to(device=aligned_goal.device, dtype=aligned_goal.dtype)
    return aligned_goal


def build_navigation_pose_command(
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    goal_command_w: torch.Tensor,
) -> torch.Tensor:
    """Convert world-frame `[x, y, z, yaw]` goals into base-frame pose commands."""

    if goal_command_w.shape[-1] != 4:
        raise ValueError(f"goal_command_w must have shape (*, 4), got {tuple(goal_command_w.shape)}")

    delta_world = goal_command_w[:, :3] - robot_pos_w[:, :3]
    robot_yaw = _yaw_from_quat_w(robot_quat_w)
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)

    pose_command = torch.zeros_like(goal_command_w)
    pose_command[:, 0] = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
    pose_command[:, 1] = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
    pose_command[:, 2] = delta_world[:, 2]
    pose_command[:, 3] = _wrap_to_pi(goal_command_w[:, 3] - robot_yaw)
    return pose_command

