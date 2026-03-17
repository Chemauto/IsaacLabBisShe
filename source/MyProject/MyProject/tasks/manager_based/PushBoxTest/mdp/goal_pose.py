# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi


def split_box_goal_command(goal_command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a box-goal command into position and optional yaw."""
    goal_pos = goal_command[:, :3]
    if goal_command.shape[1] >= 4:
        goal_yaw = wrap_to_pi(goal_command[:, 3])
    else:
        goal_yaw = torch.zeros(goal_command.shape[0], device=goal_command.device, dtype=goal_command.dtype)
    return goal_pos, goal_yaw


def yaw_error_abs(current_yaw: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
    """Compute the absolute wrapped yaw error."""
    return torch.abs(wrap_to_pi(target_yaw - current_yaw))


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract the world yaw from wxyz quaternions."""
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles to wxyz quaternions."""
    quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    half_yaw = yaw * 0.5
    quat[:, 0] = torch.cos(half_yaw)
    quat[:, 3] = torch.sin(half_yaw)
    return quat


__all__ = ["quat_to_yaw", "split_box_goal_command", "wrap_to_pi", "yaw_error_abs", "yaw_to_quat"]
