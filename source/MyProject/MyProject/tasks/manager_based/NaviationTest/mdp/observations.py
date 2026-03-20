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


def _scene_asset_size(
    env: ManagerBasedRLEnv,
    asset_name: str,
    fallback_size: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Read the configured cuboid size for an asset from the scene cfg."""

    scene_cfg = getattr(env.cfg, "scene", None)
    if scene_cfg is None or not hasattr(scene_cfg, asset_name):
        return fallback_size
    asset_cfg = getattr(scene_cfg, asset_name)
    if asset_cfg is None or not hasattr(asset_cfg, "spawn"):
        return fallback_size
    size = getattr(asset_cfg.spawn, "size", None)
    if size is None:
        return fallback_size
    return tuple(float(value) for value in size)


def _build_height_scan_grid(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Build the local 2D scan grid matching the configured RayCaster pattern."""

    sensor = env.scene.sensors[sensor_cfg.name]
    pattern_cfg = sensor.cfg.pattern_cfg
    device = env.device
    cache_key = "_navigation_height_scan_local_grid"
    cached_grid = getattr(env, cache_key, None)
    expected_num_rays = sensor.data.ray_hits_w.shape[1]
    if cached_grid is not None and cached_grid.shape == (expected_num_rays, 3) and cached_grid.device == device:
        return cached_grid

    indexing = pattern_cfg.ordering if pattern_cfg.ordering == "xy" else "ij"
    x = torch.arange(
        start=-pattern_cfg.size[0] / 2,
        end=pattern_cfg.size[0] / 2 + 1.0e-9,
        step=pattern_cfg.resolution,
        device=device,
    )
    y = torch.arange(
        start=-pattern_cfg.size[1] / 2,
        end=pattern_cfg.size[1] / 2 + 1.0e-9,
        step=pattern_cfg.resolution,
        device=device,
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)
    local_grid = torch.zeros((grid_x.numel(), 3), dtype=torch.float32, device=device)
    local_grid[:, 0] = grid_x.flatten()
    local_grid[:, 1] = grid_y.flatten()
    setattr(env, cache_key, local_grid)
    return local_grid


def _obstacle_height_scan_assets(
    env: ManagerBasedRLEnv,
    prefix: str = "obstacle_",
    fallback_size: tuple[float, float, float] = (0.5, 1.2, 0.8),
) -> list[tuple[str, tuple[float, float, float]]]:
    """Collect all configured obstacle assets that should participate in the custom scan."""

    scene_cfg = getattr(env.cfg, "scene", None)
    if scene_cfg is None:
        return []

    obstacle_assets: list[tuple[str, tuple[float, float, float]]] = []
    for asset_name, asset_cfg in vars(scene_cfg).items():
        if not asset_name.startswith(prefix):
            continue
        if asset_cfg is None or not hasattr(asset_cfg, "spawn"):
            continue
        obstacle_assets.append((asset_name, _scene_asset_size(env, asset_name, fallback_size)))

    obstacle_assets.sort(key=lambda item: item[0])
    return obstacle_assets


def obstacle_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    """Structured height scan that accounts for all active navigation obstacles."""

    obstacle_assets = _obstacle_height_scan_assets(env)

    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]
    env_origins = env.scene.env_origins
    num_envs = env.num_envs

    local_grid = _build_height_scan_grid(env, sensor_cfg)
    num_rays = local_grid.shape[0]
    repeated_quat = robot.data.root_quat_w.repeat_interleave(num_rays, dim=0)
    repeated_grid = local_grid.unsqueeze(0).repeat(num_envs, 1, 1).reshape(-1, 3)
    rotated_grid = math_utils.quat_apply_yaw(repeated_quat, repeated_grid).reshape(num_envs, num_rays, 3)

    sensor_pos_e = sensor.data.pos_w - env_origins
    sample_points_e = rotated_grid + sensor_pos_e.unsqueeze(1)
    top_heights = torch.zeros((num_envs, num_rays), dtype=torch.float32, device=env.device)

    for asset_name, size in obstacle_assets:
        try:
            asset = env.scene[asset_name]
        except KeyError:
            continue

        asset_pos_e = asset.data.root_pos_w[:, :3] - env_origins
        sample_points_local = sample_points_e - asset_pos_e.unsqueeze(1)
        repeated_asset_quat = asset.data.root_quat_w.repeat_interleave(num_rays, dim=0)
        sample_points_local = math_utils.quat_apply_inverse(
            repeated_asset_quat, sample_points_local.reshape(-1, 3)
        ).reshape(num_envs, num_rays, 3)

        in_x = torch.abs(sample_points_local[..., 0]) <= 0.5 * size[0]
        in_y = torch.abs(sample_points_local[..., 1]) <= 0.5 * size[1]
        hit_mask = in_x & in_y
        asset_top_z = asset_pos_e[:, 2].unsqueeze(1) + 0.5 * size[2]
        top_heights = torch.where(hit_mask, torch.maximum(top_heights, asset_top_z), top_heights)

    return sensor_pos_e[:, 2].unsqueeze(1) - top_heights - offset


def nearest_obstacles_state(
    env: ManagerBasedRLEnv,
    k: int = 3,
    prefix: str = "obstacle_",
    fallback_size: tuple[float, float, float] = (0.5, 1.2, 0.8),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the nearest K obstacle positions and sizes in the robot frame, padded with zeros."""

    if k <= 0:
        return torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)

    obstacle_assets = _obstacle_height_scan_assets(env, prefix=prefix, fallback_size=fallback_size)
    output = torch.zeros((env.num_envs, k, 6), dtype=torch.float32, device=env.device)
    if not obstacle_assets:
        return output.reshape(env.num_envs, k * 6)

    robot: RigidObject = env.scene[robot_cfg.name]
    positions: list[torch.Tensor] = []
    sizes: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []

    for asset_name, size in obstacle_assets:
        try:
            asset: RigidObject = env.scene[asset_name]
        except KeyError:
            continue

        asset_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            asset.data.root_pos_w[:, :3],
        )
        positions.append(asset_pos_b)
        sizes.append(torch.tensor(size, dtype=asset_pos_b.dtype, device=asset_pos_b.device).repeat(env.num_envs, 1))
        distances.append(torch.linalg.norm(asset_pos_b[:, :2], dim=-1))

    if not positions:
        return output.reshape(env.num_envs, k * 6)

    position_tensor = torch.stack(positions, dim=1)
    size_tensor = torch.stack(sizes, dim=1)
    distance_tensor = torch.stack(distances, dim=1)

    num_assets = position_tensor.shape[1]
    gather_count = min(k, num_assets)
    best_indices = torch.argsort(distance_tensor, dim=1)[:, :gather_count]
    gather_index = best_indices.unsqueeze(-1).expand(-1, -1, 3)
    output[:, :gather_count, :3] = torch.gather(position_tensor, 1, gather_index)
    output[:, :gather_count, 3:] = torch.gather(size_tensor, 1, gather_index)
    return output.reshape(env.num_envs, k * 6)


def asset_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Return the asset root position expressed in the robot root frame."""

    robot: RigidObject = env.scene[robot_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    asset_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, asset.data.root_pos_w[:, :3])
    return asset_pos_b


def asset_size(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    fallback_size: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Return the configured cuboid size of an asset, repeated for all environments."""

    size = _scene_asset_size(env, asset_cfg.name, fallback_size)
    return torch.tensor(size, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
