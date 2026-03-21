# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from MyProject.tasks.manager_based.EnvTest.scene_layout import (
    BOX_SIZE,
    HIGH_OBSTACLE_SIZE,
    LOW_OBSTACLE_SIZE,
    WALL_HEIGHT,
    WALL_LENGTH,
    WALL_THICKNESS,
)
from MyProject.tasks.manager_based.EnvTest.observation_schema import RUNTIME_BUFFER_DIMS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


GOAL_FRONT_GAP = 0.03
GOAL_LATERAL_MARGIN = 0.03
HEIGHT_SCAN_ASSETS = (
    ("left_wall", (WALL_LENGTH, WALL_THICKNESS, WALL_HEIGHT)),
    ("right_wall", (WALL_LENGTH, WALL_THICKNESS, WALL_HEIGHT)),
    ("left_low_obstacle", LOW_OBSTACLE_SIZE),
    ("right_low_obstacle", LOW_OBSTACLE_SIZE),
    ("left_high_obstacle", HIGH_OBSTACLE_SIZE),
    ("right_high_obstacle", HIGH_OBSTACLE_SIZE),
    ("support_box", BOX_SIZE),
)


def _runtime_buffer(env: "ManagerBasedEnv", attr_name: str, dim: int) -> torch.Tensor:
    """获取 EnvTest 运行时缓冲；若不存在则自动创建。

    这些缓冲用于把“外部控制器实时给出的指令”写进统一观测：
    - 低层速度命令 `velocity_commands`
    - 导航目标位姿 `pose_command`（dx, dy, dz, dyaw）
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


def _scene_asset_size(
    env: "ManagerBasedEnv",
    asset_name: str,
    fallback_size: tuple[float, float, float],
) -> tuple[float, float, float]:
    """从 scene cfg 里读取当前物体尺寸；若取不到则回退到常量。"""

    scene_cfg = getattr(env.cfg, "scene", None)
    if scene_cfg is None or not hasattr(scene_cfg, asset_name):
        return fallback_size

    asset_cfg = getattr(scene_cfg, asset_name)
    if asset_cfg is None or not hasattr(asset_cfg, "spawn"):
        return fallback_size

    spawn_cfg = asset_cfg.spawn
    size = getattr(spawn_cfg, "size", None)
    if size is None:
        return fallback_size

    return tuple(float(value) for value in size)


def velocity_commands(env: "ManagerBasedEnv") -> torch.Tensor:
    """低层策略速度命令槽位。"""

    return _runtime_buffer(env, "_envtest_velocity_commands", RUNTIME_BUFFER_DIMS["velocity_commands"])


def pose_command(env: "ManagerBasedEnv") -> torch.Tensor:
    """导航策略的 pose command 槽位。"""

    return _runtime_buffer(env, "_envtest_pose_command", RUNTIME_BUFFER_DIMS["pose_command"])


def push_goal_command(env: "ManagerBasedEnv") -> torch.Tensor:
    """推箱子高层目标点槽位。"""

    return _runtime_buffer(env, "_envtest_push_goal_command", RUNTIME_BUFFER_DIMS["push_goal_command"])


def push_actions(env: "ManagerBasedEnv") -> torch.Tensor:
    """推箱子高层上一步裁剪后动作槽位。"""

    return _runtime_buffer(env, "_envtest_push_actions", RUNTIME_BUFFER_DIMS["push_actions"])


def _build_height_scan_grid(env: "ManagerBasedEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """按 RayCaster 的 grid 配置生成与训练一致的 2D 采样网格。"""

    sensor = env.scene.sensors[sensor_cfg.name]
    pattern_cfg = sensor.cfg.pattern_cfg
    device = env.device
    cache_key = "_envtest_height_scan_local_grid"
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


def _structured_height_scan(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    include_support_box: bool = True,
) -> torch.Tensor:
    """为 EnvTest 当前场景手工构造结构化 height scan。"""

    active_assets = HEIGHT_SCAN_ASSETS if include_support_box else HEIGHT_SCAN_ASSETS[:-1]

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

    for asset_name, size in active_assets:
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


def height_scan(env: "ManagerBasedEnv", sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """为 EnvTest 当前场景手工构造结构化 height scan。

    Isaac Lab 原始 `RayCaster` 只能命中一个 mesh prim，而 EnvTest 的墙/障碍/箱子是多个 `RigidObject`。
    因此这里不再直接读取 ray hit，而是按当前场景里各障碍物的几何尺寸，在机器人 yaw 对齐的扫描网格上
    计算每个采样点对应的顶部高度。
    """

    return _structured_height_scan(env, sensor_cfg=sensor_cfg, offset=offset, include_support_box=True)


def height_scan_without_box(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    """给 push 的 low-level 使用：保留障碍物，但屏蔽 support_box。"""

    return _structured_height_scan(env, sensor_cfg=sensor_cfg, offset=offset, include_support_box=False)


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
    support_box_size = _scene_asset_size(env, "support_box", BOX_SIZE)

    candidates: list[torch.Tensor] = []
    candidate_names: list[str] = []
    candidate_size_values: list[tuple[float, float, float]] = []
    candidate_positions: list[torch.Tensor] = []
    candidate_sizes = (
        ("left_high_obstacle", HIGH_OBSTACLE_SIZE),
        ("right_high_obstacle", HIGH_OBSTACLE_SIZE),
        ("left_low_obstacle", LOW_OBSTACLE_SIZE),
        ("right_low_obstacle", LOW_OBSTACLE_SIZE),
    )
    for asset_name, fallback_size in candidate_sizes:
        try:
            asset = scene[asset_name]
        except KeyError:
            continue
        size = _scene_asset_size(env, asset_name, fallback_size)
        obstacle_pos_e = asset.data.root_pos_w[:, :3] - env_origins
        goal = torch.zeros((env.num_envs, 4), dtype=obstacle_pos_e.dtype, device=obstacle_pos_e.device)
        # x 方向：让箱子前表面尽量贴近障碍物前表面，但保留很小安全间隙。
        goal[:, 0] = obstacle_pos_e[:, 0] - 0.5 * size[0] - 0.5 * support_box_size[0] - GOAL_FRONT_GAP
        # y 方向：不再强制对齐到障碍物中心，而是在“箱子仍能贴住障碍”的可行区间内，
        # 选取距离当前箱子横向位置最近的中心点，减少无意义侧移。
        lateral_half_range = 0.5 * (size[1] - support_box_size[1]) - GOAL_LATERAL_MARGIN
        if lateral_half_range > 0.0:
            goal_y_min = obstacle_pos_e[:, 1] - lateral_half_range
            goal_y_max = obstacle_pos_e[:, 1] + lateral_half_range
            goal[:, 1] = torch.clamp(box_pos_e[:, 1], min=goal_y_min, max=goal_y_max)
        else:
            goal[:, 1] = obstacle_pos_e[:, 1]
        goal[:, 2] = 0.5 * support_box_size[2]
        goal[:, 3] = 0.0
        candidates.append(goal)
        candidate_names.append(asset_name)
        candidate_size_values.append(size)
        candidate_positions.append(obstacle_pos_e)

    if not candidates:
        raise RuntimeError("当前场景中没有可供 push_box 使用的目标障碍物。")

    candidate_tensor = torch.stack(candidates, dim=1)
    candidate_position_tensor = torch.stack(candidate_positions, dim=1)
    candidate_size_tensor = torch.tensor(candidate_size_values, dtype=box_pos_e.dtype, device=box_pos_e.device)
    # 选择“从当前箱子位姿移动量最小”的可行目标，而不是单纯最近障碍中心。
    distances = torch.linalg.norm(candidate_tensor[..., :2] - box_pos_e[:, None, :2], dim=-1)
    best_indices = torch.argmin(distances, dim=1)
    env_indices = torch.arange(env.num_envs, device=box_pos_e.device)
    selected_goal = candidate_tensor[env_indices, best_indices]
    setattr(env, "_envtest_push_goal_debug", {
        "box_size": support_box_size,
        "box_position": box_pos_e,
        "selected_obstacle_names": [candidate_names[index] for index in best_indices.detach().cpu().tolist()],
        "selected_obstacle_sizes": candidate_size_tensor[best_indices],
        "selected_obstacle_positions": candidate_position_tensor[env_indices, best_indices],
        "goal": selected_goal,
    })
    return selected_goal


def get_push_goal_debug_info(env: "ManagerBasedEnv") -> dict:
    """返回最近一次自动 push goal 计算的调试信息。"""

    goal = compute_push_goal_from_scene(env)
    debug_info = getattr(env, "_envtest_push_goal_debug", None)
    if debug_info is None:
        raise RuntimeError("Push goal debug info is unavailable.")
    debug_info = dict(debug_info)
    debug_info["goal"] = goal
    return debug_info
