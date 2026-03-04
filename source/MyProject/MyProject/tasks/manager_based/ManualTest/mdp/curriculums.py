# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_ids_to_tensor(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """将 env_ids 统一转换为当前设备上的 long tensor。"""
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _compute_column_map(terrain_gen_cfg) -> dict[str, list[int]]:
    """按照 sub_terrain 的 proportion，计算每个地形对应的列索引集合。"""
    names = list(terrain_gen_cfg.sub_terrains.keys())
    proportions = [terrain_gen_cfg.sub_terrains[name].proportion for name in names]
    total = sum(proportions)
    if total <= 0.0:
        return {name: [] for name in names}

    cumulative = []
    running = 0.0
    for p in proportions:
        running += p / total
        cumulative.append(running)

    col_map = {name: [] for name in names}
    for col in range(terrain_gen_cfg.num_cols):
        value = col / float(terrain_gen_cfg.num_cols) + 1.0e-3
        idx = 0
        while idx < len(cumulative) - 1 and value >= cumulative[idx]:
            idx += 1
        col_map[names[idx]].append(col)
    return col_map


def _select_stage(current_iteration: int, iter_stage_boundaries: tuple[int, ...]) -> int:
    """根据当前迭代数选择课程阶段索引。"""
    stage = 0
    for idx, boundary in enumerate(iter_stage_boundaries):
        if current_iteration >= boundary:
            stage = idx
    return stage


def pit_terrain_by_iteration(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    iter_stage_boundaries: tuple[int, int, int, int] = (0, 400, 900, 1300),
    steps_per_iteration: int = 8,
    stage_weights: tuple[tuple[float, float, float], ...] = (
        (0.85, 0.14, 0.01),  # easy/medium/hard
        (0.65, 0.28, 0.07),
        (0.45, 0.35, 0.20),
        (0.25, 0.35, 0.40),
    ),
    stage_max_level_ratio: tuple[float, ...] = (0.35, 0.55, 0.75, 1.0),
) -> torch.Tensor:
    """仅基于训练迭代数的坑洞课程学习。

    前期：主要采样简单坑，并限制较低地形等级；
    后期：逐步提高中/困难坑占比，并放开更高地形等级。
    """
    terrain: TerrainImporter = env.scene.terrain
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    env_ids_t = _env_ids_to_tensor(env, env_ids)
    if env_ids_t.numel() == 0:
        return torch.tensor(0.0, device=env.device)

    terrain_gen_cfg = terrain.cfg.terrain_generator
    col_map = _compute_column_map(terrain_gen_cfg)

    terrain_keys = ("easy_pit", "medium_pit", "hard_pit")
    if any(len(col_map.get(key, [])) == 0 for key in terrain_keys):
        return torch.tensor(0.0, device=env.device)

    # Isaac Lab 中 common_step_counter 是全局 step，这里换算为训练迭代轮数。
    current_iteration = int(env.common_step_counter) // max(1, int(steps_per_iteration))
    stage = _select_stage(current_iteration, iter_stage_boundaries)
    stage = min(stage, len(stage_weights) - 1, len(stage_max_level_ratio) - 1)

    # 按当前阶段的 easy/medium/hard 权重采样地形类型。
    weights = torch.tensor(stage_weights[stage], device=env.device, dtype=torch.float32)
    probs = weights / torch.clamp(weights.sum(), min=1.0e-6)
    picked_keys = torch.multinomial(probs, num_samples=env_ids_t.numel(), replacement=True)

    sampled_types = torch.empty(env_ids_t.numel(), dtype=torch.long, device=env.device)
    for key_idx, key in enumerate(terrain_keys):
        mask = picked_keys == key_idx
        if not torch.any(mask):
            continue
        columns = torch.as_tensor(col_map[key], dtype=torch.long, device=env.device)
        sampled = torch.randint(0, columns.numel(), (int(mask.sum().item()),), device=env.device)
        sampled_types[mask] = columns[sampled]

    terrain.terrain_types[env_ids_t] = sampled_types

    # 按阶段限制最大 terrain level，形成“由易到难”的课程推进。
    max_level = max(
        0,
        min(
            terrain.max_terrain_level - 1,
            int(round((terrain.max_terrain_level - 1) * float(stage_max_level_ratio[stage]))),
        ),
    )
    terrain.terrain_levels[env_ids_t] = torch.clamp(terrain.terrain_levels[env_ids_t], min=0, max=max_level)
    terrain.env_origins[env_ids_t] = terrain.terrain_origins[terrain.terrain_levels[env_ids_t], terrain.terrain_types[env_ids_t]]

    return torch.tensor(float(stage), device=env.device)


def p0_episode_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "pose_command",
    success_distance_threshold: float = 0.5,
    hard_terrain_key: str = "hard_pit",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, float]:
    """在 reset 前统计 P0 基线指标，并通过 Curriculum 日志输出。"""
    env_ids_t = _env_ids_to_tensor(env, env_ids)
    if env_ids_t.numel() == 0:
        return {
            "success_rate": 0.0,
            "hard_pit_success_rate": 0.0,
            "hard_pit_active_rate": 0.0,
            "fall_rate": 0.0,
            "timeout_rate": 0.0,
            "final_distance_mean": 0.0,
            "energy_proxy": 0.0,
        }

    # 训练启动时会先调用一次 reset，此时还没有 reset_time_outs/reset_terminated 缓冲区。
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return {
            "success_rate": 0.0,
            "hard_pit_success_rate": 0.0,
            "hard_pit_active_rate": 0.0,
            "fall_rate": 0.0,
            "timeout_rate": 0.0,
            "final_distance_mean": 0.0,
            "energy_proxy": 0.0,
        }

    command = env.command_manager.get_command(command_name)
    final_distance = torch.norm(command[env_ids_t, :2], dim=1)

    timeout = env.reset_time_outs[env_ids_t]
    terminated = env.reset_terminated[env_ids_t]
    fall = terminated & (~timeout)
    success = timeout & (final_distance < success_distance_threshold)

    terrain: TerrainImporter = env.scene.terrain
    hard_mask = torch.zeros_like(timeout)
    if terrain.cfg.terrain_type == "generator" and terrain.cfg.terrain_generator is not None:
        col_map = _compute_column_map(terrain.cfg.terrain_generator)
        hard_cols = col_map.get(hard_terrain_key, [])
        if len(hard_cols) > 0:
            hard_cols_t = torch.as_tensor(hard_cols, device=env.device, dtype=torch.long)
            hard_mask = (terrain.terrain_types[env_ids_t].unsqueeze(1) == hard_cols_t.unsqueeze(0)).any(dim=1)

    hard_timeout = timeout & hard_mask
    hard_success = success & hard_mask

    asset: Articulation = env.scene[asset_cfg.name]
    torque_term = torch.sum(torch.square(asset.data.applied_torque[env_ids_t]), dim=1)
    joint_acc_term = torch.sum(torch.square(asset.data.joint_acc[env_ids_t]), dim=1)
    action_delta = env.action_manager.action[env_ids_t] - env.action_manager.prev_action[env_ids_t]
    action_rate_term = torch.sum(torch.square(action_delta), dim=1)
    energy_proxy = torch.mean(torque_term + joint_acc_term + action_rate_term)

    hard_timeout_count = int(hard_timeout.sum().item())
    hard_success_rate = float(hard_success.sum().item()) / float(hard_timeout_count) if hard_timeout_count > 0 else 0.0

    return {
        "success_rate": torch.mean(success.float()).item(),
        "hard_pit_success_rate": hard_success_rate,
        "hard_pit_active_rate": torch.mean(hard_mask.float()).item(),
        "fall_rate": torch.mean(fall.float()).item(),
        "timeout_rate": torch.mean(timeout.float()).item(),
        "final_distance_mean": torch.mean(final_distance).item(),
        "energy_proxy": energy_proxy.item(),
    }




# -----------------------------------------------------------------------------
# 课程学习逻辑总说明（通用模板）
# -----------------------------------------------------------------------------
# 本文件目前包含两类课程策略：
# 1) pit_terrain_by_iteration:
#    - 纯“训练进度驱动”（只看迭代轮次，不看任务成功率）。
#    - 通过 iter_stage_boundaries 划分阶段。
#    - 每个阶段用 stage_weights 控制各子地形（easy/medium/hard）的采样概率。
#    - 每个阶段用 stage_max_level_ratio 限制可用 terrain_level 上限，实现“先易后难”。
#
# 2) terrain_levels_vel:
#    - “性能驱动”（看机器人是否走得足够远）。
#    - 走得好则升级地形，走得差则降级地形。
#
# 在 ManualTest 中，当前激活的是 pit_terrain_by_iteration（见 manual_rough_env_cfg.py 的 CurriculumCfg），
# terrain_levels_vel 作为备选策略保留，默认不接入。
#
# --------------------------- 如何迁移到其他地形 -------------------------------
# 若你要做 gap / obstacle / 混合复杂地形，可复用 pit_terrain_by_iteration 的框架：
#
# Step 1: 在 terrain 配置里定义子地形 key
#   例如: ("gap_easy", "gap_hard") 或 ("obs_low", "obs_mid", "obs_high")。
#
# Step 2: 在课程函数中替换 terrain_keys
#   terrain_keys = ("easy_pit", "medium_pit", "hard_pit")  -> 你的 key 列表。
#
# Step 3: 调整 stage_weights
#   - 每个阶段一个 tuple，长度等于 terrain_keys 数量；
#   - 数值越大，该子地形被采样概率越高；
#   - 建议早期“简单 key”权重大，后期“困难 key”权重大。
#
# Step 4: 调整 iter_stage_boundaries
#   - 按总训练迭代数划分阶段；
#   - 例如总 2000 iter，可设 (0, 500, 1200, 1700)。
#
# Step 5: 校准 steps_per_iteration
#   - 该值应与 PPO 配置 num_steps_per_env 对齐；
#   - 作用是把 common_step_counter（全局 step）换算为迭代轮次。
#
# Step 6: 调整 stage_max_level_ratio
#   - 控制每阶段最高 terrain_level；
#   - 典型设置为递增，如 (0.3, 0.5, 0.7, 1.0)。
#
# Step 7: 在 env cfg 中挂载 CurrTerm
#   - CurriculumCfg 内新增对应 CurrTerm；
#   - 环境类中确保 curriculum: CurriculumCfg = CurriculumCfg()；
#   - terrain_generator.curriculum = True。
#
# ----------------------------- 调参经验建议 -----------------------------------
# 1. 如果训练前期频繁摔倒：提高简单地形权重，降低 stage_max_level_ratio。
# 2. 如果中后期收敛太慢：提前提高困难地形权重，或缩短阶段边界间距。
# 3. 如果训练波动大：减少相邻阶段权重差，避免难度突变。
# 4. 若要“只按进度”课程：不要在同一任务同时启用性能驱动升级/降级项，避免策略冲突。
