# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EnvTest 的自定义事件函数。

最重要的是 `reset_structured_navigation_scene`：
它会在 reset 时根据场景编号，把不同障碍物摆到对应位置。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject

from MyProject.tasks.manager_based.EnvTest.scene_layout import (
    ACTIVE_LAYOUT_POSITIONS,
    CASE_COUNT,
    CASE_LAYOUTS,
    HIDDEN_LAYOUT_POSITIONS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _repeat_position(position: tuple[float, float, float], count: int, device: str) -> torch.Tensor:
    """把一个三维位置复制成 (count, 3) 的张量。"""
    return torch.tensor(position, dtype=torch.float, device=device).repeat(count, 1)


def _set_asset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_name: str,
    rel_positions: torch.Tensor,
):
    """把某个刚体资产移动到指定相对位置。

    这里的 rel_positions 是相对于各自 env_origin 的局部位置。
    """
    asset: RigidObject = env.scene[asset_name]
    root_state = asset.data.default_root_state[env_ids].clone()
    # 重置位置并把线速度/角速度清零，避免继承上一个 episode 的状态。
    root_state[:, 0:3] = env.scene.env_origins[env_ids] + rel_positions
    root_state[:, 7:13] = 0.0
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_structured_navigation_scene(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """按场景编号摆放结构化走廊场景。

    - scene_id=None：按 env_id 自动轮换 case1~case5
    - scene_id=0~4：所有环境都固定成同一个 case
    """

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(list(env_ids), device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    if len(env_ids) == 0:
        return

    # scene_id=None 时，批量环境自动对应 5 个 case；
    # 否则全部固定为同一个场景。
    selected_scene_id = getattr(env.cfg, "scene_id", None)
    if selected_scene_id is None:
        scenario_ids = torch.remainder(env_ids, CASE_COUNT)
    else:
        scenario_ids = torch.full_like(env_ids, fill_value=int(selected_scene_id))

    for asset_name, hidden_position in HIDDEN_LAYOUT_POSITIONS.items():
        # 默认先把所有可选障碍物放到隐藏区。
        rel_positions = _repeat_position(hidden_position, len(env_ids), env.device)
        active_position = torch.tensor(ACTIVE_LAYOUT_POSITIONS[asset_name], dtype=torch.float, device=env.device)

        for case_id, case_cfg in enumerate(CASE_LAYOUTS):
            case_mask = scenario_ids == case_id
            # 当前 case 需要这个物体时，再把它放回激活位置。
            if torch.any(case_mask) and case_cfg[asset_name]:
                rel_positions[case_mask] = active_position

        _set_asset_pose(env, env_ids, asset_name, rel_positions)
