# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg

from .env_test_env_cfg import build_scene_cfg


class EnvTestEnv(ManagerBasedEnv, gym.Env):
    """给 EnvTest 使用的轻量 Gym 包装器。

    原始 `ManagerBasedEnv` 更偏底层，不会自动提供 Gym 常用的
    `action_space` / `observation_space`。这里额外包一层，
    这样现有脚本可以直接通过 `gym.make()` 来启动。
    """

    is_vector_env: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }

    cfg: ManagerBasedEnvCfg

    def __init__(self, cfg: ManagerBasedEnvCfg, render_mode: str | None = None, **kwargs):
        # `env_cfg_entry_point` 这类注册参数会通过 kwargs 传进来，
        # 这里不需要用它们，所以直接忽略即可。
        # 由于 scene_id 可能在 parse_env_cfg 之后又被启动脚本覆盖，
        # 因此这里在真正创建环境前，再按最终 scene_id 重建一次 scene。
        if hasattr(cfg, "scene_id"):
            cfg.scene = build_scene_cfg(cfg.scene.num_envs, cfg.scene.env_spacing, cfg.scene_id)
        super().__init__(cfg=cfg)
        self.render_mode = render_mode
        # 让录视频或显示时使用和环境步长一致的帧率。
        self.metadata["render_fps"] = 1 / self.step_dt
        # 初始化统一观测中的运行时缓冲，方便外部控制器直接写值。
        self._ensure_runtime_observation_buffers()
        # 手动配置 Gym 所需的空间定义。
        self._configure_gym_env_spaces()
        print("[INFO]: Completed setting up the EnvTest environment...")

    def _configure_gym_env_spaces(self):
        """配置 Gym 所需的动作空间和观测空间。"""

        # 先构造单环境下的 observation space。
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=group_dim, dtype=np.float32
                )
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim, dtype=np.float32)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)

        # 动作空间由 action manager 的总维度决定。
        action_dim = self.action_manager.total_action_dim
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)

        # 再把单环境空间批量化成向量环境空间。
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _ensure_runtime_observation_buffers(self):
        """确保统一观测里用到的运行时缓冲已经存在。"""

        if not hasattr(self, "_envtest_velocity_commands"):
            self._envtest_velocity_commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if not hasattr(self, "_envtest_push_goal_command"):
            self._envtest_push_goal_command = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        if not hasattr(self, "_envtest_push_actions"):
            self._envtest_push_actions = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def set_runtime_observation_buffers(
        self,
        velocity_commands: torch.Tensor | None = None,
        push_goal_command: torch.Tensor | None = None,
        push_actions: torch.Tensor | None = None,
    ):
        """更新 EnvTest 统一观测中需要的运行时指令槽位。"""

        self._ensure_runtime_observation_buffers()
        updates = (
            ("_envtest_velocity_commands", velocity_commands),
            ("_envtest_push_goal_command", push_goal_command),
            ("_envtest_push_actions", push_actions),
        )
        for attr_name, value in updates:
            if value is None:
                continue
            buffer = getattr(self, attr_name)
            if value.shape != buffer.shape:
                raise ValueError(f"{attr_name} shape mismatch: expected {tuple(buffer.shape)}, got {tuple(value.shape)}.")
            buffer.copy_(value.to(device=self.device, dtype=torch.float32))
