# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Template-Push-Box-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_box_env_cfg:LocomotionPushBoxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushBoxPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_push_box_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Push-Box-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_box_env_cfg:LocomotionPushBoxEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushBoxPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_push_box_ppo_cfg.yaml",
    },
)
