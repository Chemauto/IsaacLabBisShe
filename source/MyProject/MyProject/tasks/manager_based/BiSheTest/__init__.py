# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Template-BiShe-Go2-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_rough_env_cfg:BiSheGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-BiShe-Go2-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_rough_env_cfg:BiSheGo2RoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-BiShe-Go2-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_flat_env_cfg:BiSheGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-BiShe-Go2-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_flat_env_cfg:BiSheGo2FlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_flat_ppo_cfg.yaml",
    },
)
