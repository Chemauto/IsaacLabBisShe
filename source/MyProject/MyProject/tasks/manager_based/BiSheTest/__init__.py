# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

####################################粗糙地形的设置#####################################

gym.register(
    id="Template-BiShe-Rough-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_rough_env_cfg:LocomotionBiSheRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Template-BiShe-Rough-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_rough_env_cfg:LocomotionBiSheRoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_rough_ppo_cfg.yaml",
    },
)


####################################楼梯地形的设置#####################################

gym.register(
    id="Template-BiShe-Climb-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_rough_env_cfg:LocomotionBiSheClimbEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Template-BiShe-Climb-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bishe_rough_env_cfg:LocomotionBiSheClimbEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BiSheRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bishe_rough_ppo_cfg.yaml",
    },
)