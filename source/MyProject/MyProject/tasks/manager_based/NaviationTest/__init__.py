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
    id="Template-Naviation-Rough-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_rough_env_cfg:LocomotionNaviationRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Template-Naviation-Rough-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_rough_env_cfg:LocomotionNaviationRoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_rough_ppo_cfg.yaml",
    },
)

####################################平坦地形的设置#####################################

gym.register(
    id="Template-Naviation-Flat-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_flat_env_cfg:LocomotionNaviationFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Template-Naviation-Flat-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_flat_env_cfg:LocomotionNaviationFlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_flat_ppo_cfg.yaml",
    },
)

####################################climb地形的设置#####################################

gym.register(
    id="Template-Naviation-Climb-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_climb_env_cfg:NaviationClimbEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationClimbPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_climb_ppo_cfg.yaml",
    },
)


gym.register(
    id="Template-Naviation-Climb-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_climb_env_cfg:NaviationClimbEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationClimbPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_climb_ppo_cfg.yaml",
    },
)

####################################毕设场景的设置#####################################

gym.register(
    id="Template-Naviation-BiShe-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_bishe_env_cfg:NaviationBiSheEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationBiShePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_bishe_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Naviation-BiShe-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.naviation_bishe_env_cfg:NaviationBiSheEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NaviationBiShePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_bishe_ppo_cfg.yaml",
    },
)
