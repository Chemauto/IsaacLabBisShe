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
    id="Template-Velocity-Go2-Walk-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_rough_env_cfg:VelocityGo2WalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_rough_ppo_cfg.yaml",
    },
)



gym.register(
    id="Template-Velocity-Go2-Walk-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_rough_env_cfg:VelocityGo2WalkRoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Velocity-Go2-Walk-Rough-Ros-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_rough_env_cfg:VelocityGo2WalkRoughEnvCfg_Ros",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_rough_ppo_cfg.yaml",
    },
)

####################################楼梯地形的设置#####################################

gym.register(
    id="Template-Velocity-Go2-Walk-Climb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_rough_env_cfg:LocomotionClimbEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Velocity-Go2-Walk-Climb-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_rough_env_cfg:LocomotionClimbEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_rough_ppo_cfg.yaml",
    },
)


####################################毕业坑洞地形的设置#####################################

gym.register(
    id="Template-Velocity-Go2-Walk-BiShe-Pit-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_bishe_env_cfg:LocomotionBiShePitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkBiShePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_bishe_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_bishe_env_cfg:LocomotionBiShePitEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkBiShePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_bishe_ppo_cfg.yaml",
    },
)


####################################平坦地形的设置#####################################
gym.register(
    id="Template-Velocity-Go2-Walk-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_env_cfg:VelocityGo2WalkFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_flat_ppo_cfg.yaml",
    },
)



gym.register(
    id="Template-Velocity-Go2-Walk-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_env_cfg:VelocityGo2WalkFlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Velocity-Go2-Walk-Flat-Ros-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_env_cfg:VelocityGo2WalkFlatEnvCfg_Ros",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WalkFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_flat_ppo_cfg.yaml",
    },
)
