# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


# 批量模式：通常用于一次展开多个并行环境查看所有 case。
gym.register(
    id="Template-EnvTest-Go2-v0",
    entry_point=f"{__name__}.env_test_env:EnvTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_test_env_cfg:LocomotionEnvTestEnvCfg",
    },
)

# Play 模式：通常用于 GUI 下单独检查或者演示场景。
gym.register(
    id="Template-EnvTest-Go2-Play-v0",
    entry_point=f"{__name__}.env_test_env:EnvTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_test_env_cfg:LocomotionEnvTestEnvCfg_Play",
    },
)
