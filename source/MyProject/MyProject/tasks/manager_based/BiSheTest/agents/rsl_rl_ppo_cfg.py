# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BiSheRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for rough advanced-skills task."""

    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 100
    experiment_name = "go2_bishe_advanced_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BiSheFlatPPORunnerCfg(BiSheRoughPPORunnerCfg):
    """PPO config for flat advanced-skills task."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1200
        self.experiment_name = "go2_bishe_advanced_flat"
