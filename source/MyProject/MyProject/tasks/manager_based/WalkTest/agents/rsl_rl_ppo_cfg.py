# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2WalkRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "go2_walk_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
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
class Go2WalkBiShePPORunnerCfg(Go2WalkRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000
        self.experiment_name = "go2_walk_bishe"
        # Keep scalar std to stay behavior-compatible with legacy walk checkpoints.
        self.policy.noise_std_type = "scalar"
        self.policy.init_noise_std = 0.3
        # Fine-tuning the walk checkpoint on the double-platform climb task needs a smaller,
        # non-adaptive step size to avoid sudden policy explosions.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.max_grad_norm = 0.5
        self.clip_actions = 5.0#新的代码这个加上了


# Backward-compatible alias for older references.



@configclass
class Go2WalkFlatPPORunnerCfg(Go2WalkRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "go2_walk_flat"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
