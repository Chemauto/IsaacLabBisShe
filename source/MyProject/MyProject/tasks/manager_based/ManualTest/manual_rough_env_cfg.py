# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# Custom terrain configuration for stair climbing
# 专门用于爬楼梯训练的地形配置
# from isaaclab.terrains import TerrainGeneratorCfg
# import isaaclab.terrains.config.terrain_gen as terrain_gen



##
# Scene definition
##

import MyProject.tasks.manager_based.ManualTest.mdp as mdp
from MyProject.tasks.manager_based.WalkTest.walk_rough_env_cfg import (
    VelocityGo2WalkRoughEnvCfg,
    MySceneCfg as WalkMySceneCfg,
)
from MyProject.tasks.manager_based.ManualTest.config.terrain import (
    MIXED_PIT_TERRAINS_CFG,
    EVAL_PIT_TERRAINS_CFG,
)
LOW_LEVEL_ENV_CFG = VelocityGo2WalkRoughEnvCfg()
#分层的强化学习的方式，低层的强化学习为之前已经训练好的在平地上行走的策略
#如果需要训练好的话，这个层次的策略也应该训练好一点


@configclass
class MySceneCfg(WalkMySceneCfg):
    """扩展低层环境的场景配置，使用混合坑洞地形，支持课程学习"""

    # 覆盖地形配置为混合坑洞（含三种难度：60%简单 + 30%中等 + 10%困难）
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MIXED_PIT_TERRAINS_CFG,
        max_init_terrain_level=5,  #
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)



##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path="/home/xcj/work/IsaacLab/IsaacLabBisShe/ModelBackup/TransPolicy/WalkRoughNewTransfer.pt",

        #在模型加载着一块，IsaacLab中的自带的RSL—RL训练代码的模型是checkpoint文件
        #而这个给出的示例代码则是TorchScript文件
        #在NewTools文件夹下的NewTools/model_trans.py可以转换模型
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        time_to_go = ObsTerm(func=mdp.normalized_time_to_go)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        actions = ObsTerm(func=mdp.last_action)
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # 稠密导航奖励（前期主导）
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )

    # 任务主奖励：末端位置奖励（论文风格），在回合最后一段时间激活。
    final_position = RewTerm(
        func=mdp.final_position_reward,
        weight=0.0,
        params={"command_name": "pose_command", "activate_s": 1.0, "distance_scale": 4.0},
    )
    # 早期探索引导：朝目标方向移动，收敛后自动关闭。
    exploration_bias = RewTerm(
        func=mdp.velocity_towards_target_bias,
        weight=0.0,
        params={"command_name": "pose_command", "remove_threshold": 0.35, "ema_alpha": 0.995},
    )
    stalling = RewTerm(
        func=mdp.stalling_penalty,
        weight=0.0,
        params={"command_name": "pose_command", "speed_threshold": 0.08, "distance_threshold": 0.6},
    )


    # 与 WalkTest 对齐的通用正则惩罚项
    
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    ####是否需要呢？？？
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)#关节加速度惩罚 1.原来的论文里面joint accelerations
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)#关节力矩惩罚 2.原来的论文里面joint torques
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )#碰撞惩罚  3.原来的论文里面collisions
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)#突然的动作变化 4.原来的论文里面abrupt actions changes


    feet_acc_l2 = RewTerm(
        func=mdp.feet_acc_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )#足端加速度惩罚 5.原论文额外强调 feet accelerations



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """课程学习配置：仅按训练 iter 推进坑洞难度。"""

    pit_terrain_schedule = CurrTerm(
        func=mdp.pit_terrain_by_iteration,
        params={
            # 课程阶段切换迭代点（总共 1500 iter 的默认划分）。
            "iter_stage_boundaries": (0, 400, 900, 1300),
            # ManualTest 的 PPO 配置是 num_steps_per_env=8，用于 step->iter 换算。
            "steps_per_iteration": 8,
            "stage_weights": (
                (0.85, 0.14, 0.01),  # 前期：简单坑为主
                (0.65, 0.28, 0.07),
                (0.45, 0.35, 0.20),
                (0.25, 0.35, 0.40),  # 后期：困难坑占比提高
            ),
            # 每个阶段允许的最大地形等级比例（从低到高逐步放开）。
            "stage_max_level_ratio": (0.35, 0.55, 0.75, 1.0),
        },
    )
    # P0 基线统计：在 reset 前输出 success/fall/final_distance/energy 等指标到日志。
    p0_metrics = CurrTerm(
        func=mdp.p0_episode_metrics,
        params={
            "command_name": "pose_command",
            "success_distance_threshold": 0.5,
            "hard_terrain_key": "hard_pit",
        },
    )
    # 奖励混合课程：前半程稠密导航，后半程平滑切到末端奖励。
    reward_blend = CurrTerm(
        func=mdp.blend_navigation_reward_schedule,
        params={
            "steps_per_iteration": 8,
            "blend_start_iter": 300,
            "blend_end_iter": 1100,
            "dense_position_weight": 0.5,
            "dense_position_fine_weight": 0.5,
            "dense_orientation_weight": -0.2,
            "dense_termination_weight": -400.0,
            "sparse_final_weight": 4.0,
            "sparse_exploration_weight": 0.3,
            "sparse_stalling_weight": -0.8,
        },
    )



##
# Environment configuration
##

@configclass
class LocomotionManualRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment with adaptive curriculum learning."""

    # environment settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


class LocomotionManualRoughEnvCfg_Play(LocomotionManualRoughEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4
        # disable randomization for play
        self.observations.policy.enable_corruption = False


class LocomotionManualRoughEnvCfg_Eval(LocomotionManualRoughEnvCfg):
    """固定评估配置：关闭训练课程，使用固定浅/中/深沟地形。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 256
        self.scene.env_spacing = 3.0
        self.scene.terrain.terrain_generator = EVAL_PIT_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 5
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        # 评估时关闭训练课程项，只保留 P0 指标统计。
        self.curriculum.pit_terrain_schedule = None
        self.curriculum.reward_blend = None
        self.observations.policy.enable_corruption = False
