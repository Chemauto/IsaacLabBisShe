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
    MySceneCfg as WalkMySceneCfg,
)
from MyProject.tasks.manager_based.ManualTest.config.terrain import (
    MIXED_PIT_TERRAINS_CFG,
    EVAL_PIT_TERRAINS_CFG,
)
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

    pose_command = mdp.AdvancedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        # 对齐论文：每回合一个目标，回合长度 6s。
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        radius_range=(1.0, 5.0),
        goal_height_offset=0.5,
        use_valid_target_patches=True,
        target_patch_name="target",
        max_target_height_offset=0.6,
        fallback_to_polar_sampling=False,
        log_patch_fallback=True,
        ranges=mdp.AdvancedPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 端到端动作：策略直接输出关节目标位置，由底层 PD 控制器转换为力矩。
    # 说明：这一步只替换动作接口，不改奖励和课程学习，便于你单步验收。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        """
        The observations include 1.joint positions, 2.joint velocities,3.base linear and 4.angular velocity, 5.commands, 
        6.previous actions, and 7.terrain measurements sampled around the robot.
        The commands are defined as the three-dimensional location of the target expressed in the base frame and the remaining
        time to reach that location.

        """

        # 1.关节位置 12
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 2.速度观测 12
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # 3.机体线速度观测（加入传感器噪声，提升 sim2real 鲁棒性）3
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # 4.机体角速度观测（论文要求包含 base angular velocity）3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        # 5.论文对齐：使用目标 3D 位置（机体系），不直接使用 heading 命令。3 
        pose_command = ObsTerm(func=mdp.pose_command_position_b, params={"command_name": "pose_command"})

        # 6.上一时刻动作（端到端时为关节维动作历史）12
        actions = ObsTerm(func=mdp.last_action)
        # 7.地形高度扫描（加入噪声，防止策略过拟合“完美地形感知”）187
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        # 剩余时间输入（论文中的 remaining time）这个是为了后面那个的时间奖励进行观测的 1
        time_to_go = ObsTerm(func=mdp.normalized_time_to_go)

        # 一共 12 + 12 + 3 + 3 + 3 + 12 + 187 + 1 = 233 维度

        def __post_init__(self):
            # 开启观测扰动与拼接，和 WalkTest 保持一致
            self.enable_corruption = True
            self.concatenate_terms = True
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
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 任务主奖励：末端位置奖励（论文风格），在回合最后一段时间激活。
    final_position = RewTerm(
        func=mdp.final_position_reward,
        weight=1.0,
        params={"command_name": "pose_command", "activate_s": 1.0, "distance_scale": 4.0},
    )
    # 早期探索引导：朝目标方向移动，收敛后自动关闭。
    exploration_bias = RewTerm(
        func=mdp.velocity_towards_target_bias,
        weight=1.0,
        params={
            "command_name": "pose_command",
            "remove_threshold": 0.5,
            "speed_epsilon": 1.0e-6,
        },
    )
    stalling = RewTerm(
        func=mdp.stalling_penalty,
        weight=-1.0,
        params={"command_name": "pose_command", "speed_threshold": 0.1, "distance_threshold": 0.5},
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
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"), "threshold": 1.0},
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
    """论文风格课程学习：按每回合成功/失败更新地形等级。"""

    pit_terrain_success = CurrTerm(
        func=mdp.pit_terrain_by_command_success,
        params={
            "command_name": "pose_command",
            "success_distance_threshold": 0.5,
            "level_step_up": 1,
            "level_step_down": 1,
        },
    )
    # P0 基线统计：在 reset 前仅输出 success_rate 到日志。
    p0_metrics = CurrTerm(
        func=mdp.p0_episode_metrics,
        params={
            "command_name": "pose_command",
            "success_distance_threshold": 0.5,
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
        # 端到端版本不再依赖低层环境配置，直接使用标准控制频率。
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # 对齐论文：有限时域任务，不使用无限时域 bootstrapping 假设。
        self.is_finite_horizon = True

        if self.scene.height_scanner is not None:
            # 高度扫描按策略步更新即可。
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
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
        self.curriculum.pit_terrain_success = None
        self.observations.policy.enable_corruption = False
