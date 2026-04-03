# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from sympy import im

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import MyProject.tasks.manager_based.WalkTest.mdp as walk_mdp
import MyProject.tasks.manager_based.WalkTest.mdp.curriculums as walk_curriculums
from MyProject.tasks.manager_based.WalkTest.config.terrain import (
    HIGH_DOUBLE_PLATFORM_TERRAINS_CFG,
    HIGH_DOUBLE_PLATFORM_TERRAINS_PLAY_CFG,
    HIGH_PLATFORM_TERRAINS_CFG,
    HIGH_PLATFORM_TERRAINS_PLAY_CFG,
)
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
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
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""

#     base_velocity = mdp.UniformVelocityCommandCfg(
#         asset_name="robot",
#         resampling_time_range=(10.0, 10.0),
#         rel_standing_envs=0.02,
#         rel_heading_envs=1.0,
#         heading_command=True,
#         heading_control_stiffness=0.5,
#         debug_vis=True,
#         ranges=mdp.UniformVelocityCommandCfg.Ranges(
#             lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
#         ),
#     )


@configclass
class BiSheCommandsCfg:
    """World-frame command specifications for the climb task."""

    base_velocity = walk_mdp.UniformWorldVelocityCommandCfg(
        asset_name="robot",
        # 大于 episode_length_s=20.0，保证一个 episode 内不会中途重采样方向。
        resampling_time_range=(12.0, 12.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=walk_mdp.UniformWorldVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-0.4, 0.4),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            # self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
#########################这里修改了############################
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            # self.history_length = 3
            # self.enable_corruption = True
            self.concatenate_terms = True
    # privileged observations
    critic: CriticCfg = CriticCfg()
#########################这里修改了############################

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-0.5, 0.5),
            "torque_range": (-0.5, 0.5),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 10.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

# 原来: 0.3（略降，避免下坑时为追求前进而前扑）
    move_in_command_direction = RewTerm(
        func=walk_mdp.move_in_world_command_direction,
        weight=0.3,  # 原来: 0.3（略降，避免下坑时为追求前进而前扑）
        params={"command_name": "base_velocity"},
    )
# 原来: 0.3（略降，避免下坑时为追求前进而前扑）
    # 增强机身姿态稳定性，抑制下坑时俯仰角速度过大导致前翻。
    track_lin_vel_xy_exp = RewTerm(
        func=walk_mdp.track_lin_vel_xy_world_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    #速度追踪奖励略降，避免下坑时为追求速度而前扑；同时保留一定奖励，鼓励学会在坑洞中保持一定前进速度，而不是过于保守地停在坑边。
    track_ang_vel_z_exp = RewTerm(
        func=walk_mdp.track_ang_vel_z_world_exp,
        weight=0.6,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0) # 原来: -2.0 防止上下惩罚的速度
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.10,  # 原来(父类): -0.05
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 原来为 0，不约束机身倾斜；这里开启后可明显减少“趴下再翻”的策略。
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,  # 原来(父类): 0.0
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.05,  # 原来: 0.01（大幅提升，鼓励用抬脚跨越坑沿，而不是用机身/头部顶过去）
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # air_time_variance = RewTerm(
    #     func=walk_mdp.air_time_variance_penalty,
    #     weight=-0.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )#适当的奖励函数，为了增强步态

    # 论文风格的“头部碰撞”塑形：对 base接触做惩罚，。髋关节和大腿部分，惩罚
    base_collision_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,  # 原来: -3.0（加重，抑制用腹部/机身“扑地过坑”）
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},  # 原来: 1.0
    )
    head_collision_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,  # 原来: -3.0（加重，抑制用腹部/机身"扑地过坑"）
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_.*"), "threshold": 1.0},  # 原来: 1.0
    )    # 论文风格的"头部碰撞"塑形：对head接触做惩罚，。髋关节和大腿部分，惩罚
    # 防止脚部撞击垂直表面（如墙面、障碍物侧面）
    feet_stumble = RewTerm(
        func=walk_mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # 防止后面两脚之间距离过近（避免腿部交叉或碰撞）
    Behind_feet_too_near = RewTerm(
        func=walk_mdp.feet_too_near,
        weight=-0.5,
        params={
            "threshold": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=["RL_foot", "RR_foot"]),
        },
    )
    # 防止前脚之间距离过近（避免腿部交叉或碰撞），这两个奖励函数是为了张开脚
    Front_feet_too_near = RewTerm(
        func=walk_mdp.feet_too_near,
        weight=-0.5,
        params={
            "threshold": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 头部碰撞用惩罚塑形；硬终止保持 base，避免腿/髋轻微接触导致过早终止。
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=walk_curriculums.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionBiShePitEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for the locomotion environment specialized for high-platform climbing.
    这里保留类名兼容原有 task 注册，但实际内容已经切换为高台 climb 训练。
    """
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: BiSheCommandsCfg = BiSheCommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化：覆盖为高台 climb 训练配置。"""
        # 使用高台地形。地形课程对应的是平台高度，而不是 pit 深度。
        self.scene.terrain.terrain_generator = HIGH_DOUBLE_PLATFORM_TERRAINS_CFG
        # 从最低课程等级开始，逐步提高平台高度。
        self.scene.terrain.max_init_terrain_level = 0
        # 每次 reset 都放到高台前面的低地面上，避免出生就在平台顶。
        self.events.reset_base.func = walk_mdp.reset_root_state_before_high_platform
        self.events.reset_base.params["pose_range"] = {"x": (-3.9, -3.5), "y": (-0.15, 0.15), "yaw": (-0.1, 0.1)}
        # 通用参数。
        self.decimation = 4
        self.episode_length_s = 12.0
        # 仿真参数。
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # 稳定性相关设置。
        # 传感器更新周期。
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # 根据配置启用/关闭地形课程学习。
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False




@configclass
class LocomotionBiShePitEnvCfg_Play(LocomotionBiShePitEnvCfg):
    """Play configuration for high-platform climb environment."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = HIGH_PLATFORM_TERRAINS_PLAY_CFG
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # 显示高程图信息查看
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.debug_vis = True
            self.scene.height_scanner.visualizer_cfg.prim_path = "/World/Visuals/HeightScanner"
            self.scene.height_scanner.visualizer_cfg.markers["hit"].radius = 0.06
        # 固定在高台前方同一位置，方便稳定观察 climb 行为。
        self.events.reset_base.func = walk_mdp.reset_root_state_before_high_platform
        self.events.reset_base.params["pose_range"] = {"x": (-3.7, -3.7), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        # self.commands.base_velocity.rel_standing_envs = 0.0
        # self.commands.base_velocity.rel_heading_envs = 0.0
        # self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.events.add_base_mass = None  # 评测时不增加额外质量扰动
        self.events.base_com = None  # 评测时不增加 COM 扰动
        self.events.reset_robot_joints = None  # 评测时不增加关节
        self.events.push_robot = None  # 评测时不增加随机推力扰动
        self.events.base_external_force_torque = None  # 评测时不增加随机推力扰动
