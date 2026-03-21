# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

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
from MyProject.tasks.manager_based.WalkTest.config.terrain import (
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


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
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

    # @configclass
    # class CriticCfg(ObsGroup):
    #     """Observations for critic group."""

    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    #     projected_gravity = ObsTerm(
    #         func=mdp.projected_gravity,
    #     )
    #     velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    #     joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    #     joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    #     actions = ObsTerm(func=mdp.last_action)
    #     height_scan = ObsTerm(
    #         func=mdp.height_scan,
    #         params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    #         clip=(-1.0, 1.0),
    #     )
    #     # height_scanner = ObsTerm(func=mdp.height_scan,
    #     #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    #     #     clip=(-1.0, 5.0),
    #     # )

    #     def __post_init__(self):
    #         self.history_length = 3
    #         # self.enable_corruption = True
    #         self.concatenate_terms = True
    # # privileged observations
    # critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
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
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
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
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class BiShePitRewardsCfg(RewardsCfg):
    """用于跨越坑洞技能训练的奖励项。"""

    move_in_command_direction = RewTerm(
        func=walk_mdp.move_in_command_direction,
        weight=0.3,  # 原来: 0.5（略降，避免下坑时为追求前进而前扑）
        params={"command_name": "base_velocity"},
    )
    # 增强机身姿态稳定性，抑制下坑时俯仰角速度过大导致前翻。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    #速度追踪奖励略降，避免下坑时为追求速度而前扑；同时保留一定奖励，鼓励学会在坑洞中保持一定前进速度，而不是过于保守地停在坑边。
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.50, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0) # 原来: -2.0 防止上下惩罚的速度
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.10,  # 原来(父类): -0.05
    )
    # 原来为 0，不约束机身倾斜；这里开启后可明显减少“趴下再翻”的策略。
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,  # 原来(父类): 0.0
    )
    # 启用 gated 抬脚奖励，鼓励用抬脚跨越坑沿，而不是用机身/头部顶过去。
    # feet_height = RewTerm(
    #     func=walk_mdp.feet_height_pit_gated,
    #     weight=2.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "target_height": 0.12,
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "obstacle_height_threshold": 0.10,
    #         "min_obstacle_rays": 4,
    #         "forward_min_x": 0.05,
    #         "rear_max_x": -0.05,
    #     },
    # )
    # calf_collision_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.8,  # 原来: -0.8（保持不变，小腿轻惩罚）
    #     params={
    #         # 同时兼容大小写命名，避免正则没命中导致该惩罚失效
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=".*([Cc][Aa][Ll][Ff]).*"
    #         ),
    #         "threshold": 1.0,  # 原来: 1.0（保持不变）
    #     },
    # )  # 小腿可触碰但给轻惩罚，避免把正常探坑动作误判为坏动作。

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
        weight=-1.0,
        params={
            "threshold": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=["RL_foot", "RR_foot"]),
        },
    )

    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )防止腿打滑
    # # 将原先合并的 thigh+hip 惩罚拆分为两项，便于独立调参。
    # thigh_collision_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-2.0,  # 原来(合并项): -1.5
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*([Tt][Hh][Ii][Gg][Hh]).*"),
    #         "threshold": 1.0,  # 原来(合并项): 1.0
    #     },
    # )
    # hip_collision_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-3.0,  # 原来(合并项): -1.5
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*([Hh][Ii][Pp]).*"),
    #         "threshold": 1.0,  # 原来(合并项): 1.0
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 头部碰撞用惩罚塑形；硬终止保持 base，避免腿/髋轻微接触导致过早终止。
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # # 髋关节碰撞仍视为高风险，保留硬终止但阈值适度放宽。
    # hip_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*([Hh][Ii][Pp]).*"), "threshold": 1.2},
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class VelocityGo2WalkRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Scale down terrain for smaller robot
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # Disable push robot event for stability
        self.events.push_robot = None
        # Disable base COM randomization
        self.events.base_com = None
        # Disable undesired contacts penalty
        self.rewards.undesired_contacts = None
        # Flat terrain settings (commented out for rough terrain)
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # Disable height scan (optional, for flat terrain)
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # Disable terrain curriculum (optional, for flat terrain)
        # self.curriculum.terrain_levels = None

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class VelocityGo2WalkRoughEnvCfg_Play(VelocityGo2WalkRoughEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class LocomotionBiShePitEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for the locomotion environment specialized for high-platform climbing.
    这里保留类名兼容原有 task 注册，但实际内容已经切换为高台 climb 训练。
    """
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    pit_rewards: BiShePitRewardsCfg = BiShePitRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化：覆盖为高台 climb 训练配置。"""
        # 使用高台地形。地形课程对应的是平台高度，而不是 pit 深度。
        self.scene.terrain.terrain_generator = HIGH_PLATFORM_TERRAINS_CFG
        self.rewards = self.pit_rewards
        # 从最低课程等级开始，逐步提高平台高度。
        self.scene.terrain.max_init_terrain_level = 0
        # 每次 reset 都放到高台前面的低地面上，避免出生就在平台顶。
        self.events.reset_base.func = walk_mdp.reset_root_state_before_high_platform
        self.events.reset_base.params["pose_range"] = {"x": (-3.9, -3.5), "y": (-0.15, 0.15), "yaw": (-0.1, 0.1)}
        # 将任务收窄为纯前向 climb，减少侧移/转向绕平台。
        # self.commands.base_velocity.rel_standing_envs = 0.0
        # self.commands.base_velocity.rel_heading_envs = 0.0
        # self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.15, 0.15)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # 通用参数。
        self.decimation = 4
        self.episode_length_s = 20.0
        # 仿真参数。
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # 稳定性相关设置。
        self.events.push_robot = None
        self.events.base_com = None
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

        #   1. 只有惩罚：策略可能仍“偶尔碰一下”换取前进收益。
        #   2. 只有终止：信号太稀疏，只知道“死了”，不知道“怎么更好”。
        #   3. 两者一起：既有连续梯度（惩罚），又有明确红线（终止），通常更快、更稳。
        # 使用跨越坑洞专用奖励配置
        # self.rewards = StairClimbingRewardsCfg()



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
        # remove random pushes for deterministic evaluation
        self.events.base_external_force_torque = None
        self.events.push_robot = None

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
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)





@configclass
class VelocityGo2WalkRoughEnvCfg_Ros(VelocityGo2WalkRoughEnvCfg_Play):
    """Configuration for the locomotion velocity-tracking environment."""
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
