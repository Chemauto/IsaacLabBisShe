# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push-box skill environment derived from the NavigationTest environment layout."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import MyProject.tasks.manager_based.PushBoxTest.mdp as mdp
# from MyProject.tasks.manager_based.WalkTest.walk_rough_env_cfg import VelocityGo2WalkRoughTestEnvCfg
from MyProject.tasks.manager_based.WalkTest.walk_flat_env_cfg import Go2WalkFlatEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# LOW_LEVEL_ENV_CFG = VelocityGo2WalkRoughTestEnvCfg()
LOW_LEVEL_ENV_CFG = Go2WalkFlatEnvCfg()
_REPO_ROOT = Path(__file__).resolve().parents[6]
LOW_LEVEL_POLICY_REL_PATH = Path("ModelBackup/TransPolicy/WalkFlatLowHeightTransfer.pt")
LOW_LEVEL_POLICY_PATH = str(_REPO_ROOT / LOW_LEVEL_POLICY_REL_PATH)
#低层的环境和策略配置，推箱子这个技能是基于之前训练好的走路技能进行训练的，所以这里直接引用之前走路技能的环境配置和策略路径


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene for learning the box pushing skill."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, 0.24),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.47, 0.22),
                metallic=0.1,
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 0.12)),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command term for the desired box location."""

    box_goal = mdp.BoxGoalCommandCfg(
        asset_name="box",
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=mdp.BoxGoalCommandCfg.Ranges(
            pos_x=(1.0, 3.0),
            pos_y=(-1.0, 1.0),
            yaw=(-3.1416/3, 3.1416/3),
        ),
    )


@configclass
class ActionsCfg:
    """High-level action term over the trained walk policy."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        # action_scale=(0.8, 0.6, 0.6),
        action_clip=((-0.5, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
    )


@configclass
class ObservationsCfg:
    """Policy observations for pushing."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        box_in_robot_frame_pos = ObsTerm(
            func=mdp.box_in_robot_frame_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 3
        box_in_robot_frame_yaw = ObsTerm(
            func=mdp.box_in_robot_frame_yaw,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 2
        goal_in_box_frame_pos = ObsTerm(
            func=mdp.goal_in_box_frame_pos,
            params={"command_name": "box_goal"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )  # 3
        goal_in_box_frame_yaw = ObsTerm(
            func=mdp.goal_in_box_frame_yaw,
            params={"command_name": "box_goal"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 2
        actions = ObsTerm(func=mdp.processed_last_action, params={"action_name": "pre_trained_policy_action"})  # 3

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset robot and box around a structured pushing setup."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
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

    reset_box = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.25, 0.25), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("box"),
        },
    )


@configclass
class RewardsCfg:
    """Rewards for pushing the box to the target pose."""
    ###############################惩罚奖励函数################################## 
    # 惩罚失败终止，但不对成功到达目标后的终止进行扣分。
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": ["base_contact", "box_out_of_bounds"]},
    )
    ###############################任务奖励函数################################## 
              #####################稠密奖励#######################

    # 稠密距离奖励，鼓励箱子中心始终靠近目标点。
    box_goal_distance_fine_gained = RewTerm(
        func=mdp.box_goal_distance_exp,
        weight=4.0,
        params={"std": 0.10, "command_name": "box_goal"},
    ) #原来weight4.0,std0.10

    box_goal_distance = RewTerm(
        func=mdp.box_goal_distance_exp,
        weight=2.0,
        params={"std": 0.60, "command_name": "box_goal"},
    ) #原来weight4.0,std0.10

    # 鼓励箱子最终朝向也与目标 yaw 对齐，避免只到点不转向。
    box_goal_yaw = RewTerm(
        func=mdp.box_goal_yaw_distance_exp,
        weight=2.5,
        params={"std": 0.15, "command_name": "box_goal"},
    )

    
    # 当箱子已经接近目标时，再约束机器人最终朝向与目标 yaw 对齐，避免过早干扰推箱主任务。
    # robot_goal_yaw = RewTerm(
    #     func=mdp.robot_goal_yaw_error_abs,
    #     weight=-0.50,
    #     params={"command_name": "box_goal", "activate_distance_threshold": 0.40},
    # )
    # 加入角度控制，最后0.3米内保证机器人的姿态
              #####################稀疏奖励#######################
    box_goal_success = RewTerm(
        func=mdp.box_goal_success_bonus,
        weight=15.0,
        params={
            "command_name": "box_goal",
            "distance_threshold": 0.12,#0.06
            "yaw_threshold": 0.15,#0.15
            "box_speed_threshold": 0.20,#0.06
            "robot_speed_threshold": 0.20,#0.06
        },
    )

    ###############################姿态奖励函数################################## 

    # 惩罚横滚和俯仰，让机器人身体更平稳，减少接触时侧翻风险。
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # 惩罚裁剪后高层命令变化过快，避免原始大动作数值爆炸污染训练。
    action_rate = RewTerm(
        func=mdp.processed_action_rate_l2,
        weight=-0.20,#原来-0.20
        params={"action_name": "pre_trained_policy_action"},
    )  
    face_to_object = RewTerm(
        func=mdp.face_to_object,
        weight=1.0,
    )
    # 论文 Table VIII: Negative x-velocity penalty, 实现为 max(v_b,x, 0)
    forward_x_velocity = RewTerm(
        func=mdp.forward_x_velocity_reward,
        weight=1.0,
    )
    # head_collision_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.5,  # 原来: -3.0（加重，抑制用腹部/机身"扑地过坑"）
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_.*"), "threshold": 1.0},  # 原来: 1.0
    # )    # 论文风格的"头部碰撞"塑形：对head接触做惩罚，。髋关节和大腿部分，惩罚
    # 惩罚头部刚体的 xy 投影落到箱子顶面矩形范围内，避免头越到箱子上方。
    # head_over_box = RewTerm(
    #     func=mdp.head_point_in_box_penalty,
    #     weight=-0.5,
    #     params={
    #         "head_local_offset": (0.00, 0.0, 0.0),
    #         "footprint_margin": -0.10,  # 允许头部投影稍微进入箱子边界内，因为推的过程中可能会有轻微的接触和变形。
    #         "top_surface_margin": 0.00,
    #         "head_body_cfg": SceneEntityCfg("robot", body_names="Head_.*"),
    #     },
    # )
    # 约束机身高度不要抬得过高，避免推箱子中后段为了拿进度奖励而抬身前探。
    # base_height = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-15.0,
    #     params={"target_height": 0.20},
    # )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={
            "command_name": "box_goal",
            "distance_threshold": 0.12,
            "yaw_threshold": 0.15,
            "box_speed_threshold": 0.20,
            "robot_speed_threshold": 0.20,
            "settle_steps": 4,
        },
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    box_out_of_bounds = DoneTerm(func=mdp.box_out_of_bounds)


@configclass
class CurriculumCfg:
    """Curriculum terms for expanding the box-goal range."""

    goal_range = CurrTerm(
        func=mdp.box_goal_progress_curriculum,
        params={
            # 先计算当前 batch 的平均推进比例 progress，再用
            # A <- (1 - beta) * A + beta * progress_mean
            # 做平滑更新，避免某一批 env 偶然推得好就让课程范围放大过快。
            "command_name": "box_goal",
            "progress_beta": 0.02,
        },
    )


@configclass
class LocomotionPushBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 push-box skill."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.box_goal.resampling_time_range[1]
        self.sim.physics_material = self.scene.terrain.physics_material

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class LocomotionPushBoxEnvCfg_Play(LocomotionPushBoxEnvCfg):
    """Smaller deterministic play configuration for debugging the skill."""

    def __post_init__(self):
        super().__post_init__()
        # self.decimation = LOW_LEVEL_ENV_CFG.decimation *1
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
        self.curriculum.goal_range = None
        self.commands.box_goal.ranges.pos_x = (1.7, 1.7)
        self.commands.box_goal.ranges.pos_y = (0.0, 0.0)
        self.commands.box_goal.ranges.yaw = (0, 0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_box.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


PushBoxEnvCfg = LocomotionPushBoxEnvCfg
PushBoxEnvCfg_Play = LocomotionPushBoxEnvCfg_Play
