# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push-box skill environment derived from the NavigationTest environment layout."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

import MyProject.tasks.manager_based.PushBoxTest.mdp as mdp
from MyProject.tasks.manager_based.WalkTest.walk_rough_env_cfg import VelocityGo2WalkRoughTestEnvCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

LOW_LEVEL_ENV_CFG = VelocityGo2WalkRoughTestEnvCfg()
LOW_LEVEL_POLICY_PATH = "/home/xcj/work/IsaacLab/IsaacLabBisShe/ModelBackup/TransPolicy/WalkRoughNewTransfer.pt"
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
            size=(0.4, 0.8, 0.2),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.1)),
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
            pos_x=(1.2, 3.0),
            pos_y=(-0.35, 0.35),
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
        action_scale=(0.4, 0.2, 0.3),
        action_clip=((-0.4, 0.4), (-0.2, 0.2), (-0.3, 0.3)),
    )


@configclass
class ObservationsCfg:
    """Policy observations for pushing."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 3
        box_position = ObsTerm(func=mdp.box_pose)  # 7
        robot_position = ObsTerm(func=mdp.robot_position)  # 3
        goal_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "box_goal"})  # 3
        actions = ObsTerm(func=mdp.last_action)  # 3

        def __post_init__(self):
            self.enable_corruption = False
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

    # # 小的存活奖励，用来让稳定推进的 episode 略优于中途失稳的 episode。
    # is_alive = RewTerm(func=mdp.is_alive, weight=0.2)
    # 惩罚失败终止，但不对成功到达目标后的终止进行扣分。
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": ["base_contact", "box_out_of_bounds"]},
    )
    # 稠密距离奖励，鼓励箱子中心始终靠近目标点。
    box_goal_distance = RewTerm(
        func=mdp.box_goal_distance_tanh,
        weight=3.0,
        params={"std": 0.25, "command_name": "box_goal"},
    )
    # 逐步进度奖励，只要箱子这一步朝着目标移动就能得到正反馈。
    box_goal_progress = RewTerm(
        func=mdp.box_goal_progress,
        weight=8.0,
        params={"command_name": "box_goal"},
    )
    # 鼓励机器人保持在能持续接触箱子的距离内，避免离箱子太远推不到。
    robot_box_distance = RewTerm(
        func=mdp.robot_box_distance_tanh,
        weight=0.4,
        params={"std": 0.8},
    )
    # 稀疏成功奖励，只有箱子到达目标且机器人和箱子都基本停稳时才触发。
    box_goal_success = RewTerm(
        func=mdp.box_goal_success_bonus,
        weight=10.0,
        params={
            "command_name": "box_goal",
            "distance_threshold": 0.08,
            "box_speed_threshold": 0.05,
            "robot_speed_threshold": 0.08,
        },
    )
    # 可选的额外姿态奖励；目前关闭，因为 flat_orientation 已经覆盖了相近作用。
    # upright_posture = RewTerm(
    #     func=mdp.orientation_l2,
    #     weight=1.0,
    #     params={"desired_gravity": [0.0, 0.0, -1.0]},
    # )
    # 约束机身高度不要抬得过高，避免推箱子时重心过高、姿态发飘。
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.30},
    )
    # 惩罚横滚和俯仰，让机器人身体更平稳，减少接触时侧翻风险。
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # 惩罚高层动作变化过快，减少突然发力和不稳定的推箱动作。
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.08)


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={
            "command_name": "box_goal",
            "distance_threshold": 0.08,
            "box_speed_threshold": 0.05,
            "robot_speed_threshold": 0.08,
            "settle_steps": 12,
        },
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    box_out_of_bounds = DoneTerm(func=mdp.box_out_of_bounds)


@configclass
class CurriculumCfg:
    """No curriculum terms in the first push skill version."""


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
        self.commands.box_goal.ranges.pos_x = (2.0, 2.0)
        self.commands.box_goal.ranges.pos_y = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_box.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


PushBoxEnvCfg = LocomotionPushBoxEnvCfg
PushBoxEnvCfg_Play = LocomotionPushBoxEnvCfg_Play
