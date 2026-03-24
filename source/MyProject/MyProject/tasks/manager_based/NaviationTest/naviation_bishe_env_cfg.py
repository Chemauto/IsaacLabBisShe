# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

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

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# Custom terrain configuration for stair climbing
# 专门用于爬楼梯训练的地形配置
# from isaaclab.terrains import TerrainGeneratorCfg
# import isaaclab.terrains.config.terrain_gen as terrain_gen



##
# Scene definition
##

import MyProject.tasks.manager_based.NaviationTest.mdp as mdp
from MyProject.tasks.manager_based.NaviationTest.config.terrain import TWO_PIT_TERRAINS_CFG
from MyProject.tasks.manager_based.WalkTest.walk_rough_env_cfg import VelocityGo2WalkRoughEnvCfg
LOW_LEVEL_ENV_CFG = VelocityGo2WalkRoughEnvCfg()
_REPO_ROOT = Path(__file__).resolve().parents[6]
LOW_LEVEL_POLICY_REL_PATH = Path("ModelBackup/TransPolicy/WalkRoughNewTransfer.pt")
LOW_LEVEL_POLICY_PATH = str(_REPO_ROOT / LOW_LEVEL_POLICY_REL_PATH)
#分层的强化学习的方式，低层的强化学习为之前已经训练好的在平地上行走的策略
#如果需要训练好的话，这个层次的策略也应该训练好一点


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TWO_PIT_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(4.2, 6.0), pos_y=(-1.4, 1.4), heading=(-0.6, 0.6)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        #This
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        #在模型加载着一块，IsaacLab中的自带的RSL—RL训练代码的模型是checkpoint文件
        #而这个给出的示例代码则是TorchScript文件
        #在NewTools文件夹下的NewTools/model_trans.py可以转换模型
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        action_clip=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    # observation groups
    policy: PolicyCfg = PolicyCfg()


    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        def __post_init__(self):
            self.concatenate_terms = True

    # privileged observations
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    randomize_terrain_tile = None
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
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,#原来0.8
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,#原来0.8
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.4,#原来0.3
        params={"command_name": "pose_command"},
    )
    lateral_deviation_penalty = RewTerm(
        func=mdp.lateral_deviation_penalty,
        weight=-2.0,
        params={"threshold": 0.8},
    )
    base_collision_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,  # 原来: -0.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},  
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "pose_command", "distance_threshold": 0.2, "heading_threshold": 0.15, "settle_steps": 3},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##

@configclass
class NaviationBiSheEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = 0.005
        self.sim.render_interval = self.actions.pre_trained_policy_action.low_level_decimation
        self.decimation = self.actions.pre_trained_policy_action.low_level_decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.events.reset_base.params["pose_range"] = {"x": (-0.3, 0.3), "y": (-0.45, 0.45), "yaw": (-0.35, 0.35)}
        two_pit_cfg = self.scene.terrain.terrain_generator.sub_terrains["two_pit"]
        max_pit_rear_x = max(
            two_pit_cfg.pit_1_x_range[1] + 0.5 * two_pit_cfg.pit_1_size_x_range[1],
            two_pit_cfg.pit_2_x_range[1] + 0.5 * two_pit_cfg.pit_2_size_x_range[1],
        )
        max_goal_x = min(two_pit_cfg.size[0] - two_pit_cfg.spawn_origin_x - 0.75, max_pit_rear_x + 2.5)
        min_goal_x = max_pit_rear_x + 0.45
        if max_goal_x <= min_goal_x:
            max_goal_x = min_goal_x + 0.5
        lateral_goal_limit = min(0.5 * two_pit_cfg.navigation_corridor_width - 0.15, 1.4)
        self.commands.pose_command.ranges.pos_x = (min_goal_x, max_goal_x)
        self.commands.pose_command.ranges.pos_y = (-lateral_goal_limit, lateral_goal_limit)
        self.commands.pose_command.ranges.heading = (-0.75, 0.75)

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class NaviationBiSheEnvCfg_Play(NaviationBiSheEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # keep play lightweight, but still keep a small bank of random terrain tiles for evaluation
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        play_terrain_generator = self.scene.terrain.terrain_generator.copy()
        play_terrain_generator.num_rows = 6
        play_terrain_generator.num_cols = 6
        play_terrain_generator.border_width = 2.0
        play_terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator = play_terrain_generator
        self.events.randomize_terrain_tile = EventTerm(func=mdp.randomize_terrain_tile, mode="reset")
        # disable randomization for play
        self.observations.policy.enable_corruption = False
