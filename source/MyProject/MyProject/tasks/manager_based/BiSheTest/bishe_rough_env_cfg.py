# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

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

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import MyProject.tasks.manager_based.BiSheTest.mdp as mdp
from MyProject.tasks.manager_based.BiSheTest.config.terrain import make_advanced_skills_terrains_cfg


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Rough-terrain scene with Go2 and proprioceptive + height observations."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=make_advanced_skills_terrains_cfg(),
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

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Task command: target position and heading with one sample per episode."""

    target_pose = mdp.AdvancedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        radius_range=(1.0, 5.0),
        goal_height_offset=0.5,
        use_valid_target_patches=True,
        target_patch_name="target",
        max_target_height_offset=0.6,
        fallback_to_polar_sampling=False,
        log_patch_fallback=True,
        ranges=mdp.AdvancedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Joint-position action as in the paper."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations include proprioception, target position and time-to-go."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_pos_b = ObsTerm(func=mdp.target_position_command, params={"command_name": "target_pose"})
        time_to_go = ObsTerm(func=mdp.normalized_time_to_go)
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
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization and reset events."""

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

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
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

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Final-position task reward + penalties."""

    final_position = RewTerm(
        func=mdp.final_position_reward,
        weight=4.0,
        params={"command_name": "target_pose", "activate_s": 1.0, "distance_scale": 4.0},
    )
    exploration_bias = RewTerm(
        func=mdp.velocity_towards_target_bias,
        weight=0.3,
        params={
            "command_name": "target_pose",
            "activate_s": 1.0,
            "distance_scale": 4.0,
            "remove_threshold": 0.5,
            "ema_alpha": 0.995,
        },
    )
    stalling = RewTerm(
        func=mdp.stalling_penalty,
        weight=-1.0,
        params={"command_name": "target_pose", "speed_threshold": 0.1, "distance_threshold": 0.5},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_acc_l2 = RewTerm(
        func=mdp.feet_acc_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base|.*THIGH|.*CALF"),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class CurriculumCfg:
    """Terrain curriculum based on final target error."""

    staged_terrain = CurrTerm(
        func=mdp.stage_terrain_curriculum,
        params={"mode": "auto", "command_name": "target_pose"},
    )
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_position,
        params={"command_name": "target_pose", "success_threshold": 0.5, "fail_threshold": 2.0},
    )


@configclass
class BiSheGo2RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Advanced-skills locomotion task on rough terrain."""

    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terrain_stage_mode: str = "auto"

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.is_finite_horizon = True

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Increase GPU contact buffers for large-scale (4096 envs) rough-terrain training.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**22
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22

        self.commands.target_pose.resampling_time_range = (self.episode_length_s, self.episode_length_s)

        # Keep terrain scales suitable for Go2.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.scene.terrain.max_init_terrain_level = 2
        self.curriculum.staged_terrain.params["mode"] = self.terrain_stage_mode

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class BiSheGo2RoughEnvCfg_Play(BiSheGo2RoughEnvCfg):
    """Smaller rough environment for play/eval."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 12
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class BiSheGo2RoughPhase0EnvCfg(BiSheGo2RoughEnvCfg):
    """Static stage 0: walk + basic rough terrain."""

    terrain_stage_mode = "phase0_walk"


@configclass
class BiSheGo2RoughPhase1EnvCfg(BiSheGo2RoughEnvCfg):
    """Static stage 1: harder base rough terrain."""

    terrain_stage_mode = "phase1_base"


@configclass
class BiSheGo2RoughPhase2EnvCfg(BiSheGo2RoughEnvCfg):
    """Static stage 2: add gap terrains."""

    terrain_stage_mode = "phase2_gap"


@configclass
class BiSheGo2RoughPhase3EnvCfg(BiSheGo2RoughEnvCfg):
    """Static stage 3: add pit terrains."""

    terrain_stage_mode = "phase3_pit"


@configclass
class BiSheGo2RoughPhase4EnvCfg(BiSheGo2RoughEnvCfg):
    """Static stage 4: include obstacle terrains."""

    terrain_stage_mode = "phase4_obstacle"
