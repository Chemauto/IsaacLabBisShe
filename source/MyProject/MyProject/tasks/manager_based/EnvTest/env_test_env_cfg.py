# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EnvTest 场景配置。

这里把任务当成“纯场景生成 + 机器人复位”的测试环境来使用，
不再引入奖励、终止条件、命令采样这些 RL 训练逻辑。
主要目的是把你描述的 5 类结构化走廊场景稳定地摆出来。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import MyProject.tasks.manager_based.EnvTest.mdp as mdp
from MyProject.tasks.manager_based.EnvTest.scene_layout import (
    BOX_SIZE,
    HIGH_OBSTACLE_SIZE,
    HIDDEN_LAYOUT_POSITIONS,
    LOW_OBSTACLE_SIZE,
    WALL_CENTER_X,
    WALL_CENTER_Y,
    WALL_HEIGHT,
    WALL_LENGTH,
    WALL_THICKNESS,
)

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


def _kinematic_box_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """创建不可推动的方块配置。

    用于墙壁、低障碍和高障碍。
    关键点是 `kinematic_enabled=True`，这样机器人碰到后不会把它撞走。
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.15,
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _dynamic_box_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """创建可推动箱子的方块配置。

    这里只有 case 5 会真正启用它，其它 case 会被 reset 逻辑移动到隐藏区域。
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.1,
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """结构化走廊场景。

    包含：
    - 平地
    - 左右墙壁
    - 左右低/高障碍
    - 可推动箱子
    - Go2 机器人
    """

    # 使用平地，避免 terrain generator 带来额外随机性。
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

    # 两侧墙壁：围出中间约 3m 宽的走廊。
    left_wall = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/LeftWall",
        size=(WALL_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
        pos=(WALL_CENTER_X, WALL_CENTER_Y, 0.5 * WALL_HEIGHT),
        color=(0.55, 0.55, 0.58),
    )
    right_wall = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/RightWall",
        size=(WALL_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
        pos=(WALL_CENTER_X, -WALL_CENTER_Y, 0.5 * WALL_HEIGHT),
        color=(0.55, 0.55, 0.58),
    )

    # 左右低障碍：尺寸为 1.5 x 1.0 x 0.3。
    left_low_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/LeftLowObstacle",
        size=LOW_OBSTACLE_SIZE,
        pos=HIDDEN_LAYOUT_POSITIONS["left_low_obstacle"],
        color=(0.22, 0.64, 0.34),
    )
    right_low_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/RightLowObstacle",
        size=LOW_OBSTACLE_SIZE,
        pos=HIDDEN_LAYOUT_POSITIONS["right_low_obstacle"],
        color=(0.22, 0.64, 0.34),
    )
    # 左右高障碍：尺寸为 1.5 x 1.0 x 0.5。
    left_high_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/LeftHighObstacle",
        size=HIGH_OBSTACLE_SIZE,
        pos=HIDDEN_LAYOUT_POSITIONS["left_high_obstacle"],
        color=(0.76, 0.24, 0.20),
    )
    right_high_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/RightHighObstacle",
        size=HIGH_OBSTACLE_SIZE,
        pos=HIDDEN_LAYOUT_POSITIONS["right_high_obstacle"],
        color=(0.76, 0.24, 0.20),
    )

    # 可推动箱子：尺寸为 0.8 x 0.4 x 0.2，质量在 helper 中配置为 1kg。
    support_box = _dynamic_box_cfg(
        prim_path="{ENV_REGEX_NS}/SupportBox",
        size=BOX_SIZE,
        pos=HIDDEN_LAYOUT_POSITIONS["support_box"],
        color=(0.85, 0.52, 0.18),
    )

    # 机器人主体。
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 基础环境光，便于 GUI 里观察地形和障碍物。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ActionsCfg:
    """动作配置。

    这里只保留最简单的关节位置动作接口，
    这样 `zero_agent.py` / `random_agent.py` 就能直接把环境跑起来。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """最小观测配置。

    这里只保留机器人自身状态，不包含任务目标或奖励相关量。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # 机身线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # 机身角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # 重力在机体坐标系下的投影
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 相对默认姿态的关节位置
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # 关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # 上一步动作
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # 这里只做场景测试，因此关闭噪声，并把各项观测拼成向量。
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """复位逻辑。

    reset 顺序是：
    1. reset_scene：把所有对象恢复到默认状态；
    2. reset_layout：根据场景编号重新摆放障碍和箱子；
    3. reset_base：把机器人固定回初始起点。
    """

    # 把整个 scene 恢复到默认状态。
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    # 根据 scene_id 或 env_id 选择对应 case 的障碍布局。
    reset_layout = EventTerm(
        func=mdp.reset_structured_navigation_scene,
        mode="reset",
    )

    # 固定机器人起点和初速度，避免随机 reset 干扰看场景。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
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


@configclass
class LocomotionEnvTestEnvCfg(ManagerBasedEnvCfg):
    """EnvTest 主配置。

    默认批量模式下：
    - num_envs=5
    - scene_id=None
    这样会按 env_id 自动对应 case1~case5。
    """

    scene: MySceneCfg = MySceneCfg(num_envs=5, env_spacing=8.0)
    # None 表示按 env_id 自动轮换 5 个场景；
    # 0~4 表示强制固定成某一个场景。
    scene_id: int | None = None
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 这组参数只需要保证场景稳定运行，不追求训练环境那种大吞吐。
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.decimation = 4
        self.sim.physics_material = self.scene.terrain.physics_material


class LocomotionEnvTestEnvCfg_Play(LocomotionEnvTestEnvCfg):
    """Play 模式配置。

    你当前把它改成了 5 个环境同时显示，
    并将 `scene_id=None`，因此会在 GUI 里一次看到 case1~case5。
    """

    def __post_init__(self):
        super().__post_init__()
        # Play 模式下也展开 5 个环境，方便并排看 5 个 case。
        self.scene.num_envs = 5
        self.scene.env_spacing = 8.0
        # None 表示按 env_id 自动分配场景；
        # 如果改成 0~4，则 5 个环境都会变成同一个 case。
        self.scene_id = None


EnvTestEnvCfg = LocomotionEnvTestEnvCfg
EnvTestEnvCfg_Play = LocomotionEnvTestEnvCfg_Play
