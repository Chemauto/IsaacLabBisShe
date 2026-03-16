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
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import MyProject.tasks.manager_based.EnvTest.mdp as mdp
from MyProject.tasks.manager_based.EnvTest.scene_layout import (
    ACTIVE_LAYOUT_POSITIONS,
    BOX_SIZE,
    HIGH_OBSTACLE_SIZE,
    LOW_OBSTACLE_SIZE,
    SCENE_LAYOUTS,
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

    这里只有 case 5 会真正启用它；
    其它场景下，这个资产会在 scene 构建阶段直接被裁剪掉。
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


def _get_scene_layout(scene_id: int) -> dict:
    """根据 scene_id 获取固定场景定义。"""

    if not 0 <= scene_id < len(SCENE_LAYOUTS):
        raise ValueError(f"scene_id must be in [0, {len(SCENE_LAYOUTS) - 1}], but received {scene_id}.")
    return SCENE_LAYOUTS[scene_id]


def _apply_scene_layout(scene_cfg: "MySceneCfg", scene_id: int):
    """按场景编号裁剪可选障碍物。

    这里不再使用“隐藏位置”。
    不需要的障碍物会直接被设为 None，InteractiveScene 解析时会自动跳过。
    """

    layout = _get_scene_layout(scene_id)
    optional_assets = (
        "left_low_obstacle",
        "right_low_obstacle",
        "left_high_obstacle",
        "right_high_obstacle",
        "support_box",
    )
    for asset_name in optional_assets:
        if not layout[asset_name]:
            setattr(scene_cfg, asset_name, None)


def build_scene_cfg(num_envs: int, env_spacing: float, scene_id: int) -> "MySceneCfg":
    """根据场景编号重新创建一份完整的 scene 配置。"""

    scene_cfg = MySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    _apply_scene_layout(scene_cfg, scene_id)
    return scene_cfg


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

    # 左右低障碍：尺寸由 LOW_OBSTACLE_SIZE 定义。
    left_low_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/LeftLowObstacle",
        size=LOW_OBSTACLE_SIZE,
        pos=ACTIVE_LAYOUT_POSITIONS["left_low_obstacle"],
        color=(0.22, 0.64, 0.34),
    )
    right_low_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/RightLowObstacle",
        size=LOW_OBSTACLE_SIZE,
        pos=ACTIVE_LAYOUT_POSITIONS["right_low_obstacle"],
        color=(0.22, 0.64, 0.34),
    )
    # 左右高障碍：尺寸由 HIGH_OBSTACLE_SIZE 定义。
    left_high_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/LeftHighObstacle",
        size=HIGH_OBSTACLE_SIZE,
        pos=ACTIVE_LAYOUT_POSITIONS["left_high_obstacle"],
        color=(0.76, 0.24, 0.20),
    )
    right_high_obstacle = _kinematic_box_cfg(
        prim_path="{ENV_REGEX_NS}/RightHighObstacle",
        size=HIGH_OBSTACLE_SIZE,
        pos=ACTIVE_LAYOUT_POSITIONS["right_high_obstacle"],
        color=(0.76, 0.24, 0.20),
    )

    # 可推动箱子：尺寸由 BOX_SIZE 定义，质量在 helper 中配置为 1kg。
    support_box = _dynamic_box_cfg(
        prim_path="{ENV_REGEX_NS}/SupportBox",
        size=BOX_SIZE,
        pos=ACTIVE_LAYOUT_POSITIONS["support_box"],
        color=(0.85, 0.52, 0.18),
    )

    # 机器人主体。
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 机器人前视相机。
    # 这里使用 USD Camera，挂在 Go2 的 base 上，
    # 后续可以直接通过 env.unwrapped.scene["front_camera"] 读取图像。
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        # 设为 0.0 表示每个仿真步都更新，取图最直接。
        update_period=0.0,
        height=480,
        width=640,
        # 同时提供 RGB 和深度图，后面做导航判断时会更方便。
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # 相机略微前伸并抬高，尽量看到前方走廊与障碍物。
            pos=(0.34, 0.0, 0.12),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

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
    """统一观测配置。

    这里不再只保留“最小状态”，而是直接做成 walk / climb / push_box 三个技能的观测并集。
    统一后的 policy 向量按顺序包含：

    - walk / climb 低层公共项
    - 推箱子高层特有项

    当前总维度为：
    - 低层部分：235 维
    - 推箱子额外部分：16 维
    - 并集总计：251 维

    后续 `envtest_model_use_player.py` 会按 `model_use` 从这 251 维里切出各技能真正需要的部分。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ===== walk / climb 低层公共观测 =====
        # 机身线速度（3）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # 机身角速度（3）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # 重力在机体坐标系下的投影（3）
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 外部设置的低层速度命令（3）
        velocity_commands = ObsTerm(func=mdp.velocity_commands)
        # 相对默认姿态的关节位置（12）
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # 关节速度（12）
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # 上一步关节动作（12）
        actions = ObsTerm(func=mdp.last_action)
        # 结构化高度扫描（187）
        height_scan = ObsTerm(func=mdp.structured_height_scan)

        # ===== push_box 高层额外观测 =====
        # support_box 的位姿；若当前场景没有箱子则返回 0（7）
        box_pose = ObsTerm(func=mdp.box_pose)
        # 机器人在环境坐标系中的位置（3）
        robot_position = ObsTerm(func=mdp.robot_position)
        # 推箱子目标点（3）
        goal_command = ObsTerm(func=mdp.push_goal_command)
        # 推箱子高层上一步裁剪后动作（3）
        push_actions = ObsTerm(func=mdp.push_actions)

        def __post_init__(self):
            # EnvTest 的目标是稳定复现场景和统一接口，因此关闭噪声，并把各项观测拼成向量。
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """复位逻辑。

    reset 顺序是：
    1. reset_scene：把所有对象恢复到默认状态；
    2. reset_base：把机器人固定回初始起点。
    """

    # 把整个 scene 恢复到默认状态。
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
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

    现在采用“单次启动只生成一个固定场景”的方式：
    - scene_id=0~4 对应 5 个固定 case
    - 不再使用隐藏位置
    - 不再在同一个任务里混排 5 个不同 case
    """

    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=8.0)
    # 0~4 对应 5 个固定场景。
    scene_id: int = 3
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 根据 scene_id 直接重建一份 scene，只保留当前 case 需要的障碍物。
        self.scene = build_scene_cfg(self.scene.num_envs, self.scene.env_spacing, self.scene_id)
        # 这组参数只需要保证场景稳定运行，不追求训练环境那种大吞吐。
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.decimation = 4
        self.sim.physics_material = self.scene.terrain.physics_material


class LocomotionEnvTestEnvCfg_Play(LocomotionEnvTestEnvCfg):
    """Play 模式配置。

    Play 模式默认只开 1 个环境，
    通过 `scene_id` 选择当前要查看的固定场景。
    """

    def __post_init__(self):
        super().__post_init__()
        # Play 模式默认只显示一个场景，便于单独检查。
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        # 保留父类中已经设置好的固定 scene_id。


EnvTestEnvCfg = LocomotionEnvTestEnvCfg
EnvTestEnvCfg_Play = LocomotionEnvTestEnvCfg_Play
