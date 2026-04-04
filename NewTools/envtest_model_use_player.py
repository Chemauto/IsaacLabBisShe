# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""在 EnvTest 场景中按 model_use 切换技能策略。

设计目标：
- 场景始终使用 EnvTest；
- EnvTest 的 policy 观测改成多技能观测项的并集；
- 本脚本每步先取统一大观测，再按 `model_use` 切出当前技能真正需要的输入；
- 后续 LLM 只需要改 `model_use`，不需要再关心底层观测拼接逻辑。

当前约定：
- model_use=0: idle，机器人保持静止
- model_use=1: walk（当前 walk JIT 的 232 维观测接口）
- model_use=2: climb（当前 climb checkpoint 的 232 维观测接口）
- model_use=3: push_box（19 维高层观测 + 232 维低层 walk 观测）
- model_use=4: navigation（197 维高层观测 + 232 维低层 walk 观测）
- model_use=5: nav_climb（197 维高层观测 + 232 维低层 walk 观测，可翻高台但不做导航）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run EnvTest with switchable walk / climb / push_box policies.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Template-EnvTest-Go2-Play-v0", help="EnvTest 任务名。")
parser.add_argument("--scene_id", type=int, default=0, help="EnvTest 场景编号，使用 0~4。")
parser.add_argument(
    "--model_use",
    type=int,
    default=0,
    help="技能编号：0=idle，1=walk，2=climb，3=push_box，4=navigation，5=nav_climb。",
)
parser.add_argument(
    "--model_use_file",
    type=str,
    default="/tmp/model_use.txt",
    help="可选文本文件路径；若提供，则每步从文件读取 model_use，便于后续接 LLM。",
)
parser.add_argument("--num_envs", type=int, default=1, help="环境数量。通常建议先用 1。")
parser.add_argument("--lin_vel_x", type=float, default=0.0, help="walk / climb 的前向速度命令。")
parser.add_argument("--lin_vel_y", type=float, default=0.0, help="walk / climb 的侧向速度命令。")
parser.add_argument("--ang_vel_z", type=float, default=0.0, help="walk / climb 的偏航角速度命令。")
parser.add_argument(
    "--velocity_command_file",
    type=str,
    default="/tmp/envtest_velocity_command.txt",
    help="速度指令文件，支持 3 个数：vx vy wz。",
)
parser.add_argument(
    "--goal_command_file",
    type=str,
    default="/tmp/envtest_goal_command.txt",
    help="位置指令文件，支持 3 或 4 个数：x y z [yaw]。push_box 会优先使用它。",
)
parser.add_argument(
    "--start_file",
    type=str,
    default="/tmp/envtest_start.txt",
    help="启动开关文件。写入 1 表示开始执行策略，写入 0 表示待机静止。",
)
parser.add_argument(
    "--status_json_file",
    type=str,
    default="/tmp/envtest_live_status.json",
    help="运行状态 JSON 文件。player 会持续写出 robot_pose / goal / start / model_use 等字段，供外部轮询反馈使用。",
)
parser.add_argument(
    "--reset_file",
    type=str,
    default="/tmp/envtest_reset.txt",
    help="一次性环境重置文件。写入 1 表示请求 reset，player 消费后会自动清回 0。",
)
parser.add_argument(
    "--auto_start",
    action="store_true",
    default=False,
    help="若启用，则脚本启动后立即执行策略；否则默认先静止待机。",
)
parser.add_argument(
    "--enable_front_camera",
    action="store_true",
    default=False,
    help="是否启用 EnvTest 前视相机。默认关闭以减少显存占用。",
)
parser.add_argument(
    "--front_camera_image_file",
    type=str,
    default="/tmp/envtest_front_camera.png",
    help="前视相机当前 RGB 帧输出路径；仅在 --enable_front_camera 时生效。",
)
parser.add_argument(
    "--front_camera_save_interval",
    type=float,
    default=0.2,
    help="前视相机图片写盘间隔，单位秒。",
)
parser.add_argument("--real-time", action="store_true", default=False, help="尽量按实时速度运行。")
parser.add_argument("--max_steps", type=int, default=0, help="最大仿真步数；0 表示一直运行。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 仅在需要时启用前视相机，默认关闭以避免 RTX 渲染占用过多显存。
args_cli.enable_cameras = args_cli.enable_front_camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio.v2 as imageio
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MyProject.tasks  # noqa: F401
import MyProject.tasks.manager_based.EnvTest.mdp as envtest_mdp
from MyProject.tasks.manager_based.EnvTest.config.assets import SCENE_ASSET_SIZE_FALLBACKS
from MyProject.tasks.manager_based.EnvTest.mdp.actions import (
    process_nav_climb_actions,
    process_navigation_actions,
    process_push_actions,
)
from MyProject.tasks.manager_based.EnvTest.mdp.adapters import (
    align_low_level_obs_to_play,
    align_low_level_obs_to_training,
    align_navigation_high_level_obs_to_play,
    align_push_low_level_obs_to_training,
    build_obs_slices,
    print_obs_layout,
    slice_observation,
    validate_required_terms,
)
from MyProject.tasks.manager_based.EnvTest.mdp.skill_specs import (
    CLIMB_LOW_LEVEL_OBS_DIM,
    CLIMB_LOW_LEVEL_OBS_TERMS,
    NAVIGATION_HIGH_LEVEL_OBS_DIM,
    NAVIGATION_HIGH_LEVEL_OBS_TERMS,
    PUSH_HIGH_LEVEL_OBS_DIM,
    PUSH_HIGH_LEVEL_OBS_TERMS,
    WALK_LOW_LEVEL_OBS_DIM,
    WALK_LOW_LEVEL_OBS_TERMS,
)
from MyProject.tasks.manager_based.EnvTest.utils.control_flags import consume_one_shot_value
from MyProject.tasks.manager_based.EnvTest.utils.navigation_bridge import (
    align_navigation_goal_height,
    build_navigation_pose_command,
)
from MyProject.tasks.manager_based.EnvTest.utils.player_runtime import (
    build_default_velocity_commands,
    build_status_snapshot,
    check_obs_dim,
    initialize_control_files,
    load_jit_policy,
    load_policies,
    read_explicit_goal_command_file,
    read_float_vector_file,
    reset_robot_only,
    resolve_model_use,
    resolve_runtime_velocity_commands,
    resolve_start_flag,
)
from MyProject.tasks.manager_based.EnvTest.utils.status_panel import render_status_panel, write_status_json


CLIMB_PLAY_DEFAULT_VELOCITY = (0.75, 0.0, 0.0)
STATUS_PANEL_REFRESH_INTERVAL = 0.2
NAVIGATION_HIGH_LEVEL_DECIMATION = 10
NAVIGATION_MODEL_USES = (4, 5)
STATUS_PLATFORM_ASSETS = (
    ("platform_1", ("left_high_obstacle", "left_low_obstacle")),
    ("platform_2", ("right_high_obstacle", "right_low_obstacle")),
)


@dataclass(frozen=True)
class SkillSpec:
    """单个技能模型的配置。"""

    name: str
    policy_path: str
    obs_terms: tuple[str, ...]
    obs_dim: int
    checkpoint_format: str = "jit"
    actor_hidden_dims: tuple[int, ...] = ()
    activation: str = "elu"


SKILL_REGISTRY: dict[int, SkillSpec] = {
    1: SkillSpec(
        name="walk",
        # 当前可用 walk JIT 接口为 232 维，不再包含 base_lin_vel。
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "TransPolicy", "WalkRoughTransfer.pt"),
        obs_terms=WALK_LOW_LEVEL_OBS_TERMS,
        obs_dim=WALK_LOW_LEVEL_OBS_DIM,
        checkpoint_format="jit",
    ),
    2: SkillSpec(
        name="climb",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "BiShePolicy", "Climbdouble.pt"),
        obs_terms=CLIMB_LOW_LEVEL_OBS_TERMS,
        obs_dim=CLIMB_LOW_LEVEL_OBS_DIM,
        checkpoint_format="rsl_rl_checkpoint",
        actor_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    3: SkillSpec(
        name="push_box",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "PushPolicy", "PushBox.pt"),
        obs_terms=PUSH_HIGH_LEVEL_OBS_TERMS,
        obs_dim=PUSH_HIGH_LEVEL_OBS_DIM,
        checkpoint_format="rsl_rl_checkpoint",
        actor_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    4: SkillSpec(
        name="navigation",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "NaviationPolicy", "NavigationWalk.pt"),
        obs_terms=NAVIGATION_HIGH_LEVEL_OBS_TERMS,
        obs_dim=NAVIGATION_HIGH_LEVEL_OBS_DIM,
        checkpoint_format="rsl_rl_checkpoint",
        actor_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    5: SkillSpec(
        name="nav_climb",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "NaviationPolicy", "NavigationClimb.pt"),
        obs_terms=NAVIGATION_HIGH_LEVEL_OBS_TERMS,
        obs_dim=NAVIGATION_HIGH_LEVEL_OBS_DIM,
        checkpoint_format="rsl_rl_checkpoint",
        actor_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
}

NAVIGATION_WALK_LOW_LEVEL_POLICY_PATH = os.path.join(
    REPO_ROOT, "ModelBackup", "TransPolicy", "WalkFlatHighHeightTransfer.pt"
)
NAVIGATION_CLIMB_LOW_LEVEL_POLICY_PATH = os.path.join(
    REPO_ROOT, "ModelBackup", "TransPolicy", "ClimbNewTransfer.pt"
)
PUSH_LOW_LEVEL_POLICY_PATH = os.path.join(REPO_ROOT, "ModelBackup", "TransPolicy", "WalkFlatLowHeightTransfer.pt")
PUSH_HIGH_LEVEL_DECIMATION = 10


def save_front_camera_rgb_frame(camera, image_file: str) -> None:
    """保存当前 front_camera 的第 0 个环境 RGB 图像。"""

    if camera is None or not image_file:
        return
    if "rgb" not in camera.data.output:
        return

    rgb_image = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb_image.shape[-1] > 3:
        rgb_image = rgb_image[..., :3]

    image_dir = os.path.dirname(image_file)
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)

    temp_image_file = f"{image_file}.tmp"
    imageio.imwrite(temp_image_file, rgb_image, format="png")
    os.replace(temp_image_file, image_file)


def main():
    """主入口。"""

    if not 0 <= args_cli.scene_id <= 4:
        raise ValueError("--scene_id must be in [0, 4].")
    if args_cli.model_use != 0 and args_cli.model_use not in SKILL_REGISTRY:
        raise ValueError(f"--model_use must be one of {[0, *sorted(SKILL_REGISTRY)]}.")

    initialize_control_files(args_cli)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "scene_id"):
        env_cfg.scene_id = args_cli.scene_id
    if not args_cli.enable_front_camera and hasattr(env_cfg.scene, "front_camera"):
        env_cfg.scene.front_camera = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.scene["robot"].data.joint_pos.shape[1]
    expected_action_shape = (num_envs, action_dim)
    front_camera = env.unwrapped.scene["front_camera"] if args_cli.enable_front_camera else None

    policies = load_policies(SKILL_REGISTRY, device)
    navigation_walk_low_level_policy = load_jit_policy(
        NAVIGATION_WALK_LOW_LEVEL_POLICY_PATH, "NavigationWalk low-level", device
    )
    navigation_climb_low_level_policy = load_jit_policy(
        NAVIGATION_CLIMB_LOW_LEVEL_POLICY_PATH, "NavClimb low-level", device
    )
    push_low_level_policy = load_jit_policy(PUSH_LOW_LEVEL_POLICY_PATH, "Push low-level", device)

    term_slices = build_obs_slices(env.unwrapped)
    validate_required_terms(term_slices)
    print_obs_layout(term_slices)
    unified_obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    print("[INFO] Control files have been reset at startup.")
    print(f"[INFO] model_use file: {args_cli.model_use_file}")
    print(f"[INFO] velocity command file: {args_cli.velocity_command_file}")
    print(f"[INFO] goal command file: {args_cli.goal_command_file}")
    print(f"[INFO] start file: {args_cli.start_file} (auto_start={args_cli.auto_start})")
    print(f"[INFO] reset file: {args_cli.reset_file}")

    base_velocity_commands = build_default_velocity_commands(
        args_cli.lin_vel_x,
        args_cli.lin_vel_y,
        args_cli.ang_vel_z,
        num_envs,
        device,
    )
    zero_pose_command = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
    zero_push_goal_command = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
    zero_push_actions = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    zero_actions = torch.zeros(expected_action_shape, dtype=torch.float32, device=device)
    push_last_processed_actions = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    push_current_processed_actions = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    push_low_level_last_actions = torch.zeros(expected_action_shape, dtype=torch.float32, device=device)
    push_high_level_counter = 0
    push_entry_settle_steps = 0
    navigation_current_processed_actions = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    navigation_low_level_last_actions = torch.zeros(expected_action_shape, dtype=torch.float32, device=device)
    navigation_high_level_counter = 0

    current_model_use = args_cli.model_use
    previous_model_use = None
    previous_start_flag = None
    previous_logged_push_goal = None
    last_status_panel_time = 0.0
    last_front_camera_save_time = 0.0
    dt = env.unwrapped.step_dt
    step_count = 0

    while simulation_app.is_running():
        start_time = time.time()
        current_model_use = resolve_model_use(args_cli.model_use_file, SKILL_REGISTRY, current_model_use)
        current_velocity_commands = read_float_vector_file(args_cli.velocity_command_file, 3, base_velocity_commands)
        current_velocity_commands = resolve_runtime_velocity_commands(
            args_cli.scene_id,
            current_model_use,
            current_velocity_commands,
            CLIMB_PLAY_DEFAULT_VELOCITY,
        )
        explicit_pose_command = read_explicit_goal_command_file(args_cli.goal_command_file, zero_push_goal_command)
        robot = env.unwrapped.scene["robot"]
        if explicit_pose_command is None:
            navigation_goal_command = None
        else:
            default_root_height_e = robot.data.default_root_state[:, 2:3] - env.unwrapped.scene.env_origins[:, 2:3]
            navigation_goal_command = align_navigation_goal_height(explicit_pose_command, default_root_height_e)
        current_goal_for_display = (
            navigation_goal_command if current_model_use in NAVIGATION_MODEL_USES else explicit_pose_command
        )
        current_pose_for_display = None if current_model_use in NAVIGATION_MODEL_USES else explicit_pose_command
        current_velocity_for_display = current_velocity_commands
        start_flag = resolve_start_flag(args_cli.auto_start, args_cli.start_file)
        current_policy_obs_dim = SKILL_REGISTRY[current_model_use].obs_dim if start_flag and current_model_use in SKILL_REGISTRY else None
        reset_token = consume_one_shot_value(
            args_cli.reset_file,
            accepted_tokens=("1", "2", "true", "on", "reset"),
        )
        reset_mode = 2 if reset_token == "2" else (1 if reset_token is not None else 0)
        previous_start_flag_value = previous_start_flag
        previous_model_use_value = previous_model_use
        entering_push_execution = (
            start_flag
            and current_model_use == 3
            and (previous_start_flag_value is not True or previous_model_use_value != 3)
        )
        if reset_mode != 0:
            entering_push_execution = False

        if start_flag != previous_start_flag:
            state_text = "RUN" if start_flag else "IDLE"
            print(f"[INFO] 当前执行状态: {state_text}")
            previous_start_flag = start_flag

        if current_model_use != previous_model_use:
            if current_model_use == 0:
                push_last_processed_actions.zero_()
                push_current_processed_actions.zero_()
                push_low_level_last_actions.zero_()
                push_high_level_counter = 0
                push_entry_settle_steps = 0
                navigation_current_processed_actions.zero_()
                navigation_low_level_last_actions.zero_()
                navigation_high_level_counter = 0
                print(f"[INFO] 切换技能: model_use=0, skill=idle, scene_id={args_cli.scene_id}")
            else:
                push_last_processed_actions.zero_()
                push_current_processed_actions.zero_()
                push_low_level_last_actions.zero_()
                push_high_level_counter = 0
                push_entry_settle_steps = 0
                navigation_current_processed_actions.zero_()
                navigation_low_level_last_actions.zero_()
                navigation_high_level_counter = 0
                print(
                    f"[INFO] 切换技能: model_use={current_model_use}, "
                    f"skill={SKILL_REGISTRY[current_model_use].name}, scene_id={args_cli.scene_id}"
                )
            previous_model_use = current_model_use

        if reset_mode == 1:
            with torch.inference_mode():
                env.reset()
            push_last_processed_actions.zero_()
            push_current_processed_actions.zero_()
            push_low_level_last_actions.zero_()
            push_high_level_counter = 0
            push_entry_settle_steps = 1 if start_flag and current_model_use == 3 else 0
            navigation_current_processed_actions.zero_()
            navigation_low_level_last_actions.zero_()
            navigation_high_level_counter = 0
            previous_logged_push_goal = None
            print("[INFO] 收到 reset 指令，环境已重置。")

        if reset_mode == 2:
            with torch.inference_mode():
                reset_robot_only(env.unwrapped, robot)
            push_last_processed_actions.zero_()
            push_current_processed_actions.zero_()
            push_low_level_last_actions.zero_()
            push_high_level_counter = 0
            push_entry_settle_steps = 1 if start_flag and current_model_use == 3 else 0
            navigation_current_processed_actions.zero_()
            navigation_low_level_last_actions.zero_()
            navigation_high_level_counter = 0
            previous_logged_push_goal = None
            print("[INFO] 收到 reset=2 指令，机器人已重置，环境保持不变。")

        if entering_push_execution:
            with torch.inference_mode():
                env.reset()
            push_last_processed_actions.zero_()
            push_current_processed_actions.zero_()
            push_low_level_last_actions.zero_()
            push_high_level_counter = 0
            push_entry_settle_steps = 1
            navigation_current_processed_actions.zero_()
            navigation_low_level_last_actions.zero_()
            navigation_high_level_counter = 0
            previous_logged_push_goal = None
            print("[INFO] 进入 push_box 执行态，环境已重置并预留 1 个稳定步。")

        if navigation_goal_command is None:
            navigation_pose_command = zero_pose_command
        else:
            robot_pos_e = robot.data.root_pos_w[:, :3] - env.unwrapped.scene.env_origins
            navigation_pose_command = build_navigation_pose_command(
                robot_pos_e,
                robot.data.root_quat_w,
                navigation_goal_command,
            )
            if current_model_use in NAVIGATION_MODEL_USES:
                current_pose_for_display = navigation_pose_command

        with torch.inference_mode():
            # 第一步：待机阶段。机器人保持静止，但仍然持续读取 model_use、速度指令和位置指令文件。
            if push_entry_settle_steps > 0:
                env.unwrapped.set_runtime_observation_buffers(
                    velocity_commands=current_velocity_commands,
                    pose_command=zero_pose_command,
                    push_goal_command=zero_push_goal_command,
                    push_actions=zero_push_actions,
                )
                policy_obs = env.unwrapped.observation_manager.compute_group("policy")
                actions = zero_actions
                push_entry_settle_steps -= 1
            elif not start_flag or current_model_use == 0:
                push_last_processed_actions.zero_()
                push_current_processed_actions.zero_()
                push_low_level_last_actions.zero_()
                push_high_level_counter = 0
                push_entry_settle_steps = 0
                navigation_current_processed_actions.zero_()
                navigation_low_level_last_actions.zero_()
                navigation_high_level_counter = 0
                env.unwrapped.set_runtime_observation_buffers(
                    velocity_commands=current_velocity_commands,
                    pose_command=zero_pose_command,
                    push_goal_command=zero_push_goal_command,
                    push_actions=zero_push_actions,
                )
                policy_obs = env.unwrapped.observation_manager.compute_group("policy")
                actions = zero_actions
            elif current_model_use in (1, 2):
                env.unwrapped.set_runtime_observation_buffers(
                    velocity_commands=current_velocity_commands,
                    pose_command=zero_pose_command,
                    push_goal_command=zero_push_goal_command,
                    push_actions=zero_push_actions,
                )
                unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                policy_obs = slice_observation(unified_obs, term_slices, SKILL_REGISTRY[current_model_use].obs_terms)
                policy_obs = align_low_level_obs_to_play(
                    policy_obs,
                    term_slices,
                    SKILL_REGISTRY[current_model_use].obs_terms,
                )
                check_obs_dim(SKILL_REGISTRY, current_model_use, policy_obs)
                actions = policies[current_model_use](policy_obs)
            elif current_model_use in NAVIGATION_MODEL_USES:
                if navigation_goal_command is None:
                    navigation_current_processed_actions.zero_()
                    navigation_high_level_counter = 0
                elif navigation_high_level_counter == 0:
                    env.unwrapped.set_runtime_observation_buffers(
                        velocity_commands=current_velocity_commands,
                        pose_command=navigation_pose_command,
                        push_goal_command=zero_push_goal_command,
                        push_actions=navigation_current_processed_actions,
                    )
                    unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                    navigation_high_obs = slice_observation(
                        unified_obs, term_slices, SKILL_REGISTRY[current_model_use].obs_terms
                    )
                    navigation_high_obs = align_navigation_high_level_obs_to_play(navigation_high_obs)
                    check_obs_dim(SKILL_REGISTRY, current_model_use, navigation_high_obs)
                    navigation_raw_actions = policies[current_model_use](navigation_high_obs)
                    if navigation_raw_actions.shape != navigation_current_processed_actions.shape:
                        raise RuntimeError(
                            f"{SKILL_REGISTRY[current_model_use].name} high-level action dim mismatch. "
                            f"Expected {tuple(navigation_current_processed_actions.shape)}, "
                            f"got {tuple(navigation_raw_actions.shape)}."
                        )
                    if current_model_use == 5:
                        navigation_current_processed_actions.copy_(process_nav_climb_actions(navigation_raw_actions))
                    else:
                        navigation_current_processed_actions.copy_(process_navigation_actions(navigation_raw_actions))

                current_velocity_for_display = navigation_current_processed_actions
                env.unwrapped.set_runtime_observation_buffers(
                    velocity_commands=navigation_current_processed_actions,
                    pose_command=navigation_pose_command,
                    push_goal_command=zero_push_goal_command,
                    push_actions=navigation_current_processed_actions,
                )
                unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                policy_obs = slice_observation(unified_obs, term_slices, WALK_LOW_LEVEL_OBS_TERMS)
                policy_obs = align_low_level_obs_to_training(
                    env.unwrapped,
                    policy_obs,
                    term_slices,
                    WALK_LOW_LEVEL_OBS_TERMS,
                    navigation_low_level_last_actions,
                    include_support_box=True,
                )
                if policy_obs.shape[1] != WALK_LOW_LEVEL_OBS_DIM:
                    raise RuntimeError(
                        f"{SKILL_REGISTRY[current_model_use].name} low-level observation dim mismatch. "
                        f"Expected {WALK_LOW_LEVEL_OBS_DIM}, got {policy_obs.shape[1]}."
                    )
                if current_model_use == 5:
                    actions = navigation_climb_low_level_policy(policy_obs)
                else:
                    actions = navigation_walk_low_level_policy(policy_obs)
                navigation_low_level_last_actions.copy_(actions)
                navigation_high_level_counter = (navigation_high_level_counter + 1) % NAVIGATION_HIGH_LEVEL_DECIMATION
            else:
                scene_push_goal = envtest_mdp.compute_push_goal_from_scene(env.unwrapped)
                push_goal_command = scene_push_goal if explicit_pose_command is None else explicit_pose_command
                current_goal_for_display = push_goal_command
                if push_high_level_counter == 0:
                    # 对齐 PushBoxTest：高层策略每 10 个 EnvTest 步更新一次。
                    env.unwrapped.set_runtime_observation_buffers(
                        velocity_commands=current_velocity_commands,
                        pose_command=zero_pose_command,
                        push_goal_command=push_goal_command,
                        push_actions=push_last_processed_actions,
                    )
                    unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                    push_high_obs = slice_observation(unified_obs, term_slices, SKILL_REGISTRY[3].obs_terms)
                    check_obs_dim(SKILL_REGISTRY, 3, push_high_obs)

                    current_push_goal_tuple = tuple(round(v, 4) for v in push_goal_command[0].detach().cpu().tolist())
                    if current_push_goal_tuple != previous_logged_push_goal:
                        goal_debug = envtest_mdp.get_push_goal_debug_info(env.unwrapped)
                        box_pose_tuple = tuple(round(v, 4) for v in goal_debug["box_position"][0].detach().cpu().tolist())
                        box_size_tuple = tuple(round(v, 4) for v in goal_debug["box_size"])
                        obstacle_name = goal_debug["selected_obstacle_names"][0]
                        obstacle_pos_tuple = tuple(
                            round(v, 4) for v in goal_debug["selected_obstacle_positions"][0].detach().cpu().tolist()
                        )
                        obstacle_size_tuple = tuple(
                            round(v, 4) for v in goal_debug["selected_obstacle_sizes"][0].detach().cpu().tolist()
                        )
                        print(
                            "[INFO] push_box auto goal:"
                            f" box_pos={box_pose_tuple},"
                            f" box_size={box_size_tuple},"
                            f" obstacle={obstacle_name},"
                            f" obstacle_pos={obstacle_pos_tuple},"
                            f" obstacle_size={obstacle_size_tuple},"
                            f" goal={current_push_goal_tuple}"
                        )
                        previous_logged_push_goal = current_push_goal_tuple

                    push_raw_actions = policies[3](push_high_obs)
                    if push_raw_actions.shape != push_last_processed_actions.shape:
                        raise RuntimeError(
                            f"Push high-level action dim mismatch. "
                            f"Expected {tuple(push_last_processed_actions.shape)}, got {tuple(push_raw_actions.shape)}."
                        )

                    push_current_processed_actions.copy_(process_push_actions(push_raw_actions))
                    push_last_processed_actions.copy_(push_current_processed_actions)

                # 再把裁剪后的高层动作写入低层速度命令槽位，供 rough-walk 低层策略执行。
                env.unwrapped.set_runtime_observation_buffers(
                    velocity_commands=push_current_processed_actions,
                    pose_command=zero_pose_command,
                    push_goal_command=push_goal_command,
                    push_actions=push_current_processed_actions,
                )
                unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                policy_obs = slice_observation(unified_obs, term_slices, WALK_LOW_LEVEL_OBS_TERMS)
                policy_obs = align_push_low_level_obs_to_training(
                    env.unwrapped, policy_obs, term_slices, push_low_level_last_actions
                )
                if policy_obs.shape[1] != WALK_LOW_LEVEL_OBS_DIM:
                    raise RuntimeError(
                        f"Push low-level observation dim mismatch. "
                        f"Expected {WALK_LOW_LEVEL_OBS_DIM}, got {policy_obs.shape[1]}."
                    )
                actions = push_low_level_policy(policy_obs)
                push_low_level_last_actions.copy_(actions)
                push_high_level_counter = (push_high_level_counter + 1) % PUSH_HIGH_LEVEL_DECIMATION

            if actions.shape != expected_action_shape:
                raise RuntimeError(
                    f"Action dim mismatch for model_use={current_model_use}. "
                    f"Expected {expected_action_shape}, got {tuple(actions.shape)}."
                )

            env.step(actions)

        step_count += 1
        if (
            front_camera is not None
            and args_cli.front_camera_image_file
            and time.time() - last_front_camera_save_time >= args_cli.front_camera_save_interval
        ):
            save_front_camera_rgb_frame(front_camera, args_cli.front_camera_image_file)
            last_front_camera_save_time = time.time()

        if step_count == 1:
            print(f"[INFO] EnvTest unified observation dim: {unified_obs_dim}")
            if start_flag and current_model_use in SKILL_REGISTRY:
                print(f"[INFO] 当前策略输入维度: {current_policy_obs_dim}")
            else:
                print("[INFO] 当前处于待机阶段，机器人保持静止。")
            print(f"[INFO] 当前策略输出维度: {actions.shape[1]}")

        if time.time() - last_status_panel_time >= STATUS_PANEL_REFRESH_INTERVAL:
            status_snapshot = build_status_snapshot(
                env.unwrapped,
                model_use=current_model_use,
                start_flag=start_flag,
                unified_obs_dim=unified_obs_dim,
                policy_obs_dim=current_policy_obs_dim,
                velocity_command=current_velocity_for_display,
                pose_command=current_pose_for_display,
                goal_command=current_goal_for_display,
                scene_id=args_cli.scene_id,
                skill_registry=SKILL_REGISTRY,
                platform_assets=STATUS_PLATFORM_ASSETS,
                fallback_sizes=SCENE_ASSET_SIZE_FALLBACKS,
            )
            render_status_panel(status_snapshot)
            write_status_json(status_snapshot, args_cli.status_json_file)
            last_status_panel_time = time.time()

        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
