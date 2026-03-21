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
- model_use=1: walk（按 walk_rough 的 235 维观测接口）
- model_use=2: climb（235 维观测接口）
- model_use=3: push_box（23 维高层观测 + 235 维低层 walk 观测）
- model_use=4: navigation（197 维高层观测 + 235 维低层 walk 观测）
"""

from __future__ import annotations

import argparse
import math
import os
import re
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
parser.add_argument("--model_use", type=int, default=0, help="技能编号：0=idle，1=walk，2=climb，3=push_box，4=navigation。")
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
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MyProject.tasks  # noqa: F401
import MyProject.tasks.manager_based.EnvTest.mdp as envtest_mdp
from MyProject.tasks.manager_based.EnvTest.observation_schema import (
    NAVIGATION_HIGH_LEVEL_OBS_DIM,
    NAVIGATION_HIGH_LEVEL_OBS_TERMS,
)
from MyProject.tasks.manager_based.EnvTest.scene_layout import BOX_SIZE, HIGH_OBSTACLE_SIZE, LOW_OBSTACLE_SIZE
from NewTools.envtest_control_flags import consume_one_shot_value
from NewTools.envtest_navigation_bridge import align_navigation_goal_height, build_navigation_pose_command
from NewTools.envtest_status_panel import AssetStatus, StatusSnapshot, render_status_block


LOW_LEVEL_OBS_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
)
PUSH_HIGH_LEVEL_OBS_TERMS = (
    "base_lin_vel",
    "projected_gravity",
    "box_pose",
    "robot_position",
    "goal_command",
    "push_actions",
)
LOW_LEVEL_OBS_DIM = 235
CLIMB_PLAY_DEFAULT_VELOCITY = (0.75, 0.0, 0.0)
STATUS_PANEL_REFRESH_INTERVAL = 0.2
NAVIGATION_HIGH_LEVEL_DECIMATION = 10
STATUS_PLATFORM_ASSETS = (
    ("platform_1", ("left_high_obstacle", "left_low_obstacle")),
    ("platform_2", ("right_high_obstacle", "right_low_obstacle")),
)
SCENE_ASSET_SIZE_FALLBACKS = {
    "left_low_obstacle": LOW_OBSTACLE_SIZE,
    "right_low_obstacle": LOW_OBSTACLE_SIZE,
    "left_high_obstacle": HIGH_OBSTACLE_SIZE,
    "right_high_obstacle": HIGH_OBSTACLE_SIZE,
    "support_box": BOX_SIZE,
}


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
        # walk 这里明确对齐 walk_rough 的 235 维接口，直接使用 rough 低层策略。
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "TransPolicy", "WalkRoughNewTransfer.pt"),
        obs_terms=LOW_LEVEL_OBS_TERMS,
        obs_dim=LOW_LEVEL_OBS_DIM,
        checkpoint_format="jit",
    ),
    2: SkillSpec(
        name="climb",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "BiShePolicy", "BiSheClimbPolicy.pt"),
        obs_terms=LOW_LEVEL_OBS_TERMS,
        obs_dim=LOW_LEVEL_OBS_DIM,
        checkpoint_format="rsl_rl_checkpoint",
        actor_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    3: SkillSpec(
        name="push_box",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "PushPolicy", "PushNewNoHeightRestrain.pt"),
        obs_terms=PUSH_HIGH_LEVEL_OBS_TERMS,
        obs_dim=23,
        checkpoint_format="rsl_rl_checkpoint",
        actor_hidden_dims=(256, 128),
        activation="elu",
    ),
    4: SkillSpec(
        name="navigation",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "NaviationPolicy", "NavigationBishe.pt"),
        obs_terms=NAVIGATION_HIGH_LEVEL_OBS_TERMS,
        obs_dim=NAVIGATION_HIGH_LEVEL_OBS_DIM,
        checkpoint_format="rsl_rl_checkpoint",
        actor_hidden_dims=(128, 128),
        activation="elu",
    ),
}

PUSH_LOW_LEVEL_POLICY_PATH = os.path.join(REPO_ROOT, "ModelBackup", "TransPolicy", "WalkRoughNewTransfer.pt")
PUSH_HIGH_LEVEL_DECIMATION = 10


def _make_activation(name: str) -> nn.Module:
    """根据名称创建激活层。"""

    activation_name = name.lower()
    if activation_name == "elu":
        return nn.ELU()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation for checkpoint policy loading: {name}")


def _load_rsl_rl_actor_from_checkpoint(spec: SkillSpec, device: torch.device | str) -> nn.Module:
    """从 RSL-RL checkpoint 中恢复 actor，用于推理。"""

    checkpoint = torch.load(spec.policy_path, map_location=device)
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {spec.policy_path}")

    actor_state_dict = {
        key[len("actor."):]: value for key, value in model_state_dict.items() if key.startswith("actor.")
    }
    if not actor_state_dict:
        raise KeyError(f"Checkpoint does not contain actor weights: {spec.policy_path}")

    bias_layer_indices = sorted(int(key.split(".")[0]) for key in actor_state_dict if key.endswith(".bias"))
    if not bias_layer_indices:
        raise KeyError(f"Checkpoint actor is missing bias tensors: {spec.policy_path}")
    output_dim = actor_state_dict[f"{bias_layer_indices[-1]}.bias"].shape[0]
    layer_dims = [spec.obs_dim, *spec.actor_hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for layer_index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if layer_index < len(layer_dims) - 2:
            layers.append(_make_activation(spec.activation))

    actor = nn.Sequential(*layers)
    actor.load_state_dict(actor_state_dict)
    actor.to(device)
    actor.eval()
    return actor


def _load_policies(device: torch.device | str) -> dict[int, nn.Module]:
    """加载所有高层技能策略。"""

    policies: dict[int, nn.Module] = {}
    for model_use, spec in SKILL_REGISTRY.items():
        if not os.path.isfile(spec.policy_path):
            raise FileNotFoundError(f"Policy file not found: {spec.policy_path}")
        if spec.checkpoint_format == "jit":
            policies[model_use] = torch.jit.load(spec.policy_path, map_location=device).eval()
        elif spec.checkpoint_format == "rsl_rl_checkpoint":
            policies[model_use] = _load_rsl_rl_actor_from_checkpoint(spec, device)
        else:
            raise ValueError(f"Unsupported checkpoint format '{spec.checkpoint_format}' for skill '{spec.name}'.")
    return policies


def _load_push_low_level_policy(device: torch.device | str) -> torch.jit.ScriptModule:
    """加载推箱子技能依赖的低层 rough-walk 策略。"""

    if not os.path.isfile(PUSH_LOW_LEVEL_POLICY_PATH):
        raise FileNotFoundError(f"Push low-level policy file not found: {PUSH_LOW_LEVEL_POLICY_PATH}")
    return torch.jit.load(PUSH_LOW_LEVEL_POLICY_PATH, map_location=device).eval()


def _resolve_model_use(current_model_use: int) -> int:
    """从 CLI 或外部文件中决定当前 model_use。"""

    if args_cli.model_use_file and os.path.isfile(args_cli.model_use_file):
        try:
            with open(args_cli.model_use_file, "r", encoding="utf-8") as file:
                file_value = int(file.read().strip())
        except (OSError, ValueError):
            file_value = current_model_use
        if file_value == 0 or file_value in SKILL_REGISTRY:
            return file_value
    return current_model_use


def _read_float_vector_file(file_path: str, expected_dim: int, fallback: torch.Tensor) -> torch.Tensor:
    """从文本文件里读取定长浮点向量。

    支持：
    - `0.6 0.0 0.0`
    - `0.6,0.0,0.0`
    - `vx=0.6 vy=0.0 wz=0.0`
    """

    if not file_path or not os.path.isfile(file_path):
        return fallback

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except OSError:
        return fallback

    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(values) < expected_dim:
        return fallback

    try:
        parsed = [float(value) for value in values[:expected_dim]]
    except ValueError:
        return fallback

    vector = torch.tensor(parsed, dtype=fallback.dtype, device=fallback.device)
    return vector.unsqueeze(0).repeat(fallback.shape[0], 1)


def _resolve_start_flag() -> bool:
    """决定当前是否进入执行阶段。"""

    if args_cli.auto_start:
        return True
    if not args_cli.start_file or not os.path.isfile(args_cli.start_file):
        return False

    try:
        with open(args_cli.start_file, "r", encoding="utf-8") as file:
            text = file.read().strip().lower()
    except OSError:
        return False

    return text in ("1", "true", "run", "start", "yes", "y")


def _ensure_parent_dir(file_path: str):
    """确保控制文件父目录存在。"""

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_text_file(file_path: str, content: str):
    """写入控制文件。"""

    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content.strip() + "\n")


def _initialize_control_files():
    """启动时重置控制文件，避免读取到上一次运行残留的旧值。"""

    _write_text_file(args_cli.model_use_file, str(args_cli.model_use))
    _write_text_file(
        args_cli.velocity_command_file,
        f"{args_cli.lin_vel_x} {args_cli.lin_vel_y} {args_cli.ang_vel_z}",
    )
    # push_box 默认使用场景自动推导的目标点，不直接写死成 0 0 0。
    _write_text_file(args_cli.goal_command_file, "auto")
    _write_text_file(args_cli.start_file, "1" if args_cli.auto_start else "0")
    _write_text_file(args_cli.reset_file, "0")


def _read_goal_command_file(file_path: str, fallback: torch.Tensor) -> torch.Tensor:
    """读取 push_box 目标点。

    行为：
    - 文件不存在：使用场景自动目标
    - 内容是 `auto` / `scene` / 空：使用场景自动目标
    - 内容是三个数：使用显式目标点，yaw 默认补 0
    - 内容是四个数：使用显式目标点和目标 yaw
    """

    explicit_goal = _read_explicit_goal_command_file(file_path, fallback)
    return fallback if explicit_goal is None else explicit_goal


def _parse_goal_command_text(text: str) -> list[float] | None:
    """把目标点文本解析成 `[x, y, z, yaw]`。"""

    normalized_text = text.strip().lower()
    if normalized_text in ("", "auto", "scene", "default"):
        return None

    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized_text)
    if len(values) < 3:
        return None

    try:
        parsed = [float(value) for value in values[:4]]
    except ValueError:
        return None

    if len(parsed) == 3:
        parsed.append(0.0)
    elif len(parsed) < 4:
        return None
    return parsed


def _read_explicit_goal_command_file(file_path: str, template: torch.Tensor) -> torch.Tensor | None:
    """只读取显式 pose command；`auto` / 空 / 缺失都返回 `None`。"""

    if not file_path or not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except OSError:
        return None

    parsed = _parse_goal_command_text(text)
    if parsed is None:
        return None

    vector = torch.tensor(parsed, dtype=template.dtype, device=template.device)
    return vector.unsqueeze(0).repeat(template.shape[0], 1)


def _build_default_velocity_commands(num_envs: int, device: torch.device | str) -> torch.Tensor:
    """构造 walk / climb 使用的默认速度命令。"""

    single_command = torch.tensor(
        [args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z], dtype=torch.float32, device=device
    )
    return single_command.unsqueeze(0).repeat(num_envs, 1)


def _resolve_runtime_velocity_commands(model_use: int, velocity_commands: torch.Tensor) -> torch.Tensor:
    """对齐各技能 Play 配置的默认速度命令。"""

    if args_cli.scene_id == 2 and model_use == 2 and torch.allclose(velocity_commands, torch.zeros_like(velocity_commands)):
        climb_velocity = torch.tensor(
            CLIMB_PLAY_DEFAULT_VELOCITY, dtype=velocity_commands.dtype, device=velocity_commands.device
        )
        return climb_velocity.unsqueeze(0).repeat(velocity_commands.shape[0], 1)
    return velocity_commands


def _default_push_goal(env) -> torch.Tensor:
    """按当前场景中的箱子/障碍尺寸自动生成 push goal。"""

    return envtest_mdp.compute_push_goal_from_scene(env)


def _process_push_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """对齐当前 `PushBoxTest` 配置：高层动作不再额外缩放或裁剪。"""

    return raw_actions


def _process_navigation_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """对齐 NavigationTest：高层动作按训练时的 `[-1, 1]` 范围裁剪。"""

    return torch.clamp(raw_actions, min=-1.0, max=1.0)


def _align_low_level_obs_to_training(
    env,
    policy_obs: torch.Tensor,
    term_slices: dict[str, slice],
    last_actions: torch.Tensor,
    include_support_box: bool = True,
) -> torch.Tensor:
    """把切片后的低层观测对齐到 rough-walk 训练时的 ObservationManager 行为。"""

    aligned_obs = policy_obs.clone()
    local_slices: dict[str, slice] = {}
    start = 0
    for term_name in LOW_LEVEL_OBS_TERMS:
        term_slice = term_slices[term_name]
        term_dim = term_slice.stop - term_slice.start
        local_slices[term_name] = slice(start, start + term_dim)
        start += term_dim

    if hasattr(env, "episode_length_buf"):
        last_actions[env.episode_length_buf == 0, :] = 0.0

    aligned_obs[:, local_slices["actions"]] = last_actions
    if include_support_box:
        aligned_obs[:, local_slices["height_scan"]] = policy_obs[:, local_slices["height_scan"]]
    else:
        aligned_obs[:, local_slices["height_scan"]] = envtest_mdp.height_scan_without_box(env)
    aligned_obs[:, local_slices["base_lin_vel"]] += torch.empty_like(
        aligned_obs[:, local_slices["base_lin_vel"]]
    ).uniform_(-0.1, 0.1)
    aligned_obs[:, local_slices["base_ang_vel"]] += torch.empty_like(
        aligned_obs[:, local_slices["base_ang_vel"]]
    ).uniform_(-0.2, 0.2)
    aligned_obs[:, local_slices["projected_gravity"]] += torch.empty_like(
        aligned_obs[:, local_slices["projected_gravity"]]
    ).uniform_(-0.05, 0.05)
    aligned_obs[:, local_slices["joint_pos"]] += torch.empty_like(
        aligned_obs[:, local_slices["joint_pos"]]
    ).uniform_(-0.01, 0.01)
    aligned_obs[:, local_slices["joint_vel"]] += torch.empty_like(
        aligned_obs[:, local_slices["joint_vel"]]
    ).uniform_(-1.5, 1.5)
    aligned_obs[:, local_slices["height_scan"]] += torch.empty_like(
        aligned_obs[:, local_slices["height_scan"]]
    ).uniform_(-0.1, 0.1)
    aligned_obs[:, local_slices["height_scan"]] = torch.clamp(
        aligned_obs[:, local_slices["height_scan"]], min=-1.0, max=1.0
    )
    return aligned_obs


def _align_push_low_level_obs_to_training(
    env,
    policy_obs: torch.Tensor,
    term_slices: dict[str, slice],
    push_low_level_last_actions: torch.Tensor,
) -> torch.Tensor:
    """Push-box 分支复用 rough-walk 低层观测，但要屏蔽 support_box。"""

    return _align_low_level_obs_to_training(
        env,
        policy_obs,
        term_slices,
        push_low_level_last_actions,
        include_support_box=False,
    )


def _align_navigation_high_level_obs_to_play(policy_obs: torch.Tensor) -> torch.Tensor:
    """对齐 NaviationBiSheEnvCfg_Play：不加噪声，只保留裁剪。"""

    aligned_obs = policy_obs.clone()
    local_slices = {"height_scan": slice(10, NAVIGATION_HIGH_LEVEL_OBS_DIM)}
    aligned_obs[:, local_slices["height_scan"]] = torch.clamp(aligned_obs[:, local_slices["height_scan"]], -1.0, 1.0)
    return aligned_obs


def _build_obs_slices(env, group_name: str = "policy") -> dict[str, slice]:
    """把统一观测中的每个 term 映射到切片区间。"""

    obs_manager = env.observation_manager
    term_names = obs_manager.active_terms[group_name]
    term_dims = obs_manager.group_obs_term_dim[group_name]

    slices: dict[str, slice] = {}
    start = 0
    for term_name, term_dim in zip(term_names, term_dims):
        flat_dim = math.prod(term_dim)
        slices[term_name] = slice(start, start + flat_dim)
        start += flat_dim
    return slices


def _build_local_obs_slices(term_names: tuple[str, ...], term_slices: dict[str, slice]) -> dict[str, slice]:
    """为切片后的局部观测重新建立 term 对应区间。"""

    local_slices: dict[str, slice] = {}
    start = 0
    for term_name in term_names:
        term_slice = term_slices[term_name]
        term_dim = term_slice.stop - term_slice.start
        local_slices[term_name] = slice(start, start + term_dim)
        start += term_dim
    return local_slices


def _align_low_level_obs_to_play(policy_obs: torch.Tensor, term_slices: dict[str, slice]) -> torch.Tensor:
    """对齐 walk/climb Play 模式下的低层观测后处理。"""

    aligned_obs = policy_obs.clone()
    local_slices = _build_local_obs_slices(LOW_LEVEL_OBS_TERMS, term_slices)
    aligned_obs[:, local_slices["height_scan"]] = torch.clamp(
        aligned_obs[:, local_slices["height_scan"]], min=-1.0, max=1.0
    )
    return aligned_obs


def _slice_observation(unified_obs: torch.Tensor, term_slices: dict[str, slice], term_names: tuple[str, ...]) -> torch.Tensor:
    """按给定 term 名称顺序，从统一观测中拼出某个技能需要的输入。"""

    parts = [unified_obs[:, term_slices[term_name]] for term_name in term_names]
    return torch.cat(parts, dim=-1)


def _validate_required_terms(term_slices: dict[str, slice]):
    """确保统一观测中已经包含各技能需要的全部 term。"""

    required_terms = set(LOW_LEVEL_OBS_TERMS) | set(PUSH_HIGH_LEVEL_OBS_TERMS) | set(NAVIGATION_HIGH_LEVEL_OBS_TERMS)
    missing_terms = sorted(required_terms.difference(term_slices))
    if missing_terms:
        raise RuntimeError(f"EnvTest unified observation is missing terms: {missing_terms}")


def _print_obs_layout(term_slices: dict[str, slice]):
    """打印统一观测的 term 切片布局，便于后续排查。"""

    print("[INFO] EnvTest unified observation layout:")
    for term_name, term_slice in term_slices.items():
        print(f"  - {term_name:<18} -> [{term_slice.start:>3}, {term_slice.stop:>3})")


def _check_obs_dim(model_use: int, obs: torch.Tensor):
    """检查切片后的观测维度是否和策略要求一致。"""

    expected_dim = SKILL_REGISTRY[model_use].obs_dim
    if obs.shape[1] != expected_dim:
        raise RuntimeError(
            f"Observation dim mismatch for model_use={model_use}. "
            f"Expected {expected_dim}, got {obs.shape[1]}."
        )


def _tensor_row_to_tuple(tensor: torch.Tensor | None) -> tuple[float, ...] | None:
    """把单环境 tensor 转成 Python tuple，便于终端显示。"""

    if tensor is None:
        return None

    row = tensor[0] if tensor.ndim > 1 else tensor
    return tuple(float(value) for value in row.detach().cpu().tolist())


def _scene_asset_size(env, asset_name: str) -> tuple[float, float, float]:
    """从当前 scene cfg 读取物体尺寸，取不到时回退到场景常量。"""

    fallback_size = SCENE_ASSET_SIZE_FALLBACKS[asset_name]
    scene_cfg = getattr(env.cfg, "scene", None)
    if scene_cfg is None or not hasattr(scene_cfg, asset_name):
        return fallback_size

    asset_cfg = getattr(scene_cfg, asset_name)
    if asset_cfg is None or not hasattr(asset_cfg, "spawn"):
        return fallback_size

    size = getattr(asset_cfg.spawn, "size", None)
    if size is None:
        return fallback_size
    return tuple(float(value) for value in size)


def _scene_asset_status(env, asset_name: str) -> AssetStatus | None:
    """读取当前场景物体的位置和尺寸；物体缺失时返回 `None`。"""

    try:
        asset = env.scene[asset_name]
    except KeyError:
        return None

    asset_pos_e = asset.data.root_pos_w[:, :3] - env.scene.env_origins
    return AssetStatus(
        name=asset_name,
        position=_tensor_row_to_tuple(asset_pos_e),
        size=_scene_asset_size(env, asset_name),
    )


def _select_platform_status(env, asset_names: tuple[str, ...]) -> AssetStatus | None:
    """按固定槽位顺序选取左/右通路中的障碍物。"""

    for asset_name in asset_names:
        asset_status = _scene_asset_status(env, asset_name)
        if asset_status is not None:
            return asset_status
    return None


def _skill_name_for_model_use(model_use: int) -> str | None:
    """把当前 model_use 映射成技能名。"""

    if model_use == 0:
        return "idle"
    if model_use in SKILL_REGISTRY:
        return SKILL_REGISTRY[model_use].name
    return None


def _build_status_snapshot(
    env,
    model_use: int,
    start_flag: bool,
    unified_obs_dim: int,
    policy_obs_dim: int | None,
    velocity_command: torch.Tensor,
    pose_command: torch.Tensor | None,
    goal_command: torch.Tensor | None,
) -> StatusSnapshot:
    """汇总当前终端状态面板需要的运行时信息。"""

    robot = env.scene["robot"]
    robot_pose_e = robot.data.root_pos_w[:, :3] - env.scene.env_origins
    platform_statuses = {
        label: _select_platform_status(env, asset_names) for label, asset_names in STATUS_PLATFORM_ASSETS
    }

    return StatusSnapshot(
        model_use=model_use,
        skill=_skill_name_for_model_use(model_use),
        scene_id=args_cli.scene_id,
        start=start_flag,
        unified_obs_dim=unified_obs_dim,
        policy_obs_dim=policy_obs_dim,
        pose_command=_tensor_row_to_tuple(pose_command),
        vel_command=_tensor_row_to_tuple(velocity_command),
        robot_pose=_tensor_row_to_tuple(robot_pose_e),
        goal=_tensor_row_to_tuple(goal_command),
        platform_1=platform_statuses["platform_1"],
        platform_2=platform_statuses["platform_2"],
        box=_scene_asset_status(env, "support_box"),
    )


def _render_status_panel(snapshot: StatusSnapshot):
    """在终端中覆盖刷新当前状态。"""

    if not sys.stdout.isatty():
        return

    try:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(render_status_block(snapshot))
        sys.stdout.write("\n")
        sys.stdout.flush()
    except OSError:
        pass


def _reset_robot_only(env, robot):
    """只重置机器人，不恢复箱子和障碍物。"""

    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += env.scene.env_origins
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:13])

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)


def main():
    """主入口。"""

    if not 0 <= args_cli.scene_id <= 4:
        raise ValueError("--scene_id must be in [0, 4].")
    if args_cli.model_use != 0 and args_cli.model_use not in SKILL_REGISTRY:
        raise ValueError(f"--model_use must be one of {[0, *sorted(SKILL_REGISTRY)]}.")

    _initialize_control_files()

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

    policies = _load_policies(device)
    rough_low_level_policy = _load_push_low_level_policy(device)

    term_slices = _build_obs_slices(env.unwrapped)
    _validate_required_terms(term_slices)
    _print_obs_layout(term_slices)
    unified_obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    print("[INFO] Control files have been reset at startup.")
    print(f"[INFO] model_use file: {args_cli.model_use_file}")
    print(f"[INFO] velocity command file: {args_cli.velocity_command_file}")
    print(f"[INFO] goal command file: {args_cli.goal_command_file}")
    print(f"[INFO] start file: {args_cli.start_file} (auto_start={args_cli.auto_start})")
    print(f"[INFO] reset file: {args_cli.reset_file}")

    base_velocity_commands = _build_default_velocity_commands(num_envs, device)
    zero_pose_command = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
    zero_push_goal_command = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
    zero_push_actions = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    zero_navigation_actions = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
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
    dt = env.unwrapped.step_dt
    step_count = 0

    while simulation_app.is_running():
        start_time = time.time()
        current_model_use = _resolve_model_use(current_model_use)
        current_velocity_commands = _read_float_vector_file(args_cli.velocity_command_file, 3, base_velocity_commands)
        current_velocity_commands = _resolve_runtime_velocity_commands(current_model_use, current_velocity_commands)
        explicit_pose_command = _read_explicit_goal_command_file(args_cli.goal_command_file, zero_push_goal_command)
        robot = env.unwrapped.scene["robot"]
        if explicit_pose_command is None:
            navigation_goal_command = None
        else:
            default_root_height_e = robot.data.default_root_state[:, 2:3] - env.unwrapped.scene.env_origins[:, 2:3]
            navigation_goal_command = align_navigation_goal_height(explicit_pose_command, default_root_height_e)
        current_goal_for_display = navigation_goal_command if current_model_use == 4 else explicit_pose_command
        current_pose_for_display = None if current_model_use == 4 else explicit_pose_command
        current_velocity_for_display = current_velocity_commands
        start_flag = _resolve_start_flag()
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
                _reset_robot_only(env.unwrapped, robot)
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
            if current_model_use == 4:
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
                policy_obs = _slice_observation(unified_obs, term_slices, SKILL_REGISTRY[current_model_use].obs_terms)
                policy_obs = _align_low_level_obs_to_play(policy_obs, term_slices)
                _check_obs_dim(current_model_use, policy_obs)
                actions = policies[current_model_use](policy_obs)
            elif current_model_use == 4:
                if navigation_goal_command is None:
                    navigation_current_processed_actions.zero_()
                    navigation_high_level_counter = 0
                elif navigation_high_level_counter == 0:
                    env.unwrapped.set_runtime_observation_buffers(
                        velocity_commands=current_velocity_commands,
                        pose_command=navigation_pose_command,
                        push_goal_command=zero_push_goal_command,
                        push_actions=zero_push_actions,
                    )
                    unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                    navigation_high_obs = _slice_observation(unified_obs, term_slices, SKILL_REGISTRY[4].obs_terms)
                    navigation_high_obs = _align_navigation_high_level_obs_to_play(navigation_high_obs)
                    _check_obs_dim(4, navigation_high_obs)
                    navigation_raw_actions = policies[4](navigation_high_obs)
                    if navigation_raw_actions.shape != navigation_current_processed_actions.shape:
                        raise RuntimeError(
                            f"Navigation high-level action dim mismatch. "
                            f"Expected {tuple(navigation_current_processed_actions.shape)}, "
                            f"got {tuple(navigation_raw_actions.shape)}."
                        )
                    navigation_current_processed_actions.copy_(_process_navigation_actions(navigation_raw_actions))

                current_velocity_for_display = navigation_current_processed_actions
                env.unwrapped.set_runtime_observation_buffers(
                    velocity_commands=navigation_current_processed_actions,
                    pose_command=navigation_pose_command,
                    push_goal_command=zero_push_goal_command,
                    push_actions=zero_navigation_actions,
                )
                unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                policy_obs = _slice_observation(unified_obs, term_slices, LOW_LEVEL_OBS_TERMS)
                policy_obs = _align_low_level_obs_to_training(
                    env.unwrapped,
                    policy_obs,
                    term_slices,
                    navigation_low_level_last_actions,
                    include_support_box=True,
                )
                if policy_obs.shape[1] != LOW_LEVEL_OBS_DIM:
                    raise RuntimeError(
                        f"Navigation low-level observation dim mismatch. "
                        f"Expected {LOW_LEVEL_OBS_DIM}, got {policy_obs.shape[1]}."
                    )
                actions = rough_low_level_policy(policy_obs)
                navigation_low_level_last_actions.copy_(actions)
                navigation_high_level_counter = (navigation_high_level_counter + 1) % NAVIGATION_HIGH_LEVEL_DECIMATION
            else:
                scene_push_goal = _default_push_goal(env.unwrapped)
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
                    push_high_obs = _slice_observation(unified_obs, term_slices, SKILL_REGISTRY[3].obs_terms)
                    _check_obs_dim(3, push_high_obs)

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

                    push_current_processed_actions.copy_(_process_push_actions(push_raw_actions))
                    push_last_processed_actions.copy_(push_current_processed_actions)

                # 再把裁剪后的高层动作写入低层速度命令槽位，供 rough-walk 低层策略执行。
                env.unwrapped.set_runtime_observation_buffers(
                    velocity_commands=push_current_processed_actions,
                    pose_command=zero_pose_command,
                    push_goal_command=push_goal_command,
                    push_actions=push_current_processed_actions,
                )
                unified_obs = env.unwrapped.observation_manager.compute_group("policy")
                policy_obs = _slice_observation(unified_obs, term_slices, LOW_LEVEL_OBS_TERMS)
                policy_obs = _align_push_low_level_obs_to_training(
                    env.unwrapped, policy_obs, term_slices, push_low_level_last_actions
                )
                if policy_obs.shape[1] != LOW_LEVEL_OBS_DIM:
                    raise RuntimeError(
                        f"Push low-level observation dim mismatch. "
                        f"Expected {LOW_LEVEL_OBS_DIM}, got {policy_obs.shape[1]}."
                    )
                actions = rough_low_level_policy(policy_obs)
                push_low_level_last_actions.copy_(actions)
                push_high_level_counter = (push_high_level_counter + 1) % PUSH_HIGH_LEVEL_DECIMATION

            if actions.shape != expected_action_shape:
                raise RuntimeError(
                    f"Action dim mismatch for model_use={current_model_use}. "
                    f"Expected {expected_action_shape}, got {tuple(actions.shape)}."
                )

            env.step(actions)

        step_count += 1
        if step_count == 1:
            print(f"[INFO] EnvTest unified observation dim: {unified_obs_dim}")
            if start_flag and current_model_use in SKILL_REGISTRY:
                print(f"[INFO] 当前策略输入维度: {current_policy_obs_dim}")
            else:
                print("[INFO] 当前处于待机阶段，机器人保持静止。")
            print(f"[INFO] 当前策略输出维度: {actions.shape[1]}")

        if time.time() - last_status_panel_time >= STATUS_PANEL_REFRESH_INTERVAL:
            status_snapshot = _build_status_snapshot(
                env.unwrapped,
                model_use=current_model_use,
                start_flag=start_flag,
                unified_obs_dim=unified_obs_dim,
                policy_obs_dim=current_policy_obs_dim,
                velocity_command=current_velocity_for_display,
                pose_command=current_pose_for_display,
                goal_command=current_goal_for_display,
            )
            _render_status_panel(status_snapshot)
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
