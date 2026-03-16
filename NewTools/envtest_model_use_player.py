# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""在 EnvTest 场景中按 model_use 切换技能策略。

设计目标：
- 场景始终使用 EnvTest；
- 策略选择不再由 scene_id 决定，而是由 model_use 决定；
- 这样后续 LLM 只需要输出 model_use，即可切换 walk / climb。

当前支持：
- model_use=1: 平地行走策略
- model_use=2: 攀爬策略
- model_use=3: 推箱子策略

说明：
- 这里直接加载导出的 TorchScript 策略，而不是训练 checkpoint。
- climb 所需的 height_scan 采用结构化场景几何直接构造，
  这样不需要把 EnvTest 改成带多 mesh RayCaster 的复杂版本。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass


# 把仓库根目录加入 Python 路径，避免直接运行脚本时出现导入问题。
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run EnvTest with switchable walk/climb policies.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Template-EnvTest-Go2-Play-v0", help="EnvTest 任务名。")
parser.add_argument("--scene_id", type=int, default=0, help="EnvTest 场景编号，使用 0~4。")
parser.add_argument("--model_use", type=int, default=1, help="技能编号：1=walk，2=climb，3=push_box。")
parser.add_argument(
    "--model_use_file",
    type=str,
    default="",
    help="可选文本文件路径；若提供，则每步从文件读取 model_use，便于后续接 LLM。",
)
parser.add_argument("--num_envs", type=int, default=1, help="环境数量。通常建议先用 1。")
parser.add_argument("--lin_vel_x", type=float, default=0.6, help="给低层策略的前向速度命令。")
parser.add_argument("--lin_vel_y", type=float, default=0.0, help="给低层策略的侧向速度命令。")
parser.add_argument("--ang_vel_z", type=float, default=0.0, help="给低层策略的偏航角速度命令。")
parser.add_argument("--real-time", action="store_true", default=False, help="尽量按实时速度运行。")
parser.add_argument("--max_steps", type=int, default=0, help="最大仿真步数；0 表示一直运行。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# EnvTest 里固定挂了前视相机，因此这里默认强制开启相机渲染。
args_cli.enable_cameras = True

# 启动 Isaac Sim。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MyProject.tasks  # noqa: F401
from MyProject.tasks.manager_based.EnvTest.scene_layout import BOX_SIZE, HIGH_OBSTACLE_SIZE, LOW_OBSTACLE_SIZE


@dataclass(frozen=True)
class SkillSpec:
    """单个技能模型的配置。"""

    name: str
    policy_path: str
    obs_dim: int
    use_height_scan: bool


SKILL_REGISTRY: dict[int, SkillSpec] = {
    1: SkillSpec(
        name="walk",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "WalkPolicy", "exported", "policy.pt"),
        obs_dim=48,
        use_height_scan=False,
    ),
    2: SkillSpec(
        name="climb",
        policy_path=os.path.join(REPO_ROOT, "ModelBackup", "BiShePolicy", "exported", "policy.pt"),
        obs_dim=235,
        use_height_scan=True,
    ),
    3: SkillSpec(
        name="push_box",
        policy_path=os.path.join(
            REPO_ROOT, "scripts", "rsl_rl", "logs", "rsl_rl", "push_box", "2026-03-15_10-14-05", "exported", "policy.pt"
        ),
        obs_dim=22,
        use_height_scan=False,
    ),
}

PUSH_LOW_LEVEL_POLICY_PATH = os.path.join(REPO_ROOT, "ModelBackup", "TransPolicy", "WalkRoughNewTransfer.pt")
PUSH_LOW_LEVEL_OBS_DIM = 235
PUSH_ACTION_SCALE = (0.4, 0.2, 0.3)
PUSH_ACTION_CLIP = ((-0.4, 0.4), (-0.2, 0.2), (-0.3, 0.3))


class StructuredHeightScanBuilder:
    """针对 EnvTest 规则化障碍的高度扫描构造器。

    训练中的 climb 策略需要 187 维 height_scan。
    这里直接对当前场景中的长方体障碍做几何投影，生成同尺寸的局部高度图。
    """

    def __init__(self, device: torch.device | str):
        self.device = device
        self.offset = 0.5
        self.local_points = self._build_local_points()
        self.asset_specs = {
            "left_low_obstacle": LOW_OBSTACLE_SIZE,
            "right_low_obstacle": LOW_OBSTACLE_SIZE,
            "left_high_obstacle": HIGH_OBSTACLE_SIZE,
            "right_high_obstacle": HIGH_OBSTACLE_SIZE,
            "support_box": BOX_SIZE,
        }

    def _build_local_points(self) -> torch.Tensor:
        """构造与训练中 GridPatternCfg 一致的 17x11 网格。"""

        x = torch.arange(start=-0.8, end=0.8 + 1.0e-9, step=0.1, device=self.device)
        y = torch.arange(start=-0.5, end=0.5 + 1.0e-9, step=0.1, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        local_points = torch.zeros(grid_x.numel(), 3, device=self.device)
        local_points[:, 0] = grid_x.flatten()
        local_points[:, 1] = grid_y.flatten()
        return local_points

    def compute(self, scene, robot) -> torch.Tensor:
        """根据当前机器人姿态和障碍物位置计算 height_scan。"""

        num_envs = robot.data.root_pos_w.shape[0]
        num_points = self.local_points.shape[0]

        # 网格点跟随机器人位置和平面偏航角一起变换。
        local_points = self.local_points.unsqueeze(0).expand(num_envs, -1, -1)
        quat_yaw = robot.data.root_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
        world_points = math_utils.quat_apply_yaw(quat_yaw, local_points.reshape(-1, 3)).view(num_envs, num_points, 3)
        world_points = world_points + robot.data.root_pos_w.unsqueeze(1)

        # 默认地面高度为 0。
        max_heights = torch.zeros(num_envs, num_points, device=self.device)

        for asset_name, size in self.asset_specs.items():
            try:
                asset = scene[asset_name]
            except KeyError:
                continue

            centers = asset.data.root_pos_w
            half_x = 0.5 * size[0]
            half_y = 0.5 * size[1]
            top_height = centers[:, 2].unsqueeze(1) + 0.5 * size[2]

            inside_x = torch.abs(world_points[..., 0] - centers[:, 0].unsqueeze(1)) <= half_x
            inside_y = torch.abs(world_points[..., 1] - centers[:, 1].unsqueeze(1)) <= half_y
            inside = inside_x & inside_y

            max_heights = torch.where(inside, torch.maximum(max_heights, top_height), max_heights)

        # 与 IsaacLab 原始 height_scan 定义保持一致：base_z - hit_z - offset。
        return robot.data.root_pos_w[:, 2].unsqueeze(1) - max_heights - self.offset


def _load_policies(device: torch.device | str) -> dict[int, torch.jit.ScriptModule]:
    """加载所有支持的技能策略。"""

    policies: dict[int, torch.jit.ScriptModule] = {}
    for model_use, spec in SKILL_REGISTRY.items():
        if not os.path.isfile(spec.policy_path):
            raise FileNotFoundError(f"Policy file not found: {spec.policy_path}")
        policies[model_use] = torch.jit.load(spec.policy_path, map_location=device).eval()
    return policies


def _load_push_low_level_policy(device: torch.device | str) -> torch.jit.ScriptModule:
    """加载推箱子技能依赖的低层 rough-walk 策略。"""

    if not os.path.isfile(PUSH_LOW_LEVEL_POLICY_PATH):
        raise FileNotFoundError(f"Push low-level policy file not found: {PUSH_LOW_LEVEL_POLICY_PATH}")
    return torch.jit.load(PUSH_LOW_LEVEL_POLICY_PATH, map_location=device).eval()


def _resolve_model_use(current_model_use: int) -> int:
    """从 CLI 或外部文件中决定当前 model_use。"""

    if args_cli.model_use_file:
        if os.path.isfile(args_cli.model_use_file):
            try:
                with open(args_cli.model_use_file, "r", encoding="utf-8") as file:
                    file_value = int(file.read().strip())
            except (OSError, ValueError):
                file_value = current_model_use
            if file_value in SKILL_REGISTRY:
                return file_value
    return current_model_use


def _build_command_tensor(num_envs: int, device: torch.device | str) -> torch.Tensor:
    """构造低层策略使用的速度命令。"""

    single_command = torch.tensor(
        [args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z], dtype=torch.float32, device=device
    )
    return single_command.unsqueeze(0).repeat(num_envs, 1)


def _build_walk_obs(robot, last_actions: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
    """构造平地行走策略所需观测。"""

    return torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            commands,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel - robot.data.default_joint_vel,
            last_actions,
        ),
        dim=-1,
    )


def _build_climb_obs(
    robot,
    last_actions: torch.Tensor,
    commands: torch.Tensor,
    height_scan: torch.Tensor,
) -> torch.Tensor:
    """构造攀爬策略所需观测。"""

    return torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            commands,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel - robot.data.default_joint_vel,
            last_actions,
            height_scan,
        ),
        dim=-1,
    )


def _build_push_high_obs(env, push_last_processed_actions: torch.Tensor) -> torch.Tensor:
    """构造推箱子高层策略所需观测。"""

    robot = env.unwrapped.scene["robot"]
    try:
        box = env.unwrapped.scene["support_box"]
    except KeyError as err:
        raise RuntimeError("当前 EnvTest 场景中没有 support_box，无法执行 push_box 策略。") from err

    env_origins = env.unwrapped.scene.env_origins
    box_position = box.data.root_pos_w[:, :3] - env_origins
    box_quat = math_utils.quat_unique(box.data.root_quat_w)
    box_pose = torch.cat((box_position, box_quat), dim=-1)
    robot_position = robot.data.root_pos_w[:, :3] - env_origins
    goal_command = _compute_push_goal(env)

    return torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.projected_gravity_b,
            box_pose,
            robot_position,
            goal_command,
            push_last_processed_actions,
        ),
        dim=-1,
    )


def _compute_push_goal(env) -> torch.Tensor:
    """根据当前障碍物位置，生成一个把箱子推到障碍物前方的目标点。"""

    scene = env.unwrapped.scene
    env_origins = scene.env_origins
    box = scene["support_box"]
    box_pos_e = box.data.root_pos_w[:, :3] - env_origins

    candidates: list[torch.Tensor] = []
    candidate_sizes = (
        ("left_high_obstacle", HIGH_OBSTACLE_SIZE),
        ("right_high_obstacle", HIGH_OBSTACLE_SIZE),
        ("left_low_obstacle", LOW_OBSTACLE_SIZE),
        ("right_low_obstacle", LOW_OBSTACLE_SIZE),
    )
    for asset_name, size in candidate_sizes:
        try:
            asset = scene[asset_name]
        except KeyError:
            continue
        obstacle_pos_e = asset.data.root_pos_w[:, :3] - env_origins
        goal = torch.zeros_like(obstacle_pos_e)
        # 把箱子目标点放在障碍物前沿，便于后续从箱子爬上去。
        goal[:, 0] = obstacle_pos_e[:, 0] - 0.5 * size[0] - 0.5 * BOX_SIZE[0] - 0.02
        goal[:, 1] = obstacle_pos_e[:, 1]
        goal[:, 2] = 0.5 * BOX_SIZE[2]
        candidates.append(goal)

    if not candidates:
        raise RuntimeError("当前场景中没有可供 push_box 使用的目标障碍物。")

    candidate_tensor = torch.stack(candidates, dim=1)
    distances = torch.linalg.norm(candidate_tensor[..., :2] - box_pos_e[:, None, :2], dim=-1)
    best_indices = torch.argmin(distances, dim=1)
    selected_goals = candidate_tensor[torch.arange(env.unwrapped.num_envs, device=box_pos_e.device), best_indices]
    return selected_goals


def _process_push_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """对推箱子高层动作做与训练时一致的缩放和裁剪。"""

    action_scale = torch.tensor(PUSH_ACTION_SCALE, dtype=torch.float32, device=raw_actions.device).view(1, 3)
    clip_tensor = torch.tensor(PUSH_ACTION_CLIP, dtype=torch.float32, device=raw_actions.device)
    clip_min = clip_tensor[:, 0].view(1, 3)
    clip_max = clip_tensor[:, 1].view(1, 3)
    processed_actions = raw_actions * action_scale
    processed_actions = torch.max(torch.min(processed_actions, clip_max), clip_min)
    return processed_actions


def _build_policy_obs(model_use: int, env, last_actions: torch.Tensor, height_scan_builder) -> torch.Tensor:
    """根据当前技能编号生成对应观测。"""

    robot = env.unwrapped.scene["robot"]
    commands = _build_command_tensor(env.unwrapped.num_envs, env.unwrapped.device)
    skill_spec = SKILL_REGISTRY[model_use]

    if model_use == 3:
        raise RuntimeError("push_box 观测请使用 _build_push_high_obs，不要走通用分支。")

    if skill_spec.use_height_scan:
        height_scan = height_scan_builder.compute(env.unwrapped.scene, robot)
        obs = _build_climb_obs(robot, last_actions, commands, height_scan)
    else:
        obs = _build_walk_obs(robot, last_actions, commands)

    if obs.shape[1] != skill_spec.obs_dim:
        raise RuntimeError(
            f"Observation dim mismatch for model_use={model_use}. "
            f"Expected {skill_spec.obs_dim}, got {obs.shape[1]}."
        )
    return obs


def main():
    """主入口。"""

    if not 0 <= args_cli.scene_id <= 4:
        raise ValueError("--scene_id must be in [0, 4].")
    if args_cli.model_use not in SKILL_REGISTRY:
        raise ValueError(f"--model_use must be one of {sorted(SKILL_REGISTRY)}.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "scene_id"):
        env_cfg.scene_id = args_cli.scene_id

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    policies = _load_policies(device)
    push_low_level_policy = _load_push_low_level_policy(device)
    height_scan_builder = StructuredHeightScanBuilder(device)

    action_dim = env.unwrapped.scene["robot"].data.joint_pos.shape[1]
    last_joint_actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=device)
    push_last_processed_actions = torch.zeros(env.unwrapped.num_envs, 3, device=device)
    current_model_use = args_cli.model_use
    previous_model_use = None
    dt = env.unwrapped.step_dt
    step_count = 0

    while simulation_app.is_running():
        start_time = time.time()
        current_model_use = _resolve_model_use(current_model_use)

        if current_model_use != previous_model_use:
            skill_name = SKILL_REGISTRY[current_model_use].name
            last_joint_actions.zero_()
            push_last_processed_actions.zero_()
            print(f"[INFO] 切换技能: model_use={current_model_use}, skill={skill_name}, scene_id={args_cli.scene_id}")
            previous_model_use = current_model_use

        with torch.inference_mode():
            if current_model_use == 3:
                push_high_obs = _build_push_high_obs(env, push_last_processed_actions)
                if push_high_obs.shape[1] != SKILL_REGISTRY[3].obs_dim:
                    raise RuntimeError(
                        f"Observation dim mismatch for model_use=3. "
                        f"Expected {SKILL_REGISTRY[3].obs_dim}, got {push_high_obs.shape[1]}."
                    )
                push_raw_actions = policies[3](push_high_obs)
                if push_raw_actions.shape != push_last_processed_actions.shape:
                    raise RuntimeError(
                        f"Push high-level action dim mismatch. "
                        f"Expected {tuple(push_last_processed_actions.shape)}, got {tuple(push_raw_actions.shape)}."
                    )
                push_processed_actions = _process_push_actions(push_raw_actions)
                height_scan = height_scan_builder.compute(env.unwrapped.scene, env.unwrapped.scene["robot"])
                policy_obs = _build_climb_obs(
                    env.unwrapped.scene["robot"],
                    last_joint_actions,
                    push_processed_actions,
                    height_scan,
                )
                if policy_obs.shape[1] != PUSH_LOW_LEVEL_OBS_DIM:
                    raise RuntimeError(
                        f"Push low-level observation dim mismatch. "
                        f"Expected {PUSH_LOW_LEVEL_OBS_DIM}, got {policy_obs.shape[1]}."
                    )
                actions = push_low_level_policy(policy_obs)
                push_last_processed_actions.copy_(push_processed_actions)
            else:
                policy_obs = _build_policy_obs(current_model_use, env, last_joint_actions, height_scan_builder)
                actions = policies[current_model_use](policy_obs)
            if actions.shape != last_joint_actions.shape:
                raise RuntimeError(
                    f"Action dim mismatch for model_use={current_model_use}. "
                    f"Expected {tuple(last_joint_actions.shape)}, got {tuple(actions.shape)}."
                )
            env.step(actions)
            last_joint_actions.copy_(actions)

        step_count += 1
        if step_count == 1:
            print(f"[INFO] 当前策略输入维度: {policy_obs.shape[1]}")
            print(f"[INFO] 当前策略输出维度: {actions.shape[1]}")

        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
