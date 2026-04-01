from __future__ import annotations

import os
import re
from typing import Any

import torch
import torch.nn as nn

from MyProject.tasks.manager_based.EnvTest.utils.status_panel import AssetStatus, StatusSnapshot


def make_activation(name: str) -> nn.Module:
    """Create activation layers by name."""

    activation_name = name.lower()
    if activation_name == "elu":
        return nn.ELU()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation for checkpoint policy loading: {name}")


def load_rsl_rl_actor_from_checkpoint(spec: Any, device: torch.device | str) -> nn.Module:
    """Restore an actor network from an RSL-RL checkpoint."""

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
            layers.append(make_activation(spec.activation))

    actor = nn.Sequential(*layers)
    actor.load_state_dict(actor_state_dict)
    actor.to(device)
    actor.eval()
    return actor


def load_policies(skill_registry: dict[int, Any], device: torch.device | str) -> dict[int, nn.Module]:
    """Load all high-level policies declared in `skill_registry`."""

    policies: dict[int, nn.Module] = {}
    for model_use, spec in skill_registry.items():
        if not os.path.isfile(spec.policy_path):
            raise FileNotFoundError(f"Policy file not found: {spec.policy_path}")
        if spec.checkpoint_format == "jit":
            policies[model_use] = torch.jit.load(spec.policy_path, map_location=device).eval()
        elif spec.checkpoint_format == "rsl_rl_checkpoint":
            policies[model_use] = load_rsl_rl_actor_from_checkpoint(spec, device)
        else:
            raise ValueError(f"Unsupported checkpoint format '{spec.checkpoint_format}' for skill '{spec.name}'.")
    return policies


def load_jit_policy(policy_path: str, description: str, device: torch.device | str) -> torch.jit.ScriptModule:
    """Load one exported JIT policy."""

    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"{description} policy file not found: {policy_path}")
    return torch.jit.load(policy_path, map_location=device).eval()


def resolve_model_use(model_use_file: str, skill_registry: dict[int, Any], current_model_use: int) -> int:
    """Resolve the current model id from an optional text file."""

    if model_use_file and os.path.isfile(model_use_file):
        try:
            with open(model_use_file, "r", encoding="utf-8") as file:
                file_value = int(file.read().strip())
        except (OSError, ValueError):
            file_value = current_model_use
        if file_value == 0 or file_value in skill_registry:
            return file_value
    return current_model_use


def read_float_vector_file(file_path: str, expected_dim: int, fallback: torch.Tensor) -> torch.Tensor:
    """Read fixed-length float vectors from text files."""

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


def resolve_start_flag(auto_start: bool, start_file: str) -> bool:
    """Resolve whether the player should currently execute policies."""

    if auto_start:
        return True
    if not start_file or not os.path.isfile(start_file):
        return False

    try:
        with open(start_file, "r", encoding="utf-8") as file:
            text = file.read().strip().lower()
    except OSError:
        return False

    return text in ("1", "true", "run", "start", "yes", "y")


def ensure_parent_dir(file_path: str) -> None:
    """Ensure the parent directory exists before writing files."""

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_text_file(file_path: str, content: str) -> None:
    """Write text files used by the EnvTest player."""

    ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content.strip() + "\n")


def initialize_control_files(args) -> None:
    """Reset control files at startup to avoid stale commands."""

    write_text_file(args.model_use_file, str(args.model_use))
    write_text_file(
        args.velocity_command_file,
        f"{args.lin_vel_x} {args.lin_vel_y} {args.ang_vel_z}",
    )
    write_text_file(args.goal_command_file, "auto")
    write_text_file(args.start_file, "1" if args.auto_start else "0")
    write_text_file(args.reset_file, "0")


def parse_goal_command_text(text: str) -> list[float] | None:
    """Parse target text into `[x, y, z, yaw]`."""

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


def read_explicit_goal_command_file(file_path: str, template: torch.Tensor) -> torch.Tensor | None:
    """Read explicit goal commands; `auto` and empty content return `None`."""

    if not file_path or not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except OSError:
        return None

    parsed = parse_goal_command_text(text)
    if parsed is None:
        return None

    vector = torch.tensor(parsed, dtype=template.dtype, device=template.device)
    return vector.unsqueeze(0).repeat(template.shape[0], 1)


def build_default_velocity_commands(
    lin_vel_x: float,
    lin_vel_y: float,
    ang_vel_z: float,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the default low-level velocity commands."""

    single_command = torch.tensor([lin_vel_x, lin_vel_y, ang_vel_z], dtype=torch.float32, device=device)
    return single_command.unsqueeze(0).repeat(num_envs, 1)


def resolve_runtime_velocity_commands(
    scene_id: int,
    model_use: int,
    velocity_commands: torch.Tensor,
    climb_play_default_velocity: tuple[float, float, float],
) -> torch.Tensor:
    """Apply skill-specific Play defaults to runtime velocity commands."""

    if scene_id == 2 and model_use == 2 and torch.allclose(velocity_commands, torch.zeros_like(velocity_commands)):
        climb_velocity = torch.tensor(
            climb_play_default_velocity, dtype=velocity_commands.dtype, device=velocity_commands.device
        )
        return climb_velocity.unsqueeze(0).repeat(velocity_commands.shape[0], 1)
    return velocity_commands


def check_obs_dim(skill_registry: dict[int, Any], model_use: int, obs: torch.Tensor) -> None:
    """Validate the sliced observation dimension against the selected policy."""

    expected_dim = skill_registry[model_use].obs_dim
    if obs.shape[1] != expected_dim:
        raise RuntimeError(
            f"Observation dim mismatch for model_use={model_use}. "
            f"Expected {expected_dim}, got {obs.shape[1]}."
        )


def tensor_row_to_tuple(tensor: torch.Tensor | None) -> tuple[float, ...] | None:
    """Convert a single-env tensor row into a Python tuple."""

    if tensor is None:
        return None

    row = tensor[0] if tensor.ndim > 1 else tensor
    return tuple(float(value) for value in row.detach().cpu().tolist())


def scene_asset_size(env, asset_name: str, fallback_sizes: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    """Read scene asset size from cfg and fall back to the static defaults."""

    fallback_size = fallback_sizes[asset_name]
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


def scene_asset_status(env, asset_name: str, fallback_sizes: dict[str, tuple[float, float, float]]) -> AssetStatus | None:
    """Read current scene asset position and size."""

    try:
        asset = env.scene[asset_name]
    except KeyError:
        return None

    asset_pos_e = asset.data.root_pos_w[:, :3] - env.scene.env_origins
    return AssetStatus(
        name=asset_name,
        position=tensor_row_to_tuple(asset_pos_e),
        size=scene_asset_size(env, asset_name, fallback_sizes),
    )


def select_platform_status(
    env,
    asset_names: tuple[str, ...],
    fallback_sizes: dict[str, tuple[float, float, float]],
) -> AssetStatus | None:
    """Pick the first active obstacle for one left/right platform slot."""

    for asset_name in asset_names:
        asset_status = scene_asset_status(env, asset_name, fallback_sizes)
        if asset_status is not None:
            return asset_status
    return None


def skill_name_for_model_use(model_use: int, skill_registry: dict[int, Any]) -> str | None:
    """Map the current `model_use` id to a skill name."""

    if model_use == 0:
        return "idle"
    if model_use in skill_registry:
        return skill_registry[model_use].name
    return None


def build_status_snapshot(
    env,
    model_use: int,
    start_flag: bool,
    unified_obs_dim: int,
    policy_obs_dim: int | None,
    velocity_command: torch.Tensor,
    pose_command: torch.Tensor | None,
    goal_command: torch.Tensor | None,
    scene_id: int,
    skill_registry: dict[int, Any],
    platform_assets: tuple[tuple[str, tuple[str, ...]], ...],
    fallback_sizes: dict[str, tuple[float, float, float]],
) -> StatusSnapshot:
    """Build the terminal/JSON runtime status snapshot."""

    robot = env.scene["robot"]
    robot_pose_e = robot.data.root_pos_w[:, :3] - env.scene.env_origins
    platform_statuses = {
        label: select_platform_status(env, asset_names, fallback_sizes) for label, asset_names in platform_assets
    }
    return StatusSnapshot(
        model_use=model_use,
        skill=skill_name_for_model_use(model_use, skill_registry),
        scene_id=scene_id,
        start=start_flag,
        unified_obs_dim=unified_obs_dim,
        policy_obs_dim=policy_obs_dim,
        pose_command=tensor_row_to_tuple(pose_command),
        vel_command=tensor_row_to_tuple(velocity_command),
        robot_pose=tensor_row_to_tuple(robot_pose_e),
        goal=tensor_row_to_tuple(goal_command),
        platform_1=platform_statuses["platform_1"],
        platform_2=platform_statuses["platform_2"],
        box=scene_asset_status(env, "support_box", fallback_sizes),
    )


def reset_robot_only(env, robot) -> None:
    """Reset only the robot while leaving obstacles and box untouched."""

    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += env.scene.env_origins
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:13])

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)

