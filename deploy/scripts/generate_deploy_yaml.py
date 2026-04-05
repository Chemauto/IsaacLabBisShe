#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "MyProject"

for path in [str(REPO_ROOT), str(SOURCE_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher


DEFAULT_OUTPUT_DIR = REPO_ROOT / "deploy" / "robots" / "go2" / "config"
DEFAULT_GO2_JOINT_SDK_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deploy assets for Unitree deploy runtime: deploy.yaml and optionally policy.onnx/policy.pt."
    )
    parser.add_argument("--task", type=str, required=True, help="Gym task id used to build the training environment.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Agent registry entry used to resolve the runner config when exporting policy artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, the latest checkpoint for the task's experiment is used.",
    )
    parser.add_argument(
        "--env-yaml",
        type=str,
        default=None,
        help="Optional env.yaml. If omitted, the script tries checkpoint-adjacent params/env.yaml then output-dir/params/env.yaml.",
    )
    parser.add_argument(
        "--agent-yaml",
        type=str,
        default=None,
        help="Optional agent.yaml. If omitted, the script tries checkpoint-adjacent params/agent.yaml then output-dir/params/agent.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Deploy policy directory root. Outputs are written under <output-dir>/params and <output-dir>/exported.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments used when creating the export env.")
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations when creating the export env.",
    )
    parser.add_argument(
        "--command-observation-name",
        type=str,
        default="keyboard_velocity_commands",
        help="Rename the exported velocity command observation to this deploy-side observation name.",
    )
    parser.add_argument(
        "--skip-policy-export",
        action="store_true",
        default=False,
        help="Only generate deploy.yaml and skip exporting policy.onnx/policy.pt.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


def _to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _to_serializable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _to_serializable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(f"{float(value):.3g}")
    if isinstance(value, float):
        return float(f"{value:.3g}")
    if isinstance(value, Mapping):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _expand_scale(scale: Any, dim: int) -> list[float]:
    if scale is None:
        return [1.0] * dim
    if isinstance(scale, torch.Tensor):
        scale = scale.detach().cpu().tolist()
    elif isinstance(scale, np.ndarray):
        scale = scale.tolist()
    if isinstance(scale, (int, float, np.integer, np.floating)):
        return [float(scale)] * dim
    return list(scale)


def _obs_dim(obs_tensor: torch.Tensor) -> int:
    shape = tuple(obs_tensor.shape)
    if not shape:
        return 1
    if len(shape) == 1:
        return int(np.prod(shape))
    return int(np.prod(shape[1:]))


def _export_deploy_cfg(env, output_file: Path, velocity_command_obs_name: str | None) -> Path:
    from isaaclab.assets import Articulation
    from isaaclab.utils.string import resolve_matching_names

    asset: Articulation = env.scene["robot"]
    if not hasattr(env.cfg.scene.robot, "joint_sdk_names"):
        raise AttributeError(
            "env.cfg.scene.robot.joint_sdk_names is missing. Pass the training env.yaml with --env-yaml so the deploy joint mapping can be restored."
        )

    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg: dict[str, Any] = {}
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation

    stiffness = np.zeros(len(joint_sdk_names), dtype=np.float32)
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy()
    cfg["stiffness"] = stiffness.tolist()

    damping = np.zeros(len(joint_sdk_names), dtype=np.float32)
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    cfg["commands"] = {}
    if hasattr(env.cfg, "commands") and hasattr(env.cfg.commands, "base_velocity"):
        base_velocity_cfg = env.cfg.commands.base_velocity
        ranges = base_velocity_cfg.limit_ranges.to_dict() if hasattr(base_velocity_cfg, "limit_ranges") else base_velocity_cfg.ranges.to_dict()
        cfg["commands"]["base_velocity"] = {"ranges": _to_serializable(ranges)}

    cfg["actions"] = {}
    action_names = env.action_manager.active_terms
    action_terms = env.action_manager._terms.values()
    for action_name, action_term in zip(action_names, action_terms):
        term_cfg = action_term.cfg.copy()
        action_key = action_term.__class__.__name__ or action_name

        if isinstance(term_cfg.scale, Mapping):
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()
        else:
            term_cfg.scale = _expand_scale(term_cfg.scale, action_term.action_dim)

        if term_cfg.clip is not None:
            term_cfg.clip = _to_serializable(action_term._clip[0])

        if action_key in {"JointPositionAction", "JointVelocityAction"}:
            use_default_offset = getattr(term_cfg, "use_default_offset", False)
            if use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0] * action_term.action_dim

        term_cfg = term_cfg.to_dict()
        for key in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            term_cfg.pop(key, None)

        joint_ids = action_term._joint_ids
        term_cfg["joint_ids"] = None if isinstance(joint_ids, slice) and joint_ids == slice(None) else _to_serializable(joint_ids)
        cfg["actions"][action_key] = term_cfg

    cfg["observations"] = {}
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    for obs_name, obs_cfg in zip(obs_names, obs_cfgs):
        obs_dim = _obs_dim(obs_cfg.func(env, **obs_cfg.params))
        term_cfg = obs_cfg.copy()
        term_cfg.scale = _expand_scale(term_cfg.scale, obs_dim)

        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        term_cfg = term_cfg.to_dict()
        for key in ["func", "modifiers", "noise", "flatten_history_dim"]:
            term_cfg.pop(key, None)

        func_name = getattr(obs_cfg.func, "__name__", obs_name)
        canonical_obs_name = {
            "generated_commands": "velocity_commands",
            "joint_pos_rel": "joint_pos_rel",
            "joint_vel_rel": "joint_vel_rel",
            "last_action": "last_action",
        }.get(func_name, obs_name)

        output_obs_name = canonical_obs_name
        if output_obs_name == "velocity_commands" and velocity_command_obs_name:
            output_obs_name = velocity_command_obs_name

        if func_name == "height_scan":
            term_cfg["params"] = {"expected_dim": obs_dim}
        cfg["observations"][output_obs_name] = term_cfg

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        yaml.dump(_to_serializable(cfg), file, default_flow_style=None, sort_keys=False)
    return output_file


def _read_yaml(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.Loader)


def _maybe_copy(src: Path | None, dst: Path) -> None:
    if src is None or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _relaxed_update_cfg(cfg_obj, data: Any) -> None:
    if not isinstance(data, Mapping):
        return

    for key, value in data.items():
        if isinstance(cfg_obj, dict):
            if key not in cfg_obj:
                continue
            current = cfg_obj[key]
        else:
            if not hasattr(cfg_obj, key):
                continue
            current = getattr(cfg_obj, key)

        if current is None and isinstance(value, Mapping):
            continue

        if isinstance(value, Mapping) and current is not None and (isinstance(current, dict) or hasattr(current, "__dict__")):
            _relaxed_update_cfg(current, value)
            continue

        if callable(current) and isinstance(value, str):
            continue

        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        elif isinstance(current, list) and isinstance(value, tuple):
            value = list(value)

        if isinstance(cfg_obj, dict):
            cfg_obj[key] = value
        else:
            setattr(cfg_obj, key, value)


def _maybe_apply_yaml(cfg_obj, yaml_path: Path | None) -> None:
    if yaml_path is None:
        return
    _relaxed_update_cfg(cfg_obj, _read_yaml(yaml_path))


def _load_cfg(task_name: str, entry_point_key: str):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_name, entry_point_key)
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for task '{task_name}' and key '{entry_point_key}' must be a config class, not a dict.")
    return cfg


def _resolve_checkpoint(agent_cfg, explicit_checkpoint: str | None) -> Path | None:
    from isaaclab_tasks.utils import get_checkpoint_path

    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint).expanduser().resolve()
        return checkpoint_path if checkpoint_path.is_file() else None

    log_root_path = (REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if not log_root_path.exists():
        return None

    try:
        checkpoint_path = get_checkpoint_path(
            str(log_root_path),
            getattr(agent_cfg, "load_run", ".*"),
            getattr(agent_cfg, "load_checkpoint", "model_.*.pt"),
        )
    except Exception:
        return None

    checkpoint_path = Path(checkpoint_path).resolve()
    return checkpoint_path if checkpoint_path.is_file() else None


def _resolve_params_path(
    explicit_path: str | None,
    checkpoint_path: Path | None,
    output_dir: Path,
    filename: str,
    include_output_dir_fallback: bool = False,
) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    if checkpoint_path is not None:
        candidates.append(checkpoint_path.parent / "params" / filename)
    if include_output_dir_fallback:
        candidates.append(output_dir / "params" / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _export_policy_artifacts(args_cli, agent_cfg, env, checkpoint_path: Path, output_dir: Path) -> None:
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(str(checkpoint_path), map_location=agent_cfg.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_dir = output_dir / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.onnx")


def main() -> None:
    parser = _build_parser()
    args_cli, hydra_args = parser.parse_known_args()
    args_cli.enable_cameras = False
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import MyProject.tasks  # noqa: F401
    from isaaclab.utils.io import dump_yaml

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = _load_cfg(args_cli.task, args_cli.agent)
    bootstrap_agent_yaml_path = _resolve_params_path(
        args_cli.agent_yaml,
        None,
        output_dir,
        "agent.yaml",
        include_output_dir_fallback=False,
    )
    _maybe_apply_yaml(agent_cfg, bootstrap_agent_yaml_path)

    checkpoint_path = None if args_cli.skip_policy_export else _resolve_checkpoint(agent_cfg, args_cli.checkpoint)
    agent_yaml_path = _resolve_params_path(
        args_cli.agent_yaml,
        checkpoint_path,
        output_dir,
        "agent.yaml",
        include_output_dir_fallback=False,
    )
    if agent_yaml_path != bootstrap_agent_yaml_path:
        _maybe_apply_yaml(agent_cfg, agent_yaml_path)

    env_yaml_path = _resolve_params_path(args_cli.env_yaml, checkpoint_path, output_dir, "env.yaml")

    env_cfg = _load_cfg(args_cli.task, "env_cfg_entry_point")
    _maybe_apply_yaml(env_cfg, env_yaml_path)
    _maybe_apply_yaml(agent_cfg, agent_yaml_path)

    if not hasattr(env_cfg.scene.robot, "joint_sdk_names"):
        env_cfg.scene.robot.joint_sdk_names = DEFAULT_GO2_JOINT_SDK_NAMES

    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    env_cfg.log_dir = str(output_dir)

    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    _maybe_copy(env_yaml_path, params_dir / "env.input.yaml")
    _maybe_copy(agent_yaml_path, params_dir / "agent.input.yaml")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        deploy_yaml_path = _export_deploy_cfg(
            env.unwrapped,
            params_dir / "deploy.yaml",
            velocity_command_obs_name=args_cli.command_observation_name,
        )
        print(f"[INFO] Generated deploy config: {deploy_yaml_path}")

        if checkpoint_path is not None and not args_cli.skip_policy_export:
            _export_policy_artifacts(args_cli, agent_cfg, env, checkpoint_path, output_dir)
            print(f"[INFO] Exported policy assets to: {output_dir / 'exported'}")
        elif args_cli.skip_policy_export:
            print("[INFO] Skip policy export by request.")
        else:
            print("[INFO] No checkpoint found. Generated deploy.yaml only.")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
