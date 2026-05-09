# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export IO descriptors for an Isaac Lab environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Export IO descriptors for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory.")
parser.add_argument("--scene_id", type=int, default=None, help="Scene id for EnvTest play mode. Valid values: 0-4.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MyProject.tasks  # noqa: F401


def _to_yaml_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, tuple):
        return [_to_yaml_safe(item) for item in value]
    if isinstance(value, list):
        return [_to_yaml_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_yaml_safe(item) for key, item in value.items()}
    return value


def _flatten_singleton_row(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _term_dim(term):
    dim = 1
    for item in term.get("shape", []):
        dim *= int(item)
    return dim


def _expand_scalar_or_singleton(value, dim):
    value = _flatten_singleton_row(value)
    if isinstance(value, (int, float)):
        return [float(value)] * dim
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float)) and dim != 1:
        return [float(value[0])] * dim
    return value


def _normalize_action_descriptors(actions):
    for action in actions:
        action_dim = _term_dim(action)
        for key in ("scale", "offset"):
            if key in action:
                action[key] = _expand_scalar_or_singleton(action[key], action_dim)
        if "clip" in action:
            action["clip"] = _flatten_singleton_row(action["clip"])


def _normalize_observation_descriptors(observations):
    for obs_group in observations.values():
        for obs_term in obs_group:
            overloads = obs_term.get("overloads")
            if not isinstance(overloads, dict):
                continue
            obs_dim = _term_dim(obs_term)
            if "scale" in overloads:
                overloads["scale"] = _expand_scalar_or_singleton(overloads["scale"], obs_dim)
            if "clip" in overloads:
                overloads["clip"] = _flatten_singleton_row(overloads["clip"])


def main():
    """Export IO descriptors from an Isaac Lab environment."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.scene_id is not None:
        if not 0 <= args_cli.scene_id <= 4:
            raise ValueError("--scene_id must be in [0, 4].")
        if hasattr(env_cfg, "scene_id"):
            env_cfg.scene_id = args_cli.scene_id

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    outs = _to_yaml_safe(env.unwrapped.get_IO_descriptors)
    out_observations = outs["observations"]
    out_actions = outs["actions"]
    out_articulations = outs["articulations"]
    out_scene = outs["scene"]
    _normalize_action_descriptors(out_actions)
    _normalize_observation_descriptors(out_observations)

    task_name = args_cli.task.lower().replace("-", "_").replace(" ", "_")
    output_dir = args_cli.output_dir or os.path.join(os.getcwd(), "io_descriptors")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{task_name}_IO_descriptors.yaml")
    print(f"[INFO]: Exporting IO descriptors to {output_path}")
    with open(output_path, "w") as f:
        yaml.safe_dump(outs, f)

    for action_term in out_actions:
        print(f"--- Action term: {action_term['name']} ---")
        action_term.pop("name")
        for key, value in action_term.items():
            print(f"{key}: {value}")

    for obs_group_name, obs_group in out_observations.items():
        print(f"--- Obs group: {obs_group_name} ---")
        for obs_term in obs_group:
            print(f"--- Obs term: {obs_term['name']} ---")
            obs_term.pop("name")
            for key, value in obs_term.items():
                print(f"{key}: {value}")

    for articulation_name, articulation_data in out_articulations.items():
        print(f"--- Articulation: {articulation_name} ---")
        for key, value in articulation_data.items():
            print(f"{key}: {value}")

    for key, value in out_scene.items():
        print(f"{key}: {value}")

    env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
