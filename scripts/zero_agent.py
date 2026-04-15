# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# EnvTest 专用参数：传 0~4，对应 case1~case5。
parser.add_argument("--scene_id", type=int, default=None, help="Scene id for EnvTest play mode. Valid values: 0-4.")
parser.add_argument("--hm3d_scene_name", type=str, default=None, help="Optional HM3D scene name for EnvTest.")
parser.add_argument(
    "--hm3d_robot_pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional HM3D robot spawn position for EnvTest.",
)
parser.add_argument("--hm3d_robot_yaw", type=float, default=None, help="Optional HM3D robot spawn yaw for EnvTest.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# 默认开启相机渲染，避免带相机的环境因缺少 --enable_cameras 报错。
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MyProject.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.scene_id is not None:
        if not 0 <= args_cli.scene_id <= 4:
            raise ValueError("--scene_id must be in [0, 4].")
        if hasattr(env_cfg, "scene_id"):
            # 这里直接使用 EnvTest 内部的 0~4 场景编号。
            env_cfg.scene_id = args_cli.scene_id
    if args_cli.hm3d_scene_name is not None and hasattr(env_cfg, "hm3d_scene_name"):
        env_cfg.hm3d_scene_name = args_cli.hm3d_scene_name
    if args_cli.hm3d_robot_pos is not None and hasattr(env_cfg, "hm3d_robot_pos"):
        env_cfg.hm3d_robot_pos = tuple(args_cli.hm3d_robot_pos)
    if args_cli.hm3d_robot_yaw is not None and hasattr(env_cfg, "hm3d_robot_yaw"):
        env_cfg.hm3d_robot_yaw = args_cli.hm3d_robot_yaw
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
