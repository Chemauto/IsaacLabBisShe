# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""读取 EnvTest 机器人前视相机图像的最小示例。"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数。
parser = argparse.ArgumentParser(description="预览并保存 EnvTest 前视相机图像。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="关闭 Fabric，改用 USD I/O。"
)
parser.add_argument("--num_envs", type=int, default=1, help="要启动的环境数量。")
parser.add_argument("--task", type=str, default="Template-EnvTest-Go2-Play-v0", help="任务名。")
parser.add_argument("--scene_id", type=int, default=0, help="EnvTest 场景编号，可选 0-4。")
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="最多运行多少个仿真步，0 表示一直运行到手动关闭。",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/tmp/envtest_front_camera.png",
    help="保存一帧 RGB 图像的路径，传空字符串表示不保存。",
)
# 复用 Isaac Lab 的启动参数，并强制打开相机支持。
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# 启动 Omniverse / Isaac Sim。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio.v2 as imageio
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MyProject.tasks  # noqa: F401


def _save_rgb_frame(rgb_tensor: torch.Tensor, save_path: str):
    """把第一张环境图像保存到磁盘。"""

    rgb_image = rgb_tensor.detach().cpu().numpy()
    # 某些情况下相机会返回 RGBA，这里统一截成 RGB 三通道。
    if rgb_image.shape[-1] > 3:
        rgb_image = rgb_image[..., :3]
    imageio.imwrite(save_path, rgb_image)


def main():
    """启动环境并持续读取前视相机图像。"""

    if not 0 <= args_cli.scene_id <= 4:
        raise ValueError("--scene_id must be in [0, 4].")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if hasattr(env_cfg, "scene_id"):
        # 这里直接使用 EnvTest 内部的 0~4 场景编号。
        env_cfg.scene_id = args_cli.scene_id

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    camera = env.unwrapped.scene["front_camera"]
    print(f"[INFO]: 当前场景编号: {args_cli.scene_id}")
    print(f"[INFO]: 相机数据类型: {list(camera.data.output.keys())}")

    has_saved_frame = False
    frame_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # 这里不控制机器人，只保持零动作并持续刷新传感器。
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

            rgb = camera.data.output["rgb"]
            depth = camera.data.output["distance_to_image_plane"]

            # 每隔一段时间打印一次图像尺寸，避免终端刷屏过快。
            if frame_count % 30 == 0:
                print(f"[INFO]: RGB shape={tuple(rgb.shape)}, depth shape={tuple(depth.shape)}")

            if args_cli.save_path and not has_saved_frame and frame_count >= 5:
                _save_rgb_frame(rgb[0], args_cli.save_path)
                print(f"[INFO]: 已保存一帧 RGB 图像到: {args_cli.save_path}")
                has_saved_frame = True

            frame_count += 1
            if args_cli.max_steps > 0 and frame_count >= args_cli.max_steps:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
