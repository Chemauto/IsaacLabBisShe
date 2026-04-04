# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""直接从当前 EnvTest player 导出的前视相机图片保存一张快照。"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Save one front-camera snapshot from the running EnvTest player.")
    parser.add_argument(
        "--camera_image_file",
        type=str,
        default="/tmp/envtest_front_camera.png",
        help="NewTools/envtest_model_use_player.py 实时导出的前视相机图片路径。",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/tmp/envtest_front_camera_snapshot.png",
        help="这次快照保存到哪里。",
    )
    parser.add_argument("--wait_timeout", type=float, default=5.0, help="等待最新相机图片的最长时间，单位秒。")
    parser.add_argument("--poll_interval", type=float, default=0.1, help="轮询相机图片文件的间隔，单位秒。")
    return parser.parse_args()


def wait_for_camera_image(camera_image_file: str, wait_timeout: float, poll_interval: float) -> None:
    """等待 player 刷新出一张当前相机图片。"""

    start_time = time.time()
    deadline_time = start_time + wait_timeout
    while time.time() <= deadline_time:
        if (
            os.path.isfile(camera_image_file)
            and os.path.getsize(camera_image_file) > 0
            and os.path.getmtime(camera_image_file) >= start_time
        ):
            return
        time.sleep(poll_interval)

    raise TimeoutError(
        f"没有等到新的相机图片: {camera_image_file}。"
        "请先用 --enable_front_camera 启动 NewTools/envtest_model_use_player.py。"
    )


def main() -> None:
    """保存当前 front_camera 快照。"""

    args_cli = parse_args()
    wait_for_camera_image(args_cli.camera_image_file, args_cli.wait_timeout, args_cli.poll_interval)

    save_dir = os.path.dirname(args_cli.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    rgb_image = None
    last_error = None
    for _ in range(20):
        shutil.copy2(args_cli.camera_image_file, args_cli.save_path)
        try:
            rgb_image = imageio.imread(args_cli.save_path)
            break
        except OSError as error:
            last_error = error
            time.sleep(args_cli.poll_interval)

    if rgb_image is None:
        raise OSError(f"保存后的图片仍然无法读取: {args_cli.save_path}") from last_error

    print(f"[INFO] 已读取当前相机图片: {args_cli.camera_image_file}")
    print(f"[INFO] 已保存快照到: {args_cli.save_path}")
    print(f"[INFO] 图像尺寸: {tuple(rgb_image.shape)}")


if __name__ == "__main__":
    main()
