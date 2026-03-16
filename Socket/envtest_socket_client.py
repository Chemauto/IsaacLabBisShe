#!/usr/bin/env python3
"""EnvTest UDP 控制客户端。

示例：
- `python Socket/envtest_socket_client.py --model_use 1`
- `python Socket/envtest_socket_client.py --velocity 0.6 0 0`
- `python Socket/envtest_socket_client.py --goal 1.8 0 0.1`
- `python Socket/envtest_socket_client.py --model_use 3 --goal 1.8 0 0.1 --start 1`
- `python Socket/envtest_socket_client.py --text "model_use=2; start=1"`
"""

from __future__ import annotations

import argparse
import socket


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Send EnvTest control commands through UDP socket.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="目标主机地址。")
    parser.add_argument("--port", type=int, default=5566, help="目标 UDP 端口。")
    parser.add_argument("--model_use", type=int, choices=(0, 1, 2, 3), help="技能编号：0/1/2/3。")
    parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "WZ"), help="速度指令。")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"), help="位置指令。")
    parser.add_argument("--goal_auto", action="store_true", default=False, help="把 goal 切回自动场景目标。")
    parser.add_argument("--start", type=int, choices=(0, 1), help="启动开关：0=待机，1=开始。")
    parser.add_argument("--text", type=str, default="", help="直接发送原始文本，不走字段拼接。")
    return parser.parse_args()


def build_message(args: argparse.Namespace) -> str:
    """根据命令行参数拼接 UDP 消息。"""

    if args.text:
        return args.text

    fields: list[str] = []
    if args.model_use is not None:
        fields.append(f"model_use={args.model_use}")
    if args.velocity is not None:
        vx, vy, wz = args.velocity
        fields.append(f"velocity={vx},{vy},{wz}")
    if args.goal_auto:
        fields.append("goal=auto")
    if args.goal is not None:
        x, y, z = args.goal
        fields.append(f"goal={x},{y},{z}")
    if args.start is not None:
        fields.append(f"start={args.start}")

    if not fields:
        raise ValueError("至少提供一个控制字段，或使用 --text。")
    return "; ".join(fields)


def main():
    """主入口。"""

    args = parse_args()
    message = build_message(args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode("utf-8"), (args.host, args.port))
    finally:
        sock.close()

    print(f"[INFO] Sent -> {message}")
    print(f"[INFO] Target -> {args.host}:{args.port}")


if __name__ == "__main__":
    main()
