#!/usr/bin/env python3
"""通过 UDP Socket 接收 model_use 并写入文件。"""

from __future__ import annotations

import argparse
import re
import socket


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Receive model_use through UDP socket and write it to a file.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址。")
    parser.add_argument("--port", type=int, default=5566, help="监听端口。")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/model_use.txt",
        help="写入 model_use 的目标文件路径。",
    )
    return parser.parse_args()


def extract_model_use(message: str) -> int:
    """从消息里提取 model_use。

    支持的输入示例：
    - `1`
    - `model_use=2`
    - `skill: 3`
    """

    text = message.strip()
    if not text:
        raise ValueError("收到空消息。")

    match = re.search(r"(-?\d+)", text)
    if match is None:
        raise ValueError(f"无法从消息中解析整数: {message!r}")

    model_use = int(match.group(1))
    if model_use not in (1, 2, 3):
        raise ValueError(f"model_use 必须是 1、2 或 3，收到: {model_use}")
    return model_use


def write_model_use(output_path: str, model_use: int):
    """把 model_use 写入文件。"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"{model_use}\n")


def main():
    """主入口。"""

    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print(f"[INFO] UDP model_use server listening on {args.host}:{args.port}")
    print(f"[INFO] Output file: {args.output}")
    print("[INFO] Send examples: '1', '2', '3', 'model_use=2'")

    try:
        while True:
            data, address = sock.recvfrom(1024)
            raw_message = data.decode("utf-8", errors="ignore").strip()
            try:
                model_use = extract_model_use(raw_message)
                write_model_use(args.output, model_use)
                print(f"[INFO] {address[0]}:{address[1]} -> model_use={model_use}")
            except ValueError as err:
                print(f"[WARN] Ignore invalid message from {address}: {raw_message!r} ({err})")
    except KeyboardInterrupt:
        print("\n[INFO] UDP model_use server stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
