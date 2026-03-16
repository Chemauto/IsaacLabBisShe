#!/usr/bin/env python3
"""通过键盘 1/2/3 切换 model_use 并写入文件。"""

from __future__ import annotations

import argparse
import sys
import termios
import tty


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Switch model_use by pressing keyboard keys 1/2/3.")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/model_use.txt",
        help="写入 model_use 的目标文件路径。",
    )
    return parser.parse_args()


def write_model_use(output_path: str, model_use: int):
    """把 model_use 写入文件。"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"{model_use}\n")


def get_key() -> str:
    """读取单个按键，不需要回车。"""

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def main():
    """主入口。"""

    args = parse_args()
    current_model_use = None

    print("=" * 48)
    print("Keyboard model_use switch")
    print("=" * 48)
    print(f"Output file: {args.output}")
    print("Press 1 -> walk")
    print("Press 2 -> climb")
    print("Press 3 -> push_box")
    print("Press q or ESC -> quit")
    print("=" * 48)

    try:
        while True:
            key = get_key()
            if key in ("\x1b", "q", "Q"):
                print("\n[INFO] Keyboard switch stopped.")
                break
            if key not in ("1", "2", "3"):
                continue

            model_use = int(key)
            if model_use != current_model_use:
                write_model_use(args.output, model_use)
                current_model_use = model_use
                print(f"\r[INFO] model_use -> {model_use}    ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard switch interrupted.")


if __name__ == "__main__":
    main()
