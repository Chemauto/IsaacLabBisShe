#!/usr/bin/env python3
"""通过 UDP 发送速度命令。

适用场景：
- 键盘实时微调 `vx / vy / wz`
- 在终端中输入 `forward`、`stop`、`set 0.5 0 0` 这类文本指令

默认发送到 `127.0.0.1:5555`，和 `SocketVelocityCommandCfg(port=5555)` 对应。
"""

from __future__ import annotations

import argparse
import socket
import sys
import termios
import tty


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Send velocity commands through UDP socket.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="目标主机地址。")
    parser.add_argument("--port", type=int, default=5555, help="目标 UDP 端口。")
    parser.add_argument(
        "--mode",
        type=str,
        default="keyboard",
        choices=("keyboard", "repl"),
        help="keyboard 表示按键控制，repl 表示文本指令输入。",
    )
    parser.add_argument("--vx-step", type=float, default=0.1, help="每次按键修改的前向速度步长。")
    parser.add_argument("--vy-step", type=float, default=0.1, help="每次按键修改的横向速度步长。")
    parser.add_argument("--wz-step", type=float, default=0.1, help="每次按键修改的角速度步长。")
    parser.add_argument("--max-vx", type=float, default=1.0, help="前向速度绝对值上限。")
    parser.add_argument("--max-vy", type=float, default=1.0, help="横向速度绝对值上限。")
    parser.add_argument("--max-wz", type=float, default=1.5, help="角速度绝对值上限。")
    return parser.parse_args()


class UdpCommandSender:
    """简单的 UDP 速度命令发送器。"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, vx: float, vy: float, wz: float):
        """发送一条 `vx,vy,wz` 形式的速度命令。"""

        message = f"{vx:.4f},{vy:.4f},{wz:.4f}"
        self.socket.sendto(message.encode("utf-8"), (self.host, self.port))
        print(f"[INFO] Sent -> vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")

    def close(self):
        """关闭 socket。"""

        self.socket.close()


def clip(value: float, limit: float) -> float:
    """把数值裁剪到给定范围。"""

    return max(-limit, min(limit, value))


def get_key() -> str:
    """读取单个按键，不需要按回车。"""

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def print_keyboard_help(args: argparse.Namespace):
    """打印键盘控制说明。"""

    print("=" * 56)
    print("Keyboard Velocity Command Sender")
    print("=" * 56)
    print("w/s : 增加/减小 vx")
    print("a/d : 增加/减小 vy")
    print("q/e : 增加/减小 wz")
    print("space: 速度清零")
    print("p    : 打印当前速度")
    print("ESC  : 退出")
    print("-" * 56)
    print(f"vx_step={args.vx_step}, vy_step={args.vy_step}, wz_step={args.wz_step}")
    print(f"max_vx={args.max_vx}, max_vy={args.max_vy}, max_wz={args.max_wz}")
    print("=" * 56)


def run_keyboard_mode(args: argparse.Namespace, sender: UdpCommandSender):
    """通过按键实时调整并发送速度命令。"""

    state = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    print_keyboard_help(args)
    sender.send(state["vx"], state["vy"], state["wz"])

    try:
        while True:
            key = get_key()
            changed = False

            if key == "\x1b":
                print("\n[INFO] Keyboard sender stopped.")
                break
            if key.lower() == "w":
                state["vx"] = clip(state["vx"] + args.vx_step, args.max_vx)
                changed = True
            elif key.lower() == "s":
                state["vx"] = clip(state["vx"] - args.vx_step, args.max_vx)
                changed = True
            elif key.lower() == "a":
                state["vy"] = clip(state["vy"] + args.vy_step, args.max_vy)
                changed = True
            elif key.lower() == "d":
                state["vy"] = clip(state["vy"] - args.vy_step, args.max_vy)
                changed = True
            elif key.lower() == "q":
                state["wz"] = clip(state["wz"] + args.wz_step, args.max_wz)
                changed = True
            elif key.lower() == "e":
                state["wz"] = clip(state["wz"] - args.wz_step, args.max_wz)
                changed = True
            elif key == " ":
                state = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
                changed = True
            elif key.lower() == "p":
                print(f"\n[INFO] Current -> vx={state['vx']:.3f}, vy={state['vy']:.3f}, wz={state['wz']:.3f}")

            if changed:
                sender.send(state["vx"], state["vy"], state["wz"])
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard sender interrupted.")
    finally:
        # 退出前主动发一次 0，避免机器人继续保留上一条速度命令。
        sender.send(0.0, 0.0, 0.0)


def parse_text_command(command: str) -> tuple[float, float, float] | None:
    """把文本指令转换成 `vx, vy, wz`。"""

    parts = command.strip().split()
    if not parts:
        return None

    keyword = parts[0].lower()
    if keyword in ("quit", "exit"):
        raise SystemExit
    if keyword == "stop":
        return 0.0, 0.0, 0.0
    if keyword in ("forward", "f"):
        speed = float(parts[1]) if len(parts) > 1 else 0.5
        return speed, 0.0, 0.0
    if keyword in ("backward", "back", "b"):
        speed = float(parts[1]) if len(parts) > 1 else 0.3
        return -abs(speed), 0.0, 0.0
    if keyword in ("left", "l"):
        speed = float(parts[1]) if len(parts) > 1 else 0.3
        return 0.0, abs(speed), 0.0
    if keyword in ("right", "r"):
        speed = float(parts[1]) if len(parts) > 1 else 0.3
        return 0.0, -abs(speed), 0.0
    if keyword in ("turn_left", "tl"):
        speed = float(parts[1]) if len(parts) > 1 else 0.4
        return 0.0, 0.0, abs(speed)
    if keyword in ("turn_right", "tr"):
        speed = float(parts[1]) if len(parts) > 1 else 0.4
        return 0.0, 0.0, -abs(speed)
    if keyword in ("set", "send"):
        if len(parts) != 4:
            raise ValueError("`set` / `send` 需要 3 个数值，例如: set 0.5 0.0 0.0")
        return float(parts[1]), float(parts[2]), float(parts[3])

    raise ValueError(
        "不支持的指令。可用示例：forward 0.5 / left 0.3 / turn_left 0.4 / stop / set 0.5 0 0"
    )


def run_repl_mode(sender: UdpCommandSender):
    """通过文本命令发送速度。"""

    print("=" * 56)
    print("Command REPL")
    print("=" * 56)
    print("支持的指令：")
    print("  forward 0.5")
    print("  backward 0.3")
    print("  left 0.3")
    print("  right 0.3")
    print("  turn_left 0.4")
    print("  turn_right 0.4")
    print("  set 0.5 0.0 0.0")
    print("  stop")
    print("  quit")
    print("=" * 56)

    try:
        while True:
            try:
                user_input = input("cmd> ")
                result = parse_text_command(user_input)
                if result is None:
                    continue
                sender.send(*result)
            except ValueError as err:
                print(f"[WARN] {err}")
    except SystemExit:
        print("[INFO] Command sender stopped.")
    except KeyboardInterrupt:
        print("\n[INFO] Command sender interrupted.")
    finally:
        sender.send(0.0, 0.0, 0.0)


def main():
    """主入口。"""

    args = parse_args()
    sender = UdpCommandSender(args.host, args.port)

    print(f"[INFO] Target -> {args.host}:{args.port}")
    print(f"[INFO] Mode   -> {args.mode}")

    try:
        if args.mode == "keyboard":
            run_keyboard_mode(args, sender)
        else:
            run_repl_mode(sender)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
