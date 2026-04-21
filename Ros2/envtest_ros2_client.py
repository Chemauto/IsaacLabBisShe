#!/usr/bin/env python3
"""EnvTest ROS2 控制客户端。

用于手动向 `Ros2/envtest_ros2_server.py` 发送控制命令。
正常联调 FinalProject 时不需要启动这个 client。
"""

from __future__ import annotations

import argparse
import json
import time

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    ROS2_AVAILABLE = True
except Exception:
    rclpy = None
    Node = object
    String = None
    ROS2_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send EnvTest control commands through ROS2 /go2/skill_command.")
    parser.add_argument("--skill-command-topic", default="/go2/skill_command")
    parser.add_argument("--model_use", type=int, choices=(0, 1, 2, 3, 4, 5), help="技能编号：0/1/2/3/4/5。")
    parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "VZ"), help="速度指令。")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"), help="位置指令。")
    parser.add_argument("--goal_auto", action="store_true", default=False, help="把 goal 切回自动场景目标。")
    parser.add_argument("--start", type=int, choices=(0, 1), help="启动开关：0=待机，1=开始。")
    parser.add_argument("--reset", type=int, choices=(1, 2), help="一次性重置：1=重置环境，2=只重置机器人。")
    parser.add_argument("--text", type=str, default="", help="直接发送原始文本，不走 JSON 拼接。")
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> str:
    if args.text:
        return args.text

    payload = {}
    if args.model_use is not None:
        payload["model_use"] = args.model_use
    if args.velocity is not None:
        payload["velocity"] = list(args.velocity)
    if args.model_use == 3 and args.goal is None and not args.goal_auto:
        payload["goal"] = "auto"
    if args.goal_auto:
        payload["goal"] = "auto"
    if args.goal is not None:
        payload["goal"] = list(args.goal)
    if args.start is not None:
        payload["start"] = bool(args.start)
    if args.reset is not None:
        payload["reset"] = args.reset

    if not payload:
        raise ValueError("至少提供一个控制字段，或使用 --text。")
    return json.dumps(payload, ensure_ascii=False)


def main() -> int:
    if not ROS2_AVAILABLE:
        raise SystemExit("ROS2 Python packages are not available. Source your ROS2 environment first.")

    args = parse_args()
    payload = build_payload(args)

    rclpy.init()
    node = rclpy.create_node("envtest_ros2_control_client")
    publisher = node.create_publisher(String, args.skill_command_topic, 10)

    msg = String()
    msg.data = payload

    # Give ROS2 discovery a brief moment before the one-shot publish.
    end_time = time.time() + 0.5
    while time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.05)

    publisher.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)
    print(f"[INFO] Sent -> {payload}")
    print(f"[INFO] Topic -> {args.skill_command_topic}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
