#!/usr/bin/env python3
"""通过 ROS topic 接收 model_use 并写入文件。

支持：
- ROS2 `rclpy`
- ROS1 `rospy`

默认消息类型是 `std_msgs/Int32`。
如果你的 topic 用的是字符串，也可以通过 `--msg-type string` 切换。
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Receive model_use from ROS topic and write it to a file.")
    parser.add_argument("--topic", type=str, default="/model_use", help="订阅的 topic 名称。")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/model_use.txt",
        help="写入 model_use 的目标文件路径。",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "ros2", "ros1"),
        help="优先使用的 ROS 后端。",
    )
    parser.add_argument(
        "--msg-type",
        type=str,
        default="int32",
        choices=("int32", "string"),
        help="topic 的消息类型。",
    )
    return parser.parse_args()


def normalize_model_use(value) -> int:
    """把接收到的消息转换成合法的 model_use。"""

    model_use = int(str(value).strip())
    if model_use not in (1, 2, 3):
        raise ValueError(f"model_use 必须是 1、2 或 3，收到: {model_use}")
    return model_use


def write_model_use(output_path: str, model_use: int):
    """把 model_use 写入文件。"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"{model_use}\n")


def run_ros2(args: argparse.Namespace):
    """使用 ROS2 订阅 topic。"""

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Int32, String

    msg_cls = Int32 if args.msg_type == "int32" else String

    class ModelUseRos2Node(Node):
        def __init__(self):
            super().__init__("model_use_switch_node")
            self.subscription = self.create_subscription(msg_cls, args.topic, self.callback, 10)
            self.last_model_use = None
            self.get_logger().info(f"Listening ROS2 topic: {args.topic}")
            self.get_logger().info(f"Output file: {args.output}")

        def callback(self, msg):
            raw_value = msg.data
            try:
                model_use = normalize_model_use(raw_value)
                if model_use != self.last_model_use:
                    write_model_use(args.output, model_use)
                    self.last_model_use = model_use
                    self.get_logger().info(f"model_use -> {model_use}")
            except ValueError as err:
                self.get_logger().warning(f"Ignore invalid message: {raw_value!r} ({err})")

    rclpy.init(args=None)
    node = ModelUseRos2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_ros1(args: argparse.Namespace):
    """使用 ROS1 订阅 topic。"""

    import rospy
    from std_msgs.msg import Int32, String

    msg_cls = Int32 if args.msg_type == "int32" else String
    state = {"last_model_use": None}

    def callback(msg):
        raw_value = msg.data
        try:
            model_use = normalize_model_use(raw_value)
            if model_use != state["last_model_use"]:
                write_model_use(args.output, model_use)
                state["last_model_use"] = model_use
                rospy.loginfo("model_use -> %s", model_use)
        except ValueError as err:
            rospy.logwarn("Ignore invalid message: %r (%s)", raw_value, err)

    rospy.init_node("model_use_switch_node", anonymous=True)
    rospy.loginfo("Listening ROS1 topic: %s", args.topic)
    rospy.loginfo("Output file: %s", args.output)
    rospy.Subscriber(args.topic, msg_cls, callback, queue_size=10)
    rospy.spin()


def main():
    """主入口。"""

    args = parse_args()

    if args.backend in ("auto", "ros2"):
        try:
            run_ros2(args)
            return
        except ModuleNotFoundError:
            if args.backend == "ros2":
                raise

    if args.backend in ("auto", "ros1"):
        try:
            run_ros1(args)
            return
        except ModuleNotFoundError:
            if args.backend == "ros1":
                raise

    print("[ERROR] Neither ROS2 (rclpy) nor ROS1 (rospy) is available in the current Python environment.")
    sys.exit(1)


if __name__ == "__main__":
    main()
