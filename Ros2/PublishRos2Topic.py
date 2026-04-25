#!/usr/bin/env python3
"""EnvTest ROS2 topic bridge.

职责：
- 订阅 FinalProject 发布的 `/go2/*` 控制 topic
- 转成 `FinalSim.py` 支持的文本控制协议
- 写入 `FinalSim.py` 默认读取的控制文件

这个脚本让 EnvTest 仿真对外表现得像真实 ROS2 机器人。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import String

    ROS2_AVAILABLE = True
except Exception:
    rclpy = None
    Node = object
    PoseStamped = Twist = Odometry = String = None
    ROS2_AVAILABLE = False


SKILL_NAME_TO_ID = {
    "idle": 0,
    "walk": 1,
    "climb": 2,
    "push": 3,
    "push_box": 3,
    "nav": 4,
    "navigation": 4,
    "navigation_bishe": 4,
    "nav_climb": 5,
    "navigation_climb": 5,
}
BOOL_TRUE = {"1", "true", "on", "run", "start", "yes", "y"}
BOOL_FALSE = {"0", "false", "off", "stop", "idle", "no", "n"}


@dataclass
class OutputPaths:
    model_use: str
    velocity: str
    goal: str
    start: str
    reset: str


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_text(file_path: str, text: str) -> None:
    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text.strip() + "\n")


def _parse_named_int(text: str, field_name: str) -> int | None:
    match = re.search(rf"\b{field_name}\b\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    return None if match is None else int(match.group(1))


def _parse_named_vector(text: str, field_name: str) -> tuple[float, float, float] | None:
    pattern = (
        rf"\b{field_name}\b\s*[:=]\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"[\s,]+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"[\s,]+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _parse_auto_goal(text: str) -> bool:
    match = re.search(r"\b(goal|position|pos|target)\b\s*[:=]\s*([A-Za-z_]+)", text, flags=re.IGNORECASE)
    return match is not None and match.group(2).strip().lower() in ("auto", "scene", "default")


def _parse_start(text: str) -> bool | None:
    match = re.search(r"\b(start|run)\b\s*[:=]\s*([A-Za-z0-9_]+)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    token = match.group(2).strip().lower()
    if token in BOOL_TRUE:
        return True
    if token in BOOL_FALSE:
        return False
    raise ValueError(f"无法识别 start 值: {token}")


def _parse_reset(text: str) -> int | None:
    normalized = text.strip().lower()
    if normalized == "reset":
        return 1
    match = re.search(r"\breset\b\s*[:=]\s*([A-Za-z0-9_]+)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    token = match.group(1).strip().lower()
    if token == "2":
        return 2
    if token == "reset" or token in BOOL_TRUE:
        return 1
    if token in BOOL_FALSE:
        return None
    raise ValueError(f"无法识别 reset 值: {token}")


def _parse_skill_name(text: str) -> int | None:
    return SKILL_NAME_TO_ID.get(text.strip().lower())


def _format_file_vector(values: tuple[float, float, float]) -> str:
    return f"{values[0]} {values[1]} {values[2]}"


def apply_message(text: str, output_paths: OutputPaths) -> list[str]:
    updates: list[str] = []
    normalized = text.strip()
    if not normalized:
        raise ValueError("收到空消息。")

    skill_id = _parse_named_int(normalized, "model_use")
    if skill_id is None:
        skill_id = _parse_named_int(normalized, "skill")
    if skill_id is None:
        skill_id = _parse_skill_name(normalized)
    if skill_id is not None:
        if skill_id not in (0, 1, 2, 3, 4, 5):
            raise ValueError(f"model_use 必须是 0/1/2/3/4/5，收到: {skill_id}")
        _write_text(output_paths.model_use, str(skill_id))
        updates.append(f"model_use={skill_id}")

    velocity = _parse_named_vector(normalized, "velocity")
    if velocity is None:
        velocity = _parse_named_vector(normalized, "vel")
    if velocity is not None:
        _write_text(output_paths.velocity, _format_file_vector(velocity))
        updates.append(f"velocity={velocity}")

    goal = _parse_named_vector(normalized, "goal")
    if goal is None:
        goal = _parse_named_vector(normalized, "position")
    if goal is None:
        goal = _parse_named_vector(normalized, "pos")
    if goal is None:
        goal = _parse_named_vector(normalized, "target")
    auto_goal_requested = _parse_auto_goal(normalized)
    if goal is not None:
        _write_text(output_paths.goal, _format_file_vector(goal))
        updates.append(f"goal={goal}")
    elif auto_goal_requested:
        _write_text(output_paths.goal, "auto")
        updates.append("goal=auto")
    elif skill_id == 3:
        _write_text(output_paths.goal, "auto")
        updates.append("goal=auto(scene)")

    start = _parse_start(normalized)
    if start is not None:
        _write_text(output_paths.start, "1" if start else "0")
        updates.append(f"start={int(start)}")

    reset_mode = _parse_reset(normalized)
    if reset_mode is not None:
        _write_text(output_paths.reset, str(reset_mode))
        updates.append(f"reset={reset_mode}")

    if not updates:
        raise ValueError("未识别到有效字段。支持：model_use / skill / velocity / goal / start / reset。")
    return updates


def _float_list(values: Iterable[Any], max_len: int | None = None) -> list[float]:
    items = list(values)
    if max_len is not None:
        items = items[:max_len]
    return [float(value) for value in items]


def _format_vector(values: Any) -> str:
    return ",".join(str(float(value)) for value in list(values)[:3])


def _bool_to_start(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y", "on", "run", "start"} else 0
    return 1 if bool(value) else 0


def payload_to_text(payload: dict[str, Any]) -> str:
    """Convert FinalProject `/go2/skill_command` JSON into EnvTest text protocol."""
    fields: list[str] = []

    if payload.get("model_use") is not None:
        fields.append(f"model_use={int(payload['model_use'])}")
    elif payload.get("skill") is not None:
        fields.append(f"skill={payload['skill']}")

    velocity = payload.get("velocity") or payload.get("vel_command")
    if velocity is not None:
        fields.append(f"velocity={_format_vector(velocity)}")

    goal = payload.get("goal")
    if goal is not None:
        fields.append("goal=auto" if goal == "auto" else f"goal={_format_vector(goal)}")

    if payload.get("start") is not None:
        fields.append(f"start={_bool_to_start(payload['start'])}")

    if payload.get("reset") is not None:
        fields.append(f"reset={int(payload['reset'])}")

    return "; ".join(fields)


def skill_command_to_text(data: str, output_paths: OutputPaths) -> list[str]:
    """Apply one `/go2/skill_command` String payload to EnvTest control files."""
    text = data.strip()
    if not text:
        raise ValueError("收到空 skill_command。")

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None

    if isinstance(decoded, dict):
        text = payload_to_text(decoded)

    return apply_message(text, output_paths)


def twist_message_to_text(msg: Any) -> str:
    linear = msg.linear
    return f"velocity={float(linear.x)},{float(linear.y)},{float(linear.z)}"


def pose_message_to_text(msg: Any) -> str:
    position = msg.pose.position
    return f"goal={float(position.x)},{float(position.y)},{float(position.z)}"


def read_status_snapshot(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _asset_to_alignment(asset: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(asset, dict):
        return None
    return {
        "name": asset.get("name"),
        "position": _float_list(asset.get("position") or [], max_len=3),
        "size": _float_list(asset.get("size") or [], max_len=3),
    }


def build_envtest_alignment(snapshot: dict[str, Any]) -> dict[str, Any]:
    alignment = {}
    for key in ("platform_1", "platform_2", "box"):
        asset = _asset_to_alignment(snapshot.get(key))
        if asset is not None:
            alignment[key] = asset
    return alignment


def build_skill_status(snapshot: dict[str, Any]) -> dict[str, Any]:
    status = {
        "timestamp": snapshot.get("timestamp"),
        "model_use": snapshot.get("model_use"),
        "skill": snapshot.get("skill"),
        "scene_id": snapshot.get("scene_id"),
        "start": snapshot.get("start"),
        "goal": snapshot.get("goal"),
        "vel_command": snapshot.get("vel_command"),
    }
    alignment = build_envtest_alignment(snapshot)
    if alignment:
        status["envtest_alignment"] = alignment
    return status


def _asset_to_scene_object(asset: dict[str, Any] | None, object_type: str) -> dict[str, Any] | None:
    if not isinstance(asset, dict):
        return None
    position = asset.get("position")
    size = asset.get("size")
    if not isinstance(position, list) or not isinstance(size, list):
        return None
    return {
        "id": str(asset.get("name") or object_type),
        "type": object_type,
        "center": _float_list(position, max_len=3),
        "size": _float_list(size, max_len=3),
        "movable": object_type == "box",
    }


def build_scene_objects(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for key in ("platform_1", "platform_2"):
        obj = _asset_to_scene_object(snapshot.get(key), "platform")
        if obj is not None:
            objects.append(obj)
    box = _asset_to_scene_object(snapshot.get("box"), "box")
    if box is not None:
        objects.append(box)
    return objects


def box_pose_values(snapshot: dict[str, Any]) -> list[float] | None:
    """Return the simulated box xyz position from the EnvTest status snapshot."""
    box = snapshot.get("box")
    if not isinstance(box, dict):
        return None
    position = box.get("position")
    if not isinstance(position, list) or len(position) < 3:
        return None
    return _float_list(position, max_len=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge EnvTest JSON status and control files with ROS2 topics.")
    parser.add_argument("--model-use-file", type=str, default="/tmp/model_use.txt", help="model_use 文件路径。")
    parser.add_argument(
        "--velocity-file",
        type=str,
        default="/tmp/envtest_velocity_command.txt",
        help="速度指令文件路径。",
    )
    parser.add_argument(
        "--goal-file",
        type=str,
        default="/tmp/envtest_goal_command.txt",
        help="位置指令文件路径。",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        default="/tmp/envtest_start.txt",
        help="启动开关文件路径。",
    )
    parser.add_argument(
        "--reset-file",
        type=str,
        default="/tmp/envtest_reset.txt",
        help="一次性环境重置文件路径。",
    )
    parser.add_argument("--skill-command-topic", default="/go2/skill_command")
    parser.add_argument("--cmd-vel-topic", default="/go2/cmd_vel")
    parser.add_argument("--goal-pose-topic", default="/go2/goal_pose")
    parser.add_argument("--status-json", default="/tmp/envtest_live_status.json", help="EnvTest live status JSON 路径。")
    parser.add_argument("--publish-hz", type=float, default=10.0, help="状态 topic 发布频率。")
    parser.add_argument("--odom-topic", default="/go2/odom")
    parser.add_argument("--box-pose-topic", default="/go2/box_pose")
    parser.add_argument("--skill-status-topic", default="/go2/skill_status")
    parser.add_argument("--scene-objects-topic", default="/go2/scene_objects")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--child-frame-id", default="base")
    return parser.parse_args()


class EnvTestRos2ControlNode(Node):
    def __init__(self, output_paths: OutputPaths, args: argparse.Namespace):
        super().__init__("publish_ros2_topic")
        self.output_paths = output_paths
        self.args = args
        self.status_path = Path(args.status_json)

        self.create_subscription(String, args.skill_command_topic, self._on_skill_command, 10)
        self.create_subscription(Twist, args.cmd_vel_topic, self._on_cmd_vel, 10)
        self.create_subscription(PoseStamped, args.goal_pose_topic, self._on_goal_pose, 10)

        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        self.box_pose_pub = self.create_publisher(PoseStamped, args.box_pose_topic, 10)
        self.skill_status_pub = self.create_publisher(String, args.skill_status_topic, 10)
        self.scene_objects_pub = self.create_publisher(String, args.scene_objects_topic, 10)
        period = 1.0 / max(float(args.publish_hz), 0.1)
        self.create_timer(period, self._publish_status_snapshot)

        self.get_logger().info("EnvTest ROS2 control server started.")
        self.get_logger().info(f"skill_command: {args.skill_command_topic}")
        self.get_logger().info(f"cmd_vel      : {args.cmd_vel_topic}")
        self.get_logger().info(f"goal_pose    : {args.goal_pose_topic}")
        self.get_logger().info(f"odom         : {args.odom_topic}")
        self.get_logger().info(f"box_pose     : {args.box_pose_topic}")
        self.get_logger().info(f"skill_status : {args.skill_status_topic}")
        self.get_logger().info(f"scene_objects: {args.scene_objects_topic}")
        self.get_logger().info(f"status json  : {self.status_path}")
        self.get_logger().info(f"model_use file: {output_paths.model_use}")
        self.get_logger().info(f"velocity file : {output_paths.velocity}")
        self.get_logger().info(f"goal file     : {output_paths.goal}")
        self.get_logger().info(f"start file    : {output_paths.start}")
        self.get_logger().info(f"reset file    : {output_paths.reset}")

    def _apply_text(self, text: str, source: str) -> None:
        try:
            updates = apply_message(text, self.output_paths)
        except ValueError as exc:
            self.get_logger().warn(f"Ignore invalid {source}: {text!r} ({exc})")
            return
        self.get_logger().info(f"{source} -> " + ", ".join(updates))

    def _on_skill_command(self, msg: Any) -> None:
        try:
            updates = skill_command_to_text(msg.data, self.output_paths)
        except ValueError as exc:
            self.get_logger().warn(f"Ignore invalid skill_command: {msg.data!r} ({exc})")
            return
        self.get_logger().info("skill_command -> " + ", ".join(updates))

    def _on_cmd_vel(self, msg: Any) -> None:
        self._apply_text(twist_message_to_text(msg), "cmd_vel")

    def _on_goal_pose(self, msg: Any) -> None:
        self._apply_text(pose_message_to_text(msg), "goal_pose")

    def _publish_status_snapshot(self) -> None:
        snapshot = read_status_snapshot(self.status_path)
        if snapshot is None:
            return
        self._publish_odom(snapshot)
        self._publish_box_pose(snapshot)
        self._publish_json(self.skill_status_pub, build_skill_status(snapshot))
        self._publish_json(self.scene_objects_pub, build_scene_objects(snapshot))

    def _publish_odom(self, snapshot: dict[str, Any]) -> None:
        pose = snapshot.get("robot_pose")
        if not isinstance(pose, list) or len(pose) < 3:
            return
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.child_frame_id = self.args.child_frame_id
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = float(pose[2])
        msg.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(msg)

    def _publish_box_pose(self, snapshot: dict[str, Any]) -> None:
        position = box_pose_values(snapshot)
        if position is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = 1.0
        self.box_pose_pub.publish(msg)

    @staticmethod
    def _publish_json(publisher: Any, payload: Any) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)


def main() -> int:
    if not ROS2_AVAILABLE:
        raise SystemExit("ROS2 Python packages are not available. Source your ROS2 environment first.")

    args = parse_args()
    output_paths = OutputPaths(
        model_use=args.model_use_file,
        velocity=args.velocity_file,
        goal=args.goal_file,
        start=args.start_file,
        reset=args.reset_file,
    )

    rclpy.init()
    node = EnvTestRos2ControlNode(output_paths, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("EnvTest ROS2 control server stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
