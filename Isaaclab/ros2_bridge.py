#!/usr/bin/env python3
"""ROS2 bridge for IsaacLab (FinalSim).

Subscribes to /go2/* ROS2 topics from robot_service.py and writes control
files that FinalSim.py reads.  Reads FinalSim's status JSON and publishes
/go2/* ROS2 topics back for robot_service.py.

Pure ROS2 -- no DDS dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 <-> FinalSim file bridge.")
    parser.add_argument("--status-json", default="/tmp/envtest_live_status.json")
    parser.add_argument("--control-dir", default="/tmp")
    parser.add_argument("--odom-topic", default="/go2/odom")
    parser.add_argument("--box-pose-topic", default="/go2/box_pose")
    parser.add_argument("--scene-objects-topic", default="/go2/scene_objects")
    parser.add_argument("--skill-status-topic", default="/go2/skill_status")
    parser.add_argument("--skill-command-topic", default="/go2/skill_command")
    parser.add_argument("--goal-pose-topic", default="/go2/goal_pose")
    parser.add_argument("--cmd-vel-topic", default="/rl_cmd_vel")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--publish-hz", type=float, default=10.0)
    return parser.parse_args()


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.strip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_status_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class IsaacLabBridge(Node):
    """ROS2 <-> FinalSim file bridge."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("isaaclab_ros2_bridge")
        self.args = args
        self.control_dir = Path(args.control_dir)
        self.status_json = Path(args.status_json)

        # ── ROS2 Publishers (state -> robot_service.py) ──
        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        self.box_pose_pub = self.create_publisher(PoseStamped, args.box_pose_topic, 10)
        self.scene_objects_pub = self.create_publisher(String, args.scene_objects_topic, 10)
        self.skill_status_pub = self.create_publisher(String, args.skill_status_topic, 10)

        # ── ROS2 Subscribers (commands <- robot_service.py) ──
        self.create_subscription(String, args.skill_command_topic, self._on_skill_command, 10)
        self.create_subscription(PoseStamped, args.goal_pose_topic, self._on_goal_pose, 10)
        self.create_subscription(String, args.cmd_vel_topic, self._on_cmd_vel, 10)

        # ── Timer: publish state at fixed rate ──
        self.create_timer(1.0 / max(args.publish_hz, 0.1), self._publish_state)

        self.get_logger().info(f"IsaacLab bridge started")
        self.get_logger().info(f"  status_json:     {args.status_json}")
        self.get_logger().info(f"  control_dir:     {args.control_dir}")
        self.get_logger().info(f"  skill_command:   {args.skill_command_topic}")
        self.get_logger().info(f"  cmd_vel:         {args.cmd_vel_topic}")

    # ── ROS2 callbacks: write files for FinalSim ──

    def _on_skill_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return

        _write_file(self.control_dir / "skill_command.json", json.dumps(payload, ensure_ascii=False))
        if payload.get("model_use") is not None:
            _write_file(self.control_dir / "model_use.txt", str(int(payload["model_use"])))
        velocity = payload.get("velocity")
        if isinstance(velocity, list) and len(velocity) >= 3:
            _write_file(self.control_dir / "envtest_velocity_command.txt", f"{velocity[0]} {velocity[1]} {velocity[2]}")
        goal = payload.get("goal")
        if isinstance(goal, list) and len(goal) >= 3:
            _write_file(self.control_dir / "envtest_goal_command.txt", f"{goal[0]} {goal[1]} {goal[2]}")
        elif goal == "auto":
            _write_file(self.control_dir / "envtest_goal_command.txt", "auto")
        if payload.get("start") is not None:
            _write_file(self.control_dir / "envtest_start.txt", "1" if payload["start"] else "0")
        if payload.get("reset") is not None:
            _write_file(self.control_dir / "envtest_reset.txt", str(int(payload["reset"])))

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        _write_file(self.control_dir / "envtest_goal_command.txt", f"{p.x} {p.y} {p.z}")

    def _on_cmd_vel(self, msg: String) -> None:
        # cmd_vel comes as Twist on /rl_cmd_vel, but we also get it as String
        # from ros2_state.py's skill_command. Handle Twist format here.
        pass

    # ── Timer: read FinalSim status, publish ROS2 ──

    def _publish_state(self) -> None:
        snap = _read_status_json(self.status_json)
        if snap is None:
            return
        self._publish_odom(snap)
        self._publish_box_pose(snap)
        self._publish_scene_objects(snap)
        self._publish_skill_status(snap)

    def _publish_odom(self, snap: dict) -> None:
        pose = snap.get("robot_pose")
        if not isinstance(pose, list) or len(pose) < 3:
            return
        yaw = float(snap.get("robot_yaw") or 0.0)
        qx, qy, qz, qw = _yaw_to_quat_xyzw(yaw)
        vel = snap.get("robot_vel", snap.get("vel_command", [0, 0, 0]))
        if not isinstance(vel, list) or len(vel) < 3:
            vel = [0, 0, 0]

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.child_frame_id = "base"
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = float(pose[2])
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])
        self.odom_pub.publish(msg)

    def _publish_box_pose(self, snap: dict) -> None:
        box = snap.get("box")
        if not isinstance(box, dict):
            return
        pos = box.get("position")
        if not isinstance(pos, list) or len(pos) < 3:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = 1.0
        self.box_pose_pub.publish(msg)

    def _publish_scene_objects(self, snap: dict) -> None:
        objects = []
        for key in ("platform_1", "platform_2"):
            asset = snap.get(key)
            if isinstance(asset, dict) and asset.get("position") and asset.get("size"):
                objects.append({
                    "id": asset.get("name", key),
                    "type": "platform",
                    "center": [float(v) for v in asset["position"][:3]],
                    "size": [float(v) for v in asset["size"][:3]],
                    "movable": False,
                })
        box = snap.get("box")
        if isinstance(box, dict) and box.get("position") and box.get("size"):
            objects.append({
                "id": box.get("name", "box"),
                "type": "box",
                "center": [float(v) for v in box["position"][:3]],
                "size": [float(v) for v in box["size"][:3]],
                "movable": True,
            })
        msg = String()
        msg.data = json.dumps(objects, ensure_ascii=False)
        self.scene_objects_pub.publish(msg)

    def _publish_skill_status(self, snap: dict) -> None:
        status = {
            "timestamp": snap.get("timestamp"),
            "model_use": snap.get("model_use"),
            "skill": snap.get("skill"),
            "start": snap.get("start"),
            "goal": snap.get("goal"),
            "velocity": snap.get("velocity") or snap.get("vel_command"),
        }
        msg = String()
        msg.data = json.dumps(status, ensure_ascii=False)
        self.skill_status_pub.publish(msg)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = IsaacLabBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
