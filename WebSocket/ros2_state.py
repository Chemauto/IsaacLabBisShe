from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

from WebSocket.protocol import coerce_bool, normalize_skill, position_payload, walk_velocity, world_to_body

DEFAULT_CLIMB_VELOCITY = (0.4, 0.0, 0.0)


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


class Ros2TopicState(Node):
    def __init__(self, args):
        super().__init__("llm_websocket_ros2_bridge")
        self.args = args
        self._lock = threading.Lock()
        self._revision = 0
        self._robot = None
        self._box_world = None
        self._skill_status = {}
        self._scene_objects = []

        self.skill_command_pub = self.create_publisher(String, args.skill_command_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, args.goal_pose_topic, 10)

        self.create_subscription(Odometry, args.odom_topic, self._on_odom, 10)
        self.create_subscription(PoseStamped, args.box_pose_topic, self._on_box_pose, 10)
        self.create_subscription(String, args.skill_status_topic, self._on_skill_status, 10)
        self.create_subscription(String, args.scene_objects_topic, self._on_scene_objects, 10)

    def publish_command(self, command: dict) -> None:
        skill = normalize_skill(command.get("skill"))
        payload = dict(command.get("args") or {})

        if skill == "nav":
            self._publish_goal_pose(payload)
            self._publish_skill_command({
                "model_use": 4,
                "goal": [float(payload.get("x", 0.0)), float(payload.get("y", 0.0)), float(payload.get("z", 0.0))],
                "start": True,
            })
            return
        if skill == "nav_climb":
            self._publish_goal_pose(payload)
            self._publish_skill_command({
                "model_use": 5,
                "goal": [float(payload.get("x", 0.0)), float(payload.get("y", 0.0)), float(payload.get("z", 0.0))],
                "start": True,
            })
            return
        if skill == "walk_skill":
            velocity = walk_velocity(payload)
            self._publish_cmd_vel(velocity)
            self._publish_skill_command({"model_use": 1, "velocity": list(velocity), "start": True})
            return
        if skill == "push":
            self._publish_skill_command({
                "model_use": 3,
                "goal": [float(payload.get("x", 0.0)), float(payload.get("y", 0.0)), float(payload.get("z", 0.0))],
                "start": True,
            })
            return
        if skill == "climb":
            self._publish_cmd_vel(DEFAULT_CLIMB_VELOCITY)
            self._publish_skill_command({"model_use": 2, "velocity": list(DEFAULT_CLIMB_VELOCITY), "start": True})
            return
        raise ValueError(f"unsupported skill: {command.get('skill')}")

    def publish_stop(self) -> None:
        self._publish_cmd_vel((0.0, 0.0, 0.0))
        self._publish_skill_command({"model_use": 1, "velocity": [0.0, 0.0, 0.0], "start": True})

    def snapshot(self) -> dict:
        with self._lock:
            robot = dict(self._robot or {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0})
            box_world = dict(self._box_world or self._box_from_scene_objects() or {"x": 0.0, "y": 0.0, "z": 0.0})
            status = dict(self._skill_status)
            scene_objects = list(self._scene_objects)
            revision = self._revision

        return {
            "type": "state",
            "timestamp": float(status.get("timestamp") or time.time()),
            "robot": robot,
            "box_world": box_world,
            "box_relative": world_to_body(robot, box_world),
            "skill": status.get("skill"),
            "current_skill": status.get("skill"),
            "model_use": status.get("model_use"),
            "start": coerce_bool(status.get("start")),
            "scene_objects": scene_objects,
            "raw": {
                "skill_status": status,
                "scene_objects": scene_objects,
            },
            "_revision": revision,
        }

    def ready(self) -> bool:
        with self._lock:
            return self._robot is not None or bool(self._skill_status)

    def _publish_skill_command(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.skill_command_pub.publish(msg)

    def _publish_cmd_vel(self, velocity: tuple[float, float, float]) -> None:
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.linear.y = float(velocity[1])
        msg.linear.z = float(velocity[2])
        self.cmd_vel_pub.publish(msg)

    def _publish_goal_pose(self, payload: dict) -> None:
        goal = position_payload(payload)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x = goal["x"]
        msg.pose.position.y = goal["y"]
        msg.pose.position.z = goal["z"]
        msg.pose.orientation.w = 1.0
        self.goal_pose_pub.publish(msg)

    def _on_odom(self, msg: Any) -> None:
        pose = msg.pose.pose
        robot = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "yaw": _yaw_from_quaternion(pose.orientation),
        }
        self._update(robot=robot)

    def _on_box_pose(self, msg: Any) -> None:
        self._update(box_world={
            "x": float(msg.pose.position.x),
            "y": float(msg.pose.position.y),
            "z": float(msg.pose.position.z),
        })

    def _on_skill_status(self, msg: Any) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            self._update(skill_status=payload)

    def _on_scene_objects(self, msg: Any) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = []
        if isinstance(payload, list):
            self._update(scene_objects=payload)

    def _update(self, robot=None, box_world=None, skill_status=None, scene_objects=None) -> None:
        with self._lock:
            if robot is not None:
                self._robot = robot
            if box_world is not None:
                self._box_world = box_world
            if skill_status is not None:
                self._skill_status = skill_status
            if scene_objects is not None:
                self._scene_objects = scene_objects
            self._revision += 1

    def _box_from_scene_objects(self) -> dict | None:
        for item in self._scene_objects:
            if isinstance(item, dict) and item.get("type") == "box":
                center = item.get("center")
                if isinstance(center, list) and len(center) >= 3:
                    return position_payload(center)
        return None


def _yaw_from_quaternion(orientation: Any) -> float:
    x = float(getattr(orientation, "x", 0.0))
    y = float(getattr(orientation, "y", 0.0))
    z = float(getattr(orientation, "z", 0.0))
    w = float(getattr(orientation, "w", 1.0))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
