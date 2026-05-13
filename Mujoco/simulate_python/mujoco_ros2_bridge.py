#!/usr/bin/env python3
"""ROS2 bridge for MuJoCo simulation.

Reads state JSON written by mujoco_dds_state.py and publishes /go2/* ROS2
topics.  Subscribes to /go2/skill_command and writes control files for
MuJoCo (if needed).

Pure ROS2 -- no DDS dependency.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo ROS2 bridge")
    parser.add_argument("--state-file", default="/tmp/mujoco_ros2_state.json")
    parser.add_argument("--control-dir", default="/tmp/mujoco_go2_control")
    parser.add_argument("--odom-topic", default="/go2/odom")
    parser.add_argument("--box-pose-topic", default="/go2/box_pose")
    parser.add_argument("--scene-objects-topic", default="/go2/scene_objects")
    parser.add_argument("--skill-status-topic", default="/go2/skill_status")
    parser.add_argument("--skill-command-topic", default="/go2/skill_command")
    parser.add_argument("--goal-pose-topic", default="/go2/goal_pose")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--publish-hz", type=float, default=10.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.08)
    return parser.parse_args()


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _write_file(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.strip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _quat_wxyz_to_xyzw(quat):
    return float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])


def _rotate_body_to_world(vec, quat_wxyz):
    w, x, y, z = [float(q) for q in quat_wxyz[:4]]
    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    return [
        (1 - 2*(y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
        2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
        2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz,
    ]


SKILL_ID_TO_NAME = {0: "idle", 1: "walk_skill", 2: "climb", 3: "push", 4: "nav", 5: "nav_climb"}


class MujocoRos2Bridge(Node):
    def __init__(self, args):
        super().__init__("mujoco_ros2_bridge")
        self.args = args
        self.state_file = Path(args.state_file)
        self.control_dir = Path(args.control_dir)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        self.box_pose_pub = self.create_publisher(PoseStamped, args.box_pose_topic, 10)
        self.scene_objects_pub = self.create_publisher(String, args.scene_objects_topic, 10)
        self.skill_status_pub = self.create_publisher(String, args.skill_status_topic, 10)

        # Subscribers
        self.create_subscription(String, args.skill_command_topic, self._on_skill_command, 10)
        self.create_subscription(PoseStamped, args.goal_pose_topic, self._on_goal_pose, 10)

        self._last_command = {}

        self.create_timer(1.0 / max(args.publish_hz, 0.1), self._publish)

        self.get_logger().info(f"MuJoCo ROS2 bridge started")
        self.get_logger().info(f"  state_file: {args.state_file}")
        self.get_logger().info(f"  control_dir: {args.control_dir}")

    # ── Subscribers: write files for MuJoCo ──

    def _on_skill_command(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        self._last_command = payload

        _write_file(self.control_dir / "skill_command.json", json.dumps(payload, ensure_ascii=False))
        if payload.get("model_use") is not None:
            _write_file(self.control_dir / "model_use.txt", str(int(payload["model_use"])))
        velocity = payload.get("velocity")
        if isinstance(velocity, list) and len(velocity) >= 3:
            _write_file(self.control_dir / "velocity.txt", f"{velocity[0]} {velocity[1]} {velocity[2]}")
        goal = payload.get("goal")
        if isinstance(goal, list) and len(goal) >= 3:
            _write_file(self.control_dir / "goal.txt", f"{goal[0]} {goal[1]} {goal[2]}")
        if payload.get("start") is not None:
            _write_file(self.control_dir / "start.txt", "1" if payload["start"] else "0")

    def _on_goal_pose(self, msg: PoseStamped):
        p = msg.pose.position
        _write_file(self.control_dir / "goal.txt", f"{p.x} {p.y} {p.z}")

    # ── Timer: read state JSON, publish ROS2 ──

    def _publish(self):
        snap = _read_json(self.state_file)
        if snap is None:
            return
        self._publish_odom(snap)
        self._publish_box_pose(snap)
        self._publish_scene_objects(snap)
        self._publish_skill_status(snap)

    def _publish_odom(self, snap):
        pos = snap.get("robot_pos")
        if not isinstance(pos, list) or len(pos) < 3:
            return
        quat = snap.get("robot_quat", [1, 0, 0, 0])
        vel = snap.get("robot_vel", [0, 0, 0])
        qx, qy, qz, qw = _quat_wxyz_to_xyzw(quat)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.child_frame_id = "base"
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])
        self.odom_pub.publish(msg)

    def _publish_box_pose(self, snap):
        pos = snap.get("robot_pos")
        quat = snap.get("robot_quat")
        obs = snap.get("push_box_obs")
        if not isinstance(pos, list) or not isinstance(quat, list) or not isinstance(obs, list) or len(obs) < 9:
            return
        box_in_robot = obs[6:9]
        box_world = _rotate_body_to_world(box_in_robot, quat)
        box_x = float(pos[0]) + box_world[0]
        box_y = float(pos[1]) + box_world[1]
        box_z = float(pos[2]) + box_world[2]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x = box_x
        msg.pose.position.y = box_y
        msg.pose.position.z = box_z
        msg.pose.orientation.w = 1.0
        self.box_pose_pub.publish(msg)

    def _publish_scene_objects(self, snap):
        pos = snap.get("robot_pos")
        quat = snap.get("robot_quat")
        obs = snap.get("push_box_obs")
        objects = []
        if isinstance(pos, list) and isinstance(quat, list) and isinstance(obs, list) and len(obs) >= 9:
            box_in_robot = obs[6:9]
            box_world = _rotate_body_to_world(box_in_robot, quat)
            objects.append({
                "id": "box",
                "type": "box",
                "center": [
                    float(pos[0]) + box_world[0],
                    float(pos[1]) + box_world[1],
                    float(pos[2]) + box_world[2],
                ],
                "size": [0.6, 0.8, 0.24],
                "movable": True,
            })
        msg = String()
        msg.data = json.dumps(objects, ensure_ascii=False)
        self.scene_objects_pub.publish(msg)

    def _publish_skill_status(self, snap):
        cmd = self._last_command
        model_use = int(cmd.get("model_use") or 0)
        skill = cmd.get("skill") or SKILL_ID_TO_NAME.get(model_use, "unknown")
        status = {
            "timestamp": snap.get("sim_time"),
            "model_use": model_use,
            "skill": skill,
            "start": bool(cmd.get("start")),
            "goal": cmd.get("goal"),
            "velocity": cmd.get("velocity"),
        }
        msg = String()
        msg.data = json.dumps(status, ensure_ascii=False)
        self.skill_status_pub.publish(msg)


def main():
    args = parse_args()
    rclpy.init()
    node = MujocoRos2Bridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
