import json
import tempfile
import unittest
from pathlib import Path

from envtest_ros2_server import (
    OutputPaths,
    build_scene_objects,
    build_skill_status,
    pose_message_to_text,
    skill_command_to_text,
    twist_message_to_text,
)


class EnvTestRos2ServerTests(unittest.TestCase):
    def test_skill_command_json_writes_same_control_files_as_socket_protocol(self):
        payload = {
            "model_use": 1,
            "velocity": [0.6, 0.0, 0.0],
            "start": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = OutputPaths(
                model_use=str(root / "model_use.txt"),
                velocity=str(root / "velocity.txt"),
                goal=str(root / "goal.txt"),
                start=str(root / "start.txt"),
                reset=str(root / "reset.txt"),
            )
            updates = skill_command_to_text(json.dumps(payload), paths)

            self.assertEqual(updates, ["model_use=1", "velocity=(0.6, 0.0, 0.0)", "start=1"])
            self.assertEqual((root / "model_use.txt").read_text(encoding="utf-8"), "1\n")
            self.assertEqual((root / "velocity.txt").read_text(encoding="utf-8"), "0.6 0.0 0.0\n")
            self.assertEqual((root / "start.txt").read_text(encoding="utf-8"), "1\n")

    def test_twist_message_to_text_uses_linear_xyz(self):
        class Linear:
            x = 0.2
            y = -0.3
            z = 0.0

        class Twist:
            linear = Linear()

        self.assertEqual(twist_message_to_text(Twist()), "velocity=0.2,-0.3,0.0")

    def test_pose_message_to_text_uses_position_xyz(self):
        class Position:
            x = 4.5
            y = 0.75
            z = 0.1

        class Pose:
            position = Position()

        class PoseStamped:
            pose = Pose()

        self.assertEqual(pose_message_to_text(PoseStamped()), "goal=4.5,0.75,0.1")

    def test_status_snapshot_builds_finalproject_skill_status_and_scene_objects(self):
        snapshot = {
            "timestamp": 1776753740.0,
            "model_use": 0,
            "skill": "idle",
            "scene_id": 1,
            "start": False,
            "goal": None,
            "vel_command": [0.0, 0.0, 0.0],
            "platform_1": {
                "name": "left_low_obstacle",
                "position": [3.0, 0.75, 0.15],
                "size": [2.0, 1.5, 0.3],
            },
            "platform_2": None,
            "box": None,
        }

        self.assertEqual(
            build_skill_status(snapshot),
            {
                "timestamp": 1776753740.0,
                "model_use": 0,
                "skill": "idle",
                "scene_id": 1,
                "start": False,
                "goal": None,
                "vel_command": [0.0, 0.0, 0.0],
                "envtest_alignment": {
                    "platform_1": {
                        "name": "left_low_obstacle",
                        "position": [3.0, 0.75, 0.15],
                        "size": [2.0, 1.5, 0.3],
                    }
                },
            },
        )
        self.assertEqual(
            build_scene_objects(snapshot),
            [
                {
                    "id": "left_low_obstacle",
                    "type": "platform",
                    "center": [3.0, 0.75, 0.15],
                    "size": [2.0, 1.5, 0.3],
                    "movable": False,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
