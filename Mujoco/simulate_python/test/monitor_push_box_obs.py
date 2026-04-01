#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, LowState_, SportModeState_


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import config


@dataclass
class TopicCache:
    msg: Any = None
    recv_wall_time: float = 0.0


class MonitorState:
    def __init__(self):
        self._lock = threading.Lock()
        self.push_box_obs = TopicCache()
        self.lowstate = TopicCache()
        self.sportstate = TopicCache()

    def update(self, name: str, msg: Any):
        with self._lock:
            setattr(self, name, TopicCache(msg, time.time()))

    def snapshot(self):
        with self._lock:
            return {
                "push_box_obs": TopicCache(self.push_box_obs.msg, self.push_box_obs.recv_wall_time),
                "lowstate": TopicCache(self.lowstate.msg, self.lowstate.recv_wall_time),
                "sportstate": TopicCache(self.sportstate.msg, self.sportstate.recv_wall_time),
            }


def _clear_screen(enabled: bool):
    if enabled and sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def _fmt_vec(values, precision=3):
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _quat_wxyz_to_rot(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw_from_quat(quat):
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_from_sin_cos(sin_value, cos_value):
    return math.atan2(float(sin_value), float(cos_value))


def _topic_age_text(cache: TopicCache) -> str:
    if cache.recv_wall_time <= 0.0:
        return "n/a"
    return f"{time.time() - cache.recv_wall_time:.3f}s"


def _print_header(snapshot: dict[str, TopicCache], topic_name: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(
        "topics: "
        f"{topic_name}={_topic_age_text(snapshot['push_box_obs'])}, "
        f"lowstate={_topic_age_text(snapshot['lowstate'])}, "
        f"sportstate={_topic_age_text(snapshot['sportstate'])}"
    )
    print("-" * 96)


def _print_world_state(snapshot: dict[str, TopicCache], obs_vector: np.ndarray):
    lowstate_msg = snapshot["lowstate"].msg
    sportstate_msg = snapshot["sportstate"].msg
    if lowstate_msg is None or sportstate_msg is None:
        print("world_state: <need both rt/lowstate and rt/sportmodestate>")
        return

    robot_pos_w = np.asarray(sportstate_msg.position, dtype=np.float64)
    robot_quat_w = np.asarray(lowstate_msg.imu_state.quaternion, dtype=np.float64)
    robot_rot_w = _quat_wxyz_to_rot(robot_quat_w)
    robot_yaw = _yaw_from_quat(robot_quat_w)

    box_in_robot_frame_pos = obs_vector[6:9].astype(np.float64)
    box_relative_yaw = _yaw_from_sin_cos(obs_vector[9], obs_vector[10])

    box_pos_w = robot_pos_w + robot_rot_w @ box_in_robot_frame_pos
    box_yaw = robot_yaw + box_relative_yaw

    goal_cfg_pos_w = np.asarray(config.PUSH_BOX_GOAL_POSITION, dtype=np.float64)
    goal_relative_yaw = _yaw_from_sin_cos(obs_vector[14], obs_vector[15])
    goal_yaw = box_yaw + goal_relative_yaw

    print(f"robot_pos_w : {_fmt_vec(robot_pos_w, 4)}")
    print(f"box_pos_w   : {_fmt_vec(box_pos_w, 4)}")
    print(f"goal_pos_w  : {_fmt_vec(goal_cfg_pos_w, 4)}")
    print(f"robot_yaw   : {robot_yaw: .4f} rad")
    print(f"box_yaw     : {box_yaw: .4f} rad")
    print(f"goal_yaw    : {float(config.PUSH_BOX_GOAL_YAW): .4f} rad")


def main():
    parser = argparse.ArgumentParser(description="Monitor push-box sim2sim observation topic.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--topic", type=str, default="rt/push_box_obs")
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    state = MonitorState()
    ChannelFactoryInitialize(args.domain_id, args.interface)
    ChannelSubscriber(args.topic, HeightMap_).Init(lambda msg: state.update("push_box_obs", msg), 10)
    ChannelSubscriber("rt/lowstate", LowState_).Init(lambda msg: state.update("lowstate", msg), 10)
    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(lambda msg: state.update("sportstate", msg), 10)

    period = 1.0 / max(args.rate, 1e-3)
    while True:
        start = time.time()
        snapshot = state.snapshot()
        _clear_screen(not args.no_clear)
        _print_header(snapshot, args.topic)
        cache = snapshot["push_box_obs"]
        if cache.msg is None:
            print("push_box_obs: <no data>")
        else:
            age = time.time() - cache.recv_wall_time
            print(
                f"topic={args.topic} age={age:.3f}s stamp={cache.msg.stamp:.3f} "
                f"frame={cache.msg.frame_id} size={cache.msg.width}x{cache.msg.height}"
            )
            obs_vector = np.asarray(cache.msg.data, dtype=np.float32)
            _print_world_state(snapshot, obs_vector)
        elapsed = time.time() - start
        time.sleep(max(0.0, period - elapsed))


if __name__ == "__main__":
    main()
