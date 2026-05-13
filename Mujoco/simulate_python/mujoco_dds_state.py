#!/usr/bin/env python3
"""Subscribe MuJoCo DDS topics, write state snapshot to JSON file.

Runs as a standalone process (no ROS2) to avoid CycloneDDS conflict.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, LowState_, SportModeState_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--state-file", default="/tmp/mujoco_ros2_state.json")
    parser.add_argument("--write-hz", type=float, default=50.0)
    return parser.parse_args()


class MujocoDdsState:
    def __init__(self):
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.quat_wxyz = [1.0, 0.0, 0.0, 0.0]
        self.box_obs = None  # push_box_obs data

    def snapshot(self) -> dict:
        yaw = self._quat_to_yaw(self.quat_wxyz)
        snap = {
            "robot_pos": list(self.pos),
            "robot_vel": list(self.vel),
            "robot_quat": list(self.quat_wxyz),
            "robot_yaw": yaw,
            "sim_time": time.time(),
        }
        if self.box_obs is not None:
            snap["push_box_obs"] = list(self.box_obs)
        return snap

    @staticmethod
    def _quat_to_yaw(quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def write_json(path, data: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def main():
    args = parse_args()
    running = True

    def stop(_sig, _frm):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ChannelFactoryInitialize(args.domain_id, args.interface)
    state = MujocoDdsState()

    def on_sportmode(msg: SportModeState_):
        state.pos = [float(msg.position[0]), float(msg.position[1]), float(msg.position[2])]
        state.vel = [float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])]

    def on_lowstate(msg: LowState_):
        state.quat_wxyz = [float(msg.imu_state.quaternion[i]) for i in range(4)]

    def on_push_box(msg: HeightMap_):
        if isinstance(msg.data, list) and len(msg.data) >= 9:
            state.box_obs = [float(v) for v in msg.data]

    sub_sport = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub_sport.Init(on_sportmode, 10)

    sub_low = ChannelSubscriber("rt/lowstate", LowState_)
    sub_low.Init(on_lowstate, 10)

    sub_box = ChannelSubscriber("rt/push_box_obs", HeightMap_)
    sub_box.Init(on_push_box, 10)

    print(f"[mujoco_dds_state] writing to {args.state_file}")
    period = 1.0 / max(args.write_hz, 0.1)
    while running:
        write_json(args.state_file, state.snapshot())
        time.sleep(period)

    print("[mujoco_dds_state] stopped.")


if __name__ == "__main__":
    main()
