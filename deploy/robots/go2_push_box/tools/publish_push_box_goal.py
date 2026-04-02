#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__HeightMap_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_


DEFAULT_GOAL = (1.7, 0.0, 0.12, 0.0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously publish a push-box goal to rt/push_box_goal.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--topic", type=str, default="rt/push_box_goal")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate in Hz.")
    parser.add_argument(
        "--goal",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "YAW"),
        default=DEFAULT_GOAL,
        help="World-frame push-box goal [x y z yaw].",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)

    publisher = ChannelPublisher(args.topic, HeightMap_)
    publisher.Init()

    goal_msg = unitree_go_msg_dds__HeightMap_()
    goal_msg.frame_id = "world"
    goal_msg.width = 4
    goal_msg.height = 1
    goal_msg.resolution = 1.0
    goal_msg.origin = [0.0, 0.0]
    goal_msg.data = [float(value) for value in args.goal]

    publish_period = 1.0 / max(args.rate, 1.0e-3)

    print(f"Publishing push-box goal to {args.topic}")
    print(f"goal=[{goal_msg.data[0]:.3f}, {goal_msg.data[1]:.3f}, {goal_msg.data[2]:.3f}, {goal_msg.data[3]:.3f}]")
    print("Press Ctrl+C to stop.")

    while True:
        goal_msg.stamp = time.time()
        publisher.Write(goal_msg)
        time.sleep(publish_period)


if __name__ == "__main__":
    main()
