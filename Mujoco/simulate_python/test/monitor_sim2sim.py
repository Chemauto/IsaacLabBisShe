#!/usr/bin/env python3
import argparse
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, LowCmd_, LowState_, SportModeState_


GO2_JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]


@dataclass
class TopicCache:
    msg: Any = None
    recv_wall_time: float = 0.0


class MonitorState:
    def __init__(self):
        self._lock = threading.Lock()
        self.lowstate = TopicCache()
        self.sportstate = TopicCache()
        self.lowcmd = TopicCache()
        self.heightmap = TopicCache()

    def update(self, name: str, msg: Any):
        with self._lock:
            cache = getattr(self, name)
            cache.msg = msg
            cache.recv_wall_time = time.time()

    def snapshot(self):
        with self._lock:
            return {
                "lowstate": TopicCache(self.lowstate.msg, self.lowstate.recv_wall_time),
                "sportstate": TopicCache(self.sportstate.msg, self.sportstate.recv_wall_time),
                "lowcmd": TopicCache(self.lowcmd.msg, self.lowcmd.recv_wall_time),
                "heightmap": TopicCache(self.heightmap.msg, self.heightmap.recv_wall_time),
            }


def _fmt_vec(data, precision=3):
    return "[" + ", ".join(f"{float(x):.{precision}f}" for x in data) + "]"


def _topic_age_text(cache: TopicCache) -> str:
    if cache.recv_wall_time <= 0.0:
        return "n/a"
    return f"{time.time() - cache.recv_wall_time:.3f}s"


def _print_header(snapshot: dict[str, TopicCache]):
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(
        "topics: "
        f"lowstate={_topic_age_text(snapshot['lowstate'])}, "
        f"sportstate={_topic_age_text(snapshot['sportstate'])}, "
        f"lowcmd={_topic_age_text(snapshot['lowcmd'])}, "
        f"heightmap={_topic_age_text(snapshot['heightmap'])}"
    )
    print("-" * 96)


def _print_sportstate(cache: TopicCache):
    msg = cache.msg
    if msg is None:
        print("sportstate: <no data>")
        return
    print(
        "sportstate: "
        f"pos={_fmt_vec(msg.position, 3)} "
        f"vel={_fmt_vec(msg.velocity, 3)}"
    )


def _print_lowstate(cache: TopicCache):
    msg = cache.msg
    if msg is None:
        print("lowstate: <no data>")
        return
    imu = msg.imu_state
    print(
        "lowstate: "
        f"quat={_fmt_vec(imu.quaternion, 3)} "
        f"gyro={_fmt_vec(imu.gyroscope, 3)} "
        f"acc={_fmt_vec(imu.accelerometer, 3)}"
    )


def _print_lowcmd(cache: TopicCache, joint_count: int):
    msg = cache.msg
    if msg is None:
        print("lowcmd: <no data>")
        return
    print("lowcmd / lowstate joints:")
    print("name         q_cmd    dq_cmd    kp      kd      tau_ff")
    for idx in range(joint_count):
        cmd = msg.motor_cmd[idx]
        print(
            f"{GO2_JOINT_NAMES[idx]:10s} "
            f"{cmd.q:8.3f} {cmd.dq:8.3f} {cmd.kp:7.3f} {cmd.kd:7.3f} {cmd.tau:9.3f}"
        )


def _print_joint_state(lowstate_cache: TopicCache, joint_count: int):
    msg = lowstate_cache.msg
    if msg is None:
        return
    print("joint state:")
    print("name         q        dq       tau_est")
    for idx in range(joint_count):
        motor = msg.motor_state[idx]
        print(
            f"{GO2_JOINT_NAMES[idx]:10s} "
            f"{motor.q:8.3f} {motor.dq:8.3f} {motor.tau_est:9.3f}"
        )


def _print_heightmap(cache: TopicCache, sample_count: int):
    msg = cache.msg
    if msg is None:
        print("heightmap: <no data>")
        return
    data = np.asarray(msg.data, dtype=np.float32)
    if data.size == 0:
        print(f"heightmap: stamp={msg.stamp:.3f} frame={msg.frame_id} empty")
        return
    center_idx = data.size // 2
    print(
        "heightmap: "
        f"stamp={msg.stamp:.3f} frame={msg.frame_id} size={msg.width}x{msg.height} "
        f"min={data.min():.3f} max={data.max():.3f} mean={data.mean():.3f} center={data[center_idx]:.3f}"
    )
    sample = data[:sample_count].tolist()
    print(f"heightmap sample[{sample_count}]: {_fmt_vec(sample, 3)}")


def _print_heightmap_grid(cache: TopicCache):
    msg = cache.msg
    if msg is None:
        return
    data = np.asarray(msg.data, dtype=np.float32)
    if data.size == 0 or int(msg.width) <= 0 or int(msg.height) <= 0:
        return

    grid = data.reshape(int(msg.height), int(msg.width))
    x_coords = [msg.origin[0] + idx * msg.resolution for idx in range(int(msg.width))]
    y_coords = [msg.origin[1] + idx * msg.resolution for idx in range(int(msg.height))]

    print("heightmap grid: x(back->front), y(top=left, bottom=right)")
    header = "y\\x   " + " ".join(f"{x:>6.2f}" for x in x_coords)
    print(header)
    for row_idx in range(int(msg.height) - 1, -1, -1):
        row = " ".join(f"{value:>6.2f}" for value in grid[row_idx])
        print(f"{y_coords[row_idx]:>5.2f} {row}")


def _print_heightmap_delta_grid(cache: TopicCache):
    msg = cache.msg
    if msg is None:
        return
    data = np.asarray(msg.data, dtype=np.float32)
    if data.size == 0 or int(msg.width) <= 0 or int(msg.height) <= 0:
        return

    grid = data.reshape(int(msg.height), int(msg.width))
    x_coords = [msg.origin[0] + idx * msg.resolution for idx in range(int(msg.width))]
    y_coords = [msg.origin[1] + idx * msg.resolution for idx in range(int(msg.height))]
    reference = float(np.max(grid))
    delta_grid = reference - grid

    print(f"heightmap delta grid: reference=max(grid)={reference:.3f}, positive means higher obstacle")
    header = "y\\x   " + " ".join(f"{x:>6.2f}" for x in x_coords)
    print(header)
    for row_idx in range(int(msg.height) - 1, -1, -1):
        row = " ".join(f"{value:>6.2f}" for value in delta_grid[row_idx])
        print(f"{y_coords[row_idx]:>5.2f} {row}")


def _clear_screen(enabled: bool):
    if enabled and sys.stdout.isatty():
        print("[2J[H", end="")


def main():
    parser = argparse.ArgumentParser(description="Continuously print MuJoCo DDS topics for sim2sim debugging.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--rate", type=float, default=2.0, help="Refresh rate in Hz.")
    parser.add_argument("--joint-count", type=int, default=12, choices=[3, 6, 12])
    parser.add_argument("--heightmap-sample", type=int, default=8)
    parser.add_argument(
        "--no-heightmap-grid",
        action="store_false",
        dest="heightmap_grid",
        help="Disable the 17x11 heightmap grid printout.",
    )
    parser.add_argument(
        "--no-heightmap-delta-grid",
        action="store_false",
        dest="heightmap_delta_grid",
        help="Disable the heightmap delta grid printout.",
    )
    parser.add_argument("--no-clear", action="store_true", help="Do not clear terminal between refreshes.")
    parser.set_defaults(heightmap_grid=True, heightmap_delta_grid=True)
    args = parser.parse_args()

    state = MonitorState()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    ChannelSubscriber("rt/lowstate", LowState_).Init(lambda msg: state.update("lowstate", msg), 10)
    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(lambda msg: state.update("sportstate", msg), 10)
    ChannelSubscriber("rt/lowcmd", LowCmd_).Init(lambda msg: state.update("lowcmd", msg), 10)
    ChannelSubscriber("rt/heightmap", HeightMap_).Init(lambda msg: state.update("heightmap", msg), 10)

    period = 1.0 / max(args.rate, 1e-3)
    while True:
        start = time.time()
        snapshot = state.snapshot()
        _clear_screen(not args.no_clear)
        _print_header(snapshot)
        _print_sportstate(snapshot["sportstate"])
        _print_lowstate(snapshot["lowstate"])
        _print_joint_state(snapshot["lowstate"], args.joint_count)
        _print_lowcmd(snapshot["lowcmd"], args.joint_count)
        _print_heightmap(snapshot["heightmap"], args.heightmap_sample)
        if args.heightmap_grid:
            _print_heightmap_grid(snapshot["heightmap"])
        if args.heightmap_delta_grid:
            _print_heightmap_delta_grid(snapshot["heightmap"])
        elapsed = time.time() - start
        time.sleep(max(0.0, period - elapsed))


if __name__ == "__main__":
    main()
