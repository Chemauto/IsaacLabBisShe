from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from WebSocket.feedback import evaluate_feedback
from WebSocket.ros2_state import ROS2_AVAILABLE, Ros2TopicState, rclpy


DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_WS_PORT = 8765
DEFAULT_POLL_INTERVAL_SEC = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge FinalProject WebSocket commands to ROS2 topics.")
    parser.add_argument("--host", type=str, default=os.getenv("ROBOT_WS_HOST", DEFAULT_WS_HOST), help="WebSocket监听地址。")
    parser.add_argument("--port", type=int, default=int(os.getenv("ROBOT_WS_PORT", DEFAULT_WS_PORT)), help="WebSocket监听端口。")
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("ROBOT_WS_POLL_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC)), help="状态推送轮询间隔秒数。")
    parser.add_argument("--skill-command-topic", default="/go2/skill_command")
    parser.add_argument("--cmd-vel-topic", default="/go2/cmd_vel")
    parser.add_argument("--goal-pose-topic", default="/go2/goal_pose")
    parser.add_argument("--odom-topic", default="/go2/odom")
    parser.add_argument("--box-pose-topic", default="/go2/box_pose")
    parser.add_argument("--skill-status-topic", default="/go2/skill_status")
    parser.add_argument("--scene-objects-topic", default="/go2/scene_objects")
    parser.add_argument("--frame-id", default="map")
    return parser.parse_args()


async def handle_client(ws, ros2_node: Ros2TopicState, poll_interval: float | None = None):
    try:
        payload = _parse_message(await ws.recv())
        if payload.get("type") == "healthcheck":
            await ws.send(json.dumps(build_health_payload(ros2_node), ensure_ascii=False))
            return
        if payload.get("type") == "get_state":
            await ws.send(json.dumps(build_state_payload(ros2_node), ensure_ascii=False))
            return
        command = _parse_command(payload)
        if not ros2_node.ready():
            raise ValueError("ROS2 state topics not ready")
        start_state = ros2_node.snapshot()
        ros2_node.publish_command(command)
        await _stream_until_feedback(ws, command, start_state, ros2_node, poll_interval=poll_interval)
    except Exception as error:
        await ws.send(json.dumps({"type": "error", "message": str(error)}, ensure_ascii=False))


async def _stream_until_feedback(
    ws,
    command: dict,
    start_state: dict,
    ros2_node: Ros2TopicState,
    poll_interval: float | None = None,
):
    last_revision = -1
    start_time = time.time()
    interval = float(poll_interval or os.getenv("ROBOT_WS_POLL_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC))

    while True:
        current_state = ros2_node.snapshot()
        revision = int(current_state.pop("_revision", 0))
        if revision > last_revision:
            await ws.send(json.dumps(current_state, ensure_ascii=False))
            last_revision = revision

        feedback = evaluate_feedback(
            command,
            start_state=start_state,
            current_state=current_state,
            elapsed_sec=time.time() - start_time,
        )
        if feedback is not None:
            ros2_node.publish_stop()
            await ws.send(json.dumps(feedback, ensure_ascii=False))
            return
        await asyncio.sleep(interval)


def build_health_payload(ros2_node: Ros2TopicState) -> dict:
    snapshot = ros2_node.snapshot()
    return {
        "type": "health",
        "signal": "SUCCESS",
        "message": "robot websocket ros2 bridge online",
        "status_json_ready": ros2_node.ready(),
        "current_skill": snapshot.get("skill"),
        "model_use": snapshot.get("model_use"),
        "start": snapshot.get("start"),
        "timestamp": snapshot.get("timestamp"),
    }


def build_state_payload(ros2_node: Ros2TopicState) -> dict:
    snapshot = ros2_node.snapshot()
    snapshot.pop("_revision", None)
    snapshot["signal"] = "SUCCESS"
    return snapshot


def _parse_message(raw_message) -> dict:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("invalid message payload")
    return payload


def _parse_command(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("type") != "command":
        raise ValueError("invalid command payload")
    if not payload.get("skill"):
        raise ValueError("command.skill is required")
    return payload


async def serve(ros2_node: Ros2TopicState, host: str, port: int, poll_interval: float):
    try:
        import websockets
    except ImportError as error:
        raise RuntimeError("未安装websockets，请先执行 pip install websockets") from error

    async def _handler(ws):
        await handle_client(ws, ros2_node, poll_interval=poll_interval)

    async with websockets.serve(_handler, host, port):
        await asyncio.Future()


def main() -> int:
    if not ROS2_AVAILABLE:
        raise SystemExit("ROS2 Python packages are not available. Source your ROS2 environment first.")

    args = parse_args()
    rclpy.init()
    node = Ros2TopicState(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        asyncio.run(serve(node, args.host, args.port, args.poll_interval))
    except KeyboardInterrupt:
        node.get_logger().info("WebSocket ROS2 bridge stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
