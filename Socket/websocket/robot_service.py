from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    socket_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(socket_root))
    from envtest_socket_server import OutputPaths, apply_message
    from websocket.feedback import evaluate_feedback
    from websocket.protocol import build_state_payload, build_udp_message
    from websocket.status_reader import load_status_snapshot, status_timestamp
else:
    from envtest_socket_server import OutputPaths, apply_message
    from .feedback import evaluate_feedback
    from .protocol import build_state_payload, build_udp_message
    from .status_reader import load_status_snapshot, status_timestamp


DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_WS_PORT = 8765
DEFAULT_POLL_INTERVAL_SEC = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified robot service for EnvTest simulation.")
    parser.add_argument("--host", type=str, default=os.getenv("ROBOT_WS_HOST", DEFAULT_WS_HOST), help="WebSocket监听地址。")
    parser.add_argument("--port", type=int, default=int(os.getenv("ROBOT_WS_PORT", DEFAULT_WS_PORT)), help="WebSocket监听端口。")
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("ROBOT_WS_POLL_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC)), help="轮询状态JSON的间隔秒数。")
    parser.add_argument("--model-use-file", type=str, default="/tmp/model_use.txt", help="model_use 文件路径。")
    parser.add_argument("--velocity-file", type=str, default="/tmp/envtest_velocity_command.txt", help="速度指令文件路径。")
    parser.add_argument("--goal-file", type=str, default="/tmp/envtest_goal_command.txt", help="位置指令文件路径。")
    parser.add_argument("--start-file", type=str, default="/tmp/envtest_start.txt", help="启动开关文件路径。")
    parser.add_argument("--reset-file", type=str, default="/tmp/envtest_reset.txt", help="一次性重置文件路径。")
    return parser.parse_args()
#解析统一机器人服务的启动参数


def build_output_paths(args: argparse.Namespace) -> OutputPaths:
    return OutputPaths(
        model_use=args.model_use_file,
        velocity=args.velocity_file,
        goal=args.goal_file,
        start=args.start_file,
        reset=args.reset_file,
    )
#把命令行参数转换成控制文件路径对象


def apply_command_text(text: str, output_paths: OutputPaths) -> list[str]:
    return apply_message(text, output_paths)
#复用旧UDP server的解析和写文件逻辑，但不再需要启动UDP进程


async def handle_client(ws, output_paths: OutputPaths, poll_interval: float | None = None):
    try:
        payload = _parse_message(await ws.recv())
        if payload.get("type") == "healthcheck":
            await ws.send(json.dumps(build_health_payload(load_status_snapshot()), ensure_ascii=False))
            return
        command = _parse_command(payload)
        start_state = build_state_payload(load_status_snapshot())
        if not start_state.get("raw"):
            raise ValueError("status json not ready")
        apply_command_text(build_udp_message(command["skill"], command.get("args")), output_paths)
        await _stream_until_feedback(ws, command, start_state, output_paths, poll_interval=poll_interval)
    except Exception as error:
        await ws.send(json.dumps({"type": "error", "message": str(error)}, ensure_ascii=False))
#处理一个WebSocket客户端：收命令、写控制文件、推状态、回反馈


async def _stream_until_feedback(
    ws,
    command: dict,
    start_state: dict,
    output_paths: OutputPaths,
    poll_interval: float | None = None,
):
    last_timestamp = -1.0
    start_time = time.time()
    interval = float(poll_interval or os.getenv("ROBOT_WS_POLL_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC))

    while True:
        raw_status = load_status_snapshot()
        current_timestamp = status_timestamp(raw_status)
        current_state = build_state_payload(raw_status)
        if current_timestamp > last_timestamp:
            await ws.send(json.dumps(current_state, ensure_ascii=False))
            last_timestamp = current_timestamp

        feedback = evaluate_feedback(
            command,
            start_state=start_state,
            current_state=current_state,
            elapsed_sec=time.time() - start_time,
        )
        if feedback is not None:
            apply_command_text("start=0", output_paths)
            await ws.send(json.dumps(feedback, ensure_ascii=False))
            return
        await asyncio.sleep(interval)
#持续轮询状态JSON并输出state，直到动作成功或失败


def build_health_payload(status: dict | None) -> dict:
    raw_status = dict(status or {})
    ready = bool(raw_status.get("timestamp"))
    return {
        "type": "health",
        "signal": "SUCCESS",
        "message": "robot service online",
        "status_json_ready": ready,
        "current_skill": raw_status.get("skill"),
        "model_use": raw_status.get("model_use"),
        "start": raw_status.get("start"),
        "timestamp": raw_status.get("timestamp"),
    }
#构建健康检查响应，区分服务在线和仿真状态是否就绪


def _parse_message(raw_message) -> dict:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("invalid message payload")
    return payload
#解析客户端原始消息，要求消息体必须是JSON对象


def _parse_command(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("type") != "command":
        raise ValueError("invalid command payload")
    if not payload.get("skill"):
        raise ValueError("command.skill is required")
    return payload
#解析规划器发来的命令，确保字段结构正确


async def serve(output_paths: OutputPaths, host: str | None = None, port: int | None = None, poll_interval: float | None = None):
    try:
        import websockets
    except ImportError as error:
        raise RuntimeError("未安装websockets，请先执行 pip install websockets") from error

    async def _handler(ws):
        await handle_client(ws, output_paths, poll_interval=poll_interval)

    async with websockets.serve(_handler, host or DEFAULT_WS_HOST, int(port or DEFAULT_WS_PORT)):
        await asyncio.Future()
#启动统一机器人服务，对外暴露WebSocket协议


def main():
    args = parse_args()
    asyncio.run(serve(build_output_paths(args), args.host, args.port, args.poll_interval))
#命令行入口


if __name__ == "__main__":
    main()
