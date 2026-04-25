from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from websocket.feedback import evaluate_feedback
    from websocket.protocol import build_state_payload, build_udp_message
    from websocket.status_reader import load_status_snapshot, status_timestamp
    from websocket.udp_bridge import send_stop_message, send_udp_message
else:
    from .feedback import evaluate_feedback
    from .protocol import build_state_payload, build_udp_message
    from .status_reader import load_status_snapshot, status_timestamp
    from .udp_bridge import send_stop_message, send_udp_message


DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_WS_PORT = 8765
DEFAULT_POLL_INTERVAL_SEC = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge FinalProject WebSocket commands into EnvTest UDP controls.")
    parser.add_argument("--host", type=str, default=os.getenv("ROBOT_WS_HOST", DEFAULT_WS_HOST), help="WebSocket监听地址。")
    parser.add_argument("--port", type=int, default=int(os.getenv("ROBOT_WS_PORT", DEFAULT_WS_PORT)), help="WebSocket监听端口。")
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("ROBOT_WS_POLL_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC)), help="轮询状态JSON的间隔秒数。")
    return parser.parse_args()
#解析WebSocket bridge的启动参数


async def handle_client(ws, poll_interval: float | None = None):
    try:
        command = _parse_command(await ws.recv())
        start_state = build_state_payload(load_status_snapshot())
        if not start_state.get("raw"):
            raise ValueError("status json not ready")
        send_udp_message(build_udp_message(command["skill"], command.get("args")))
        await _stream_until_feedback(ws, command, start_state, poll_interval=poll_interval)
    except Exception as error:
        await ws.send(json.dumps({"type": "error", "message": str(error)}, ensure_ascii=False))
#处理一个客户端连接：收命令、转UDP、推状态、回feedback


async def _stream_until_feedback(ws, command: dict, start_state: dict, poll_interval: float | None = None):
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
            send_stop_message()
            await ws.send(json.dumps(feedback, ensure_ascii=False))
            return
        await asyncio.sleep(interval)
#持续轮询状态JSON，增量推送state，直到动作成功或失败


def _parse_command(raw_message) -> dict:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    payload = json.loads(raw_message)
    if not isinstance(payload, dict) or payload.get("type") != "command":
        raise ValueError("invalid command payload")
    if not payload.get("skill"):
        raise ValueError("command.skill is required")
    return payload
#解析客户端发来的命令，并做最小字段校验


async def serve(host: str | None = None, port: int | None = None, poll_interval: float | None = None):
    try:
        import websockets
    except ImportError as error:
        raise RuntimeError("未安装websockets，请先执行 pip install websockets") from error

    async def _handler(ws):
        await handle_client(ws, poll_interval=poll_interval)

    async with websockets.serve(_handler, host or DEFAULT_WS_HOST, int(port or DEFAULT_WS_PORT)):
        await asyncio.Future()
#启动WebSocket服务并持续监听客户端连接


def main():
    args = parse_args()
    asyncio.run(serve(args.host, args.port, args.poll_interval))
#命令行入口


if __name__ == "__main__":
    main()
