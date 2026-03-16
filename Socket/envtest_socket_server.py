#!/usr/bin/env python3
"""EnvTest UDP 控制服务。

职责：
- 接收外部 UDP 消息
- 解析 `model_use / velocity / goal / start`
- 写入 `envtest_model_use_player.py` 默认读取的控制文件
"""

from __future__ import annotations

import argparse
import os
import re
import socket
from dataclasses import dataclass


SKILL_NAME_TO_ID = {
    "idle": 0,
    "walk": 1,
    "climb": 2,
    "push": 3,
    "push_box": 3,
}
BOOL_TRUE = {"1", "true", "on", "run", "start", "yes", "y"}
BOOL_FALSE = {"0", "false", "off", "stop", "idle", "no", "n"}


@dataclass
class OutputPaths:
    """服务写入的目标文件。"""

    model_use: str
    velocity: str
    goal: str
    start: str


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Receive EnvTest control commands through UDP socket.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址。")
    parser.add_argument("--port", type=int, default=5566, help="监听端口。")
    parser.add_argument("--model-use-file", type=str, default="/tmp/model_use.txt", help="model_use 文件路径。")
    parser.add_argument(
        "--velocity-file",
        type=str,
        default="/tmp/envtest_velocity_command.txt",
        help="速度指令文件路径。",
    )
    parser.add_argument(
        "--goal-file",
        type=str,
        default="/tmp/envtest_goal_command.txt",
        help="位置指令文件路径。",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        default="/tmp/envtest_start.txt",
        help="启动开关文件路径。",
    )
    return parser.parse_args()


def _ensure_parent_dir(file_path: str):
    """确保目标文件父目录存在。"""

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_text(file_path: str, text: str):
    """写文本到目标文件。"""

    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text.strip() + "\n")


def _parse_named_int(text: str, field_name: str) -> int | None:
    """解析形如 `field=1` 的整数。"""

    match = re.search(rf"\b{field_name}\b\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _parse_named_vector(text: str, field_name: str) -> tuple[float, float, float] | None:
    """解析形如 `field=0.6,0,0` 的三维向量。"""

    pattern = (
        rf"\b{field_name}\b\s*[:=]\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"[\s,]+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"[\s,]+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _parse_auto_goal(text: str) -> bool:
    """判断是否要求把 goal 切回自动模式。"""

    match = re.search(r"\b(goal|position|pos|target)\b\s*[:=]\s*([A-Za-z_]+)", text, flags=re.IGNORECASE)
    if match is None:
        return False
    token = match.group(2).strip().lower()
    return token in ("auto", "scene", "default")


def _parse_start(text: str) -> bool | None:
    """解析启动开关。"""

    match = re.search(r"\b(start|run)\b\s*[:=]\s*([A-Za-z0-9_]+)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    token = match.group(2).strip().lower()
    if token in BOOL_TRUE:
        return True
    if token in BOOL_FALSE:
        return False
    raise ValueError(f"无法识别 start 值: {token}")


def _parse_skill_name(text: str) -> int | None:
    """解析 `walk / climb / push_box / idle` 这类消息。"""

    return SKILL_NAME_TO_ID.get(text.strip().lower())


def _format_vector(values: tuple[float, float, float]) -> str:
    """把三维向量格式化成文件内容。"""

    return f"{values[0]} {values[1]} {values[2]}"


def apply_message(text: str, output_paths: OutputPaths) -> list[str]:
    """解析一条 UDP 消息，并把结果写入对应文件。"""

    updates: list[str] = []
    normalized = text.strip()
    if not normalized:
        raise ValueError("收到空消息。")

    skill_id = _parse_named_int(normalized, "model_use")
    if skill_id is None:
        skill_id = _parse_named_int(normalized, "skill")
    if skill_id is None:
        skill_id = _parse_skill_name(normalized)
    if skill_id is not None:
        if skill_id not in (0, 1, 2, 3):
            raise ValueError(f"model_use 必须是 0/1/2/3，收到: {skill_id}")
        _write_text(output_paths.model_use, str(skill_id))
        updates.append(f"model_use={skill_id}")

    velocity = _parse_named_vector(normalized, "velocity")
    if velocity is None:
        velocity = _parse_named_vector(normalized, "vel")
    if velocity is not None:
        _write_text(output_paths.velocity, _format_vector(velocity))
        updates.append(f"velocity={velocity}")

    goal = _parse_named_vector(normalized, "goal")
    if goal is None:
        goal = _parse_named_vector(normalized, "position")
    if goal is None:
        goal = _parse_named_vector(normalized, "pos")
    if goal is None:
        goal = _parse_named_vector(normalized, "target")
    if goal is not None:
        _write_text(output_paths.goal, _format_vector(goal))
        updates.append(f"goal={goal}")
    elif _parse_auto_goal(normalized):
        _write_text(output_paths.goal, "auto")
        updates.append("goal=auto")

    start = _parse_start(normalized)
    if start is not None:
        _write_text(output_paths.start, "1" if start else "0")
        updates.append(f"start={int(start)}")

    if not updates:
        raise ValueError(
            "未识别到有效字段。支持：model_use / skill / velocity / goal / start，"
            "例如 `model_use=3; goal=1.8,0,0.1; start=1`。"
        )
    return updates


def main():
    """主入口。"""

    args = parse_args()
    output_paths = OutputPaths(
        model_use=args.model_use_file,
        velocity=args.velocity_file,
        goal=args.goal_file,
        start=args.start_file,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print(f"[INFO] EnvTest UDP control server listening on {args.host}:{args.port}")
    print(f"[INFO] model_use file : {output_paths.model_use}")
    print(f"[INFO] velocity file  : {output_paths.velocity}")
    print(f"[INFO] goal file      : {output_paths.goal}")
    print(f"[INFO] start file     : {output_paths.start}")
    print("[INFO] Example: model_use=3; goal=1.8,0,0.1; start=1")

    try:
        while True:
            data, address = sock.recvfrom(2048)
            message = data.decode("utf-8", errors="ignore").strip()
            try:
                updates = apply_message(message, output_paths)
                print(f"[INFO] {address[0]}:{address[1]} -> " + ", ".join(updates))
            except ValueError as err:
                print(f"[WARN] Ignore invalid message from {address}: {message!r} ({err})")
    except KeyboardInterrupt:
        print("\n[INFO] EnvTest UDP control server stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
