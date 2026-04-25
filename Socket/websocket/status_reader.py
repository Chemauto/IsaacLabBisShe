from __future__ import annotations

import json
import os


DEFAULT_STATUS_JSON = "/tmp/envtest_live_status.json"


def load_status_snapshot(file_path: str | None = None) -> dict:
    status_path = os.path.abspath(file_path or os.getenv("ENVTEST_STATUS_JSON", DEFAULT_STATUS_JSON))
    if not os.path.isfile(status_path):
        return {}

    try:
        with open(status_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
#读取player持续写出的状态JSON，失败时返回空字典


def status_timestamp(payload: dict) -> float:
    if not isinstance(payload, dict):
        return 0.0
    try:
        return float(payload.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        return 0.0
#提取状态时间戳，给服务端判断是否有新状态可发
