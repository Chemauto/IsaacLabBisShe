from __future__ import annotations

import math


DEFAULT_TIMEOUT_SEC = {
    "nav": 60.0,
    "push": 90.0,
    "walk_skill": 10.0,
    "climb": 20.0,
}
DEFAULT_DISTANCE_TOL = {
    "nav": 0.15,
    "push": 0.08,
    "climb": 0.05,
    "walk_skill": 0.10,
}


def evaluate_feedback(command: dict, start_state: dict, current_state: dict, elapsed_sec: float, timeout_sec: float | None = None):
    skill = _normalize_skill(command.get("skill"))
    timeout = float(timeout_sec or DEFAULT_TIMEOUT_SEC.get(skill, 60.0))

    if skill == "nav" and _distance(current_state.get("robot"), command.get("args")) <= DEFAULT_DISTANCE_TOL["nav"]:
        return _feedback(command, "SUCCESS", "nav reached target", _summary(command, current_state, "robot"))
    if skill == "push" and _distance(current_state.get("box_world"), command.get("args")) <= DEFAULT_DISTANCE_TOL["push"]:
        return _feedback(command, "SUCCESS", "push reached target", _summary(command, current_state, "box_world"))
    if skill == "climb" and _climb_height(current_state, start_state) >= _target_height(command) - DEFAULT_DISTANCE_TOL["climb"]:
        return _feedback(command, "SUCCESS", "climb reached target height", _summary(command, current_state, "robot"))
    if skill == "walk_skill" and _walk_progress(command, start_state, current_state) >= DEFAULT_DISTANCE_TOL["walk_skill"]:
        return _feedback(command, "SUCCESS", "walk reached minimum progress", _summary(command, current_state, "robot"))
    if current_state.get("start") is False and elapsed_sec > 0.5:
        return _feedback(command, "FAILURE", f"{skill} stopped before reaching target", _summary(command, current_state))
    if elapsed_sec >= timeout:
        return _feedback(command, "FAILURE", f"{skill} timeout", _summary(command, current_state))
    return None
#根据当前状态和起始状态判断技能是否成功、失败，未结束时返回None


def _normalize_skill(skill: str | None) -> str:
    normalized = str(skill or "").strip().lower()
    if normalized == "walk":
        return "walk_skill"
    if normalized == "navigation":
        return "nav"
    if normalized == "push_box":
        return "push"
    return normalized
#统一反馈模块内部使用的技能名


def _distance(position: dict | None, goal: dict | None) -> float:
    pos = _position(position)
    target = _position(goal)
    return math.sqrt((pos["x"] - target["x"]) ** 2 + (pos["y"] - target["y"]) ** 2 + (pos["z"] - target["z"]) ** 2)
#计算当前位置和目标位置的欧氏距离


def _climb_height(current_state: dict, start_state: dict) -> float:
    current_z = _position(current_state.get("robot")).get("z", 0.0)
    start_z = _position(start_state.get("robot")).get("z", 0.0)
    return current_z - start_z
#计算攀爬技能自启动以来的高度变化


def _target_height(command: dict) -> float:
    args = dict(command.get("args") or {})
    return _coerce_number(args.get("height"))
#读取climb目标高度


def _walk_progress(command: dict, start_state: dict, current_state: dict) -> float:
    start = _position(start_state.get("robot"))
    current = _position(current_state.get("robot"))
    dx = current["x"] - start["x"]
    dy = current["y"] - start["y"]
    yaw = _coerce_number((start_state.get("robot") or {}).get("yaw"))
    body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    direction = str((command.get("args") or {}).get("direction") or "").strip().lower()
    if direction == "front":
        return body_x
    if direction == "back":
        return -body_x
    if direction == "left":
        return body_y
    if direction == "right":
        return -body_y
    return 0.0
#按walk方向投影位移，判断是否至少走出一个最小步长


def _position(value) -> dict:
    if isinstance(value, dict):
        return {
            "x": _coerce_number(value.get("x")),
            "y": _coerce_number(value.get("y")),
            "z": _coerce_number(value.get("z")),
        }
    return {"x": 0.0, "y": 0.0, "z": 0.0}
#把反馈判断里使用的位置对象统一成x/y/z字典


def _feedback(command: dict, signal: str, message: str, summary: dict | None):
    return {
        "type": "feedback",
        "action_id": command.get("action_id"),
        "skill": _normalize_skill(command.get("skill")),
        "signal": signal,
        "message": message,
        "summary": summary or {},
    }
#构建统一feedback对象，和LLM客户端约定的格式保持一致


def _summary(command: dict, current_state: dict, key: str | None = None) -> dict:
    summary = {
        "target": dict(command.get("args") or {}),
    }
    if key:
        summary[f"final_{key}"] = _position(current_state.get(key))
        if {"x", "y", "z"}.issubset(set((command.get("args") or {}).keys())):
            summary["distance"] = round(_distance(current_state.get(key), command.get("args")), 4)
    return summary
#为feedback补一个轻量摘要，便于客户端显示和回灌LLM


def _coerce_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
#把输入转成float，失败时回退0
