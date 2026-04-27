from __future__ import annotations

import math

from WebSocket.protocol import coerce_number, normalize_skill


DEFAULT_TIMEOUT_SEC = {
    "nav": 60.0,
    "nav_climb": 60.0,
    "push": 90.0,
    "walk_skill": 30.0,
    "climb": 30.0,
}
DEFAULT_DISTANCE_TOL = {
    "nav": 0.15,
    "nav_climb": 0.15,
    "push": 0.08,
    "climb": 0.05,
    "walk_skill": 0.08,
}
CLIMB_STABLE_HITS = 10
CLIMB_STABLE_DELTA_M = 0.02
CLIMB_SETTLE_SEC = 1.0


class FeedbackTracker:
    def __init__(self, command: dict, start_state: dict):
        self.command = command
        self.start_state = start_state
        self._last_robot = None
        self._stable_hits = 0
        self._reached_at = None

    def evaluate(self, current_state: dict, elapsed_sec: float, timeout_sec: float | None = None):
        skill = normalize_skill(self.command.get("skill"))
        if skill != "climb":
            return evaluate_feedback(self.command, self.start_state, current_state, elapsed_sec, timeout_sec)
        return self._evaluate_climb(current_state, elapsed_sec, timeout_sec)

    def _evaluate_climb(self, current_state: dict, elapsed_sec: float, timeout_sec: float | None):
        timeout = float(timeout_sec or DEFAULT_TIMEOUT_SEC["climb"])
        robot = _position(current_state.get("robot"))
        reached = _climb_height(current_state, self.start_state) >= _target_height(self.command) - DEFAULT_DISTANCE_TOL["climb"]
        if reached and self._reached_at is None:
            self._reached_at = elapsed_sec
        stable = self._last_robot is not None and _distance(robot, self._last_robot) <= CLIMB_STABLE_DELTA_M
        settled = self._reached_at is not None and elapsed_sec - self._reached_at >= CLIMB_SETTLE_SEC
        self._last_robot = robot

        if reached and settled and stable:
            self._stable_hits += 1
            if self._stable_hits >= CLIMB_STABLE_HITS:
                summary = _summary(self.command, current_state, "robot")
                summary["stable_hits"] = self._stable_hits
                summary["settle_sec"] = round(elapsed_sec - self._reached_at, 3)
                return _feedback(self.command, "SUCCESS", "climb reached target height and stabilized", summary)
        else:
            self._stable_hits = 0

        if current_state.get("start") is False and elapsed_sec > 0.5:
            return _feedback(self.command, "FAILURE", "climb stopped before reaching target", _summary(self.command, current_state))
        if elapsed_sec >= timeout:
            return _feedback(self.command, "FAILURE", "climb timeout", _summary(self.command, current_state))
        return None
#对climb做连续稳定判定，避免瞬间高度达到就返回成功


def evaluate_feedback(command: dict, start_state: dict, current_state: dict, elapsed_sec: float, timeout_sec: float | None = None):
    skill = normalize_skill(command.get("skill"))
    timeout = float(timeout_sec or DEFAULT_TIMEOUT_SEC.get(skill, 60.0))

    if skill in {"nav", "nav_climb"} and _distance_xy(current_state.get("robot"), command.get("args")) <= DEFAULT_DISTANCE_TOL[skill]:
        return _feedback(command, "SUCCESS", f"{skill} reached target", _summary(command, current_state, "robot"))
    if skill == "push" and _distance_xy(current_state.get("box_world"), command.get("args")) <= DEFAULT_DISTANCE_TOL["push"]:
        return _feedback(command, "SUCCESS", "push reached target", _summary(command, current_state, "box_world"))
    if skill == "climb" and _climb_height(current_state, start_state) >= _target_height(command) - DEFAULT_DISTANCE_TOL["climb"]:
        return _feedback(command, "SUCCESS", "climb reached target height", _summary(command, current_state, "robot"))
    if skill == "walk_skill" and _walk_progress(command, start_state, current_state) >= _walk_target(command):
        return _feedback(command, "SUCCESS", "walk reached minimum progress", _summary(command, current_state, "robot"))
    if current_state.get("start") is False and elapsed_sec > 0.5:
        return _feedback(command, "FAILURE", f"{skill} stopped before reaching target", _summary(command, current_state))
    if elapsed_sec >= timeout:
        return _feedback(command, "FAILURE", f"{skill} timeout", _summary(command, current_state))
    return None


def _distance(position: dict | None, goal: dict | None) -> float:
    pos = _position(position)
    target = _position(goal)
    return math.sqrt((pos["x"] - target["x"]) ** 2 + (pos["y"] - target["y"]) ** 2 + (pos["z"] - target["z"]) ** 2)


def _distance_xy(position: dict | None, goal: dict | None) -> float:
    pos = _position(position)
    target = _position(goal)
    return math.hypot(pos["x"] - target["x"], pos["y"] - target["y"])


def _climb_height(current_state: dict, start_state: dict) -> float:
    return _position(current_state.get("robot")).get("z", 0.0) - _position(start_state.get("robot")).get("z", 0.0)


def _target_height(command: dict) -> float:
    return coerce_number(dict(command.get("args") or {}).get("height"))


def _walk_target(command: dict) -> float:
    distance = abs(coerce_number(dict(command.get("args") or {}).get("distance")))
    return max(DEFAULT_DISTANCE_TOL["walk_skill"], distance - DEFAULT_DISTANCE_TOL["walk_skill"])


def _walk_progress(command: dict, start_state: dict, current_state: dict) -> float:
    start = _position(start_state.get("robot"))
    current = _position(current_state.get("robot"))
    dx = current["x"] - start["x"]
    dy = current["y"] - start["y"]
    yaw = coerce_number((start_state.get("robot") or {}).get("yaw"))
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


def _position(value) -> dict:
    if isinstance(value, dict):
        return {
            "x": coerce_number(value.get("x")),
            "y": coerce_number(value.get("y")),
            "z": coerce_number(value.get("z")),
        }
    return {"x": 0.0, "y": 0.0, "z": 0.0}


def _feedback(command: dict, signal: str, message: str, summary: dict | None):
    return {
        "type": "feedback",
        "action_id": command.get("action_id"),
        "skill": normalize_skill(command.get("skill")),
        "signal": signal,
        "message": message,
        "summary": summary or {},
    }


def _summary(command: dict, current_state: dict, key: str | None = None) -> dict:
    summary = {"target": dict(command.get("args") or {})}
    if key:
        summary[f"final_{key}"] = _position(current_state.get(key))
        if {"x", "y", "z"}.issubset(set((command.get("args") or {}).keys())):
            summary["distance"] = round(_distance(current_state.get(key), command.get("args")), 4)
            summary["distance_xy"] = round(_distance_xy(current_state.get(key), command.get("args")), 4)
    return summary
