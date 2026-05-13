from __future__ import annotations

import math


WALK_DIRECTION_TO_VELOCITY = {
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}
DEFAULT_WALK_SPEED = 0.5
DEFAULT_PUSH_GOAL_Z = 0.12


def normalize_skill(skill: str | None) -> str:
    normalized = str(skill or "").strip().lower()
    if normalized == "navigation":
        return "nav"
    if normalized in {"navigation_climb", "navclimb"}:
        return "nav_climb"
    if normalized == "walk":
        return "walk_skill"
    if normalized == "push_box":
        return "push"
    return normalized


def walk_velocity(args: dict | None) -> tuple[float, float, float]:
    payload = dict(args or {})
    direction = str(payload.get("direction") or "").strip().lower()
    if direction not in WALK_DIRECTION_TO_VELOCITY:
        raise ValueError(f"unsupported walk direction: {direction}")
    scale = coerce_number(payload.get("v", DEFAULT_WALK_SPEED))
    base = WALK_DIRECTION_TO_VELOCITY[direction]
    return tuple(component * scale for component in base)


def position_payload(value) -> dict:
    if isinstance(value, dict):
        return {
            "x": coerce_number(value.get("x")),
            "y": coerce_number(value.get("y")),
            "z": coerce_number(value.get("z")),
        }
    if isinstance(value, (list, tuple)):
        values = list(value) + [0.0, 0.0, 0.0]
        return {
            "x": coerce_number(values[0]),
            "y": coerce_number(values[1]),
            "z": coerce_number(values[2]),
        }
    return {"x": 0.0, "y": 0.0, "z": 0.0}


def push_goal(args: dict | None, box_world: dict | None = None) -> list[float]:
    payload = dict(args or {})
    z = coerce_number(payload.get("z"))
    if z <= 0.05:
        box_z = coerce_number((box_world or {}).get("z"))
        z = box_z if box_z > 0.05 else DEFAULT_PUSH_GOAL_Z
    return [coerce_number(payload.get("x")), coerce_number(payload.get("y")), z]


def stop_command_payload() -> dict:
    return {
        "model_use": 0,
        "skill": "idle",
        "start": False,
        "velocity": [0.0, 0.0, 0.0],
    }


def idle_command_payload() -> dict:
    return {
        "velocity": [0.0, 0.0, 0.0],
        "start": True,
    }


def world_to_body(robot: dict, world: dict) -> dict:
    yaw = coerce_number(robot.get("yaw"))
    dx = coerce_number(world.get("x")) - coerce_number(robot.get("x"))
    dy = coerce_number(world.get("y")) - coerce_number(robot.get("y"))
    return {
        "x": math.cos(yaw) * dx + math.sin(yaw) * dy,
        "y": -math.sin(yaw) * dx + math.cos(yaw) * dy,
        "z": coerce_number(world.get("z")) - coerce_number(robot.get("z")),
    }


def coerce_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def coerce_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in ("1", "true", "on", "yes", "run", "start"):
        return True
    if token in ("0", "false", "off", "no", "idle", "stop"):
        return False
    return None
