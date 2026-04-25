from __future__ import annotations

import math


SKILL_TO_MODEL_USE = {
    "nav": 4,
    "navigation": 4,
    "walk_skill": 1,
    "walk": 1,
    "climb": 2,
    "push": 3,
    "push_box": 3,
}
WALK_DIRECTION_TO_VELOCITY = {
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}


def build_udp_message(skill: str, args: dict | None) -> str:
    normalized_skill = _normalize_skill(skill)
    payload = dict(args or {})

    if normalized_skill == "nav":
        return f"model_use=4; goal={_format_xyz(payload)}; start=1"
    if normalized_skill == "push":
        return f"model_use=3; goal={_format_xyz(payload)}; start=1"
    if normalized_skill == "walk_skill":
        velocity = _walk_velocity(payload)
        return f"model_use=1; velocity={_format_values(velocity)}; start=1"
    if normalized_skill == "climb":
        return "model_use=2; start=1"
    raise ValueError(f"unsupported skill: {skill}")
#把LLM侧技能命令翻译成现有UDP server能理解的文本消息


def build_state_payload(status: dict | None) -> dict:
    raw = dict(status or {})
    robot = _robot_payload(raw)
    box_world = _box_world_payload(raw, robot)
    box_relative = _box_relative_payload(raw, robot, box_world)
    return {
        "type": "state",
        "timestamp": _coerce_number(raw.get("timestamp")),
        "robot": robot,
        "box_relative": box_relative,
        "box_world": box_world,
        "skill": raw.get("skill"),
        "current_skill": raw.get("skill"),
        "model_use": raw.get("model_use"),
        "start": _coerce_bool(raw.get("start")),
        "raw": raw,
    }
#把player状态JSON标准化成WebSocket推给客户端的统一state协议


def _normalize_skill(skill: str | None) -> str:
    normalized = str(skill or "").strip().lower()
    if normalized in ("navigation",):
        return "nav"
    if normalized in ("walk",):
        return "walk_skill"
    if normalized in ("push_box",):
        return "push"
    return normalized
#统一技能名，避免LLM侧和仿真侧别名不一致


def _format_xyz(args: dict) -> str:
    return _format_values((_coerce_number(args.get("x")), _coerce_number(args.get("y")), _coerce_number(args.get("z"))))
#把x/y/z参数格式化成goal文本


def _format_values(values: tuple[float, float, float]) -> str:
    return ",".join(_fmt_number(value) for value in values)
#把三维值格式化成UDP消息里的逗号分隔文本


def _fmt_number(value: float) -> str:
    value = round(_coerce_number(value), 6)
    return str(int(value)) if value == int(value) else str(value)
#格式化数字，尽量保持消息短且稳定


def _walk_velocity(args: dict) -> tuple[float, float, float]:
    direction = str(args.get("direction") or "").strip().lower()
    if direction not in WALK_DIRECTION_TO_VELOCITY:
        raise ValueError(f"unsupported walk direction: {direction}")
    scale = _coerce_number(args.get("v"))
    base = WALK_DIRECTION_TO_VELOCITY[direction]
    return tuple(component * scale for component in base)
#把front/back/left/right和速度映射成EnvTest需要的vx/vy/wz


def _robot_payload(status: dict) -> dict:
    pose = _coerce_vector(status.get("robot_pose"))
    yaw = status.get("robot_yaw")
    if yaw is None and len(pose) >= 4:
        yaw = pose[3]
    return {
        "x": pose[0],
        "y": pose[1],
        "z": pose[2],
        "yaw": _coerce_number(yaw),
    }
#提取机器人世界坐标和偏航角


def _box_world_payload(status: dict, robot: dict) -> dict:
    if isinstance(status.get("box_world"), dict):
        return _position_payload(status["box_world"])
    if isinstance(status.get("box"), dict):
        position = status["box"].get("position")
        if position is not None:
            return _position_payload(position)
    relative = status.get("box_relative")
    if relative is not None:
        return _body_to_world(robot, _position_payload(relative))
    return {"x": 0.0, "y": 0.0, "z": 0.0}
#优先用绝对箱子坐标，没有时再由相对坐标和yaw换回世界系


def _box_relative_payload(status: dict, robot: dict, box_world: dict) -> dict:
    if status.get("box_relative") is not None:
        return _position_payload(status.get("box_relative"))
    return _world_to_body(robot, box_world)
#优先用状态里已有的箱子相对位姿，否则由世界坐标反算


def _position_payload(value) -> dict:
    if isinstance(value, dict):
        return {
            "x": _coerce_number(value.get("x")),
            "y": _coerce_number(value.get("y")),
            "z": _coerce_number(value.get("z")),
        }
    vector = _coerce_vector(value)
    return {"x": vector[0], "y": vector[1], "z": vector[2]}
#把list/tuple/dict都标准化成x/y/z字典


def _coerce_vector(value) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        values = list(value) + [0.0, 0.0, 0.0]
        return _coerce_number(values[0]), _coerce_number(values[1]), _coerce_number(values[2])
    return 0.0, 0.0, 0.0
#把输入转成固定三维向量，缺失时回退零向量


def _body_to_world(robot: dict, relative: dict) -> dict:
    yaw = _coerce_number(robot.get("yaw"))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return {
        "x": _coerce_number(robot.get("x")) + cos_yaw * relative["x"] - sin_yaw * relative["y"],
        "y": _coerce_number(robot.get("y")) + sin_yaw * relative["x"] + cos_yaw * relative["y"],
        "z": _coerce_number(robot.get("z")) + relative["z"],
    }
#把机器人本体系下的箱子相对位姿转换成世界系坐标


def _world_to_body(robot: dict, box_world: dict) -> dict:
    yaw = _coerce_number(robot.get("yaw"))
    dx = box_world["x"] - _coerce_number(robot.get("x"))
    dy = box_world["y"] - _coerce_number(robot.get("y"))
    return {
        "x": math.cos(yaw) * dx + math.sin(yaw) * dy,
        "y": -math.sin(yaw) * dx + math.cos(yaw) * dy,
        "z": box_world["z"] - _coerce_number(robot.get("z")),
    }
#把箱子世界坐标转换成机器人本体系相对坐标


def _coerce_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
#把任意输入尽量转成float，失败时回退0


def _coerce_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in ("1", "true", "on", "yes", "run"):
        return True
    if token in ("0", "false", "off", "no", "idle", "stop"):
        return False
    return None
#把start字段尽量转成布尔值，未知格式则保留None
