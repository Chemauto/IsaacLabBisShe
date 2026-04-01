from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class AssetStatus:
    """Single scene asset shown in the runtime status panel."""

    name: str
    position: tuple[float, ...]
    size: tuple[float, ...]


@dataclass(frozen=True)
class StatusSnapshot:
    """Runtime snapshot written to terminal and JSON status files."""

    model_use: int | None
    skill: str | None
    scene_id: int | None
    start: bool | None
    unified_obs_dim: int | None
    policy_obs_dim: int | None
    pose_command: tuple[float, ...] | None
    vel_command: tuple[float, ...] | None
    robot_pose: tuple[float, ...] | None
    goal: tuple[float, ...] | None
    platform_1: AssetStatus | None
    platform_2: AssetStatus | None
    box: AssetStatus | None


def _format_scalar(value: object | None) -> str:
    return "None" if value is None else str(value)


def _format_vector(values: tuple[float, ...] | None) -> str:
    if values is None:
        return "None"
    return "(" + ", ".join(f"{value:.3f}" for value in values) + ")"


def _format_start(start: bool | None) -> str:
    if start is None:
        return "None"
    return "1 (RUN)" if start else "0 (IDLE)"


def _format_asset(asset: AssetStatus | None) -> str:
    if asset is None:
        return "None"
    return f"{asset.name}, pos={_format_vector(asset.position)}, size={_format_vector(asset.size)}"


def build_status_lines(snapshot: StatusSnapshot) -> list[str]:
    """Render a snapshot into fixed terminal rows."""

    return [
        "=== EnvTest Live Status ===",
        f"model_use: {_format_scalar(snapshot.model_use)}",
        f"skill: {_format_scalar(snapshot.skill)}",
        f"scene_id: {_format_scalar(snapshot.scene_id)}",
        f"start: {_format_start(snapshot.start)}",
        f"unified_obs_dim: {_format_scalar(snapshot.unified_obs_dim)}",
        f"policy_obs_dim: {_format_scalar(snapshot.policy_obs_dim)}",
        f"pose_command: {_format_vector(snapshot.pose_command)}",
        f"vel_command: {_format_vector(snapshot.vel_command)}",
        f"robot_pose: {_format_vector(snapshot.robot_pose)}",
        f"goal: {_format_vector(snapshot.goal)}",
        f"platform_1: {_format_asset(snapshot.platform_1)}",
        f"platform_2: {_format_asset(snapshot.platform_2)}",
        f"box: {_format_asset(snapshot.box)}",
    ]


def render_status_block(snapshot: StatusSnapshot) -> str:
    """Render the status panel as plain multi-line text."""

    return "\n".join(build_status_lines(snapshot))


def render_status_panel(snapshot: StatusSnapshot) -> None:
    """Refresh the terminal status panel in-place when running in a TTY."""

    if not sys.stdout.isatty():
        return

    try:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(render_status_block(snapshot))
        sys.stdout.write("\n")
        sys.stdout.flush()
    except OSError:
        pass


def write_status_json(snapshot: StatusSnapshot, file_path: str) -> None:
    """Atomically write runtime status to JSON for external polling."""

    if not file_path:
        return

    payload = {
        "timestamp": round(time.time(), 6),
        "model_use": snapshot.model_use,
        "skill": snapshot.skill,
        "scene_id": snapshot.scene_id,
        "start": snapshot.start,
        "unified_obs_dim": snapshot.unified_obs_dim,
        "policy_obs_dim": snapshot.policy_obs_dim,
        "pose_command": list(snapshot.pose_command) if snapshot.pose_command is not None else None,
        "vel_command": list(snapshot.vel_command) if snapshot.vel_command is not None else None,
        "robot_pose": list(snapshot.robot_pose) if snapshot.robot_pose is not None else None,
        "goal": list(snapshot.goal) if snapshot.goal is not None else None,
        "platform_1": None if snapshot.platform_1 is None else {
            "name": snapshot.platform_1.name,
            "position": list(snapshot.platform_1.position),
            "size": list(snapshot.platform_1.size),
        },
        "platform_2": None if snapshot.platform_2 is None else {
            "name": snapshot.platform_2.name,
            "position": list(snapshot.platform_2.position),
            "size": list(snapshot.platform_2.size),
        },
        "box": None if snapshot.box is None else {
            "name": snapshot.box.name,
            "position": list(snapshot.box.position),
            "size": list(snapshot.box.size),
        },
    }

    status_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    temp_path = f"{status_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
        file.write("\n")
    os.replace(temp_path, status_path)

