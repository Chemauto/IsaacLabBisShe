from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetStatus:
    """终端状态面板中的单个场景物体。"""

    name: str
    position: tuple[float, ...]
    size: tuple[float, ...]


@dataclass(frozen=True)
class StatusSnapshot:
    """EnvTest 运行时快照。"""

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
    """把当前快照转成固定终端面板。"""

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
    """渲染为多行纯文本。"""

    return "\n".join(build_status_lines(snapshot))
