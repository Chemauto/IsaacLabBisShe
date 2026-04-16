"""Observation specs for EnvTest runtime skill adapters."""

from __future__ import annotations

from dataclasses import dataclass

from MyProject.tasks.manager_based.EnvTest.observation_schema import (
    NAVIGATION_HIGH_LEVEL_OBS_DIM,
    NAVIGATION_HIGH_LEVEL_OBS_TERMS,
)


@dataclass(frozen=True)
class SkillObservationSpec:
    """Flat observation contract expected by one policy."""

    name: str
    obs_terms: tuple[str, ...]
    obs_dim: int


WALK_LOW_LEVEL_OBS_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
)
CLIMB_LOW_LEVEL_OBS_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
)
PUSH_HIGH_LEVEL_OBS_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "box_in_robot_frame_pos",
    "box_in_robot_frame_yaw",
    "goal_in_box_frame_pos",
    "goal_in_box_frame_yaw",
    "push_actions",
)

WALK_LOW_LEVEL_OBS_DIM = 232
CLIMB_LOW_LEVEL_OBS_DIM = 232
PUSH_HIGH_LEVEL_OBS_DIM = 19

SKILL_OBSERVATION_SPECS = {
    "walk": SkillObservationSpec("walk", WALK_LOW_LEVEL_OBS_TERMS, WALK_LOW_LEVEL_OBS_DIM),
    "climb": SkillObservationSpec("climb", CLIMB_LOW_LEVEL_OBS_TERMS, CLIMB_LOW_LEVEL_OBS_DIM),
    "push_box": SkillObservationSpec("push_box", PUSH_HIGH_LEVEL_OBS_TERMS, PUSH_HIGH_LEVEL_OBS_DIM),
    "navigation": SkillObservationSpec("navigation", NAVIGATION_HIGH_LEVEL_OBS_TERMS, NAVIGATION_HIGH_LEVEL_OBS_DIM),
}

__all__ = [
    "CLIMB_LOW_LEVEL_OBS_DIM",
    "CLIMB_LOW_LEVEL_OBS_TERMS",
    "NAVIGATION_HIGH_LEVEL_OBS_DIM",
    "NAVIGATION_HIGH_LEVEL_OBS_TERMS",
    "PUSH_HIGH_LEVEL_OBS_DIM",
    "PUSH_HIGH_LEVEL_OBS_TERMS",
    "SKILL_OBSERVATION_SPECS",
    "SkillObservationSpec",
    "WALK_LOW_LEVEL_OBS_DIM",
    "WALK_LOW_LEVEL_OBS_TERMS",
]
