"""EnvTest unified observation schema.

This module is intentionally pure-Python so tests and runtime code can share the
same observation layout without importing Isaac Lab or Isaac Sim.
"""

from __future__ import annotations


UNIFIED_POLICY_TERM_DIMS: dict[str, int] = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos": 12,
    "joint_vel": 12,
    "actions": 12,
    "height_scan": 187,
    "box_pose": 7,
    "robot_position": 3,
    "pose_command": 4,
    "goal_command": 4,
    "push_actions": 3,
}

UNIFIED_POLICY_DIM = sum(UNIFIED_POLICY_TERM_DIMS.values())

NAVIGATION_HIGH_LEVEL_OBS_TERMS = (
    "base_lin_vel",
    "projected_gravity",
    "pose_command",
    "height_scan",
)
NAVIGATION_HIGH_LEVEL_OBS_DIM = sum(UNIFIED_POLICY_TERM_DIMS[term_name] for term_name in NAVIGATION_HIGH_LEVEL_OBS_TERMS)

RUNTIME_BUFFER_DIMS: dict[str, int] = {
    "velocity_commands": 3,
    "pose_command": 4,
    "push_goal_command": 4,
    "push_actions": 3,
}
