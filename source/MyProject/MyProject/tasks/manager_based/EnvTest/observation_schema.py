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
    "pose_command": 4,
    "box_in_robot_frame_pos": 3,
    "box_in_robot_frame_yaw": 2,
    "goal_in_box_frame_pos": 3,
    "goal_in_box_frame_yaw": 2,
    "push_actions": 3,
}

UNIFIED_POLICY_DIM = sum(UNIFIED_POLICY_TERM_DIMS.values())

NAVIGATION_HIGH_LEVEL_OBS_TERMS = (
    "projected_gravity",
    "pose_command",
    "height_scan",
    "push_actions",
)
NAVIGATION_HIGH_LEVEL_OBS_DIM = sum(UNIFIED_POLICY_TERM_DIMS[term_name] for term_name in NAVIGATION_HIGH_LEVEL_OBS_TERMS)

RUNTIME_BUFFER_DIMS: dict[str, int] = {
    "velocity_commands": 3,
    "pose_command": 4,
    "push_goal_command": 4,
    "push_actions": 3,
}
