"""Asset geometry and default placement constants for EnvTest scenes."""

from __future__ import annotations

# Corridor geometry.
CORRIDOR_CLEAR_WIDTH = 3.0
WALL_THICKNESS = 0.2
WALL_HEIGHT = 1.0
WALL_LENGTH = 12.0
WALL_CENTER_X = 3.0
WALL_CENTER_Y = 0.5 * CORRIDOR_CLEAR_WIDTH + 0.5 * WALL_THICKNESS

# Static obstacle and movable box sizes.
LOW_OBSTACLE_SIZE = (2.0, 1.5, 0.3)
HIGH_OBSTACLE_SIZE = (2.0, 1.5, 0.5)
BOX_SIZE = (0.6, 0.8, 0.2)

# Default placement anchors.
OBSTACLE_CENTER_X = 3.0
LEFT_LANE_Y = 0.75
RIGHT_LANE_Y = -0.75
BOX_CENTER_X = 1.0
BOX_CENTER_Y = 0.0

# Asset positions when the scene layout marks them as active.
ACTIVE_LAYOUT_POSITIONS = {
    "left_low_obstacle": (OBSTACLE_CENTER_X, LEFT_LANE_Y, 0.5 * LOW_OBSTACLE_SIZE[2]),
    "right_low_obstacle": (OBSTACLE_CENTER_X, RIGHT_LANE_Y, 0.5 * LOW_OBSTACLE_SIZE[2]),
    "left_high_obstacle": (OBSTACLE_CENTER_X, LEFT_LANE_Y, 0.5 * HIGH_OBSTACLE_SIZE[2]),
    "right_high_obstacle": (OBSTACLE_CENTER_X, RIGHT_LANE_Y, 0.5 * HIGH_OBSTACLE_SIZE[2]),
    "support_box": (BOX_CENTER_X, BOX_CENTER_Y, 0.5 * BOX_SIZE[2]),
}

# Fallback sizes used by status/debug helpers when scene assets are optional.
SCENE_ASSET_SIZE_FALLBACKS = {
    "left_low_obstacle": LOW_OBSTACLE_SIZE,
    "right_low_obstacle": LOW_OBSTACLE_SIZE,
    "left_high_obstacle": HIGH_OBSTACLE_SIZE,
    "right_high_obstacle": HIGH_OBSTACLE_SIZE,
    "support_box": BOX_SIZE,
}

__all__ = [
    "ACTIVE_LAYOUT_POSITIONS",
    "BOX_CENTER_X",
    "BOX_CENTER_Y",
    "BOX_SIZE",
    "CORRIDOR_CLEAR_WIDTH",
    "HIGH_OBSTACLE_SIZE",
    "LEFT_LANE_Y",
    "LOW_OBSTACLE_SIZE",
    "OBSTACLE_CENTER_X",
    "RIGHT_LANE_Y",
    "SCENE_ASSET_SIZE_FALLBACKS",
    "WALL_CENTER_X",
    "WALL_CENTER_Y",
    "WALL_HEIGHT",
    "WALL_LENGTH",
    "WALL_THICKNESS",
]

