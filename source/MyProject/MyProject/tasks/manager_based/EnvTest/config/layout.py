"""Scene layout presets for EnvTest."""

from __future__ import annotations

OPTIONAL_SCENE_ASSET_NAMES = (
    "left_low_obstacle",
    "right_low_obstacle",
    "left_high_obstacle",
    "right_high_obstacle",
    "support_box",
)

SCENE_LAYOUTS = (
    {
        "name": "case_1_clear",
        "name_zh": "场景1_双侧无障碍",
        "left_low_obstacle": False,
        "right_low_obstacle": False,
        "left_high_obstacle": False,
        "right_high_obstacle": False,
        "support_box": False,
    },
    {
        "name": "case_2_left_low_right_clear",
        "name_zh": "场景2_左低障碍右侧畅通",
        "left_low_obstacle": True,
        "right_low_obstacle": False,
        "left_high_obstacle": False,
        "right_high_obstacle": False,
        "support_box": False,
    },
    {
        "name": "case_3_both_low",
        "name_zh": "场景3_左右均为低障碍",
        "left_low_obstacle": True,
        "right_low_obstacle": True,
        "left_high_obstacle": False,
        "right_high_obstacle": False,
        "support_box": False,
    },
    {
        "name": "case_4_left_low_right_high",
        "name_zh": "场景4_左低障碍右高障碍",
        "left_low_obstacle": True,
        "right_low_obstacle": False,
        "left_high_obstacle": False,
        "right_high_obstacle": True,
        "support_box": False,
    },
    {
        "name": "case_5_both_high_with_box",
        "name_zh": "场景5_双高障碍加可推动箱子",
        "left_low_obstacle": False,
        "right_low_obstacle": False,
        "left_high_obstacle": True,
        "right_high_obstacle": True,
        "support_box": True,
    },
)

CASE_COUNT = len(SCENE_LAYOUTS)


def get_scene_layout(scene_id: int) -> dict[str, str | bool]:
    """Return the fixed layout definition for one EnvTest scene id."""

    if not 0 <= scene_id < CASE_COUNT:
        raise ValueError(f"scene_id must be in [0, {CASE_COUNT - 1}], but received {scene_id}.")
    return SCENE_LAYOUTS[scene_id]


__all__ = [
    "CASE_COUNT",
    "OPTIONAL_SCENE_ASSET_NAMES",
    "SCENE_LAYOUTS",
    "get_scene_layout",
]

