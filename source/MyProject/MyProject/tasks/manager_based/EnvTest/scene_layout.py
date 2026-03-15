"""EnvTest 的场景常量。

所有尺寸、位置和 case 开关都集中放在这里，
后面改走廊宽度、障碍大小或障碍位置时只需要改这一处。
"""

# 一共 5 个场景。
CASE_COUNT = 5

# 走廊净宽约 3m，两边各有一堵墙。
CORRIDOR_CLEAR_WIDTH = 3.0
WALL_THICKNESS = 0.2
WALL_HEIGHT = 1.0
WALL_LENGTH = 8.0
WALL_CENTER_X = 3.0
WALL_CENTER_Y = 0.5 * CORRIDOR_CLEAR_WIDTH + 0.5 * WALL_THICKNESS

# 两类不可移动障碍的尺寸。
LOW_OBSTACLE_SIZE = (1.0, 1.5, 0.3)
HIGH_OBSTACLE_SIZE = (1.0, 1.5, 0.5)
# 可推动箱子的尺寸。
BOX_SIZE = (0.4, 0.8, 0.2)

# 所有主要障碍的中心都放在机器人前方 2m 左右。
OBSTACLE_CENTER_X = 2.0
# 左右两条通路中心线。
LEFT_LANE_Y = 0.75
RIGHT_LANE_Y = -0.75
# 箱子默认放在机器人与障碍之间。
BOX_CENTER_X = 1.0
BOX_CENTER_Y = 0.0

# 各类物体在“激活状态”下的摆放位置。
ACTIVE_LAYOUT_POSITIONS = {
    "left_low_obstacle": (OBSTACLE_CENTER_X, LEFT_LANE_Y, 0.5 * LOW_OBSTACLE_SIZE[2]),
    "right_low_obstacle": (OBSTACLE_CENTER_X, RIGHT_LANE_Y, 0.5 * LOW_OBSTACLE_SIZE[2]),
    "left_high_obstacle": (OBSTACLE_CENTER_X, LEFT_LANE_Y, 0.5 * HIGH_OBSTACLE_SIZE[2]),
    "right_high_obstacle": (OBSTACLE_CENTER_X, RIGHT_LANE_Y, 0.5 * HIGH_OBSTACLE_SIZE[2]),
    "support_box": (BOX_CENTER_X, BOX_CENTER_Y, 0.5 * BOX_SIZE[2]),
}

# 5 个固定场景的布尔开关表。
# True 表示该场景需要这个物体，False 表示该物体根本不生成。
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
