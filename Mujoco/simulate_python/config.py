ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 0 # Domain id (set to 0 for RL deployment)
INTERFACE = "lo" # Interface

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, required for RL deployment

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer


ENABLE_HEIGHTMAP = True
HEIGHTMAP_TOPIC = "rt/heightmap"
HEIGHTMAP_FRAME_ID = "base_link"
HEIGHTMAP_UPDATE_DT = 0.02
HEIGHTMAP_SIZE = (1.6, 1.0)
HEIGHTMAP_RESOLUTION = 0.1
HEIGHTMAP_RAY_OFFSET_Z = 20.0
HEIGHTMAP_HEIGHT_OFFSET = 0.5
HEIGHTMAP_GEOM_GROUP = (1, 1, 0, 0, 0, 0)


ENABLE_HEIGHTMAP_VIS = True
HEIGHTMAP_VIS_POINT_RADIUS = 0.015
HEIGHTMAP_VIS_POINT_STRIDE = 1
HEIGHTMAP_VIS_DRAW_RAYS = True
HEIGHTMAP_VIS_RAY_STRIDE = 8

# Push-box high-level observation publishing.
# For normal walk/climb scenes, either set this to False or use a scene without support_box.
ENABLE_PUSH_BOX_OBS = True
PUSH_BOX_OBS_TOPIC = "rt/push_box_obs"
PUSH_BOX_OBS_FRAME_ID = "base_link"
PUSH_BOX_OBS_UPDATE_DT = 0.02
PUSH_BOX_BODY_NAME = "support_box"
PUSH_BOX_GOAL_POSITION = (1.7, 0.0, 0.12)
PUSH_BOX_GOAL_YAW = 0.0
PUSH_BOX_GOAL_TOPIC = "rt/push_box_goal"
