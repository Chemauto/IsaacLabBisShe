import mujoco
import numpy as np
import pygame
import sys
import struct

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__HeightMap_
from unitree_sdk2py.utils.thread import RecurrentThread

import config
if config.ROBOT=="g1":
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
TOPIC_HEIGHTMAP = "rt/heightmap"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35

class UnitreeSdk2Bridge:

    def __init__(self, mj_model, mj_data, mj_lock=None):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.mj_lock = mj_lock

        self.num_motor = self.mj_model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.dt = self.mj_model.opt.timestep
        self.idl_type = (self.num_motor > NUM_MOTOR_IDL_GO) # 0: unitree_go, 1: unitree_hg

        self.joystick = None

        # Check sensor
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == "imu_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        )
        self.HighStateThread.Start()

        self.enable_heightmap = getattr(config, "ENABLE_HEIGHTMAP", False) and not self.idl_type
        self.height_map = None
        self.height_map_puber = None
        self.HeightMapThread = None
        self.height_map_hit_points_w = None
        self.height_map_hit_valid = None
        self.height_map_ray_start = None
        if self.enable_heightmap:
            self._init_height_map()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }

    def _lock(self):
        if self.mj_lock is None:
            from contextlib import nullcontext
            return nullcontext()
        return self.mj_lock

    def LowCmdHandler(self, msg: LowCmd_):
        with self._lock():
            if self.mj_data is not None:
                for i in range(self.num_motor):
                    self.mj_data.ctrl[i] = (
                        msg.motor_cmd[i].tau
                        + msg.motor_cmd[i].kp
                        * (msg.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + msg.motor_cmd[i].kd
                        * (
                            msg.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.num_motor]
                        )
                    )

    def PublishLowState(self):
        with self._lock():
            if self.mj_data is not None:
                for i in range(self.num_motor):
                    self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
                    self.low_state.motor_state[i].dq = self.mj_data.sensordata[
                        i + self.num_motor
                    ]
                    self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[
                        i + 2 * self.num_motor
                    ]

                if self.have_frame_sensor_:

                    self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 0
                    ]
                    self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 1
                    ]
                    self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 2
                    ]
                    self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 3
                    ]

                    self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 4
                    ]
                    self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 5
                    ]
                    self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 6
                    ]

                    self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 7
                    ]
                    self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 8
                    ]
                    self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[
                        self.dim_motor_sensor + 9
                    ]

                if self.joystick != None:
                    pygame.event.get()
                    self.low_state.wireless_remote[2] = int(
                        "".join(
                            [
                                f"{key}"
                                for key in [
                                    0,
                                    0,
                                    int(self.joystick.get_axis(self.axis_id["LT"]) > 0),
                                    int(self.joystick.get_axis(self.axis_id["RT"]) > 0),
                                    int(self.joystick.get_button(self.button_id["SELECT"])),
                                    int(self.joystick.get_button(self.button_id["START"])),
                                    int(self.joystick.get_button(self.button_id["LB"])),
                                    int(self.joystick.get_button(self.button_id["RB"])),
                                ]
                            ]
                        ),
                        2,
                    )
                    self.low_state.wireless_remote[3] = int(
                        "".join(
                            [
                                f"{key}"
                                for key in [
                                    int(self.joystick.get_hat(0)[0] < 0),
                                    int(self.joystick.get_hat(0)[1] < 0),
                                    int(self.joystick.get_hat(0)[0] > 0),
                                    int(self.joystick.get_hat(0)[1] > 0),
                                    int(self.joystick.get_button(self.button_id["Y"])),
                                    int(self.joystick.get_button(self.button_id["X"])),
                                    int(self.joystick.get_button(self.button_id["B"])),
                                    int(self.joystick.get_button(self.button_id["A"])),
                                ]
                            ]
                        ),
                        2,
                    )
                    sticks = [
                        self.joystick.get_axis(self.axis_id["LX"]),
                        self.joystick.get_axis(self.axis_id["RX"]),
                        -self.joystick.get_axis(self.axis_id["RY"]),
                        -self.joystick.get_axis(self.axis_id["LY"]),
                    ]
                    packs = list(map(lambda x: struct.pack("f", x), sticks))
                    self.low_state.wireless_remote[4:8] = packs[0]
                    self.low_state.wireless_remote[8:12] = packs[1]
                    self.low_state.wireless_remote[12:16] = packs[2]
                    self.low_state.wireless_remote[20:24] = packs[3]

                self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):
        with self._lock():
            if self.mj_data is not None:
                self.high_state.position[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 10
                ]
                self.high_state.position[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 11
                ]
                self.high_state.position[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 12
                ]

                self.high_state.velocity[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 13
                ]
                self.high_state.velocity[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 14
                ]
                self.high_state.velocity[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 15
                ]

            self.high_state_puber.Write(self.high_state)

    def _init_height_map(self):
        self.height_map = unitree_go_msg_dds__HeightMap_()
        topic_name = getattr(config, "HEIGHTMAP_TOPIC", TOPIC_HEIGHTMAP)
        self.height_map_puber = ChannelPublisher(topic_name, HeightMap_)
        self.height_map_puber.Init()

        self.height_map_frame_id = getattr(config, "HEIGHTMAP_FRAME_ID", "base_link")
        self.height_map_resolution = float(getattr(config, "HEIGHTMAP_RESOLUTION", 0.1))
        self.height_map_size = tuple(getattr(config, "HEIGHTMAP_SIZE", (1.6, 1.0)))
        self.height_map_ray_offset_z = float(getattr(config, "HEIGHTMAP_RAY_OFFSET_Z", 20.0))
        self.height_map_height_offset = float(getattr(config, "HEIGHTMAP_HEIGHT_OFFSET", 0.5))
        self.height_map_geom_group = np.array(
            getattr(config, "HEIGHTMAP_GEOM_GROUP", (1, 1, 0, 0, 0, 0)),
            dtype=np.uint8,
        )

        x_coords = np.arange(
            -self.height_map_size[0] / 2,
            self.height_map_size[0] / 2 + 1.0e-9,
            self.height_map_resolution,
            dtype=np.float64,
        )
        y_coords = np.arange(
            -self.height_map_size[1] / 2,
            self.height_map_size[1] / 2 + 1.0e-9,
            self.height_map_resolution,
            dtype=np.float64,
        )
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
        self.height_map_local_grid = np.zeros((grid_x.size, 3), dtype=np.float64)
        self.height_map_local_grid[:, 0] = grid_x.flatten()
        self.height_map_local_grid[:, 1] = grid_y.flatten()

        self.height_map.width = int(len(x_coords))
        self.height_map.height = int(len(y_coords))
        self.height_map.resolution = float(self.height_map_resolution)
        self.height_map.origin = [float(x_coords[0]), float(y_coords[0])]
        self.height_map.frame_id = self.height_map_frame_id
        self.height_map.data = [0.0] * self.height_map_local_grid.shape[0]
        self.height_map_hit_points_w = np.zeros_like(self.height_map_local_grid)
        self.height_map_hit_valid = np.zeros(self.height_map_local_grid.shape[0], dtype=bool)
        self.height_map_ray_start = np.zeros(3, dtype=np.float64)
        self.height_map_ray_starts_w = np.zeros_like(self.height_map_local_grid)

        update_dt = float(getattr(config, "HEIGHTMAP_UPDATE_DT", max(self.dt, 0.02)))
        self.HeightMapThread = RecurrentThread(
            interval=update_dt, target=self.PublishHeightMap, name="sim_heightmap"
        )
        self.HeightMapThread.Start()

    def _base_yaw(self):
        quat = self.mj_data.qpos[3:7]
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _compute_height_map(self):
        base_pos = self.mj_data.qpos[0:3].copy()
        yaw = self._base_yaw()
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        yaw_rot = np.array(
            [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        rotated_grid = self.height_map_local_grid @ yaw_rot.T
        ray_starts = base_pos + rotated_grid
        ray_starts[:, 2] += self.height_map_ray_offset_z
        ray_dirs = np.zeros_like(ray_starts)
        ray_dirs[:, 2] = -1.0

        geomid = np.full(ray_dirs.shape[0], -1, dtype=np.int32)
        dist = np.full(ray_dirs.shape[0], -1.0, dtype=np.float64)
        single_geomid = np.zeros(1, dtype=np.int32)
        for idx, ray_dir in enumerate(ray_dirs):
            dist[idx] = mujoco.mj_ray(
                self.mj_model,
                self.mj_data,
                ray_starts[idx],
                ray_dir,
                self.height_map_geom_group,
                1,
                -1,
                single_geomid,
            )
            geomid[idx] = int(single_geomid[0])

        hit_points = ray_starts.copy()
        hit_mask = dist > 0.0
        if np.any(hit_mask):
            hit_points[hit_mask] = ray_starts[hit_mask] + dist[hit_mask, None] * ray_dirs[hit_mask]

        self.height_map_ray_start = base_pos + np.array([0.0, 0.0, self.height_map_ray_offset_z], dtype=np.float64)
        self.height_map_ray_starts_w = ray_starts
        self.height_map_hit_points_w = hit_points
        self.height_map_hit_valid = hit_mask.copy()

        heights = np.zeros(ray_dirs.shape[0], dtype=np.float32)
        if np.any(hit_mask):
            hit_z = hit_points[hit_mask, 2]
            heights[hit_mask] = (base_pos[2] - hit_z - self.height_map_height_offset).astype(np.float32)
        return heights

    def _draw_sphere(self, viewer, pos, radius, rgba):
        if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
            return False
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1
        return True

    def _draw_line(self, viewer, start, end, width, rgba):
        if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
            return False
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        viewer.user_scn.ngeom += 1
        return True

    def RenderDebugViewer(self, viewer):
        viewer.user_scn.ngeom = 0
        if not getattr(config, "ENABLE_HEIGHTMAP_VIS", False):
            return
        if self.height_map_hit_points_w is None or self.height_map_hit_valid is None:
            return

        point_stride = max(1, int(getattr(config, "HEIGHTMAP_VIS_POINT_STRIDE", 1)))
        ray_stride = max(1, int(getattr(config, "HEIGHTMAP_VIS_RAY_STRIDE", 8)))
        point_radius = float(getattr(config, "HEIGHTMAP_VIS_POINT_RADIUS", 0.015))
        draw_rays = bool(getattr(config, "HEIGHTMAP_VIS_DRAW_RAYS", True))

        self._draw_sphere(viewer, self.height_map_ray_start, point_radius * 1.5, [1.0, 0.85, 0.1, 0.9])

        for idx in range(0, len(self.height_map_hit_points_w), point_stride):
            hit = self.height_map_hit_points_w[idx]
            valid = bool(self.height_map_hit_valid[idx])
            if valid:
                height = float(self.height_map.data[idx]) if idx < len(self.height_map.data) else 0.0
                norm = np.clip((height + 1.0) / 2.0, 0.0, 1.0)
                rgba = [norm, 0.2, 1.0 - norm, 0.9]
            else:
                rgba = [1.0, 0.1, 0.1, 0.35]
            if not self._draw_sphere(viewer, hit, point_radius, rgba):
                break

        if draw_rays:
            for idx in range(0, len(self.height_map_hit_points_w), ray_stride):
                hit = self.height_map_hit_points_w[idx]
                valid = bool(self.height_map_hit_valid[idx])
                rgba = [0.3, 0.8, 1.0, 0.35] if valid else [1.0, 0.1, 0.1, 0.2]
                if not self._draw_line(viewer, self.height_map_ray_starts_w[idx], hit, 1.5, rgba):
                    break

    def PublishHeightMap(self):
        with self._lock():
            if self.mj_data is None or self.height_map is None or self.height_map_puber is None:
                return

            heights = self._compute_height_map()
            self.height_map.stamp = float(self.mj_data.time)
            self.height_map.frame_id = self.height_map_frame_id
            self.height_map.data = heights.tolist()
            self.height_map_puber.Write(self.height_map)

    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 3,  # Right stick axis x
                "RY": 4,  # Right stick axis y
                "LT": 2,  # Left trigger
                "RT": 5,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        print(" ")

        print("<<------------- Link ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                print("actuator_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name:
                print(
                    "sensor_index:",
                    index,
                    ", name:",
                    name,
                    ", dim:",
                    self.mj_model.sensor_dim[i],
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
