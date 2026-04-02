import math
import threading

import numpy as np

import config
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__HeightMap_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py_bridge import UnitreeSdk2Bridge


class PushBoxSdk2Bridge(UnitreeSdk2Bridge):
    """Bridge with extra push-box observation publishing."""

    def __init__(self, mj_model, mj_data, mj_lock=None):
        super().__init__(mj_model, mj_data, mj_lock)

        self.enable_push_box_obs = getattr(config, "ENABLE_PUSH_BOX_OBS", False) and not self.idl_type
        self.push_box_obs = None
        self.push_box_obs_puber = None
        self.push_box_goal_suber = None
        self.PushBoxObsThread = None
        self.push_box_goal_lock = threading.Lock()
        if self.enable_push_box_obs:
            self._init_push_box_obs()

    @staticmethod
    def _quat_wxyz_to_rot(quat):
        w, x, y, z = quat
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _yaw_from_quat(quat):
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _init_push_box_obs(self):
        self.push_box_obs = unitree_go_msg_dds__HeightMap_()
        topic_name = getattr(config, "PUSH_BOX_OBS_TOPIC", "rt/push_box_obs")
        self.push_box_obs_puber = ChannelPublisher(topic_name, HeightMap_)
        self.push_box_obs_puber.Init()

        self.push_box_obs.frame_id = getattr(config, "PUSH_BOX_OBS_FRAME_ID", "base_link")
        self.push_box_obs.width = 16
        self.push_box_obs.height = 1
        self.push_box_obs.resolution = 1.0
        self.push_box_obs.origin = [0.0, 0.0]
        self.push_box_obs.data = [0.0] * 16

        try:
            self.push_box_body_id = self.mj_model.body(getattr(config, "PUSH_BOX_BODY_NAME", "support_box")).id
        except Exception:
            print("Push-box body not found in current scene. Disable rt/push_box_obs publishing.")
            self.enable_push_box_obs = False
            self.push_box_obs = None
            self.push_box_obs_puber = None
            return

        self.push_box_goal_position = np.asarray(
            getattr(config, "PUSH_BOX_GOAL_POSITION", (1.7, 0.0, 0.12)),
            dtype=np.float64,
        )
        self.push_box_goal_yaw = float(getattr(config, "PUSH_BOX_GOAL_YAW", 0.0))
        goal_topic = getattr(config, "PUSH_BOX_GOAL_TOPIC", "rt/push_box_goal")
        self.push_box_goal_suber = ChannelSubscriber(goal_topic, HeightMap_)
        self.push_box_goal_suber.Init(self.PushBoxGoalHandler, 10)

        update_dt = float(getattr(config, "PUSH_BOX_OBS_UPDATE_DT", max(self.dt, 0.02)))
        self.PushBoxObsThread = RecurrentThread(
            interval=update_dt,
            target=self.PublishPushBoxObs,
            name="sim_push_box_obs",
        )
        self.PushBoxObsThread.Start()

    def PushBoxGoalHandler(self, msg: HeightMap_):
        goal_data = getattr(msg, "data", None)
        if goal_data is None or len(goal_data) != 4:
            return

        with self.push_box_goal_lock:
            self.push_box_goal_position = np.asarray(goal_data[:3], dtype=np.float64)
            self.push_box_goal_yaw = float(goal_data[3])

    def _compute_push_box_obs(self):
        base_pos_w = self.mj_data.qpos[0:3].copy()
        base_quat_w = self.mj_data.qpos[3:7].copy()
        base_rot_w = self._quat_wxyz_to_rot(base_quat_w)
        base_rot_bw = base_rot_w.T
        base_lin_vel_b = base_rot_bw @ self.mj_data.qvel[0:3]
        projected_gravity_b = base_rot_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

        box_pos_w = self.mj_data.xpos[self.push_box_body_id].copy()
        box_quat_w = self.mj_data.xquat[self.push_box_body_id].copy()
        box_rot_w = self._quat_wxyz_to_rot(box_quat_w)
        box_rot_bw = box_rot_w.T

        with self.push_box_goal_lock:
            goal_position_w = self.push_box_goal_position.copy()
            goal_yaw = float(self.push_box_goal_yaw)

        box_in_robot_frame_pos = base_rot_bw @ (box_pos_w - base_pos_w)
        base_yaw = self._yaw_from_quat(base_quat_w)
        box_yaw = self._yaw_from_quat(box_quat_w)
        box_relative_yaw = self._wrap_to_pi(box_yaw - base_yaw)
        box_in_robot_frame_yaw = np.array([math.sin(box_relative_yaw), math.cos(box_relative_yaw)], dtype=np.float64)

        goal_in_box_frame_pos = box_rot_bw @ (goal_position_w - box_pos_w)
        goal_relative_yaw = self._wrap_to_pi(goal_yaw - box_yaw)
        goal_in_box_frame_yaw = np.array([math.sin(goal_relative_yaw), math.cos(goal_relative_yaw)], dtype=np.float64)

        return np.concatenate(
            [
                base_lin_vel_b.astype(np.float32),
                projected_gravity_b.astype(np.float32),
                box_in_robot_frame_pos.astype(np.float32),
                box_in_robot_frame_yaw.astype(np.float32),
                goal_in_box_frame_pos.astype(np.float32),
                goal_in_box_frame_yaw.astype(np.float32),
            ]
        )

    def PublishPushBoxObs(self):
        with self._lock():
            if self.mj_data is None or self.push_box_obs is None or self.push_box_obs_puber is None:
                return

            obs = self._compute_push_box_obs()
            self.push_box_obs.stamp = float(self.mj_data.time)
            self.push_box_obs.data = obs.tolist()
            self.push_box_obs_puber.Write(self.push_box_obs)
