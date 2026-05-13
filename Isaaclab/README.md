# Isaaclab 目录

IsaacLab 仿真入口和 ROS2 桥接。

## 文件

- `FinalSim.py`：启动 IsaacLab 仿真器，读取 `/tmp` 控制文件，写出 `/tmp/envtest_live_status.json`
- `ros2_bridge.py`：读取 FinalSim 状态 JSON → 发布 ROS2 `/go2/*` 话题；订阅 ROS2 控制话题 → 写文件给 FinalSim
- `PublishRos2Topic.py`：旧版桥接（已弃用，保留备用）

## 数据流

```
FinalSim.py → /tmp/envtest_live_status.json → ros2_bridge.py → /go2/* ROS2 → robot_service.py
robot_service.py → /go2/skill_command → ros2_bridge.py → /tmp 控制文件 → FinalSim.py
```

控制文件：

- `model_use`：`/tmp/model_use.txt`
- `velocity`：`/tmp/envtest_velocity_command.txt`
- `goal`：`/tmp/envtest_goal_command.txt`
- `start`：`/tmp/envtest_start.txt`
- `reset`：`/tmp/envtest_reset.txt`

## 启动

一键启动（包含 robot_service）：

```bash
bash run_isaaclab.sh
```

单独启动：

```bash
# 终端 1: FinalSim (env_isaaclab 环境)
conda run -n env_isaaclab python Isaaclab/FinalSim.py --scene_id 3 --enable_front_camera

# 终端 2: ros2_bridge (ros2_env 环境)
conda run -n ros2_env python Isaaclab/ros2_bridge.py

# 终端 3: robot_service (ros2_env 环境)
conda run -n ros2_env python WebSocket/robot_service.py
```

## 调试命令

不经过 FinalProject，直接用 ROS2 命令发布控制 topic：

```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# walk
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"model_use\": 1, \"velocity\": [0.6, 0.0, 0.0], \"start\": true}'}"

# stop
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"start\": false}'}"

# push_box
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"model_use\": 3, \"goal\": \"auto\", \"start\": true}'}"

# navigation
ros2 topic pub --once /go2/goal_pose geometry_msgs/msg/PoseStamped "{pose: {position: {x: 4.5, y: 0.0, z: 0.1}, orientation: {w: 1.0}}}"

# reset
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"reset\": 1}'}"
```

## model_use 对应

| ID | 技能 |
|----|------|
| 0 | idle |
| 1 | walk |
| 2 | climb |
| 3 | push_box |
| 4 | navigation |
| 5 | nav_climb |
