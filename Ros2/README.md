# Ros2 目录说明

这个目录保留 EnvTest 仿真入口、ROS2 桥接程序和本说明文档。`FinalSim.py` 负责启动仿真器并持续写出状态 JSON，`PublishRos2Topic.py` 负责把这些数据转换成和实机一致的 ROS2 topic，同时把 ROS2 控制 topic 写回 EnvTest 使用的控制文件。

## 文件

- `FinalSim.py`：启动 EnvTest 仿真器，读取 `/tmp` 控制文件，并持续写出 `/tmp/envtest_live_status.json`
- `PublishRos2Topic.py`：读取 EnvTest JSON 状态、发布 ROS2 topic、订阅 ROS2 控制 topic
- `README.md`：当前说明

## ROS2 仿真链路

```text
WebSocket/robot_service.py 或 ROS2 调试命令
             -> /go2/skill_command, /go2/cmd_vel, /go2/goal_pose
             -> Ros2/PublishRos2Topic.py
             -> /tmp 控制文件
             -> Ros2/FinalSim.py
             -> /tmp/envtest_live_status.json
             -> Ros2/PublishRos2Topic.py
             -> /go2/odom, /go2/box_pose, /go2/skill_status, /go2/scene_objects
             -> WebSocket/robot_service.py 或 ROS2 调试工具
```

EnvTest player 不需要改，仍然读取这些文件：

- `model_use`: `/tmp/model_use.txt`
- `velocity`: `/tmp/envtest_velocity_command.txt`
- `goal`: `/tmp/envtest_goal_command.txt`
- `start`: `/tmp/envtest_start.txt`
- `reset`: `/tmp/envtest_reset.txt`

## 启动流程

1. 启动 EnvTest 仿真器：

```bash
cd /home/robot/work/IsaacLabBisShe
python Ros2/FinalSim.py --scene_id 3 --enable_front_camera
```

2. 启动 ROS2 桥接程序：

```bash
cd /home/robot/work/IsaacLabBisShe
source /opt/ros/jazzy/setup.bash
python Ros2/PublishRos2Topic.py
```

3. 启动 LLM WebSocket 服务端：

```bash
cd /home/robot/work/IsaacLabBisShe
source /opt/ros/jazzy/setup.bash
python WebSocket/robot_service.py
```

4. 启动 FinalProject：

```bash
cd /home/robot/work/FinalProject
source /opt/ros/jazzy/setup.bash
export FINALPROJECT_ROBOT_TYPE=go2
export FINALPROJECT_NAV_BACKEND=ros
python run.py
```

## 调试命令

不经过 FinalProject，可以直接用 ROS2 命令发布控制 topic：

```bash
source /opt/ros/jazzy/setup.bash

ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"model_use\": 1, \"velocity\": [0.6, 0.0, 0.0], \"start\": true}'}"
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"start\": false}'}"
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"model_use\": 3, \"goal\": \"auto\", \"start\": true}'}"
ros2 topic pub --once /go2/goal_pose geometry_msgs/msg/PoseStamped "{pose: {position: {x: 4.5, y: 0.0, z: 0.1}, orientation: {w: 1.0}}}"
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"reset\": 1}'}"
```

也可以直接向 `/go2/skill_command` 发原始文本，桥接程序会复用内部文本解析逻辑：

```bash
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: 'model_use=1; velocity=0.6,0,0; start=1'}"
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: 'model_use=3; goal=auto; start=1'}"
```

## 订阅 topic

- `/go2/skill_command`：`std_msgs/String`
  - JSON 示例：`{"model_use": 1, "velocity": [0.6, 0.0, 0.0], "start": true}`
  - 原始文本示例：`model_use=1; velocity=0.6,0,0; start=1`
- `/go2/cmd_vel`：`geometry_msgs/Twist`
  - 读取 `linear.x / linear.y / linear.z` 写入 velocity 文件
- `/go2/goal_pose`：`geometry_msgs/PoseStamped`
  - 读取 `pose.position.x / y / z` 写入 goal 文件

## 发布 topic

- `/go2/odom`：`nav_msgs/Odometry`
  - 来自 `/tmp/envtest_live_status.json` 的 `robot_pose`
- `/go2/box_pose`：`geometry_msgs/PoseStamped`
  - 来自 `/tmp/envtest_live_status.json` 的 `box.position`
- `/go2/skill_status`：`std_msgs/String`
  - JSON 包含 `timestamp/model_use/skill/scene_id/start/goal/vel_command/envtest_alignment`
- `/go2/scene_objects`：`std_msgs/String`
  - JSON 数组，来自 `platform_1/platform_2/box`

## 快速检查

```bash
cd /home/robot/work/IsaacLabBisShe
python -B -c "from pathlib import Path; [compile(Path(name).read_text(encoding='utf-8'), name, 'exec') for name in ('Ros2/FinalSim.py', 'Ros2/PublishRos2Topic.py')]"
```
