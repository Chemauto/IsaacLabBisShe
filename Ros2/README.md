# Ros2 目录说明

这个目录提供 EnvTest 的 ROS2 控制入口，包含两件事：

- 订阅 FinalProject 的 `/go2/*` 控制 topic，并写入 EnvTest player 使用的 `/tmp` 控制文件
- 读取 `/tmp/envtest_live_status.json`，发布 FinalProject Go2 后端需要的 `/go2/odom`、`/go2/skill_status`、`/go2/scene_objects`

## 文件

- `envtest_ros2_server.py`：订阅 `/go2/*` 控制 topic、写控制文件，并发布 EnvTest 状态 topic
- `envtest_ros2_client.py`：调试用的一次性 ROS2 发送工具
- `test_envtest_ros2_server.py`：核心协议转换测试

## 和 Socket 版的关系

Socket 版链路：

```text
FinalProject -> deploy/go2_skill_bridge.py -> UDP 5566 -> Socket/envtest_socket_server.py -> /tmp 控制文件 -> EnvTest player
```

ROS2 版链路：

```text
FinalProject -> /go2/skill_command, /go2/cmd_vel, /go2/goal_pose
             -> Ros2/envtest_ros2_server.py
             -> /tmp 控制文件
             -> EnvTest player
             -> /tmp/envtest_live_status.json
             -> Ros2/envtest_ros2_server.py
             -> /go2/odom, /go2/skill_status, /go2/scene_objects
             -> FinalProject
```

EnvTest player 不需要改，仍然读取这些文件：

- `model_use`: `/tmp/model_use.txt`
- `velocity`: `/tmp/envtest_velocity_command.txt`
- `goal`: `/tmp/envtest_goal_command.txt`
- `start`: `/tmp/envtest_start.txt`
- `reset`: `/tmp/envtest_reset.txt`

## 启动流程

1. 启动 EnvTest player：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python NewTools/envtest_model_use_player.py --scene_id 3
```

2. 启动 ROS2 控制 server：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
source /opt/ros/jazzy/setup.bash
python Ros2/envtest_ros2_server.py
```

3. 启动 FinalProject：

```bash
cd /home/xcj/work/FinalProject
source /opt/ros/jazzy/setup.bash
export FINALPROJECT_ROBOT_TYPE=go2
export FINALPROJECT_NAV_BACKEND=ros
python run.py
```

这条链路不需要再启动：

```bash
python Socket/envtest_socket_server.py
python /home/xcj/work/FinalProject/deploy/go2_skill_bridge.py
```

## 调试命令

不经过 FinalProject，直接发 ROS2 控制命令：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
source /opt/ros/jazzy/setup.bash

python Ros2/envtest_ros2_client.py --model_use 1 --velocity 0.6 0.0 0.0 --start 1
python Ros2/envtest_ros2_client.py --start 0
python Ros2/envtest_ros2_client.py --model_use 3 --goal_auto --start 1
python Ros2/envtest_ros2_client.py --model_use 4 --goal 4.5 0.0 0.1 --start 1
python Ros2/envtest_ros2_client.py --reset 1
```

也可以直接发原始文本，server 会复用 Socket server 的解析逻辑：

```bash
python Ros2/envtest_ros2_client.py --text "model_use=1; velocity=0.6,0,0; start=1"
python Ros2/envtest_ros2_client.py --text "model_use=3; goal=auto; start=1"
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
- `/go2/skill_status`：`std_msgs/String`
  - JSON 包含 `timestamp/model_use/skill/scene_id/start/goal/vel_command/envtest_alignment`
- `/go2/scene_objects`：`std_msgs/String`
  - JSON 数组，来自 `platform_1/platform_2/box`

## 测试

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/Ros2
python -B -m unittest test_envtest_ros2_server
```
