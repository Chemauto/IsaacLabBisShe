# WebSocket 目录说明

这个目录提供 FinalProject/LLM 使用的 WebSocket 服务端。服务端只和 ROS2 topic 通信，不直接读写 EnvTest 的 `/tmp` 控制文件。

## 文件

- `robot_service.py`：WebSocket 服务端入口，默认监听 `ws://0.0.0.0:8765`
- `ros2_state.py`：订阅 ROS2 状态 topic，缓存机器人、箱子和技能状态，并发布 ROS2 控制 topic
- `protocol.py`：技能名、速度和状态字段的轻量转换
- `feedback.py`：根据 ROS2 状态判断动作成功、失败或超时

## 运行

先确保 ROS2 环境里已经能看到 `/go2/*` topic：

```bash
ros2 topic list
```

启动服务端：

```bash
cd /home/robot/work/IsaacLabBisShe
source /opt/ros/jazzy/setup.bash
python WebSocket/robot_service.py
```

FinalProject 保持使用：

```env
ROBOT_WS_URL=ws://127.0.0.1:8765
```

## 链路

```text
FinalProject
-> ws://127.0.0.1:8765
-> WebSocket/robot_service.py
-> /go2/skill_command, /go2/cmd_vel, /go2/goal_pose
-> 真实机器人或 Ros2/PublishRos2Topic.py

真实机器人或 Ros2/PublishRos2Topic.py
-> /go2/odom, /go2/box_pose, /go2/scene_objects, /go2/skill_status
-> WebSocket/robot_service.py
-> FinalProject
```
