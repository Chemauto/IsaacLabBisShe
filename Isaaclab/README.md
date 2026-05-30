# IsaacLab 后端

该目录包含 IsaacLab 仿真入口和 ROS2 桥接代码，用于为 LLMPlanner 提供统一技能执行环境。

## 核心文件

- `FinalSim.py`：启动 IsaacLab 仿真，读取 `/tmp` 控制文件，写出 `/tmp/envtest_live_status.json`。
- `ros2_bridge.py`：将 FinalSim 状态发布到 `/go2/*` ROS2 话题，并把 ROS2 控制命令写回 `/tmp` 控制文件。
- `PublishRos2Topic.py`：旧版桥接脚本，保留备用。

## 数据流

```text
FinalSim.py -> /tmp/envtest_live_status.json -> ros2_bridge.py -> /go2/* -> robot_service.py
robot_service.py -> /go2/skill_command 或 /go2/goal_pose -> ros2_bridge.py -> /tmp 控制文件 -> FinalSim.py
```

## 推荐启动

```bash
bash run_isaaclab.sh
```

## 单独启动

```bash
conda run -n env_isaaclab python Isaaclab/FinalSim.py --scene_id 4 --enable_front_camera
conda run -n ros2_env python Isaaclab/ros2_bridge.py
conda run -n ros2_env python WebSocket/robot_service.py
```

## 任务 ID

- climb：`Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0`
- navigation：`Template-Naviation-BiShe-Go2-Play-v0`
- push_box：`Template-Push-Box-Go2-Play-v0`

## model_use

| ID | 技能 |
|---:|---|
| 0 | idle |
| 1 | walk |
| 2 | climb |
| 3 | push_box |
| 4 | navigation |
| 5 | nav_climb |
