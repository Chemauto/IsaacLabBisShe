# WebSocket 技能服务

该目录提供机器人技能服务端，负责连接 LLMPlanner 与 ROS2/IsaacLab/MuJoCo 后端。

## 文件说明

- `robot_service.py`：WebSocket 服务端，默认监听 `ws://0.0.0.0:8765`。
- `ros2_state.py`：订阅 `/go2/*` 状态话题，并发布控制话题。
- `protocol.py`：技能名称、参数和控制量转换。
- `feedback.py`：技能成功、失败和超时判定。

## 数据流

```text
LLMPlanner -> WebSocket -> robot_service.py -> /go2/skill_command, /go2/goal_pose, /rl_cmd_vel
/go2/odom, /go2/box_pose, /go2/scene_objects, /go2/skill_status -> robot_service.py -> LLMPlanner
```

## 启动

通常由根目录脚本一起启动：

```bash
bash run_isaaclab.sh
# 或
bash run_mujoco.sh
```

单独启动：

```bash
conda run -n ros2_env python WebSocket/robot_service.py
```

## 命令格式

```json
{"type": "command", "action_id": "nav-xxxx", "skill": "nav", "args": {"x": 4.0, "y": 0.0, "z": 0.3}}
```

服务端返回实时状态和最终反馈：

```json
{"type": "feedback", "action_id": "nav-xxxx", "skill": "nav", "signal": "SUCCESS", "message": "arrived"}
```

## 技能映射

LLMPlanner 使用的 `skill` 会在服务端映射到底层 `model_use`：`walk=1`，`climb=2`，`push_box=3`，`navigation=4`。
