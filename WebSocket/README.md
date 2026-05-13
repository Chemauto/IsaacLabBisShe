# WebSocket 目录

WebSocket 服务端，连接 Planner (FinalProject) 和 ROS2 话题。

## 文件

- `robot_service.py`：WebSocket 服务端，默认 `ws://0.0.0.0:8765`
- `ros2_state.py`：订阅 ROS2 `/go2/*` 状态话题，发布控制话题
- `protocol.py`：技能名、速度等转换工具
- `feedback.py`：动作成功/失败/超时判定

## 数据流

```
Planner → WebSocket → robot_service.py → /go2/skill_command, /rl_cmd_vel, /go2/goal_pose (ROS2)
/go2/odom, /go2/box_pose, /go2/scene_objects, /go2/skill_status (ROS2) → robot_service.py → WebSocket → Planner
```

## 启动

通常由 `run_isaaclab.sh` 或 `run_mujoco.sh` 一起启动。单独启动：

```bash
conda run -n ros2_env python WebSocket/robot_service.py
```

Planner 连接：`ws://127.0.0.1:8765`

## WebSocket 协议

发送命令：

```json
{"type": "command", "skill": "walk", "args": {"direction": "front", "v": 0.5}}
{"type": "command", "skill": "push_box", "args": {"goal": "auto"}}
{"type": "command", "skill": "navigation", "args": {"x": 4.5, "y": 0.0, "z": 0.1}}
```

接收状态：

```json
{"type": "state", "robot": {"x": 0, "y": 0, "z": 0.3, "yaw": 0}, "box_world": {...}, "skill": "walk", ...}
```

接收反馈：

```json
{"type": "feedback", "action_id": "...", "signal": "SUCCESS", "message": "walk completed"}
```
