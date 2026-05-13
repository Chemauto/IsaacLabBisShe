# IsaacLabBisShe

Unitree Go2 多技能训练与统一仿真运行框架，支持 IsaacLab、MuJoCo 和实物。

## 架构

```
Planner (FinalProject) ──WebSocket──> robot_service.py ──ROS2──> go2_main_loop (legged_ws)
                                        ↑ /go2/* topics
                    ┌───────────────────┬┴──────────────────┐
              IsaacLab (ros2_bridge)  MuJoCo (mujoco_ros2_bridge)
                    ↑                    ↑
            FinalSim.py JSON      mujoco_dds_state.py JSON
```

robot_service.py 通过 ROS2 `/go2/*` 话题统一接收数据，不关心后端是哪个。

## 快速启动

**IsaacLab**（需要 `env_isaaclab` + `ros2_env` conda 环境）：

```bash
bash run_isaaclab.sh
```

**MuJoCo**（需要 `ros2_env` conda 环境）：

```bash
bash run_mujoco.sh
```

Ctrl+C 停止所有进程。

## 目录结构

| 目录 | 说明 |
|------|------|
| `Isaaclab/` | IsaacLab 仿真入口 (`FinalSim.py`) 和 ROS2 桥接 (`ros2_bridge.py`) |
| `Mujoco/` | MuJoCo 仿真、DDS 状态读取、ROS2 桥接 |
| `WebSocket/` | WebSocket 服务端 (`robot_service.py`)，连接 Planner |
| `deploy/` | 策略导出与 C++ 部署控制器 |
| `ModelBackup/` | 运行时加载的策略模型 |
| `source/` | IsaacLab 任务定义（WalkTest、PushBoxTest、EnvTest 等） |

## ROS2 话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/go2/odom` | Odometry | 机器人位姿和速度 |
| `/go2/box_pose` | PoseStamped | 箱子世界坐标 |
| `/go2/scene_objects` | String (JSON) | 场景物体列表 |
| `/go2/skill_status` | String (JSON) | 当前技能状态 |
| `/go2/skill_command` | String (JSON) | 技能指令 |
| `/go2/goal_pose` | PoseStamped | 导航目标点 |
| `/rl_cmd_vel` | Twist | 速度指令 |

## model_use 对应

| ID | 技能 |
|----|------|
| 0 | idle |
| 1 | walk |
| 2 | climb |
| 3 | push_box |
| 4 | navigation |
| 5 | nav_climb |
