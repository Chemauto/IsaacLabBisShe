# llm-legged-lab

llm-legged-lab 是 LLMPlanner 使用的腿足机器人仿真、策略模型与技能服务仓库。仓库提供 IsaacLab/MuJoCo 仿真入口、训练好的 Go2 策略模型、ROS2 桥接、WebSocket 技能服务，以及导航、攀爬、推箱等底层技能执行能力。

LLMPlanner 负责高层自然语言任务规划；本仓库负责底层仿真、策略推理、状态发布和技能反馈。两者通过 WebSocket 协议连接。

## 演示视频

### IsaacLab 仿真

https://github.com/user-attachments/assets/387c4f83-7d37-41c6-b78f-7a54f4fe6f19

### Sim2Sim

https://github.com/user-attachments/assets/458fbd56-1e15-4b9e-84bd-57142dfc4fa4

## 任务 ID

| 技能 | IsaacLab Play 任务 ID | 说明 |
|---|---|---|
| climb | `Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0` | 坑洞/台阶地形上的攀爬与越障策略 |
| navigation | `Template-Naviation-BiShe-Go2-Play-v0` | 毕设固定场景导航策略 |
| push_box | `Template-Push-Box-Go2-Play-v0` | 推箱到目标位姿策略 |

训练任务对应去掉 `-Play` 后缀，例如 `Template-Push-Box-Go2-v0`。

## 架构

```text
LLMPlanner --WebSocket--> WebSocket/robot_service.py --ROS2--> /go2/* topics
                                      ^
                                      |
                    IsaacLab/ros2_bridge.py 或 MuJoCo/mujoco_ros2_bridge.py
                                      ^
                                      |
                         IsaacLab FinalSim.py / MuJoCo simulator
```

`robot_service.py` 统一接收 LLMPlanner 技能命令，并通过 ROS2 话题驱动 IsaacLab 或 MuJoCo 后端。

## 快速启动

IsaacLab 后端：

```bash
bash run_isaaclab.sh
```

MuJoCo 后端：

```bash
bash run_mujoco.sh
```

默认 WebSocket 地址为 `ws://127.0.0.1:8765`。使用 `Ctrl+C` 停止所有进程。

## 目录结构

| 目录 | 说明 |
|---|---|
| `Isaaclab/` | IsaacLab 仿真入口与 ROS2 桥接 |
| `Mujoco/` | MuJoCo 仿真、DDS 状态读取与 ROS2 桥接 |
| `WebSocket/` | WebSocket 技能服务端，连接 LLMPlanner |
| `deploy/` | 策略导出、ONNX/TorchScript 部署与 C++ 控制器 |
| `ModelBackup/` | 训练好的策略模型和部署模型 |
| `source/` | IsaacLab 任务定义与环境配置 |

## ROS2 话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/go2/odom` | Odometry | 机器人位姿和速度 |
| `/go2/box_pose` | PoseStamped | 箱子世界坐标 |
| `/go2/scene_objects` | String(JSON) | 场景物体列表 |
| `/go2/skill_status` | String(JSON) | 当前技能状态 |
| `/go2/skill_command` | String(JSON) | 技能指令 |
| `/go2/goal_pose` | PoseStamped | 导航目标点 |
| `/rl_cmd_vel` | Twist | 速度指令 |

## model_use 映射

| ID | 技能 |
|---:|---|
| 0 | idle |
| 1 | walk |
| 2 | climb |
| 3 | push_box |
| 4 | navigation |
| 5 | nav_climb |

## 协议说明

本仓库包含 Isaac Lab 派生代码，相关文件保留 `SPDX-License-Identifier: BSD-3-Clause`。第三方依赖和预编译库保留其原始协议。公开发布时请同时保留源码文件头、第三方目录中的许可证文件以及本仓库根目录的 `LICENSE` 文件。
