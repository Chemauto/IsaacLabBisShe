# 基于 LLM 思考的四足机器人导航项目

本项目面向 Unitree Go2，目标是实现“环境理解 + 技能选择 + 技能执行”的结构化导航。
当前实现不是让 LLM 直接输出底层关节动作，而是让它在多个已训练技能之间做切换，并通过统一场景和统一控制接口完成执行。

## 当前能力

当前已经接入 4 类技能：

- `walk`
- `climb`
- `push_box`
- `navigation`

当前运行主线是：

1. 使用 `EnvTest` 生成固定结构化场景
2. 使用 `NewTools/envtest_model_use_player.py` 在同一场景里切换技能
3. 使用 `Socket` 提供外部 UDP 控制接口
4. 通过 `model_use / velocity / goal / start / reset` 控制执行

## 关键目录

- `source/MyProject/MyProject/tasks/manager_based/WalkTest`
  行走与攀爬相关环境
- `source/MyProject/MyProject/tasks/manager_based/PushBoxTest`
  推箱子环境
- `source/MyProject/MyProject/tasks/manager_based/NaviationTest`
  导航训练环境
- `source/MyProject/MyProject/tasks/manager_based/EnvTest`
  统一结构化测试场景
- `NewTools`
  多技能运行脚本和桥接逻辑
- `Socket`
  UDP server/client 控制接口
- `ModelBackup`
  运行时加载的策略模型

## 关键入口

- `NewTools/envtest_model_use_player.py`
  统一启动脚本，负责：
  - 启动 `EnvTest`
  - 读取控制文件
  - 根据 `model_use` 切换技能
  - 在终端显示实时状态面板
- `Socket/envtest_socket_server.py`
  接收 UDP 文本命令并写入控制文件
- `Socket/envtest_socket_client.py`
  从命令行发送控制命令

## 安装

```bash
cd /home/robot/work/IsaacLabBisShe
python -m pip install -e source/MyProject
```

## 最短启动流程

先启动 player：

```bash
cd /home/robot/work/IsaacLabBisShe
python NewTools/envtest_model_use_player.py --scene_id 4
```

再启动 UDP server：

```bash
python Socket/envtest_socket_server.py
```

然后发送命令：

```bash
python Socket/envtest_socket_client.py --model_use 1 --velocity 0.6 0.0 0.0
python Socket/envtest_socket_client.py --start 1
python Socket/envtest_socket_client.py --start 0
python Socket/envtest_socket_client.py --model_use 3 --goal_auto --start 1
python Socket/envtest_socket_client.py --model_use 4 --goal 4.5 0.0 0.1 --start 1
python Socket/envtest_socket_client.py --reset 1
python Socket/envtest_socket_client.py --reset 2
```

## EnvTest 概览

`EnvTest` 是当前多技能统一运行的核心场景环境。
它本身不包含奖励和训练逻辑，重点是稳定复现固定结构化场景并提供统一观测接口。

`scene_id=0~4` 对应：

- `0`：左右都无障碍
- `1`：左低台阶，右侧空
- `2`：左右都是低台阶
- `3`：左低台阶，右高台阶
- `4`：左右都是高台阶，中间有可推动箱子

场景中关键物体尺寸：

- 低台：`(2.0, 1.5, 0.3)`
- 高台：`(2.0, 1.5, 0.5)`
- 箱子：`(0.6, 0.8, 0.2)`

## 统一观测

`EnvTest` 的 `policy` 观测已经统一成四类技能的并集，总维度是 `256`。

主要组成：

- walk / climb 低层公共观测：`235`
- navigation 额外 `pose_command`：`4`
- push_box 高层额外观测：`17`

运行时会按 `model_use` 自动切片：

- `walk`：`235`
- `climb`：`235`
- `push_box`：高层 `23` + 低层 `235`
- `navigation`：高层 `197` + 低层 `235`

其中 navigation 会把外部世界系 `goal(x, y, z, yaw)` 转成策略需要的 base-frame `pose_command(dx, dy, dz, dyaw)`。

## model_use 对应关系

- `0`：idle
- `1`：walk
- `2`：climb
- `3`：push_box
- `4`：navigation

## player 状态面板

`envtest_model_use_player.py` 启动后会在终端持续覆盖刷新状态面板，显示：

- `model_use`
- `skill`
- `scene_id`
- `start`
- `unified_obs_dim`
- `policy_obs_dim`
- `pose_command`
- `vel_command`
- `robot_pose`
- `goal`
- `platform_1`
- `platform_2`
- `box`

如果对象或命令当前不存在，会显示 `None`。

## reset 语义

当前 `reset` 支持两种一次性模式：

- `reset=1`
  重置整个环境，相当于 `env.reset()`
- `reset=2`
  只重置机器人，不重置箱子、障碍物和场景状态

## 详细文档

更具体的使用说明见：

- `source/MyProject/MyProject/tasks/manager_based/EnvTest/README.md`
- `Socket/README.md`

如果你要继续改 `EnvTest` 行为，优先看这些文件：

- `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env_cfg.py`
- `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env.py`
- `source/MyProject/MyProject/tasks/manager_based/EnvTest/observation_schema.py`
- `NewTools/envtest_model_use_player.py`
- `NewTools/envtest_navigation_bridge.py`
