# 基于 LLM 思考的四足机器人导航项目

## 项目简介

本项目面向 Unitree Go2 四足机器人，目标是实现“语音指令 + 环境理解 + 技能选择 + 技能执行”的自主导航。

系统的核心思路不是让 LLM 直接输出每一个底层动作，而是让 LLM 根据当前环境、障碍物高度、可利用物体和目标点信息，在多个已经训练好的技能之间做选择，并输出可解释的决策结果。

当前项目聚焦 3 个核心技能：

- `walk`
- `climb`
- `push_box`

其中，攀爬技能当前按最大可通过高度 `0.3 m` 设计。

## 当前整体架构

项目现在分为 4 个层次：

1. 低层技能策略：行走、攀爬、推箱子
2. 结构化测试场景：`EnvTest`
3. 技能切换执行器：`NewTools/envtest_model_use_player.py`
4. 外部控制接口：`Socket`

当前已经实现的运行逻辑是：

- `scene_id` 决定当前仿真场景
- `model_use` 决定当前调用哪个技能
- 外部可以通过 UDP Socket 发送 `model_use / velocity / goal / start`
- `EnvTest` 的统一观测是三个技能观测的并集
- `envtest_model_use_player.py` 会根据 `model_use` 从统一观测里切出各自需要的输入

## 目录结构

项目中当前最重要的目录如下：

- `source/MyProject/MyProject/tasks/manager_based/WalkTest`
  行走与攀爬技能环境
- `source/MyProject/MyProject/tasks/manager_based/PushBoxTest`
  推箱子技能环境
- `source/MyProject/MyProject/tasks/manager_based/EnvTest`
  结构化导航测试环境
- `NewTools`
  运行时多技能切换脚本
- `Socket`
  Socket 控制脚本
- `ModelBackup`
  训练得到或导出的策略模型

## 关键文件

### 1. 行走技能

- `source/MyProject/MyProject/tasks/manager_based/WalkTest/walk_rough_env_cfg.py`

注册任务：

- `Template-Velocity-Go2-Walk-Rough-v0`
- `Template-Velocity-Go2-Walk-Rough-Play-v0`

### 2. 攀爬技能

- `source/MyProject/MyProject/tasks/manager_based/WalkTest/walk_bishe_env_cfg.py`

注册任务：

- `Template-Velocity-Go2-Walk-BiShe-Pit-v0`
- `Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0`

### 3. 推箱子技能

- `source/MyProject/MyProject/tasks/manager_based/PushBoxTest/push_box_env_cfg.py`

注册任务：

- `Template-Push-Box-Go2-v0`
- `Template-Push-Box-Go2-Play-v0`

### 4. 结构化测试环境

- `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env_cfg.py`
- `source/MyProject/MyProject/tasks/manager_based/EnvTest/README.md`

注册任务：

- `Template-EnvTest-Go2-v0`
- `Template-EnvTest-Go2-Play-v0`

### 5. 多技能运行脚本

- `NewTools/envtest_model_use_player.py`

这个脚本负责：

- 启动 `EnvTest`
- 读取 `model_use`
- 读取速度指令和位置指令
- 根据技能类型切换不同策略

### 6. Socket 控制

- `Socket/envtest_socket_server.py`
- `Socket/envtest_socket_client.py`
- `Socket/README.md`

## 环境安装

进入项目根目录后，先把扩展安装为 editable package：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python -m pip install -e source/MyProject
```

## VSCode 配置

建议在 `.vscode/settings.json` 中加入 Isaac Lab 的路径引用。注意这里的绝对路径需要根据你自己的机器环境修改：

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/MyProject",
        "/home/xcj/work/IsaacLab/IsaacLab-main/source/isaaclab",
        "/home/xcj/work/IsaacLab/IsaacLab-main/source/isaaclab_assets",
        "/home/xcj/work/IsaacLab/IsaacLab-main/source/isaaclab_mimic",
        "/home/xcj/work/IsaacLab/IsaacLab-main/source/isaaclab_rl",
        "/home/xcj/work/IsaacLab/IsaacLab-main/source/isaaclab_tasks"
    ],
    "python-envs.defaultEnvManager": "ms-python.python:conda",
    "python-envs.defaultPackageManager": "ms-python.python:conda",
    "python-envs.pythonProjects": []
}
```

## 技能训练命令

推荐直接使用 `scripts/rsl_rl/train.py`。

### 1. 训练行走技能

```bash
python scripts/rsl_rl/train.py --task Template-Velocity-Go2-Walk-Rough-v0 --headless
```

### 2. 训练攀爬技能

```bash
python scripts/rsl_rl/train.py --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 --headless
```

### 3. 训练推箱子技能

```bash
python scripts/rsl_rl/train.py --task Template-Push-Box-Go2-v0 --headless
```

## 模型导出与使用

训练得到的原始 `.pt` 通常是 checkpoint，不一定能直接用于部署脚本。

运行时推荐使用导出的 TorchScript 策略，例如：

- `ModelBackup/TransPolicy/WalkRoughNewTransfer.pt`
- `ModelBackup/BiShePolicy/exported/policy.pt`
- `ModelBackup/PushPolicy/exported/policy.pt`

当前 `NewTools/envtest_model_use_player.py` 默认使用的就是这类可直接推理的模型。

## EnvTest 场景说明

`EnvTest` 目前提供 5 个固定结构化场景，`scene_id=0~4`：

- `0`：左右都无障碍
- `1`：左低台阶，右侧空
- `2`：左右都是低台阶
- `3`：左低台阶，右高台阶
- `4`：左右都是高台阶，中间有可推动箱子

打开单场景：

```bash
python scripts/zero_agent.py --task Template-EnvTest-Go2-Play-v0 --scene_id 4
```

## EnvTest 多技能运行

运行脚本：

```bash
python NewTools/envtest_model_use_player.py --scene_id 4
```

当前脚本采用两阶段流程：

1. 启动后先待机，机器人保持静止
2. 通过 `model_use / velocity / goal / start` 控制何时开始执行

其中：

- `model_use=0`：idle
- `model_use=1`：walk
- `model_use=2`：climb
- `model_use=3`：push_box

## 统一观测

`EnvTest` 的 `policy` 观测目前已经扩展成三个技能的并集，总维度为 `251`：

- 低层 walk / climb 公共观测：`235`
- push_box 高层额外观测：`16`

运行时脚本会根据 `model_use` 自动切片：

- `walk`：取 `235` 维
- `climb`：取 `235` 维
- `push_box`：先取 `22` 维高层输入，再调用 `235` 维低层策略

## Socket 控制流程

先启动 `EnvTest player`：

```bash
python NewTools/envtest_model_use_player.py --scene_id 4
```

再启动 UDP server：

```bash
python Socket/envtest_socket_server.py
```

然后通过 client 发控制命令：

```bash
python Socket/envtest_socket_client.py --model_use 1 --velocity 0.6 0.0 0.0
python Socket/envtest_socket_client.py --start 1
python Socket/envtest_socket_client.py --start 0
python Socket/envtest_socket_client.py --model_use 3 --goal_auto --start 1
```

如果不手动发 `goal`，`push_box` 默认会自动使用程序计算出的高台前目标点，把箱子推到高台旁边，供后续爬箱子再爬高台使用。

## LLM 决策目标

最终希望实现的能力是：

- 用户通过语音给出目标指令
- 系统感知环境
- LLM 判断当前场景属于哪一种情况
- LLM 输出技能选择与理由
- 系统执行对应技能

建议 LLM 重点关注的信息包括：

- 左侧障碍高度
- 右侧障碍高度
- 是否存在可推动箱子
- 箱子高度
- 目标点位置
- 当前应选择哪种技能链

## 当前项目状态

当前已经完成：

- 行走、攀爬、推箱子三个技能环境
- `EnvTest` 结构化场景
- 统一观测并集
- 按 `model_use` 切换技能
- Socket 控制 `model_use / velocity / goal / start`

当前更适合继续推进的方向是：

- 把视觉/深度/场景信息整理成 LLM 输入
- 让 LLM 输出 `model_use` 和解释文本
- 增强 `push_box` 与 `climb` 之间的串联策略
- 做整套“语音输入 -> LLM 决策 -> 技能执行”的端到端联调
