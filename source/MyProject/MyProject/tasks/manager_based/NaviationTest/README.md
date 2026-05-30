# NaviationTest

NaviationTest 定义 Go2 分层导航任务。高层导航策略输出速度命令，低层行走策略负责关节控制。毕设固定场景导航任务用于 LLMPlanner 的 `navigation` 技能。

## 核心任务 ID

| 用途 | 任务 ID |
|---|---|
| 训练 | `Template-Naviation-BiShe-Go2-v0` |
| Play/技能回放 | `Template-Naviation-BiShe-Go2-Play-v0` |

## 其他导航任务

- `Template-Naviation-Rough-Go2-v0`
- `Template-Naviation-Rough-Go2-Play-v0`
- `Template-Naviation-Flat-Go2-v0`
- `Template-Naviation-Flat-Go2-Play-v0`
- `Template-Naviation-Climb-Go2-v0`
- `Template-Naviation-Climb-Go2-Play-v0`

## 主要文件

- `naviation_bishe_env_cfg.py`：毕业设计固定场景导航配置。
- `naviation_rough_env_cfg.py`：粗糙地形导航配置。
- `naviation_flat_env_cfg.py`：平地导航配置。
- `naviation_climb_env_cfg.py`：攀爬导航配置。
- `mdp/terminations.py`：目标到达、超时和失败终止条件。

## 训练与回放

```bash
python scripts/rsl_rl/train.py --task Template-Naviation-BiShe-Go2-v0 --headless
python scripts/rsl_rl/play.py --task Template-Naviation-BiShe-Go2-Play-v0
```

## 与 LLMPlanner 的关系

LLMPlanner 调用 `nav(x, y, z)` 时，WebSocket 服务端把世界系目标点转换为导航策略使用的相对位姿命令，并切换到底层 navigation 策略执行。

## 许可证

本目录包含 Isaac Lab 派生配置，保留源码文件中的 `SPDX-License-Identifier: BSD-3-Clause` 声明。
