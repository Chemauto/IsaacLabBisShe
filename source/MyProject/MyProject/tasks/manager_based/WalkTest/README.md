# WalkTest

WalkTest 定义 Go2 行走和攀爬相关 IsaacLab 任务。其中毕业设计攀爬/坑洞任务用于 LLMPlanner 的 `climb` 技能。

## 核心任务 ID

| 用途 | 任务 ID |
|---|---|
| 训练 | `Template-Velocity-Go2-Walk-BiShe-Pit-v0` |
| Play/技能回放 | `Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0` |

## 主要文件

- `walk_bishe_env_cfg.py`：毕业设计坑洞/攀爬任务配置。
- `walk_rough_env_cfg.py`：粗糙地形行走配置。
- `walk_flat_env_cfg.py`：平地行走配置。
- `walk_climb_env_cfg.py`：攀爬补充配置。
- `agents/rsl_rl_ppo_cfg.py`：RSL-RL 训练配置。

## 训练与回放

```bash
python scripts/rsl_rl/train.py --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 --headless
python scripts/rsl_rl/play.py --task Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0
```

## 与 LLMPlanner 的关系

LLMPlanner 调用 `climb(height)` 时，WebSocket 服务端会切换到底层攀爬策略。该策略来自本任务训练得到的模型，用于跨越坑洞、台阶或辅助箱体形成的高度差。

## 许可证

本目录包含 Isaac Lab 派生配置，保留源码文件中的 `SPDX-License-Identifier: BSD-3-Clause` 声明。
