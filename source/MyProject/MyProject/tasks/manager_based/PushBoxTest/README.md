# PushBoxTest

PushBoxTest 定义 Go2 推箱任务。策略目标是让机器人接近箱体、保持有效接触，并将箱体推到目标位置和目标 yaw。该任务用于 LLMPlanner 的 `push` 技能。

## 核心任务 ID

| 用途 | 任务 ID |
|---|---|
| 训练 | `Template-Push-Box-Go2-v0` |
| Play/技能回放 | `Template-Push-Box-Go2-Play-v0` |

## 任务设置

- 机器人：Unitree Go2。
- 地形：平地。
- 箱体默认尺寸：约 `0.4 x 0.8 x 0.2 m`。
- 高层动作：`vx, vy, wz`。
- 低层控制：复用预训练 walk policy。

## 训练与回放

```bash
python scripts/rsl_rl/train.py --task Template-Push-Box-Go2-v0 --headless
python scripts/rsl_rl/play.py --task Template-Push-Box-Go2-Play-v0
```

## 成功判定

推箱成功通常同时考虑箱体位置误差、yaw 误差、箱体速度、机器人速度和稳定步数。WebSocket 运行时会进一步根据 `feedback.py` 中的阈值返回最终技能反馈。

## 与 LLMPlanner 的关系

LLMPlanner 调用 `push(x, y, yaw)` 时，WebSocket 服务端将目标位姿发送到底层推箱策略。推箱成功后，上层规划可继续执行导航或攀爬队列。

## 许可证

本目录包含 Isaac Lab 派生配置，保留源码文件中的 `SPDX-License-Identifier: BSD-3-Clause` 声明。
