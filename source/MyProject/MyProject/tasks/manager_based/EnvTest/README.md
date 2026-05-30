# EnvTest

EnvTest 是用于多技能联调和 LLMPlanner 评测的固定场景环境。它不以训练奖励为核心，而是稳定生成走廊、障碍物和可推动箱体，并为 `walk / climb / push_box / navigation` 提供统一观测。

## 场景

| scene_id | 场景说明 |
|---:|---|
| 0 | 无障碍，直接导航 |
| 1 | 单侧低台阶 |
| 2 | 双侧低台阶 |
| 3 | 一侧低台阶，一侧高台阶 |
| 4 | 双侧高台阶，中间有可推动箱体 |

## 技能与任务

- climb：`Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0`
- navigation：`Template-Naviation-BiShe-Go2-Play-v0`
- push_box：`Template-Push-Box-Go2-Play-v0`

## 统一观测

`EnvTest` 的 `policy` 观测是多技能观测并集，总维度约 `252`，包含基座速度、重力投影、关节状态、高度扫描、导航目标、箱体相对位姿、推箱目标和历史动作。

## 运行

```bash
python Isaaclab/FinalSim.py --scene_id 4 --enable_front_camera
```

或通过根目录脚本启动完整桥接：

```bash
bash run_isaaclab.sh
```

## 控制文件

- `/tmp/model_use.txt`
- `/tmp/envtest_velocity_command.txt`
- `/tmp/envtest_goal_command.txt`
- `/tmp/envtest_start.txt`
- `/tmp/envtest_reset.txt`
- `/tmp/envtest_live_status.json`

推荐通过 WebSocket/ROS2 桥接控制，而不是手动改写控制文件。
