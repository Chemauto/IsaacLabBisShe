# PushBoxTest README
对应文件：`push_box_env_cfg.py`、`mdp/commands.py`、`mdp/rewards.py`、`mdp/terminations.py`、`mdp/curriculums.py`

## 1. 任务定义
`PushBoxTest` 训练一个中层技能：让 Go2 在平地上把箱子推到目标位姿。

- 机器人：`Unitree Go2`
- 地形：平地
- 箱子尺寸：`0.4 x 0.8 x 0.2 m`
- 箱子质量：`4.0 kg`
- 箱子默认位置：`(1.0, 0.0, 0.1)`

目标不是重新学走路，而是在已有 walk policy 上学习：接近箱子、保持有效接触、把箱子推到目标位置并调整目标 yaw。

## 2. 控制结构
- 高层动作：`vx, vy, wz`
- 低层控制：调用预训练 walk policy
- 低层策略路径：`ModelBackup/TransPolicy/WalkRoughNewTransfer.pt`
- `sim.dt = 0.005`
- 低层 `decimation = 4`
- 高层 `decimation = 40`
- 高层一步 `0.2 s`
- episode 长度 `12 s`
- 每个 episode 共 `60` 个高层 step

## 3. 当前命令与重置
目标命令 `box_goal`：

- `pos_x = (0.5, 3.5)`
- `pos_y = (-1.0, 1.0)`
- `yaw = (-pi/3, pi/3)`

重置：

- 机器人：`x in [-0.1, 0.1]`, `y in [-0.2, 0.2]`, `yaw in [-0.5, 0.5]`
- 箱子：`x in [-0.2, 0.2]`, `y in [-0.25, 0.25]`, `yaw in [-0.3, 0.3]`

## 4. 当前观测
- `base_lin_vel`
- `projected_gravity`
- `box_position`
- `robot_position`
- `goal_command`
- `last_action`

## 5. 当前奖励
失败惩罚：

- `termination_penalty = -200.0`
- 只对 `base_contact` 和 `box_out_of_bounds` 生效

位置奖励：

- `box_goal_distance`: `weight=3.0`, `std=0.6`
- `box_goal_distance_fine_gained`: `weight=3.0`, `std=0.15`

```text
R_pos = 3.0 * (1 - tanh(error_pos / 0.6))
      + 3.0 * (1 - tanh(error_pos / 0.15))
```

yaw 奖励：

- `box_goal_yaw`: `weight=1.5`, `std=0.4`
- `box_goal_yaw_fine_gained`: `weight=1.5`, `std=0.15`

```text
R_yaw = 1.5 * (1 - tanh(error_yaw / 0.4))
      + 1.5 * (1 - tanh(error_yaw / 0.15))
```

稀疏成功奖励：

- `box_goal_success = 20.0`
- `distance_threshold = 0.08 m`
- `yaw_threshold = 0.18 rad`
- `box_speed_threshold = 0.08 m/s`
- `robot_speed_threshold = 0.08 m/s`

姿态与动作平滑：

- `flat_orientation = -2.0`
- `robot_goal_yaw = -0.3`，仅在 `error_pos < 0.25 m` 时启用，用于约束机器人最终朝向与目标 yaw 对齐
- `action_rate = -0.20`

## 6. 成功终止条件
`goal_reached` 与成功奖励保持一致：

- `distance_threshold = 0.08 m`
- `yaw_threshold = 0.18 rad`
- `box_speed_threshold = 0.08 m/s`
- `robot_speed_threshold = 0.08 m/s`
- `settle_steps = 6`

其他终止项：`time_out`、`base_contact`、`box_out_of_bounds`

## 7. 奖励数值解释
### 7.1 位置误差对应的总位置奖励
| 位置误差(m) | coarse std=0.6 | fine std=0.15 | 合计 |
|---|---:|---:|---:|
| 0.00 | 3.000 | 3.000 | 6.000 |
| 0.05 | 2.751 | 2.036 | 4.786 |
| 0.10 | 2.505 | 1.252 | 3.756 |
| 0.15 | 2.265 | 0.715 | 2.981 |
| 0.20 | 2.036 | 0.390 | 2.425 |
| 0.30 | 1.614 | 0.108 | 1.722 |
| 0.40 | 1.252 | 0.029 | 1.281 |
| 0.60 | 0.715 | 0.002 | 0.717 |
| 1.00 | 0.207 | 0.000 | 0.207 |

说明：

- `fine` 项主要在 `0.0 ~ 0.2 m` 内起作用
- `coarse` 项负责更大范围的引导

### 7.2 角度误差对应的总 yaw 奖励
| 角度误差(rad) | 角度误差(deg) | coarse std=0.4 | fine std=0.15 | 合计 |
|---|---:|---:|---:|---:|
| 0.00 | 0.0 | 1.500 | 1.500 | 3.000 |
| 0.05 | 2.9 | 1.314 | 1.018 | 2.331 |
| 0.10 | 5.7 | 1.133 | 0.626 | 1.758 |
| 0.15 | 8.6 | 0.963 | 0.358 | 1.320 |
| 0.18 | 10.3 | 0.867 | 0.250 | 1.117 |
| 0.20 | 11.5 | 0.807 | 0.195 | 1.002 |
| 0.30 | 17.2 | 0.547 | 0.054 | 0.601 |
| 0.40 | 22.9 | 0.358 | 0.014 | 0.372 |
| 0.60 | 34.4 | 0.142 | 0.001 | 0.143 |
| 0.785 | 45.0 | 0.058 | 0.000 | 0.058 |

说明：

- `fine` yaw 项主要在 `0.0 ~ 0.2 rad` 内起作用
- 成功阈值 `0.18 rad` 约等于 `10.3 deg`
- 这意味着策略需要把箱子 yaw 收到大约 `10 度` 以内

## 8. 课程学习
`goal_range` 使用基于 episode 改善比例的 curriculum：

- `progress = clamp(1 - final_distance / initial_distance, 0, 1)`
- `A <- (1 - beta) * A + beta * mean(progress)`
- `progress_beta = 0.02`

当前逻辑是按“推近多少”逐步放大目标采样范围，不是按成功率放大。
