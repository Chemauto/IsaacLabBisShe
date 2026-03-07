# BiSheTest - Hierarchical Navigation Task

**Unitree Go2 四足机器人分层强化学习导航任务**

基于 Isaac Lab 实现的分层强化学习导航系统，使用预训练的行走策略作为低层控制器，训练高层导航策略。

## 项目概述

本任务实现了分层强化学习（Hierarchical RL）架构：

- **高层策略**（High-level Policy）：导航控制器，输出目标速度命令
- **低层策略**（Low-level Policy）：预训练的行走控制器，接收速度命令并输出关节位置

### 参考论文

本实现基于论文：
> **"Hierarchical Reinforcement Learning for Agile Quadrupedal Locomotion with Demonstrations"**

---

## 目录结构

```
BiSheTest/
├── __init__.py                    # Gym 环境注册
├── bishe_rough_env_cfg.py     # 粗糙地形导航配置
├── bishe_flat_env_cfg.py      # 平坦地形导航配置
├── agents/
│   ├── __init__.py
│   ├── rsl_rl_ppo_cfg.py         # PPO 训练配置
│   └── skrl_bishe_*_ppo_cfg.yaml # SKRL 配置
├── mdp/
│   ├── __init__.py
│   ├── pre_trained_policy_action.py  # 分层动作管理器
│   ├── rewards.py                # 自定义奖励函数
│   ├── terminations.py           # 终止条件
│   └── curriculums.py            # 课程学习配置
└── config/
    ├── __init__.py
    └── terrain.py                # 自定义地形配置
```

---

## 已注册的 Gym 环境

| 环境 ID | 地形类型 | 用途 | 环境数量 |
|---------|---------|------|---------|
| `Template-BiShe-Rough-Go2-v0` | 粗糙地形 + 障碍物 | 训练 | 4096 |
| `Template-BiShe-Rough-Go2-Play-v0` | 粗糙地形 + 障碍物 | 推理/可视化 | 50 |
| `Template-BiShe-Flat-Go2-v0` | 平坦地形 | 训练 | 4096 |
| `Template-BiShe-Flat-Go2-Play-v0` | 平坦地形 | 推理/可视化 | 50 |
| `Template-BiShe-Climb-Go2-v0` | 楼梯地形 | 训练 | 4096 |
| `Template-BiShe-Climb-Go2-Play-v0` | 楼梯地形 | 推理/可视化 | 50 |

---

## 快速开始

### 前置条件

1. **训练低层行走策略**（必须先完成）
   ```bash
   python scripts/rsl_rl/train.py --task Template-Velocity-Go2-Walk-Rough-v0 --headless
   ```

2. **转换模型为 TorchScript 格式**
   编辑 `NewTools/model_trans.py` 设置路径：
   ```python
   CHECKPOINT_PATH = "path/to/walk_checkpoint.pth"
   OUTPUT_TS_PATH = "ModelBackup/TransPolicy/WalkRoughNewTransfer.pt"
   ```
   然后运行：
   ```bash
   python NewTools/model_trans.py
   ```

3. **更新导航配置中的模型路径**
   在 `bishe_rough_env_cfg.py:143` 更新：
   ```python
   policy_path="ModelBackup/TransPolicy/WalkRoughNewTransfer.pt"
   ```

### 训练导航策略

```bash
# 粗糙地形导航（带障碍物）
python scripts/rsl_rl/train.py --task Template-BiShe-Rough-Go2-v0 --headless

# 平坦地形导航
python scripts/rsl_rl/train.py --task Template-BiShe-Flat-Go2-v0 --headless

# 楼梯攀爬导航
python scripts/rsl_rl/train.py --task Template-BiShe-Climb-Go2-v0 --headless
```

### 推理和可视化

```bash
# 使用训练好的模型进行推理
python scripts/rsl_rl/play.py --task Template-BiShe-Climb-Go2-Play-v0 \
    --checkpoint ModelBackup/BiShePolicy/model_XXXX.pt
```

---

## 配置说明

### 命令空间

高层策略使用 **2D 位姿命令**：

```python
pose_command = mdp.UniformPose2dCommandCfg(
    asset_name="robot",
    simple_heading=False,
    resampling_time_range=(8.0, 8.0),  # 每 8 秒重新采样目标
    debug_vis=True,
    ranges=mdp.UniformPose2dCommandCfg.Ranges(
        pos_x=(-3.0, 3.0),    # 目标 x 范围（米）
        pos_y=(-3.0, 3.0),    # 目标 y 范围（米）
        heading=(-math.pi, math.pi)  # 目标航向范围（弧度）
    ),
)
```

**测试时设置固定目标**：
在 `LocomotionBiSheClimbEnvCfg_Play.__post_init__()` 中：
```python
# 设置固定目标（min==max 即为固定值）
self.commands.pose_command.ranges.pos_x = (3.0, 3.0)
self.commands.pose_command.ranges.pos_y = (0.0, 0.0)
self.commands.pose_command.ranges.heading = (0.0, 0.0)
```

### 动作空间

**高层动作**（3 维）：
- `vx`: 前进速度 (m/s)
- `vy`: 横向速度 (m/s)
- `ωz`: 转向角速度 (rad/s)

**低层动作**（12 维）：
- 12 个关节的目标位置

### 观测空间

高层策略观测（~180 维）：

| 观测项 | 维度 | 描述 |
|--------|------|------|
| `base_lin_vel` | 3 | 基座线速度 (vx, vy, vz) |
| `projected_gravity` | 3 | 重力方向投影 |
| `pose_command` | 3 | 目标位姿 (x, y, heading) |
| `height_scan` | ~168 | **地形高度扫描**（关键！） |
| `actions` | 3 | 上一次动作 |

低层策略观测（~200 维）包含：
- 基座速度、重力投影
- 速度命令
- 关节位置/速度
- **高度扫描**
- 上一次动作

---

## 地形配置

### 粗糙地形（`ROUGH_TERRAINS_CFG`）

包含多种子地形类型：
- **随机粗糙**：Perlin 噪声生成的不平地形
- **阶梯**：离散高度跳跃
- **箱体**：不同高度的盒子障碍物
- **楼梯**：正向和反向楼梯

自定义缩放（针对 Go2 机器人）：
```python
sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
sub_terrains["random_rough"].noise_range = (0.01, 0.06)
```

### 楼梯地形（`STAIR_TERRAINS_CFG`）

专门用于爬楼梯训练：
- 正向楼梯：上楼训练
- 反向楼梯：下楼训练
- 可配置阶梯高度和深度

### 平台障碍物

导航任务中添加了可爬越的平台：
```python
platform = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Platform",
    spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.26)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.13)),
)
```

---

## 奖励函数

基于论文实现的复杂奖励函数：

### 主要奖励项

| 奖励项 | 权重 | 描述 |
|--------|------|------|
| `position_tracking` | 1.0 | 位置跟踪（粗粒度） |
| `position_tracking_fine` | 2.0 | 位置跟踪（细粒度） |
| `heading_tracking` | 0.5 | 航向跟踪 |
| `climb_progress_reward` | 0.0-2.0 | 爬高台进展奖励 |
| `termination_penalty` | -400.0 | 提前终止惩罚 |

### 正则化惩罚

- `lin_vel_z_l2`：限制垂直运动（除非爬高台）
- `ang_vel_xy_l2`：限制侧倾/俯仰角速度
- `joint_torques_l2`：关节力矩惩罚
- `joint_pos_limit`：关节位置限制
- `orientation_l2`：姿态保持

自定义奖励位于：`mdp/rewards.py`

---

## 分层架构详解

### PreTrainedPolicyAction 工作流程

```
高层策略输入:
- 基座速度、重力
- 目标位姿 (x, y, heading)
- 高度扫描
- 上一次动作

    ↓

高层策略输出:
- 低层命令 (vx, vy, ωz)

    ↓

低层策略（预训练）:
- 接收命令 + 完整观测
- 输出关节目标位置

    ↓

PD 控制器:
- 转换为关节力矩
- 应用到机器人
```

### 关键参数

```python
pre_trained_policy_action = mdp.PreTrainedPolicyActionCfg(
    asset_name="robot",
    policy_path="ModelBackup/TransPolicy/WalkRoughNewTransfer.pt",
    low_level_decimation=4,      # 低层策略 decimation
    low_level_actions=...,       # 低层动作配置
    low_level_observations=...,  # 低层观测空间
)
```

---

## 训练技巧

### 超参数

- **学习率**：1e-4
- **批大小**：4096 envs × 32 steps = 131,072 transitions
- **PPO clip**：0.2
- **Epochs**：3-5
- **Episode 长度**：8 秒（与命令重采样时间匹配）

### 课程学习

在 `mdp/curriculums.py` 中配置：
- 逐渐增加目标距离
- 逐渐增加地形难度
- 动态调整奖励权重

### 调试建议

1. **检查低层策略**：确保低层行走策略已经收敛
2. **可视化命令**：`debug_vis=True` 查看目标位置
3. **检查高度扫描**：确保地形信息正确传递
4. **监控奖励**：`position_tracking` 应该逐渐上升

---

## 常见问题

### Q: 机器人不移动？

**A**: 检查：
1. 低层策略路径是否正确
2. 低层策略是否为 TorchScript 格式
3. 观测空间维度是否匹配

### Q: 训练不稳定？

**A**: 尝试：
1. 降低学习率
2. 增加批大小
3. 调整 reward weights
4. 检查是否有 NaN

### Q: 爬楼梯失败？

**A**:
1. 确保观测空间包含 `height_scan`
2. 增加 `climb_progress_reward` 权重
3. 训练更长时间
4. 调整楼梯高度（`STAIR_TERRAINS_CFG`）

### Q: 观测空间维度警告？

**A**: 修改观测后必须重新训练：
```bash
# 删除旧模型
rm -rf logs/rsl_rl/bishe_rough/

# 重新训练
python scripts/rsl_rl/train.py --task Template-BiShe-Rough-Go2-v0 --headless
```

---

## 性能基准

### 粗糙地形导航

| 指标 | 目标值 |
|------|--------|
| 平均奖励 | > 200 |
| 成功率 | > 80% |
| 平均速度 | > 0.5 m/s |

### 楼梯攀爬

| 指标 | 目标值 |
|------|--------|
| 平均奖励 | > 150 |
| 攀爬成功率 | > 70% |

---
