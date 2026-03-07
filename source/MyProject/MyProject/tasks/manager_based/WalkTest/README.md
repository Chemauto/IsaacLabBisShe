# WalkTest 配置说明（`walk_bishe_env_cfg.py`）

本文档基于：
- `MyProject/tasks/manager_based/WalkTest/walk_bishe_env_cfg.py`
- `MyProject/tasks/manager_based/WalkTest/__init__.py`

用于说明 Go2 在粗糙地形与毕业设计坑洞（Pit）任务中的环境配置、训练入口和常用操作。

## 1. 文件定位

- 环境配置文件：`walk_bishe_env_cfg.py`
- 任务注册文件：`__init__.py`
- Pit 地形参数：`config/terrain.py` 中的 `MIXED_PIT_TERRAINS_CFG`

## 2. 配置类结构

`walk_bishe_env_cfg.py` 主要由以下配置块组成：

- `MySceneCfg`
  - 地形导入器（`TerrainImporterCfg`）
  - Go2 机器人资产
  - 传感器：高度扫描 `height_scanner`、接触力 `contact_forces`
- `CommandsCfg`
  - 速度指令采样范围与重采样周期
- `ActionsCfg`
  - 关节位置动作（`JointPositionActionCfg`）
- `ObservationsCfg`
  - 速度、重力投影、关节状态、历史动作、高度扫描等观测
- `EventCfg`
  - 材质随机化、质量扰动、重置与（可选）推扰事件
- `RewardsCfg`
  - 速度跟踪奖励 + 多项运动学/动力学惩罚
- `TerminationsCfg`
  - 超时终止、机身非法接触终止
- `CurriculumCfg`
  - 默认课程项：`terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`

## 3. 关键环境类

### `VelocityGo2WalkRoughEnvCfg`
基础训练配置：

- `num_envs=4096`
- `episode_length_s=20.0`
- `sim.dt=0.005`
- 默认开启地形课程（当 `curriculum.terrain_levels` 不为 `None`）
- 默认关闭：`push_robot`、`base_com`、`undesired_contacts`

### `VelocityGo2WalkRoughEnvCfg_Play`
推理/可视化配置：

- 缩小为 `num_envs=50`
- 地形网格缩小为 `5x5`
- 强制 `terrain_generator.curriculum=False`
- 关闭观测噪声、关闭外力与推扰

### `LocomotionBiShePitEnvCfg`
Pit 训练配置（继承 Rough）：

- 在父类初始化后，将地形替换为 `MIXED_PIT_TERRAINS_CFG`
- 其余奖励/终止/事件沿用基类（除非你后续再覆盖）

### `LocomotionBiShePitEnvCfg_Play`
Pit 评估配置：

- `num_envs=50`
- 地形 `5x5` 且关闭课程
- 固定速度命令：
  - `lin_vel_x=(1.0, 1.0)`
  - `lin_vel_y=(0.0, 0.0)`
  - `ang_vel_z=(0.0, 0.0)`
  - `heading=(0.0, 0.0)`

## 4. 已注册任务 ID

在 `__init__.py` 中，和本文件直接相关的任务有：

- `Template-Velocity-Go2-Walk-BiShe-Pit-v0`
- `Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0`

此外同目录还注册了 Rough / Climb / Flat 系列任务。

## 5. 训练与评估命令

以下命令在目录 `scripts/rsl_rl` 下执行。

### 5.1 从头训练 Pit

```bash
python train.py --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 --headless
```

### 5.2 从已有模型继续训练（resume）

示例：从 `go2_walk_bishe/bootstrap_from_rough/WalkStart.pt` 继续训练。

```bash
python train.py \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --headless \
  --resume \
  --experiment_name go2_walk_bishe \
  --load_run '^bootstrap_from_rough$' \
  --checkpoint '^WalkStart\.pt$' \
  --run_name ft_from_WalkStart
```

### 5.3 Play 模式评估

```bash
python play.py --task Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0
```

## 6. 训练日志怎么看

常看指标：

- `Episode_Termination/time_out`：越高通常越稳
- `Episode_Termination/base_contact`：越低通常越好
- `Episode_Reward/track_lin_vel_xy_exp`、`track_ang_vel_z_exp`：越高越好
- `Curriculum/terrain_levels`：课程难度平均等级，升高代表在向更难地形推进

## 7. 常用调参点（Pit）

- 地形难度：`config/terrain.py` 里的 `MIXED_PIT_TERRAINS_CFG`
  - `num_rows` 控制难度层级数量
  - `num_cols` + `proportion` 控制不同坑型占比
  - `pit_depth_range` 控制坑深范围
- 稳定性：
  - 减小命令范围（先固定前进再放开）
  - 暂时降低课程上升速度（或减小地形最大深度）
- 动作“干净度”（减少刮蹭）：
  - 重新启用 `undesired_contacts`，先用较小负权重

## 8. 备注

- 当前 `CurriculumCfg` 使用的是 `mdp.terrain_levels_vel`（基于行走距离的升降级）。
- Play 配置显式关闭了 terrain curriculum，用于更可控的评估。

## 9. 可能需要优化的地方（部署安全）

当前策略已经具备“深坑可通过”能力，但仍存在“机身/腿部与 pit 边缘接触”的现象。  
仿真里能过关不代表实机长期安全，频繁刮蹭会带来电机冲击、电流峰值和结构磨损风险。

### 9.1 主要风险

- 机身或大腿与坑边碰撞，导致瞬时冲击增大
- 关节在碰撞后出现大扭矩补偿，增加电机热负载
- 长期部署时，减速器和结构件磨损加快

### 9.2 建议优化方向（按优先级）

1. 重新启用接触惩罚（首选）

- 目前基类里关闭了 `undesired_contacts`，建议重新启用并从小权重开始。
- 推荐先试：`weight = -0.1 ~ -0.3`，观察成功率变化后再逐步加大。
- 先约束 `THIGH`，再按需要扩展到 `CALF` 或其他敏感部位。

2. 加入关节功率/冲击约束

- 增大 `dof_torques_l2` 与 `dof_acc_l2` 的惩罚强度（小步调整）。
- 目标是减少“硬顶过去”的动作模式，降低电机峰值负载。

3. 调整课程节奏

- 当 `base_contact` 回升时，减慢课程难度上升速度（或降低最深坑范围）。
- 先稳定“低碰撞通过”，再继续提升坑深。

4. 增加部署前评估指标（建议作为验收门槛）

- `Episode_Termination/base_contact` 保持在低水平并稳定
- 统计每回合 body/thigh/calf 接触次数（均值和分位数）
- 记录关节峰值扭矩、峰值速度、温升代理指标（若可用）

### 9.3 实操建议（避免一次改太多）

建议每次只改一类参数，单次训练 200~500 iterations 对比：

- A 组：仅开启 `undesired_contacts`（小权重）
- B 组：A 组基础上小幅增加 `dof_torques_l2`
- C 组：B 组基础上放慢课程提升

按“通过率 + 接触次数 + 扭矩峰值”三指标联合选最优配置，再用于实机部署。
