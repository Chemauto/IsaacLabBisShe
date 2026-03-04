# ManualTest

`ManualTest` 是基于 Isaac Lab 的 Unitree Go2 粗糙坑洞导航任务，采用“高层导航策略 + 低层预训练步态策略”的分层控制方式。

当前版本重点是：
- 在混合坑洞地形上进行位置与朝向跟踪训练
- 使用按训练迭代数（iter）推进的课程学习
- 前期更多简单坑，后期逐步增加中/困难坑比例

## 已注册环境

- `Template-Manual-Rough-Go2-v0`：训练环境
- `Template-Manual-Rough-Go2-Play-v0`：可视化/回放环境
- `Template-Manual-Rough-Go2-Eval-v0`：固定评估环境（关闭训练课程）

注册位置：`__init__.py`

## 目录结构

```text
ManualTest/
├── __init__.py                        # Gym 环境注册
├── manual_rough_env_cfg.py            # 主环境配置（场景、观测、奖励、课程等）
├── config/
│   └── terrain.py                     # 坑洞地形生成配置（easy/medium/hard/mixed）
├── mdp/
│   ├── __init__.py
│   ├── pre_trained_policy_action.py   # 高层动作 -> 低层预训练策略动作
│   ├── rewards.py                     # 任务奖励
│   ├── terminations.py                # 终止项（含可选越界终止）
│   └── curriculums.py                 # 课程学习函数（iter 驱动）
└── agents/
    ├── rsl_rl_ppo_cfg.py              # RSL-RL PPO 配置
    └── skrl_manual_rough_ppo_cfg.yaml # skrl PPO 配置
```

## 任务与控制逻辑

- 高层动作：`PreTrainedPolicyAction.action_dim = 3`
  - 维度定义：`[v_x, v_y, w_z]`
- 高层命令：`pose_command`（由 `UniformPose2dCommand` 生成，4 维）
  - 维度定义：`[x, y, z, heading]`
  - 说明：虽然采样范围配置的是 `pos_x/pos_y/heading`，但 `z` 会自动使用机器人默认根高度，因此命令张量是 4 维
- 低层控制：加载 TorchScript 低层策略（`policy_path`），输出低层关节目标动作（Go2 通常为 12 维）
- 高层策略观测维度（当前配置）：
  - `base_lin_vel`: 3 维
  - `projected_gravity`: 3 维
  - `pose_command`: 4 维
  - `height_scan`: 187 维
  - `actions`: 3 维
  - 合计：`3 + 3 + 4 + 187 + 3 = 200` 维
  - 其中 `height_scan=187` 的来源：`GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`
    - x 方向采样点数：`1.6 / 0.1 + 1 = 17`
    - y 方向采样点数：`1.0 / 0.1 + 1 = 11`
    - 总射线数：`17 * 11 = 187`

## 地形配置

在 `config/terrain.py` 中定义：
- `EASY_PIT_TERRAINS_CFG`
- `MEDIUM_PIT_TERRAINS_CFG`
- `HARD_PIT_TERRAINS_CFG`
- `MIXED_PIT_TERRAINS_CFG`（默认训练使用）

`MIXED_PIT_TERRAINS_CFG` 当前比例：
- easy: `0.60`
- medium: `0.30`
- hard: `0.10`

说明：`proportion` 决定“地图池”里各类地形列分布，是静态容量；课程学习再基于 iter 动态重采样。

## 课程学习（核心）

入口：`manual_rough_env_cfg.py -> CurriculumCfg -> pit_terrain_schedule`

使用函数：`mdp/curriculums.py::pit_terrain_by_iteration`

当前默认参数：
- `iter_stage_boundaries = (0, 400, 900, 1300)`
- `steps_per_iteration = 8`（与 `agents/rsl_rl_ppo_cfg.py` 的 `num_steps_per_env=8` 对齐）
- `stage_weights`（按 easy/medium/hard）：
  - stage0: `(0.85, 0.14, 0.01)`
  - stage1: `(0.65, 0.28, 0.07)`
  - stage2: `(0.45, 0.35, 0.20)`
  - stage3: `(0.25, 0.35, 0.40)`
- `stage_max_level_ratio = (0.35, 0.55, 0.75, 1.0)`

作用机制：
1. 用 `common_step_counter // steps_per_iteration` 计算当前 iter。
2. 根据阶段边界选择当前 stage。
3. 按该 stage 权重重采样 `easy_pit/medium_pit/hard_pit`。
4. 同时限制该 stage 允许的最大 `terrain_level`。

这样可实现“前期简单多，后期困难多”的课程学习。

## P0 基线指标日志

已新增 `Curriculum/p0_metrics/*` 指标（在 reset 时统计）：
- `Curriculum/p0_metrics/success_rate`
- `Curriculum/p0_metrics/hard_pit_success_rate`
- `Curriculum/p0_metrics/hard_pit_active_rate`
- `Curriculum/p0_metrics/fall_rate`
- `Curriculum/p0_metrics/timeout_rate`
- `Curriculum/p0_metrics/final_distance_mean`
- `Curriculum/p0_metrics/energy_proxy`

## 训练与回放

在仓库根目录执行。

### 1) 安装扩展（如未安装）

```bash
python -m pip install -e source/MyProject
```

### 2) 冒烟训练

```bash
python scripts/rsl_rl/train.py \
  --task Template-Manual-Rough-Go2-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 10 \
  --run_name smoke_manual
```

### 3) 正式训练（默认 PPO 配置是 1500 iter）

```bash
python scripts/rsl_rl/train.py \
  --task Template-Manual-Rough-Go2-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 1500 \
  --run_name manual_curriculum
```

### 4) 回放

```bash
python scripts/rsl_rl/play.py --task Template-Manual-Rough-Go2-Play-v0
```

### 5) 固定评估（推荐用于改动前后对比）

```bash
python scripts/rsl_rl/play.py --task Template-Manual-Rough-Go2-Eval-v0
```

## 关键可调参数

优先修改 `manual_rough_env_cfg.py` 中 `CurriculumCfg`：
- `iter_stage_boundaries`：阶段切换时机
- `stage_weights`：各阶段地形难度采样比例
- `stage_max_level_ratio`：各阶段可用 terrain level 上限

其次修改 `config/terrain.py`：
- `pit_depth_range`：不同难度坑深
- `proportion`：地形池静态容量
- `num_rows/num_cols`：地形地图规模

## 常见问题

1. 课程感觉不生效
- 确认 `LocomotionManualRoughEnvCfg` 中已挂载 `curriculum: CurriculumCfg = CurriculumCfg()`。
- 确认 `self.scene.terrain.terrain_generator.curriculum = True`。

2. 难度上升太快
- 增大 `iter_stage_boundaries` 间隔。
- 提高早期 easy 权重，降低 hard 权重。
- 降低早期 `stage_max_level_ratio`。

3. 低层策略加载失败
- 检查 `policy_path` 是否存在且为 TorchScript `.pt` 文件。

## 备注

- `max_init_terrain_level` 仅控制初始地形等级上限，不是完整课程学习机制。
- 若迁移到 gap/obstacle 等任务，可复用 `pit_terrain_by_iteration` 的框架，替换 `terrain_keys` 与阶段权重即可。
