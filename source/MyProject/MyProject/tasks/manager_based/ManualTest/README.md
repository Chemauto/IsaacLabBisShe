# ManualTest

`ManualTest` 是基于 Isaac Lab 的 Unitree Go2 粗糙坑洞导航任务，当前为**端到端关节动作版本**（策略直接输出关节目标位置，由底层 PD 转力矩）。

当前版本重点：
- 目标导航 + 越坑洞联合训练（`pose_command`）
- 课程学习仅按训练迭代数（iter）推进难度
- 前期简单坑占比高，后期逐步提升困难坑占比

## 已注册环境

- `Template-Manual-Rough-Go2-v0`：训练环境
- `Template-Manual-Rough-Go2-Play-v0`：可视化/回放环境
- `Template-Manual-Rough-Go2-Eval-v0`：固定评估环境（关闭训练课程）

注册位置：`__init__.py`

## 目录结构（当前）

```text
ManualTest/
├── __init__.py                        # Gym 环境注册
├── manual_rough_env_cfg.py            # 主环境配置（场景、观测、奖励、课程等）
├── config/
│   └── terrain.py                     # 坑洞地形生成配置（easy/medium/hard/mixed）
├── mdp/
│   ├── __init__.py
│   ├── commands.py                    # AdvancedPose2dCommand（极坐标采样+有效目标过滤）
│   ├── commands_cfg.py                # 命令配置
│   ├── observations.py                # time_to_go、pose_command_position_b 等观测函数
│   ├── rewards.py                     # 任务奖励
│   ├── terminations.py                # 终止项（含可选越界终止）
│   └── curriculums.py                 # 课程学习函数（iter 驱动）
└── agents/
    ├── rsl_rl_ppo_cfg.py              # RSL-RL PPO 配置
    └── skrl_manual_rough_ppo_cfg.yaml # skrl PPO 配置
```

## 动作、命令与观测（当前）

- 动作：`JointPositionActionCfg`
  - 策略输出所有关节目标位置（Go2 通常 12 维）
- 命令：`AdvancedPose2dCommandCfg`
  - 目标按极坐标在机器人周围采样，半径 `[1.0, 5.0]` m
  - 带有效目标过滤（避开坑底/高障碍），失败重采样
- 策略观测项：
  - `base_lin_vel`：3 维
  - `base_ang_vel`：3 维
  - `projected_gravity`：3 维
  - `pose_command`：3 维（`dx, dy, dz`，机体系）
  - `time_to_go`：1 维
  - `joint_pos`：12 维（Go2）
  - `joint_vel`：12 维（Go2）
  - `height_scan`：187 维
  - `actions`：12 维（上一时刻动作）
  - 合计：`236` 维（按 Go2 12 关节）
- 观测噪声：已开启 `enable_corruption=True`，并对速度、重力投影、关节、高度扫描注入噪声

## 地形配置

在 `config/terrain.py` 中定义：
- `EASY_PIT_TERRAINS_CFG`
- `MEDIUM_PIT_TERRAINS_CFG`
- `HARD_PIT_TERRAINS_CFG`
- `MIXED_PIT_TERRAINS_CFG`（默认训练使用）

`MIXED_PIT_TERRAINS_CFG` 默认比例：
- easy: `0.60`
- medium: `0.30`
- hard: `0.10`

说明：`proportion` 决定地形池静态容量；训练时课程学习会按 iter 动态重采样不同难度。

## 课程学习（核心）

入口：`manual_rough_env_cfg.py -> CurriculumCfg -> pit_terrain_schedule`

使用函数：`mdp/curriculums.py::pit_terrain_by_iteration`

当前默认参数（`manual_rough_env_cfg.py`）：
- `iter_stage_boundaries = (0, 400, 900, 1300)`
- `steps_per_iteration = 48`（与 `agents/rsl_rl_ppo_cfg.py` 的 `num_steps_per_env=48` 对齐）
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

并包含 `p0_metrics` 统计项，在训练日志输出：
- `success_rate`
- `hard_pit_success_rate`
- `hard_pit_active_rate`
- `fall_rate`
- `timeout_rate`
- `final_distance_mean`
- `energy_proxy`

## 奖励（当前）

- 任务主奖励：
  - `final_position`（末端位置奖励，后 1s 激活）
  - `exploration_bias`（早期探索引导，可自动关闭）
  - `stalling`（停滞惩罚）
- 正则惩罚：
  - `dof_acc_l2`
  - `dof_torques_l2`
  - `undesired_contacts`（`.*_thigh|.*_calf`）
  - `action_rate_l2`
  - `feet_acc_l2`

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

### 3) 正式训练（默认 PPO 配置是 2000 iter）

```bash
python scripts/rsl_rl/train.py \
  --task Template-Manual-Rough-Go2-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 2000 \
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

3. `undesired_contacts` 报正则不匹配
- Go2 刚体名是小写（如 `FL_thigh`），不要写 `.*THIGH`。
- 当前推荐：`body_names=".*_thigh|.*_calf"`。

## 复现状态（只看“能否复现流程”）

- 当前已可视为“方法流程复现版”：
  - 端到端关节动作
  - 论文式 episode（6s、目标采样、time-to-go 输入）
  - iter 驱动地形课程学习
  - 可训练、可回放、可固定评估
- 若要继续追论文效果，重点在奖励权重、课程阈值与超参精调，而不是框架重写。
