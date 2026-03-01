# BiSheTest

`BiSheTest` 是基于 Isaac Lab 的 Unitree Go2 高级技能训练任务，目标是复现：
- 复杂地形稳定行走
- gap 跨越
- pit 脱困/攀爬
- obstacle 绕障

本任务已实现：
- 关节位置动作控制（Joint Position Action）
- `6s` 回合长度 + 最后 `1s` 末端位置奖励（Final Position Reward）
- 观测包含目标位置与剩余时间（time-to-go）
- 分阶段地形课程（base / gap / pit / obstacle）
- 目标点有效性过滤（优先使用平坦 patch，减少采样到坑底或障碍顶）

## 已注册环境

- `Template-BiShe-Go2-Flat-v0`
- `Template-BiShe-Go2-Flat-Play-v0`
- `Template-BiShe-Go2-Rough-v0`（自动阶段）
- `Template-BiShe-Go2-Rough-Play-v0`
- `Template-BiShe-Go2-Rough-Phase0-v0`
- `Template-BiShe-Go2-Rough-Phase1-v0`
- `Template-BiShe-Go2-Rough-Phase2-Gap-v0`
- `Template-BiShe-Go2-Rough-Phase3-Pit-v0`
- `Template-BiShe-Go2-Rough-Phase4-Obstacle-v0`

## 安装扩展

在仓库根目录执行：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python -m pip install -e source/MyProject
```

## 冒烟测试（推荐先跑）

```bash
python scripts/rsl_rl/train.py \
  --task Template-BiShe-Go2-Rough-Phase0-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 10 \
  --run_name smoke
```

## 训练方式

### 方案 A：分阶段训练（推荐）

```bash
python scripts/rsl_rl/train.py --task Template-BiShe-Go2-Rough-Phase0-v0 --headless --num_envs 2048 --max_iterations 1200 --run_name p0
python scripts/rsl_rl/train.py --task Template-BiShe-Go2-Rough-Phase1-v0 --headless --num_envs 2048 --max_iterations 1200 --run_name p1
python scripts/rsl_rl/train.py --task Template-BiShe-Go2-Rough-Phase2-Gap-v0 --headless --num_envs 2048 --max_iterations 1500 --run_name p2_gap
python scripts/rsl_rl/train.py --task Template-BiShe-Go2-Rough-Phase3-Pit-v0 --headless --num_envs 2048 --max_iterations 1500 --run_name p3_pit
python scripts/rsl_rl/train.py --task Template-BiShe-Go2-Rough-Phase4-Obstacle-v0 --headless --num_envs 2048 --max_iterations 1800 --run_name p4_obs
```

### 方案 B：自动阶段训练

```bash
python scripts/rsl_rl/train.py \
  --task Template-BiShe-Go2-Rough-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 3000 \
  --run_name auto_stage
```

## 断点续训

```bash
python scripts/rsl_rl/train.py \
  --task Template-BiShe-Go2-Rough-v0 \
  --headless \
  --resume \
  --load_run <run_folder_name> \
  --checkpoint <checkpoint_file_name>
```

`checkpoint` 通常形如 `model_XXX.pt`。

## 回放与可视化

```bash
python scripts/rsl_rl/play.py --task Template-BiShe-Go2-Rough-Play-v0
```

## 日志与模型输出目录

```text
logs/rsl_rl/go2_bishe_advanced_rough/<timestamp>/
```

## 运行日志说明

- 阶段切换时会打印日志，例如：
  - `[BiSheTest] Stage switched to phaseX at global step Y`
- 若某批次环境没有找到有效目标 patch，会打印回退日志，例如：
  - `[BiSheTest] Target patch filtering fallback for N envs at step S`
