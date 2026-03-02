# BiShe 高级技能复现 Todo（中文）

## 0. 进入项目目录

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
```

## 1. 找到最近一次训练目录和 checkpoint

```bash
# 最近一次 run 目录
RUN_DIR=$(ls -dt scripts/rsl_rl/logs/rsl_rl/go2_bishe_advanced_rough/* | head -n1)
echo "RUN_DIR=$RUN_DIR"

# 看有哪些模型
ls -lh "$RUN_DIR"/model_*.pt | tail
```

如果你要从刚跑完的 3300 迭代继续，优先用：

```bash
CKPT=model_3300.pt
```

## 2. 先做 Gap 专项续训（关键）

```bash
python scripts/rsl_rl/train.py \
  --task Template-BiShe-Go2-Rough-Phase2-Gap-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 1200 \
  --resume \
  --load_run "$(basename "$RUN_DIR")" \
  --checkpoint "$CKPT" \
  --run_name ft_phase2_gap
```

## 3. 再做 Pit 专项续训

```bash
# 自动取 gap 续训后的最新 run
RUN_DIR=$(ls -dt scripts/rsl_rl/logs/rsl_rl/go2_bishe_advanced_rough/*ft_phase2_gap* | head -n1)
CKPT=$(ls "$RUN_DIR"/model_*.pt | sort -V | tail -n1 | xargs basename)
echo "RUN_DIR=$RUN_DIR, CKPT=$CKPT"

python scripts/rsl_rl/train.py \
  --task Template-BiShe-Go2-Rough-Phase3-Pit-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 1000 \
  --resume \
  --load_run "$(basename "$RUN_DIR")" \
  --checkpoint "$CKPT" \
  --run_name ft_phase3_pit
```

## 4. 最后做 Obstacle 专项续训

```bash
# 自动取 pit 续训后的最新 run
RUN_DIR=$(ls -dt scripts/rsl_rl/logs/rsl_rl/go2_bishe_advanced_rough/*ft_phase3_pit* | head -n1)
CKPT=$(ls "$RUN_DIR"/model_*.pt | sort -V | tail -n1 | xargs basename)
echo "RUN_DIR=$RUN_DIR, CKPT=$CKPT"

python scripts/rsl_rl/train.py \
  --task Template-BiShe-Go2-Rough-Phase4-Obstacle-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 1000 \
  --resume \
  --load_run "$(basename "$RUN_DIR")" \
  --checkpoint "$CKPT" \
  --run_name ft_phase4_obstacle
```

## 5. 回放验证（看是否还会“静止不动”）

```bash
python scripts/rsl_rl/play.py --task Template-BiShe-Go2-Rough-Play-v0
```

## 6. 训练时重点观察指标

- `Metrics/target_pose/error_pos_2d`：目标是逐步下降，尽量接近 `< 0.5`
- `Episode_Termination/time_out`：可以高，但不能伴随“原地不动”
- `Episode_Reward/stalling`：绝对值应逐步变小（更少停滞）
- `Curriculum/staged_terrain`：应能进入后期阶段，不要长期卡在低阶段

## 7. 常见坑位

- `--checkpoint` 只填文件名（如 `model_300.pt`），不要填绝对路径。
- `--load_run` 填 run 文件夹名（不是完整路径）。
- 如果显存不足，把 `--num_envs` 从 `2048` 改成 `1024` 或 `512`。
