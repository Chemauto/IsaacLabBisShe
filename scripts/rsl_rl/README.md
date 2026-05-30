# scripts/rsl_rl

该目录保存 RSL-RL 训练、回放、评测和绘图脚本。

## 关键任务 ID

- climb：`Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0`
- navigation：`Template-Naviation-BiShe-Go2-Play-v0`
- push_box：`Template-Push-Box-Go2-Play-v0`

## 训练示例

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Naviation-BiShe-Go2-v0 --headless
```

## Play 示例

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py --task Template-Naviation-BiShe-Go2-Play-v0 --checkpoint /path/to/model.pt --num_envs 1
```

## 评测输出

使用 `--eval_episodes` 会在 checkpoint 目录下生成 `eval/<timestamp>/`，包含：

- `summary.json`：平均指标。
- `episodes.csv`：逐回合指标。

## 绘图

```bash
python scripts/rsl_rl/plot_eval.py /path/to/eval/<timestamp>
```
