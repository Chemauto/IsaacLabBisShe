# `scripts/rsl_rl` 用法
## 1. 训练
```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Velocity-Go2-Walk-Rough-v0 \
  --headless
```
## 2. 评估并导出误差数据
```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Template-Velocity-Go2-Walk-Rough-Play-v0 \
  --checkpoint /home/xcj/work/IsaacLab/IsaacLabBisShe/ModelBackup/WalkPolicy/WalkRoughNew0406.pt \
  --num_envs 1 \
  --eval_episodes 20 \
  --headless
```
输出目录：`<checkpoint_dir>/eval/<timestamp>/`
生成文件：
- `summary.json`：平均指标
- `episodes.csv`：每次 reset 的评估数据
## 3. 评估结果写入 TensorBoard
```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Template-Velocity-Go2-Walk-Rough-Play-v0 \
  --checkpoint /home/xcj/work/IsaacLab/IsaacLabBisShe/ModelBackup/WalkPolicy/WalkRoughNew0406.pt \
  --num_envs 1 \
  --eval_episodes 20 \
  --eval_tensorboard \
  --headless
```
```bash
tensorboard --logdir /home/xcj/work/IsaacLab/IsaacLabBisShe/ModelBackup/WalkPolicy/eval/<timestamp>/tensorboard
```
## 4. 把 `episodes.csv` 画成图片
```bash
python scripts/rsl_rl/plot_eval.py /home/xcj/work/IsaacLab/IsaacLabBisShe/ModelBackup/WalkPolicy/eval/<timestamp>
```
生成：
- `plots/overview.png`
- `plots/metrics/*.png`
