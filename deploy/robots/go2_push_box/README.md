# Go2 Push-Box Deploy

这个目录是单独给 `push_box` 用的 deploy 工程。

它不走综合技能切换，只做一件事：

1. 订阅 `rt/push_box_obs`
2. 跑 push-box 高层策略
3. 再把高层输出喂给 low-level walk 策略
4. 最后往 `rt/lowcmd` 发关节目标

## 目录里最重要的文件

- `config/config.yaml`
  - 你主要改这里
  - 包括模型路径、话题名、步长、动作裁剪

- `src/State_PushBoxRL.cpp`
  - push-box 主逻辑
  - 代码结构基本就是“原来的 go2 控制器 + 额外订阅 `rt/push_box_obs`”

- `tools/export_push_box_policies.py`
  - 把当前 push checkpoint 和 low-level walk JIT 导成 ONNX

- `tools/publish_push_box_goal.py`
  - 持续发布运行时推箱目标
  - 往 `rt/push_box_goal` 发 4 维 `[x, y, z, yaw]`

## 正确启动顺序

### 1. 先启动 MuJoCo push-box 仿真

不要用原来的普通仿真入口。

要用这个：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/Mujoco/simulate_python
python3 unitree_mujoco.py
```

这个脚本才会额外发布：

- `rt/push_box_obs`
- `rt/heightmap`
- `rt/lowstate`
- `rt/sportmodestate`

并且现在还会订阅：

- `rt/push_box_goal`

### 2. 再启动 deploy 控制器

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_push_box/build
./go2_push_box_ctrl --network lo
```

如果不是本机回环，而是真实 DDS 网卡，就把 `lo` 换成你的网卡名。

### 3. 再切状态

先进入站立：

- 按 `2`

再进入 push-box：

- 按 `3`

注意：

- 终端里现在仍然会打印旧的 `Velocity` 提示
- 但在这个工程里，`3` 实际对应的是 `PushBox`

只要日志里出现类似下面这一行，就说明已经切到了 push-box：

```text
FSM: Change state from FixStand to PushBox
```

## 如果按 3 机器人不动，先看这几条

最常见原因不是状态没切进去，而是 `push_box_obs` 没有收到。

### 1. 看有没有切到 PushBox

如果没有这行：

```text
FSM: Change state from FixStand to PushBox
```

说明你还没真正切进状态 3。

先确认：

- 当前终端窗口是 `go2_push_box_ctrl` 这个进程的前台终端
- 键盘输入没有被别的程序吃掉
- 你先按了 `2`，再按 `3`

### 2. 看有没有收到 `rt/push_box_obs`

如果没收到，控制器里会打印：

```text
Push-box observation topic is unavailable or has wrong dim; using zero fallback.
```

这时高层输入会变成全 0，加上 `last_push_actions=0`，高层基本不会给出有效推箱命令，机器人通常就不动。

这时候不要查 deploy，先查 MuJoCo 那边是不是跑的是：

```bash
python3 unitree_mujoco.py
```

而不是普通的 `run_simulator.sh` 或别的 terrain 仿真入口。

### 3. 看有没有收到 `rt/heightmap`

如果没收到，控制器里会打印：

```text
Height-map topic is unavailable or has wrong dim; using zero fallback.
```

low-level walk 还能跑，但效果会明显变差。

### 4. 看有没有收到 optional topic 日志

正常情况下，切进去以后应该能看到：

```text
Received optional topic rt/push_box_obs
Received optional topic rt/heightmap
```

如果没有这两行，基本就是 DDS 话题没接上。

## 推荐的最小联调方法

### 终端 1：MuJoCo

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/Mujoco/simulate_python
python3 unitree_mujoco.py
```

### 终端 2：看高层观测有没有发出来

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/Mujoco/simulate_python
python3 test/monitor_push_box_obs.py
```

如果这里都没有数据，deploy 一定不会动。

### 终端 3：deploy

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_push_box/build
./go2_push_box_ctrl --network lo
```

然后按：

1. `2`
2. `3`

## 你最常改的地方

都在：

- `config/config.yaml`

一般只需要改这些：

- `high_level_policy`
- `low_level_policy`
- `push_box_obs_topic`
- `height_map_topic`
- `high_level_decimation`
- `push_action_clip`

如果你只是想实时改推箱目标，不需要再改 `Mujoco/simulate_python/config.py` 里的：

- `PUSH_BOX_GOAL_POSITION`
- `PUSH_BOX_GOAL_YAW`

它们现在只是启动时的默认目标。运行中更推荐直接发话题：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_push_box
python3 tools/publish_push_box_goal.py --goal 1.7 0.0 0.12 0.0
```

行为和导航一致：

- 没收到话题前，用 `config.py` 里的默认目标
- 收到过一次有效 `rt/push_box_goal` 后，就保持最后一次有效目标
- 就算你停掉发布脚本，也不会回退到 `config.py` 里的默认目标

## 一句话判断当前问题

如果你“按 3 之后机器人不动”，我现在最先怀疑的是：

1. 没有真正切到 `PushBox`
2. 没有运行 `unitree_mujoco.py`
3. `rt/push_box_obs` 没收到，所以高层输入退化成全 0
