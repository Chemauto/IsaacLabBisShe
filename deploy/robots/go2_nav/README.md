# Go2 Navigation Deploy

这个目录是给 `navigation` 技能单独做的 deploy 工程。

结构和 `go2_push_box` 基本一致，但导航不需要额外的 MuJoCo 专用观测话题：

1. 直接订阅已有的 `rt/lowstate`、`rt/sportmodestate`、`rt/heightmap`
2. 本地根据世界坐标目标点计算训练时需要的 `pose_command`
3. 跑高层导航策略
4. 再把高层输出的速度命令喂给 low-level rough-walk 策略

## 目录里最重要的文件

- `config/config.yaml`
  - 你主要改这里
  - 包括模型路径、目标点、话题名、步长和动作裁剪

- `src/State_NavigationRL.cpp`
  - 导航主逻辑
  - 代码结构就是“原来的 go2 控制器 + 高层导航 ONNX + 低层 walk ONNX”

- `tools/export_navigation_policies.py`
  - 把 `NavigationBishe.pt` 和 `WalkRoughTransfer.pt` 导成 ONNX

- `tools/publish_navigation_goal.py`
  - 持续发布运行时导航目标
  - 往 `rt/navigation_goal` 发 4 维 `[x, y, z, yaw]`

## 怎么运行

### 1. 先准备 ONNX

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav
python3 tools/export_navigation_policies.py
```

### 2. 编译控制器

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav
cmake -S . -B build
cmake --build build -j4
```

### 3. 启动 MuJoCo

导航这套直接复用现有 MuJoCo 入口，不需要 push-box 专用 bridge：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/Mujoco/simulate_python
python3 unitree_mujoco.py
```

如果你不是要跑 push-box，请先把 [config.py](/home/xcj/work/IsaacLab/IsaacLabBisShe/Mujoco/simulate_python/config.py) 里的 `ROBOT_SCENE` 切回普通地形场景，比如 `scene.xml`。

### 4. 启动 deploy

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav/build
./go2_nav_ctrl --network lo
```

然后：

1. 按 `2` 进入 `FixStand`
2. 按 `3` 进入 `Navigation`

如果切成功，终端里会看到：

```text
FSM: Change state from FixStand to Navigation
```

## 目标点怎么改

默认目标点在：

- `config/config.yaml` 里的 `default_goal_world`

例如：

```yaml
default_goal_world: [4.8, 0.0, 0.0, 0.0]
```

含义是：

- `x, y, z, yaw` 世界坐标目标

当前默认还打开了：

```yaml
use_current_height_for_goal: true
```

这表示运行时会自动把目标点的 `z` 对齐到机器人当前高度，减少和训练时 2D 导航命令的不一致。

如果你后面想动态改目标，也可以给 `goal_command_topic` 发布 4 维 `[x, y, z, yaw]`，它会覆盖 `default_goal_world`。

当前默认配置还打开了：

```yaml
latch_last_goal_on_timeout: true
```

这表示：

- 只要收到过一次有效目标，就保持最后一次有效目标
- 就算目标发布脚本停掉，也不会回退到 `default_goal_world`
- 只有重启控制器或者收到新的目标时才更新

最直接的方式就是用这个脚本：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav
python3 tools/publish_navigation_goal.py --goal 4.8 0.0 0.0 0.0
```

它会持续发布到 `rt/navigation_goal`。

注意：

- 不要只发一次，因为 `go2_nav` 默认对目标话题做了 `200 ms` 超时检查
- 即使你停掉这个脚本，只要 `latch_last_goal_on_timeout: true`，控制器也会继续保持最后一次有效目标

## 如果按 3 机器人不动，先看这几条

1. 有没有切到 `Navigation`
2. 有没有收到 `rt/sportmodestate`
3. 有没有收到 `rt/heightmap`
4. ONNX 是否已经导出到 `config/exported/`

如果高层观测缺失，控制器会把高层导航动作置零，机器人会保持站立。
