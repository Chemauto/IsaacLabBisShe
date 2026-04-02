# Unitree Deploy 使用说明

这个目录现在包含两类部署方式：

1. 通用 `deploy.yaml` 路线
   - 对应 [go2](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2)
   - 适合单策略 deploy
   - 用 `deploy/scripts/export_policy_and_deploy.sh` 自动导出

2. 专用 sim2sim 路线
   - 对应 [go2_push_box](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_push_box)
   - 对应 [go2_nav](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav)
   - 适合高层策略 + 低层 walk 策略的两级控制
   - 不走 `deploy.yaml`，而是把观测拼接和动作后处理直接写在 C++ / Python 里

## 目录

```text
deploy/
├── scripts/
│   ├── export_policy_and_deploy.sh
│   └── generate_deploy_yaml.py
└── robots/
    ├── go2/
    ├── go2_push_box/
    └── go2_nav/
```

## 通用 Go2 Deploy

这个分支用于把 IsaacLab 训练出的单策略导出到 [go2/config](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/config)。

### 导出 deploy 配置与策略

最常用的是直接传 `task` 和 `checkpoint`：

```bash
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --checkpoint /path/to/model_XXXX.pt
```

输出会写入：

- [env.yaml](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/config/params/env.yaml)
- [agent.yaml](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/config/params/agent.yaml)
- [deploy.yaml](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/config/params/deploy.yaml)
- `deploy/robots/go2/config/exported/{policy.onnx,policy.pt}`

如果只想生成 `deploy.yaml`：

```bash
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --skip-policy-export
```

如果要覆盖默认参数路径：

```bash
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --env-yaml /path/to/params/env.yaml \
  --agent-yaml /path/to/params/agent.yaml \
  --checkpoint /path/to/model_XXXX.pt
```

### 编译与运行

先确认 [config.yaml](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/config/config.yaml) 里：

```yaml
Velocity:
  policy_dir: config
```

编译：

```bash
cd deploy/robots/go2/build
cmake ..
make -j4
```

运行：

```bash
cd deploy/robots/go2/build
./go2_ctrl --network lo
```

## Push-Box Deploy

push-box 是专用 sim2sim 路线，对应目录：

- [go2_push_box](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_push_box)

这套运行时逻辑是：

1. MuJoCo 发布 `rt/push_box_obs`
2. 控制器本地维护 `last_push_actions`
3. 高层输入拼成 `16 + 3 = 19` 维
4. 高层 push policy 输出 3 维命令
5. 再喂给 low-level walk policy

### 导出与编译

```bash
cd deploy/robots/go2_push_box
python3 tools/export_push_box_policies.py

cd build
cmake ..
make -j4
```

### 运行

MuJoCo：

```bash
cd Mujoco/simulate_python
python3 unitree_mujoco.py
```

deploy：

```bash
cd deploy/robots/go2_push_box/build
./go2_push_box_ctrl --network lo
```

状态切换：

1. `2` 进入 `FixStand`
2. `3` 进入 `PushBox`

### 实时改推箱目标

push-box 现在也支持运行中动态改目标，不再只能靠 `config.py` 里的默认值。

目标发布脚本在：

- [publish_push_box_goal.py](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_push_box/tools/publish_push_box_goal.py)

用法：

```bash
cd deploy/robots/go2_push_box
python3 tools/publish_push_box_goal.py --goal 1.7 0.0 0.12 0.0
```

当前行为：

- MuJoCo 启动时先用 `PUSH_BOX_GOAL_POSITION / PUSH_BOX_GOAL_YAW`
- 收到 `rt/push_box_goal` 后改成新的目标
- 后面即使停掉发布脚本，也保持最后一次有效目标

### 推箱自动停下阈值

配置在 [config.yaml](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_push_box/config/config.yaml)：

```yaml
enable_success_stop: true
success_distance_threshold: 0.10
success_yaw_threshold: 0.15
success_settle_steps: 10
```

含义是：

- 箱子离目标的平面距离小于 `0.10 m`
- 箱子 yaw 误差小于 `0.15 rad`
- 连续满足 `10` 个控制周期
- 就锁住站立停止

## Navigation Deploy

navigation 也是专用 sim2sim 路线，对应目录：

- [go2_nav](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav)

这套运行时逻辑是：

1. 直接订阅 `rt/lowstate`、`rt/sportmodestate`、`rt/heightmap`
2. 本地把世界坐标目标转换成训练时的 `pose_command`
3. 高层 navigation policy 输出 3 维速度命令
4. 再喂给 low-level rough-walk policy

### 导出与编译

```bash
cd deploy/robots/go2_nav
python3 tools/export_navigation_policies.py

cmake -S . -B build
cmake --build build -j4
```

### 运行

MuJoCo：

```bash
cd Mujoco/simulate_python
python3 unitree_mujoco.py
```

deploy：

```bash
cd deploy/robots/go2_nav/build
./go2_nav_ctrl --network lo
```

状态切换：

1. `2` 进入 `FixStand`
2. `3` 进入 `Navigation`

### 实时改导航目标

目标发布脚本在：

- [publish_navigation_goal.py](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav/tools/publish_navigation_goal.py)

用法：

```bash
cd deploy/robots/go2_nav
python3 tools/publish_navigation_goal.py --goal 4.8 0.0 0.0 0.0
```

当前行为：

- 默认先用 `default_goal_world`
- 收到过一次有效 `rt/navigation_goal` 后，就保持最后一次有效目标
- 后面即使停掉发布脚本，也不会回退到默认点

### 导航自动停下阈值

配置在 [config.yaml](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2_nav/config/config.yaml)：

```yaml
enable_navigation_success_stop: true
navigation_success_distance_threshold: 0.20
navigation_success_yaw_threshold: 0.15
navigation_success_settle_steps: 3
```

这里故意使用 `navigation_*` 前缀，不和 push-box 的停止阈值混用。

含义是：

- 机器人离导航目标的平面距离小于 `0.20 m`
- 机器人 yaw 误差小于 `0.15 rad`
- 连续满足 `3` 个控制周期
- 就锁住站立停止

## MuJoCo Sim2Sim 联调

当前 rough / bishe / push-box / navigation 都依赖 `Mujoco/` 里的 Python 仿真与 DDS 话题。

最常用的 MuJoCo 启动方式：

```bash
cd Mujoco/simulate_python
python3 unitree_mujoco.py
```

完整监控：

```bash
cd Mujoco/simulate_python
export LD_LIBRARY_PATH=${CYCLONEDDS_BUILD:-$HOME/cyclonedds/build}/lib:$LD_LIBRARY_PATH
python3 test/monitor_sim2sim.py
```

push-box 专用观测监控：

```bash
cd Mujoco/simulate_python
python3 test/monitor_push_box_obs.py
```

如果你跑 navigation，不要忘了先把 [config.py](/home/xcj/work/IsaacLab/IsaacLabBisShe/Mujoco/simulate_python/config.py) 里的 `ROBOT_SCENE` 切回非 push-box 场景，比如 `scene.xml`。

## 常见问题

- 报 `Observation term 'xxx' is not registered`：`deploy.yaml` 的 observation 和通用 deploy runtime 不一致。
- 报 `Height scan data is unavailable`：没有人发布 `rt/heightmap`。
- `go2_push_box` 按 `3` 不动：先看有没有收到 `rt/push_box_obs`。
- `go2_nav` 按 `3` 不动：先看有没有收到 `rt/sportmodestate` 和 `rt/heightmap`。
- MuJoCo 里 `heightmap` 全是常数：当前在平地；切到带障碍场景或走到障碍附近再看。
- 机器人默认自己往前走：检查 [deploy.yaml](/home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/config/params/deploy.yaml) 里的 `commands.base_velocity.ranges.lin_vel_x`。
- 想保留训练里的手柄命令 observation：导出时加 `--command-observation-name velocity_commands`。
