# Unitree Deploy 使用说明

这个目录用于把 IsaacLab 训练出的 Go2 策略导出到 `deploy/robots/go2/config/`，然后编译并运行部署控制器。

## 目录
```text
deploy/
├── scripts/
│   ├── export_policy_and_deploy.sh
│   └── generate_deploy_yaml.py
└── robots/go2/config/
    ├── config.yaml
    ├── params/{env.yaml,agent.yaml,deploy.yaml}
    └── exported/{policy.onnx,policy.pt}
```

## 导出 deploy 配置与策略
最常用的是直接传 `task` 和 `checkpoint`：
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --checkpoint /path/to/model_XXXX.pt
```
输出会写入 `deploy/robots/go2/config/params/{env.yaml,agent.yaml,deploy.yaml}` 和 `deploy/robots/go2/config/exported/{policy.onnx,policy.pt}`。

如果只想生成 `deploy.yaml`：
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
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

## 编译与运行 deploy
先确认 `deploy/robots/go2/config/config.yaml` 里：
```yaml
Velocity:
  policy_dir: config
```
编译：
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/build
cmake ..
make -j4
```
运行：
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe/deploy/robots/go2/build
./go2_ctrl --network lo
```

实机时把 `lo` 换成真实网卡名，例如 `enp2s0`。

## MuJoCo heightmap 联调
当前 rough / bishe 策略需要 `rt/sportmodestate` 和 `rt/heightmap`。`unitree_mujoco` 已补了 `rt/heightmap` 发布、viewer 可视化和 sim2sim 监控脚本。
启动 MuJoCo：
```bash
cd /home/xcj/work/unitree/unitree_mujoco/simulate_python
bash run_simulator.sh
```
新终端看完整 sim2sim 状态：
```bash
cd /home/xcj/work/unitree/unitree_mujoco/simulate_python
export LD_LIBRARY_PATH=/home/xcj/cyclonedds/build/lib:$LD_LIBRARY_PATH
python3 test/monitor_sim2sim.py
```
`monitor_sim2sim.py` 默认会打印 `lowstate`、`sportmodestate`、`lowcmd`、`heightmap grid` 和 `heightmap delta grid`。MuJoCo viewer 中会显示 `height_scan` 命中点和稀疏射线。详细流程见 `/home/xcj/work/unitree/unitree_mujoco/simulate_python/SIM2SIM_DEBUG.md`。

## 常见问题
- 报 `Observation term 'xxx' is not registered`：`deploy.yaml` 的 observation 和 deploy runtime 不一致。
- 报 `Height scan data is unavailable`：没有人发布 `rt/heightmap`。
- MuJoCo 里 `heightmap` 全是常数：当前在平地；切到带障碍场景或走到障碍附近再看。
- 机器人默认自己往前走：检查 `deploy/robots/go2/config/params/deploy.yaml` 里的 `commands.base_velocity.ranges.lin_vel_x`，如果最小值是 `0.2`，无按键也会被夹成前进命令。
- 想保留训练里的手柄命令 observation：导出时加 `--command-observation-name velocity_commands`。
