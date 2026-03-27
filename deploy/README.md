# Unitree Deploy 使用说明

这个目录用于把 IsaacLab 训练出的 Go2 策略整理到 `deploy/robots/go2/config/`，然后编译并运行部署控制器。

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

## 准备

至少需要训练时的 `env.yaml`，因为里面有 `joint_sdk_names`，生成 `deploy.yaml` 时会用到。

如果还要一起导出 `policy.onnx` / `policy.pt`，还需要：
- `agent.yaml`
- checkpoint，例如 `model_*.pt`

推荐先复制训练 run 的参数目录：
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
cp -r /你的训练目录/params deploy/robots/go2/config/
```

## 只生成 deploy.yaml
如果 `deploy/robots/go2/config/params/` 已经有 `env.yaml` 和 `agent.yaml`：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --skip-policy-export
```

输出文件：
```text
deploy/robots/go2/config/params/deploy.yaml
```

## 同时导出 policy 和 deploy.yaml
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --checkpoint /你的训练目录/model_XXXX.pt
```

输出文件：
```text
deploy/robots/go2/config/params/deploy.yaml
deploy/robots/go2/config/exported/policy.onnx
deploy/robots/go2/config/exported/policy.pt
```

## 自定义参数路径
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --env-yaml /path/to/params/env.yaml \
  --agent-yaml /path/to/params/agent.yaml \
  --checkpoint /path/to/model_XXXX.pt
```

## 编译和运行
先确认 `deploy/robots/go2/config/config.yaml` 中：
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

## 常见问题
- 报 `joint_sdk_names is missing`：没给训练时的 `env.yaml`。
- 报 `YAML::BadFile: .../params/deploy.yaml`：先运行上面的导出脚本。
- 想改输出目录：加 `--output-dir /你的目录`。
- 默认会把训练里的 `velocity_commands` 改成部署侧使用的 `keyboard_velocity_commands`。
