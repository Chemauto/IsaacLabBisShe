# Deploy

`deploy/` 保存策略导出和部署控制器。它主要用于把 IsaacLab 训练好的策略转换为 ONNX/TorchScript，并在 MuJoCo 或实机控制框架中运行。

## 路线

1. 通用 Go2 deploy：适合单策略导出与运行。
2. `go2_push_box`：推箱专用两级控制，高层 push policy + 低层 walk policy。
3. `go2_nav`：导航专用两级控制，高层 navigation policy + 低层 walk policy。

## 关键任务 ID

- climb：`Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0`
- navigation：`Template-Naviation-BiShe-Go2-Play-v0`
- push_box：`Template-Push-Box-Go2-Play-v0`

## 常用目录

- `robots/go2/`：通用 Go2 控制器。
- `robots/go2_push_box/`：推箱控制器和导出工具。
- `robots/go2_nav/`：导航控制器和导出工具。
- `scripts/`：策略导出和配置生成脚本。

## 导出示例

```bash
bash deploy/scripts/export_policy_and_deploy.sh --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 --checkpoint /path/to/model.pt
```

推箱策略导出：

```bash
cd deploy/robots/go2_push_box
python3 tools/export_push_box_policies.py
```

导航策略导出：

```bash
cd deploy/robots/go2_nav
python3 tools/export_navigation_policies.py
```

## 协议说明

部署代码中包含 Isaac Lab 派生配置和第三方运行库。公开发布时保留源码文件头、第三方许可证和根目录 `LICENSE`。
