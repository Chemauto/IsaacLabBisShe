# ModelBackup

该目录保存训练好的策略模型和部署转换模型。模型供 IsaacLab/MuJoCo 回放、WebSocket 技能服务和 deploy 控制器使用。

## 任务与模型

| 技能 | Play 任务 ID | 主要模型 |
|---|---|---|
| climb | `Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0` | `BiShePolicy/ClimbNew.pt` |
| navigation | `Template-Naviation-BiShe-Go2-Play-v0` | `NaviationPolicy/NavigationWalk.pt` |
| push_box | `Template-Push-Box-Go2-Play-v0` | `PushPolicy/PushBox.pt` |

## 目录

- `WalkPolicy/`：平地、粗糙地形和低机身行走策略。
- `PushPolicy/`：推箱策略。
- `NaviationPolicy/`：导航策略。
- `BiShePolicy/`：攀爬和毕业设计场景策略。
- `TransPolicy/`：部署用 TorchScript 转换模型。

## 说明

模型文件通常体积较大。公开仓库时建议使用 Release、Git LFS 或外部下载链接管理模型，并在 README 中说明来源、用途和许可证。
