# ModelBackup

这个目录用于备份训练好的策略模型和部署用的 TorchScript 转换模型。

## WalkPolicy

| 模型路径 | 说明 |
| --- | --- |
| `ModelBackup/WalkPolicy/WalkFlatHighHeight.pt` | 平地行走策略，正常机身高度版本。 |
| `ModelBackup/WalkPolicy/WalkFlatLowHeight.pt` | 平地行走策略，低机身高度版本，主要用于推箱子。 |
| `ModelBackup/WalkPolicy/WalkRough.pt` | 粗糙地形行走策略，便于后续训练爬高台。 |

## PushPolicy

| 模型路径 | 说明 |
| --- | --- |
| `ModelBackup/PushPolicy/PushBox.pt` | 推箱子策略。 |

## NaviationPolicy

| 模型路径 | 说明 |
| --- | --- |
| `ModelBackup/NaviationPolicy/NavigationWalk.pt` | 导航策略，用于行走避障。 |
| `ModelBackup/NaviationPolicy/NavigationClimb.pt` | 导航策略，用于爬高台任务。 |

## BiShePolicy

| 模型路径 | 说明 |
| --- | --- |
| `ModelBackup/BiShePolicy/ClimbNew.pt` | 爬高台策略。 |
| `ModelBackup/BiShePolicy/ClimbdoubleOld.pt` | 旧版双层高台爬台策略备份。 |

## TransPolicy

| 模型路径 | 说明 |
| --- | --- |
| `ModelBackup/TransPolicy/WalkFlatHighHeightTransfer.pt` | `WalkFlatHighHeight.pt` 转换后的部署模型。 |
| `ModelBackup/TransPolicy/WalkFlatLowHeightTransfer.pt` | `WalkFlatLowHeight.pt` 转换后的部署模型。 |
| `ModelBackup/TransPolicy/WalkRoughTransfer.pt` | `WalkRough.pt` 转换后的部署模型。 |
| `ModelBackup/TransPolicy/ClimbNewTransfer.pt` | `ClimbNew.pt` 转换后的部署模型。 |
