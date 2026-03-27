# IsaacLabBisShe

这个仓库面向 Unitree Go2，当前主要覆盖三条线：

- IsaacLab 中的多技能训练与统一场景运行
- `deploy/` 下的策略导出与部署控制器
- `Mujoco/` 下的 Python 仿真、`heightmap` 联调与 sim2sim 调试

## 当前重点能力

- `walk / climb / push_box / navigation` 四类技能已经接入统一运行链路
- `EnvTest` 提供固定结构化场景，便于技能切换和调试
- `deploy/` 可以导出 `deploy.yaml`、`policy.onnx`、`policy.pt`
- `Mujoco/` 已支持 `rt/lowstate`、`rt/sportmodestate`、`rt/heightmap`
- MuJoCo viewer 中可直接显示 `height_scan` 命中点和射线

## 目录概览

- `source/MyProject/MyProject/tasks/manager_based/WalkTest`
  - 行走、攀爬相关环境与任务配置
- `source/MyProject/MyProject/tasks/manager_based/PushBoxTest`
  - 推箱子环境
- `source/MyProject/MyProject/tasks/manager_based/NaviationTest`
  - 导航训练环境
- `source/MyProject/MyProject/tasks/manager_based/EnvTest`
  - 多技能统一测试场景
- `NewTools`
  - 多技能运行脚本和桥接逻辑
- `Socket`
  - UDP server/client 控制接口
- `ModelBackup`
  - 运行时加载的策略模型
- `deploy`
  - Go2 deploy 配置导出、ONNX/JIT 导出、C++ 控制器
- `Mujoco`
  - MuJoCo Python 仿真、terrain 工具、Go2 资源

## 安装

先安装项目 Python 包：

```bash
python -m pip install -e source/MyProject
```

如果要跑 MuJoCo terrain tool，还需要：

```bash
pip3 install noise opencv-python numpy
```

## 流程 1：多技能统一运行

最常用入口：

- `NewTools/envtest_model_use_player.py`
- `Socket/envtest_socket_server.py`
- `Socket/envtest_socket_client.py`

启动 player：

```bash
python NewTools/envtest_model_use_player.py --scene_id 4
```

另一个终端启动 UDP server：

```bash
python Socket/envtest_socket_server.py
```

再发送命令：

```bash
python Socket/envtest_socket_client.py --model_use 1 --velocity 0.6 0.0 0.0
python Socket/envtest_socket_client.py --start 1
python Socket/envtest_socket_client.py --model_use 3 --goal_auto --start 1
python Socket/envtest_socket_client.py --model_use 4 --goal 4.5 0.0 0.1 --start 1
python Socket/envtest_socket_client.py --reset 1
```

`model_use` 对应关系：

- `0`：idle
- `1`：walk
- `2`：climb
- `3`：push_box
- `4`：navigation

`EnvTest` 的 `scene_id=0~4` 对应不同平台和箱子组合，`scene_id=4` 是当前最常用的综合测试场景。

## 流程 2：deploy 导出与运行

最常用导出方式：

```bash
bash deploy/scripts/export_policy_and_deploy.sh \
  --task Template-Velocity-Go2-Walk-BiShe-Pit-v0 \
  --checkpoint /path/to/model_XXXX.pt
```

输出会写到：

- `deploy/robots/go2/config/params/{env.yaml,agent.yaml,deploy.yaml}`
- `deploy/robots/go2/config/exported/{policy.onnx,policy.pt}`

编译 deploy 控制器：

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

更完整说明见：

- `deploy/README.md`

## 流程 3：MuJoCo sim2sim 调试

MuJoCo 目录分三部分：

- `Mujoco/simulate_python`
  - Python 仿真入口、DDS bridge、监控脚本
- `Mujoco/terrain_tool`
  - terrain 生成工具
- `Mujoco/unitree_robots`
  - Go2 XML、mesh、terrain scene、height field 图片

启动 MuJoCo：

```bash
cd Mujoco/simulate_python
bash run_simulator.sh
```

另一个终端看 sim2sim 数据：

```bash
cd Mujoco/simulate_python
export LD_LIBRARY_PATH=${CYCLONEDDS_BUILD:-$HOME/cyclonedds/build}/lib:$LD_LIBRARY_PATH
python3 test/monitor_sim2sim.py
```

当前 `monitor_sim2sim.py` 默认会打印：

- `lowstate`
- `sportmodestate`
- `lowcmd`
- `heightmap grid`
- `heightmap delta grid`

如果要生成 terrain：

```bash
cd Mujoco/terrain_tool
python3 terrain_generator.py
python3 mine_terrain_generator.py
```

然后修改 `Mujoco/simulate_python/config.py` 里的 `ROBOT_SCENE`。

更完整说明见：

- `Mujoco/README.md`
- `Mujoco/simulate_python/SIM2SIM_DEBUG.md`
- `Mujoco/terrain_tool/readme_zh.md`

## 推荐先看

- `source/MyProject/MyProject/tasks/manager_based/EnvTest/README.md`
- `Socket/README.md`
- `deploy/README.md`
- `Mujoco/README.md`
