# MuJoCo Workspace

这个目录放 IsaacLabBisShe 当前使用的 MuJoCo Python 仿真、地形生成工具和 Go2 模型资源。

## 目录作用
- `simulate_python/`：MuJoCo Python 仿真入口，发布 `rt/lowstate`、`rt/sportmodestate`、`rt/heightmap`，接收 `rt/lowcmd`。
  - `unitree_sdk2py_bridge.py`：原始通用 bridge
  - `push_box_sdk2py_bridge.py`：push-box 额外观测 bridge
- `terrain_tool/`：地形生成工具，用来生成 `scene_terrain.xml` 或自定义 mine 场景。
- `unitree_robots/`：机器人 XML、mesh、terrain scene、height field 图片等运行资源。

## 常用流程

1. 启动 MuJoCo 仿真：
```bash
cd Mujoco/simulate_python
bash run_simulator.sh
```

2. 另一个终端看 sim2sim 数据：
```bash
cd Mujoco/simulate_python
export LD_LIBRARY_PATH=$HOME/cyclonedds/build/lib:$LD_LIBRARY_PATH
python3 test/monitor_sim2sim.py
```

3. 再启动你的控制器，让它往 `rt/lowcmd` 发命令。

## Push-box 场景

push-box 现在已经合并到通用 MuJoCo 入口里，不再单独维护一份仿真主脚本。

1. 启动 push-box 仿真：
```bash
cd Mujoco/simulate_python
python3 unitree_mujoco.py
```

2. 查看 push-box 高层观测：
```bash
cd Mujoco/simulate_python
python3 test/monitor_push_box_obs.py
```

需要保证 `config.py` 里这几个配置是对的：

- `ROBOT_SCENE = "../unitree_robots/go2/scene_push_box.xml"`
- `ENABLE_PUSH_BOX_OBS = True`

新增 topic：

- `rt/push_box_obs`

`rt/push_box_obs` 当前发布 16 维状态：

- `base_lin_vel(3)`
- `projected_gravity(3)`
- `box_in_robot_frame_pos(3)`
- `box_in_robot_frame_yaw(sin, cos)`
- `goal_in_box_frame_pos(3)`
- `goal_in_box_frame_yaw(sin, cos)`

说明：

- `push_actions(3)` 不从 MuJoCo 发，因为它属于 deploy / policy 本地内部状态，更适合控制器侧自己维护。
- `config.py` 默认把可移动箱子从 `rt/heightmap` 中剔除了，方便后续低层 walk policy 继续复用无箱子 height scan。

## 切换地形

1. 生成普通 terrain：
```bash
cd Mujoco/terrain_tool
python3 terrain_generator.py
```

2. 生成 mine terrain：
```bash
cd Mujoco/terrain_tool
python3 mine_terrain_generator.py
```

如果要重新生成 push-box scene：
```bash
cd Mujoco/terrain_tool
python3 push_box_scene_generator.py
```

3. 修改 `simulate_python/config.py` 里的 `ROBOT_SCENE`，例如：
```python
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/mine_scene_terrain2.xml"
```

## 进一步说明

- 仿真与调试流程：`simulate_python/SIM2SIM_DEBUG.md`
- 地形工具说明：`terrain_tool/readme_zh.md`
