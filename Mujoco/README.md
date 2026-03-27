# MuJoCo Workspace

这个目录放 IsaacLabBisShe 当前使用的 MuJoCo Python 仿真、地形生成工具和 Go2 模型资源。

## 目录作用
- `simulate_python/`：MuJoCo Python 仿真入口，发布 `rt/lowstate`、`rt/sportmodestate`、`rt/heightmap`，接收 `rt/lowcmd`。
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

3. 修改 `simulate_python/config.py` 里的 `ROBOT_SCENE`，例如：
```python
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/mine_scene_terrain2.xml"
```

## 进一步说明

- 仿真与调试流程：`simulate_python/SIM2SIM_DEBUG.md`
- 地形工具说明：`terrain_tool/readme_zh.md`
