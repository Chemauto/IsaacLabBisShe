# MuJoCo 目录

MuJoCo 仿真、DDS 状态读取和 ROS2 桥接。

## 目录结构

- `simulate_python/`：仿真入口和桥接脚本
- `terrain_tool/`：地形生成工具
- `unitree_robots/`：Go2 XML、mesh 等资源

## simulate_python 文件

- `unitree_mujoco.py`：MuJoCo 仿真主入口，发布 DDS 话题
- `unitree_sdk2py_bridge.py`：DDS bridge（rt/lowstate, rt/sportmodestate, rt/heightmap）
- `push_box_sdk2py_bridge.py`：push-box 额外观测 bridge
- `mujoco_dds_state.py`：订阅 MuJoCo DDS → 写 JSON 状态文件
- `mujoco_ros2_bridge.py`：读 JSON 状态文件 → 发布 ROS2 `/go2/*` 话题
- `config.py`：仿真配置（场景路径、push-box、heightmap 等）

## 数据流

```
unitree_mujoco.py → DDS rt/sportmodestate, rt/lowstate
                      ↓
              mujoco_dds_state.py → /tmp/mujoco_ros2_state.json
                                      ↓
                              mujoco_ros2_bridge.py → /go2/* ROS2 → robot_service.py
```

## 启动

一键启动（包含 robot_service）：

```bash
bash run_mujoco.sh
```

单独启动：

```bash
cd Mujoco/simulate_python

# 终端 1: MuJoCo 仿真 (ros2_env)
conda run -n ros2_env python unitree_mujoco.py

# 终端 2: DDS 状态写入 (ros2_env)
conda run -n ros2_env python mujoco_dds_state.py

# 终端 3: ROS2 桥接 (ros2_env)
conda run -n ros2_env python mujoco_ros2_bridge.py

# 终端 4: robot_service (ros2_env)
conda run -n ros2_env python WebSocket/robot_service.py
```

## 调试命令

```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# walk
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"model_use\": 1, \"velocity\": [0.6, 0.0, 0.0], \"start\": true}'}"

# stop
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"start\": false}'}"

# push_box
ros2 topic pub --once /go2/skill_command std_msgs/msg/String "{data: '{\"model_use\": 3, \"goal\": \"auto\", \"start\": true}'}"

# navigation
ros2 topic pub --once /go2/goal_pose geometry_msgs/msg/PoseStamped "{pose: {position: {x: 4.5, y: 0.0, z: 0.1}, orientation: {w: 1.0}}}"

# 查看状态
ros2 topic echo /go2/odom --once
ros2 topic echo /go2/box_pose --once
ros2 topic echo /go2/skill_status --once
```

## config.py 配置

- `ROBOT_SCENE`：场景 XML 路径
- `ENABLE_PUSH_BOX_OBS`：启用 push-box 观测
- `ENABLE_HEIGHTMAP`：启用高度图

## 地形生成

```bash
cd Mujoco/terrain_tool
python3 terrain_generator.py          # 普通 terrain
python3 mine_terrain_generator.py     # mine terrain
python3 push_box_scene_generator.py   # push-box scene
```

生成后修改 `config.py` 的 `ROBOT_SCENE` 指向新场景。
