# EnvTest 使用说明

`EnvTest` 是一个用于技能回放和多技能联调的纯场景环境。
它本身不负责训练奖励，而是稳定生成固定走廊场景，并提供统一观测给 `walk / climb / push_box / navigation` 复用。

## 场景说明

`scene_id` 取值为 `0~4`：

- `0`：左右都无障碍
- `1`：左低台阶，右侧空
- `2`：左右都是低台阶
- `3`：左低台阶，右高台阶
- `4`：左右都是高台阶，中间有可推动箱子

## 目录职责

核心文件分工如下：

- `__init__.py`：注册 `Template-EnvTest-Go2-v0` 和 `Template-EnvTest-Go2-Play-v0`
- `env_test_env.py`：Gym 包装层，负责 `action_space` / `observation_space` 和运行时缓冲
- `env_test_env_cfg.py`：EnvTest 的场景、传感器、统一观测配置
- `observation_schema.py`：纯 Python 的统一观测维度定义，给 runtime 和文档共用
- `config/assets.py`：墙、障碍、箱子等尺寸和默认摆位
- `config/layout.py`：`scene_id -> layout` 的固定 case 表
- `mdp/observations.py`：环境侧原始观测项和 push goal 计算
- `mdp/actions.py`：不同技能的高层动作后处理
- `mdp/skill_specs.py`：各技能原始观测接口定义和维度
- `mdp/adapters.py`：统一观测切片、训练/播放对齐逻辑
- `utils/control_flags.py`：一次性控制文件消费逻辑
- `utils/navigation_bridge.py`：世界系导航目标转 base-frame pose command
- `utils/status_panel.py`：终端状态面板和状态 JSON 输出
- `utils/player_runtime.py`：`envtest_model_use_player.py` 使用的运行时 helper
- `NewTools/envtest_model_use_player.py`：综合实验入口，主要保留配置和主循环

说明：

- 这里没有单独拆成 `terrain.py`，因为 EnvTest 的配置不只是地形，还包含障碍、箱子和固定场景 case，所以用 `config/assets.py + config/layout.py` 更准确。

## 直接打开环境

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python scripts/zero_agent.py --task Template-EnvTest-Go2-Play-v0 --scene_id 4
```

## 相机图像

前视相机 `front_camera` 默认会一起初始化，可通过 `env.unwrapped.scene["front_camera"]` 读取 `rgb` 和 `distance_to_image_plane`。

```bash
python scripts/envtest_camera_view.py --scene_id 4
python scripts/envtest_camera_view.py --scene_id 4 --headless --max_steps 10
```

默认保存路径：`/tmp/envtest_front_camera.png`

## 统一观测

`EnvTest` 的 `policy` 观测是四个技能的并集，总维度为 `252`：

- `base_lin_vel`：`3`
- `base_ang_vel`：`3`
- `projected_gravity`：`3`
- `velocity_commands`：`3`
- `joint_pos`：`12`
- `joint_vel`：`12`
- `actions`：`12`
- `height_scan`：`187`
- `pose_command`：`4`
- `box_in_robot_frame_pos`：`3`
- `box_in_robot_frame_yaw`：`2`
- `goal_in_box_frame_pos`：`3`
- `goal_in_box_frame_yaw`：`2`
- `push_actions`：`3`

其中：

- `walk` 使用 `232` 维低层观测
- `climb` 使用 `232` 维低层观测
- `navigation` 使用 `197` 维高层观测：`base_lin_vel + projected_gravity + pose_command + height_scan`
- `push_box` 使用 `19` 维高层观测：`base_lin_vel + projected_gravity + box_in_robot_frame_* + goal_in_box_frame_* + push_actions`
- `push_box` 高层输出会写回 `velocity_commands`，再调用 `232` 维 low-level walk 策略执行
- `navigation` 高层输出会写回 `velocity_commands`，再调用 `232` 维 low-level walk 策略执行

### push_box 相对观测含义

- `box_in_robot_frame_pos`：箱子在机器人根坐标系下的 `x, y, z`
- `box_in_robot_frame_yaw`：`sin(box_yaw - robot_yaw), cos(box_yaw - robot_yaw)`
- `goal_in_box_frame_pos`：目标点在箱子坐标系下的 `x, y, z`
- `goal_in_box_frame_yaw`：`sin(goal_yaw - box_yaw), cos(goal_yaw - box_yaw)`

## 按 model_use 切换技能

player 脚本：

```bash
python NewTools/envtest_model_use_player.py --scene_id 4
```

约定如下：

- `model_use=0`：idle
- `model_use=1`：walk
- `model_use=2`：climb
- `model_use=3`：push_box
- `model_use=4`：navigation

启动后默认先待机，再通过控制文件或 UDP 改写：

- `model_use`：`/tmp/model_use.txt`
- 速度指令：`/tmp/envtest_velocity_command.txt`
- 位置指令：`/tmp/envtest_goal_command.txt`
- 启动开关：`/tmp/envtest_start.txt`
- 重置指令：`/tmp/envtest_reset.txt`
- 状态 JSON：`/tmp/envtest_live_status.json`

推荐直接用 UDP 控制：

```bash
python Socket/envtest_socket_server.py
python Socket/envtest_socket_client.py --model_use 1 --velocity -0.6 0.0 0.0
python Socket/envtest_socket_client.py --start 1
```

运行中切换技能：

```bash
python Socket/envtest_socket_client.py --model_use 2
python Socket/envtest_socket_client.py --model_use 3
python Socket/envtest_socket_client.py --model_use 4
```

如果是 `push_box`，可直接发送目标点：

```bash
python Socket/envtest_socket_client.py --goal 1.8 0.0 0.1
```

如果不手动发 `goal`，`model_use=3` 会自动根据当前场景中的障碍位置生成推箱目标点。

如果是 `navigation`，可以直接发送世界系目标点：

```bash
python Socket/envtest_socket_client.py --model_use 4 --goal 4.5 0.0 0.1 --start 1
```

此时 player 会把世界系 `goal(x, y, z, yaw)` 转成导航策略使用的 base-frame `pose_command(dx, dy, dz, dyaw)`。

## reset

UDP 控制支持两种一次性 reset：

- `reset=1`：重置整个环境，相当于 `env.reset()`
- `reset=2`：只重置机器人，不重置箱子、障碍物和场景状态

示例：

```bash
python Socket/envtest_socket_client.py --reset 1
python Socket/envtest_socket_client.py --reset 2
python Socket/envtest_socket_client.py --text "reset=2"
```
