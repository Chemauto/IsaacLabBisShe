# EnvTest 使用说明

`EnvTest` 是一个用于 Go2 结构化导航实验的纯场景环境，不包含奖励函数和训练逻辑。
当前主要作为统一场景底座，供 `walk / climb / push_box / navigation` 四类技能共用。

## 场景说明

`scene_id` 取值为 `0~4`：

- `0`：左右都无障碍
- `1`：左低台阶，右侧空
- `2`：左右都是低台阶
- `3`：左低台阶，右高台阶
- `4`：左右都是高台阶，中间有可推动箱子

## 直接打开环境

```bash
cd /home/robot/work/IsaacLabBisShe
python scripts/zero_agent.py --task Template-EnvTest-Go2-Play-v0 --scene_id 4
```

## 相机图像

前视相机 `front_camera` 默认会一起初始化，可直接通过 `env.unwrapped.scene["front_camera"]` 读取 `rgb` 和 `distance_to_image_plane`。

```bash
python scripts/envtest_camera_view.py --scene_id 4
python scripts/envtest_camera_view.py --scene_id 4 --headless --max_steps 10
```

默认保存路径：`/tmp/envtest_front_camera.png`

## 统一观测

`EnvTest` 的 `policy` 观测已经改成四个技能的并集，总维度是 `256`：

- `base_lin_vel`：`3`
- `base_ang_vel`：`3`
- `projected_gravity`：`3`
- `velocity_commands`：`3`
- `joint_pos`：`12`
- `joint_vel`：`12`
- `actions`：`12`
- `height_scan`：`187`
- `pose_command`：`4`
- `box_pose`：`7`
- `robot_position`：`3`
- `goal_command`：`4`
- `push_actions`：`3`

其中：

- `walk` 使用前 `235` 维低层观测
- `climb` 也使用同样的 `235` 维低层观测
- `navigation` 使用 `197` 维高层观测：`base_lin_vel + projected_gravity + pose_command + height_scan`
- `push_box` 先使用 `23` 维高层观测，再把高层输出写回 `velocity_commands`，继续调用 `235` 维低层 walk 策略

## 按 model_use 切换技能

脚本现在默认分两步：

1. 启动后先待机，机器人保持静止
2. 你先选 `model_use`、再写速度或位置指令，最后写入启动开关

```bash
python NewTools/envtest_model_use_player.py --scene_id 4
```

player 启动后会在终端里持续覆盖刷新状态面板，显示：

- `model_use`
- `skill`
- `scene_id`
- `start`
- `unified_obs_dim`
- `policy_obs_dim`
- `pose_command`
- `vel_command`
- `robot_pose`
- `goal`
- `platform_1`
- `platform_2`
- `box`

若当前对象或命令不存在，会显示 `None`。

对应关系：

- `0`：idle
- `1`：walk
- `2`：climb
- `3`：push_box
- `4`：navigation

控制文件默认是：

- `model_use`：`/tmp/model_use.txt`
- 速度指令：`/tmp/envtest_velocity_command.txt`
- 位置指令：`/tmp/envtest_goal_command.txt`（支持 `x y z [yaw]`）
- 启动开关：`/tmp/envtest_start.txt`
- 重置指令：`/tmp/envtest_reset.txt`

推荐直接用 `Socket` 目录里的 UDP 控制：

```bash
python Socket/envtest_socket_server.py
python Socket/envtest_socket_client.py --model_use 1 --velocity -0.6 0.0 0.0
```

开始执行：

```bash
python Socket/envtest_socket_client.py --start 1
```

运行中切换技能：

```bash
python Socket/envtest_socket_client.py --model_use 2
python Socket/envtest_socket_client.py --model_use 3
python Socket/envtest_socket_client.py --model_use 4
```

如果是 `push_box`，位置指令可直接写目标点：

```bash
python Socket/envtest_socket_client.py --goal 1.8 0.0 0.1
```

如果你不手动发 `goal`，`model_use=3` 默认会自动使用程序根据当前高台位置计算出的目标点，也就是把箱子推到高台前方便后续爬上去。

如果是 `navigation`，可以直接发送世界系目标点：

```bash
python Socket/envtest_socket_client.py --model_use 4 --goal 4.5 0.0 0.1 --start 1
```

此时 player 会把世界系 `goal(x, y, z)` 转成导航策略需要的 base-frame `pose_command(dx, dy, dz, dyaw)`。

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
