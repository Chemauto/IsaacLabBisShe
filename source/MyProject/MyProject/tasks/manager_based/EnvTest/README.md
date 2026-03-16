# EnvTest 使用说明

`EnvTest` 是一个用于 Go2 结构化导航实验的纯场景环境，不包含奖励函数和训练逻辑。

## 场景说明

`scene_id` 取值为 `0~4`：

- `0`：左右都无障碍
- `1`：左低台阶，右侧空
- `2`：左右都是低台阶
- `3`：左低台阶，右高台阶
- `4`：左右都是高台阶，中间有可推动箱子

## 直接打开环境

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
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

`EnvTest` 的 `policy` 观测已经改成三个技能的并集，总维度是 `251`：

- `base_lin_vel`：`3`
- `base_ang_vel`：`3`
- `projected_gravity`：`3`
- `velocity_commands`：`3`
- `joint_pos`：`12`
- `joint_vel`：`12`
- `actions`：`12`
- `height_scan`：`187`
- `box_pose`：`7`
- `robot_position`：`3`
- `goal_command`：`3`
- `push_actions`：`3`

其中：

- `walk` 使用前 `235` 维低层观测
- `climb` 也使用同样的 `235` 维低层观测
- `push_box` 先使用 `22` 维高层观测，再把高层输出写回 `velocity_commands`，继续调用 `235` 维低层 walk 策略

## 按 model_use 切换技能

脚本现在默认分两步：

1. 启动后先待机，机器人保持静止
2. 你先选 `model_use`、再写速度或位置指令，最后写入启动开关

```bash
python NewTools/envtest_model_use_player.py --scene_id 4
```

对应关系：

- `0`：idle
- `1`：walk
- `2`：climb
- `3`：push_box

控制文件默认是：

- `model_use`：`/tmp/model_use.txt`
- 速度指令：`/tmp/envtest_velocity_command.txt`
- 位置指令：`/tmp/envtest_goal_command.txt`
- 启动开关：`/tmp/envtest_start.txt`

推荐直接用 `Socket` 目录里的 UDP 控制：

```bash
python Socket/envtest_socket_client.py --model_use 1 --velocity 0.6 0.0 0.0
```

`envtest_model_use_player.py` 已经内置监听 `0.0.0.0:5566`，不需要再单独启动 server。

开始执行：

```bash
python Socket/envtest_socket_client.py --start 1
```

运行中切换技能：

```bash
python Socket/envtest_socket_client.py --model_use 2
python Socket/envtest_socket_client.py --model_use 3
```

如果是 `push_box`，位置指令可直接写目标点：

```bash
python Socket/envtest_socket_client.py --goal 1.8 0.0 0.1
```
