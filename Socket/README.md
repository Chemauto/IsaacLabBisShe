# Socket 目录说明

这个目录现在主要用于给 `EnvTest` 提供外部 `model_use` 切换输入。

## 目标

- `scene_id` 只决定当前仿真场景
- `model_use` 只决定当前调用哪个技能策略
- `model_use=1` 表示 `walk`
- `model_use=2` 表示 `climb`
- `model_use=3` 表示 `push_box`

## 先启动 EnvTest

先在一个终端启动仿真，并让它持续读取 `/tmp/model_use.txt`：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python NewTools/envtest_model_use_player.py --scene_id 4 --model_use_file /tmp/model_use.txt
```

## 三种切换方式

### 1. UDP Socket 切换

启动监听：

```bash
python Socket/model_use_socket_switch.py
```

默认监听 `0.0.0.0:5566`，收到 `1`、`2`、`3` 或 `model_use=2` 后会写入 `/tmp/model_use.txt`。

### 2. ROS topic 切换

ROS2：

```bash
python Socket/model_use_ros_topic_switch.py --backend ros2
ros2 topic pub /model_use std_msgs/msg/Int32 "{data: 2}"
```

ROS1：

```bash
python Socket/model_use_ros_topic_switch.py --backend ros1
rostopic pub /model_use std_msgs/Int32 "data: 2"
```

如果 topic 发的是字符串，可以加 `--msg-type string`。

### 3. 键盘 1/2/3 切换

```bash
python Socket/model_use_keyboard_switch.py
```

按键说明：

- `1` -> `walk`
- `2` -> `climb`
- `3` -> `push_box`
- `q` 或 `ESC` -> 退出

## 文件说明

- `model_use_socket_switch.py`：UDP 方式切换 `model_use`
- `model_use_ros_topic_switch.py`：ROS1/ROS2 topic 方式切换 `model_use`
- `model_use_keyboard_switch.py`：本地键盘方式切换 `model_use`
- `send_cmd.py`：旧的速度命令测试脚本，和 `model_use` 切换无直接关系

## 说明

这 3 个脚本本质上都是把最新的 `model_use` 写到 `/tmp/model_use.txt`，真正执行策略切换的是 `NewTools/envtest_model_use_player.py`。

## 速度命令脚本

如果你想通过键盘或文本指令直接给 Go2 发送 `vx, vy, wz`，可以使用：

```bash
python Socket/command_sender.py --mode keyboard
```

或者：

```bash
python Socket/command_sender.py --mode repl
```

这个脚本会把速度命令通过 UDP 发到 `127.0.0.1:5555`，对应的是 `WalkTest` 里使用 `SocketVelocityCommandCfg(port=5555)` 的环境。

常见文本指令示例：

- `forward 0.5`
- `left 0.3`
- `turn_left 0.4`
- `set 0.5 0.0 0.0`
- `stop`

如果你要接这个速度命令，接收侧可以使用 `Template-Velocity-Go2-Walk-Flat-Ros-v0` 这一类带 socket 命令的任务。
