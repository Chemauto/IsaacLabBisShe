# Socket 目录说明

这个目录现在保留 `EnvTest` 的 UDP `server + client`。

## 文件

- `envtest_socket_server.py`：接收 UDP 消息并写控制文件
- `envtest_socket_client.py`：向 server 发送 UDP 控制消息

## 使用流程

先启动 `EnvTest`：

```bash
cd /home/robot/work/IsaacLabBisShe
python NewTools/envtest_model_use_player.py --scene_id 4
python NewTools/envtest_model_use_player.py --scene_id 4 --enable_front_camera
```


player 启动后会在终端里持续覆盖刷新一块状态面板，显示：

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

如果某个对象或命令当前不存在，会显示 `None`。

其中：

- `unified_obs_dim` 是 EnvTest 统一 `policy` 观测维度，当前为 `252`
- `policy_obs_dim` 是当前 `model_use` 实际切片后的策略输入维度
- walk / climb 对应 `232`
- push_box 高层对应 `19`
- navigation 高层对应 `197`

再启动 UDP 服务：

```bash
python Socket/envtest_socket_server.py
```

默认监听：

- host: `0.0.0.0`
- port: `5566`

默认控制文件：

- `model_use`: `/tmp/model_use.txt`
- `velocity`: `/tmp/envtest_velocity_command.txt`
- `goal`: `/tmp/envtest_goal_command.txt`
- `start`: `/tmp/envtest_start.txt`
- `reset`: `/tmp/envtest_reset.txt`

然后再发控制命令：

```bash
python Socket/envtest_socket_client.py --model_use 1 --velocity 0.6 0.0 0.0
python Socket/envtest_socket_client.py --start 1
python Socket/envtest_socket_client.py --start 0
python Socket/envtest_socket_client.py --model_use 3 --goal 1.8 0.0 0.1 --start 1
python Socket/envtest_socket_client.py --model_use 3 --goal_auto --start 1
python Socket/envtest_socket_client.py --model_use 4 --goal 4.5 0.0 0.1 --start 1
python Socket/envtest_socket_client.py --reset 1
python Socket/envtest_socket_client.py --reset 2
python Socket/envtest_socket_client.py --model_use 5 --goal 3.0 0.75 0.5 --start 1
```

说明：

- `--model_use 3` 且未显式给 `--goal` 时，client 会自动补 `goal=auto`
- server 端也会对 `model_use=3` 做同样兜底，保证 push_box 默认走场景自动目标
- `--model_use 4` 复用同一个 `goal` 文件；player 会把世界系 `goal(x, y, z)` 转成 navigation 需要的 base-frame `pose_command(dx, dy, dz, dyaw)`
- `--reset 1` 只会触发一次整环境重置 `env.reset()`，不会改当前 `model_use / start / goal / velocity`
- `--reset 2` 只重置机器人本体，不重置箱子、障碍物和场景状态，也不会改当前 `model_use / start / goal / velocity`
- EnvTest 的统一 `policy` 观测现在是四类技能的并集：walk / climb / push_box / navigation，总维度 `252`
- player 会先读这 `252` 维统一观测，再按 `model_use` 切出当前技能真正需要的输入

## 支持的控制字段

- `--model_use 0/1/2/3/4`
- `--velocity vx vy wz`
- `--goal x y z`
- `--goal_auto`
- `--start 0/1`
- `--reset 1/2`

也支持直接发送原始文本：

```bash
python Socket/envtest_socket_client.py --text "model_use=3; goal=1.8,0,0.1; start=1"
python Socket/envtest_socket_client.py --text "model_use=4; goal=4.5,0,0.1; start=1"
python Socket/envtest_socket_client.py --text "reset=1"
python Socket/envtest_socket_client.py --text "reset=2"
```

player 侧支持这些文本格式：

- `model_use=0/1/2/3/4`
- `skill=0/1/2/3/4`
- `velocity=0.6,0,0`
- `vel=0.6,0,0`
- `goal=1.8,0,0.1`
- `goal=auto`
- `position=1.8,0,0.1`
- `start=1`
- `start=0`
- `reset=1`
- `reset=2`
- `reset=true`
- `reset`
- `idle`
- `walk`
- `climb`
- `push_box`
- `nav`
- `navigation`
- `navigation_bishe`
