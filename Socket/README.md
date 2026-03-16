# Socket 目录说明

这个目录现在只保留 `EnvTest` 的 UDP 客户端。

`envtest_model_use_player.py` 已经内置了 UDP 监听，所以不需要再单独启动 server。

## 文件

- `envtest_socket_client.py`：向 `EnvTest` 发送 UDP 控制消息

## 使用流程

先启动 `EnvTest`：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python NewTools/envtest_model_use_player.py --scene_id 4
```

启动后，player 会默认监听：

- host: `0.0.0.0`
- port: `5566`

然后再发控制命令：

```bash
python Socket/envtest_socket_client.py --model_use 1 --velocity 0.6 0.0 0.0
python Socket/envtest_socket_client.py --start 1
python Socket/envtest_socket_client.py --start 0
python Socket/envtest_socket_client.py --model_use 3 --goal 1.8 0.0 0.1 --start 1
```

## 支持的控制字段

- `--model_use 0/1/2/3`
- `--velocity vx vy wz`
- `--goal x y z`
- `--start 0/1`

也支持直接发送原始文本：

```bash
python Socket/envtest_socket_client.py --text "model_use=3; goal=1.8,0,0.1; start=1"
```

player 侧支持这些文本格式：

- `model_use=0/1/2/3`
- `skill=0/1/2/3`
- `velocity=0.6,0,0`
- `vel=0.6,0,0`
- `goal=1.8,0,0.1`
- `position=1.8,0,0.1`
- `start=1`
- `start=0`
- `idle`
- `walk`
- `climb`
- `push_box`
