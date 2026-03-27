# MuJoCo Sim2Sim Debug Flow

## 1. 相关文件

- `config.py`
  - MuJoCo 场景、DDS domain、网卡、手柄、高程图开关都在这里配。
- `unitree_mujoco.py`
  - MuJoCo 仿真入口。启动后会发布 `rt/lowstate`、`rt/sportmodestate`、`rt/heightmap`，并订阅 `rt/lowcmd`。
- `test/monitor_sim2sim.py`
  - 持续打印 sim2sim 关心的话题数据。

## 2. 推荐配置

先确认 `config.py` 里这几项：

```python
DOMAIN_ID = 0
INTERFACE = "lo"
ENABLE_HEIGHTMAP = True
ENABLE_HEIGHTMAP_VIS = True
```

如果要看粗糙地形或高台，确认 `ROBOT_SCENE` 指向带障碍的场景，例如 `mine_scene_terrain2.xml`。

## 3. 启动 MuJoCo 仿真

推荐直接用脚本启动，这样会自动带上 CycloneDDS 的库路径：

```bash
cd Mujoco/simulate_python
bash run_simulator.sh
```

也可以直接运行：

```bash
cd Mujoco/simulate_python
export LD_LIBRARY_PATH=$HOME/cyclonedds/build/lib:$LD_LIBRARY_PATH
python3 unitree_mujoco.py
```

启动后 MuJoCo viewer 会显示机器人、射线和 `heightmap` 命中点。

## 4. 怎么控制机器人

MuJoCo 仿真本身不负责“决策”，它只做两件事：

- 发布状态：`rt/lowstate`、`rt/sportmodestate`、`rt/heightmap`
- 接收控制：`rt/lowcmd`

所以要让机器人动起来，需要另一个控制程序发 `rt/lowcmd`。常见有两种方式：

### 方式 A：简单测试控制

```bash
cd Mujoco/simulate_python
export LD_LIBRARY_PATH=$HOME/cyclonedds/build/lib:$LD_LIBRARY_PATH
python3 test/test_unitree_sdk2.py
```

这个脚本会持续给关节发简单力矩，用来确认 DDS 和控制链路是否通。

### 方式 B：你的 sim2sim 控制器

例如 IsaacLabBisShe 里的 `go2_ctrl`。只要它往 `rt/lowcmd` 发命令，MuJoCo 就会执行。

如果你要用手柄给上层控制器喂命令，把 `config.py` 里的 `USE_JOYSTICK = 1`，然后再启动 MuJoCo。此时 MuJoCo 会发布无线手柄相关数据，具体动作仍然由外部控制器决定。

## 5. 查看仿真数据

新开一个终端，直接运行：

```bash
cd Mujoco/simulate_python
export LD_LIBRARY_PATH=$HOME/cyclonedds/build/lib:$LD_LIBRARY_PATH
python3 test/monitor_sim2sim.py
```

当前默认值已经是：

- `domain_id = 0`
- `interface = lo`
- `rate = 2 Hz`
- 默认打印 `heightmap grid`
- 默认打印 `heightmap delta grid`

如果想保留历史输出：

```bash
python3 test/monitor_sim2sim.py --no-clear
```

如果只想看原始数值，不想看二维网格：

```bash
python3 test/monitor_sim2sim.py --no-heightmap-grid --no-heightmap-delta-grid
```

## 6. 怎么看这些信息

### topic 延迟

输出第一行类似：

```text
topics: lowstate=0.000s, sportstate=0.002s, lowcmd=0.000s, heightmap=0.016s
```

表示四个 topic 最近一次收到消息距现在多久。持续是很小的数，说明链路正常。

### joint state / lowcmd

- `joint state` 是 MuJoCo 当前真实关节状态
- `lowcmd` 是外部控制器给 MuJoCo 的目标命令

这两块最适合做 sim2sim 对比，例如：

- `q_cmd` 和实际 `q` 是否偏差过大
- `dq_cmd` 和实际 `dq` 是否同趋势
- 某个关节是否持续高扭矩

### heightmap grid

`heightmap grid` 里：

- 列 `x(back->front)`：从机器人后方到前方
- 行 `y(top=left, bottom=right)`：从机器人左侧到右侧
- 表格里的值是 `height_scan = base_z - hit_z - 0.5`

平地时整张图接近常数，这是正常的。

### heightmap delta grid

`heightmap delta grid` 更适合看障碍：

- `0.00` 表示和当前平地基线一样高
- `0.20` 表示这里比平地高了 20 cm

如果机器人正对一个 20 cm 高台，前方对应那几列通常会出现连续的 `0.20`。

## 7. 推荐调试顺序

1. 先启动 `run_simulator.sh`，确认 viewer 正常打开。
2. 再跑 `python3 test/monitor_sim2sim.py`，确认 `lowstate`、`sportstate`、`heightmap` 都在更新。
3. 再启动你的控制器，确认 `lowcmd` 开始更新。
4. 对照 `joint state` 和 `lowcmd` 看关节是否跟得上。
5. 把机器人移到平地、台阶、坑前面，观察 `heightmap delta grid` 是否符合直觉。

## 8. 现在这套 heightmap 语义

当前 MuJoCo 里的 `heightmap` 已经按 IsaacLab 的 `GridPattern + yaw alignment` 对齐：

- 网格范围：`1.6 x 1.0`
- 分辨率：`0.1`
- 维度：`17 x 11 = 187`
- 射线方式：每个网格点独立从上往下平行打射线

这套输出已经适合你现在的 sim2sim 联调。
