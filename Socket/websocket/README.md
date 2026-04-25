# WebSocket Robot Service

这个目录现在存放 `EnvTest` 的 WebSocket 统一服务层。

推荐入口不是直接启动这里的 `ws_server.py`，而是从 `Socket` 根目录启动：

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python Socket/robot_service.py
```

这样外部规划器只需要连一个 WebSocket 服务，不需要再单独启动 UDP server。

## 目录说明

- `robot_service.py`
  - 推荐入口对应的实际实现
  - 直接写 `/tmp/model_use.txt`、`/tmp/envtest_goal_command.txt`、`/tmp/envtest_start.txt`
  - 轮询 `/tmp/envtest_live_status.json`
  - 对外返回 `state` 和 `feedback`

- `protocol.py`
  - 把 LLM 侧技能命令转成 `EnvTest` 控制文本
  - 把 player 写出的状态 JSON 转成统一 WebSocket `state`

- `feedback.py`
  - 根据状态流判断 `nav / push / walk_skill / climb` 的成功失败

- `status_reader.py`
  - 读取 `/tmp/envtest_live_status.json`

- `udp_bridge.py`
  - 老桥接版 `ws_server.py` 使用
  - 通过 UDP 复用 `envtest_socket_server.py`

- `ws_server.py`
  - 旧的桥接入口
  - 路径是 `WebSocket -> UDP -> 文件`
  - 现在保留，但不再是推荐启动方式

## 推荐启动方式

### 终端 1：仿真器

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python NewTools/envtest_model_use_player.py --scene_id 4
```

### 终端 2：统一机器人服务端

```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
python Socket/robot_service.py
```

默认监听：

- host: `0.0.0.0`
- port: `8765`

### 终端 3：LLM 规划器

```bash
cd /home/xcj/work/Planner/FinalProject
export ROBOT_WS_URL="ws://127.0.0.1:8765"
python Tui/tui.py
```

## 整体结构

现在推荐的链路是：

```text
Planner/FinalProject
  -> WebSocket command
  -> Socket/robot_service.py
  -> 写 /tmp 控制文件
  -> envtest_model_use_player.py
  -> 写 /tmp/envtest_live_status.json
  -> robot_service.py 轮询状态
  -> WebSocket state / feedback
  -> Planner 继续规划
```

和旧桥接版相比，少了一层：

```text
WebSocket -> UDP server -> 文件
```

现在直接变成：

```text
WebSocket -> 文件
```

## 运行流程

下面按一次真实交互说明。

假设用户在 `Planner/FinalProject` 里输入：

```text
导航到 1,0,0 点
```

### 1. 规划器生成技能调用

`Tui/tui.py` 读取用户输入后，调用：

- `Planner.llm_core.make_plan(messages)`

如果 LLM 决定执行导航，会生成：

```json
{
  "name": "nav",
  "args": {"x": 1, "y": 0, "z": 0}
}
```

然后 `Executor.executor.run_plan()` 开始执行。

### 2. Executor 发送 WebSocket 命令

执行链路是：

```text
Executor.executor.run_plan()
-> Executor.tools.call_tool()
-> Executor.skills.Nav()
-> Executor.robot_ws.send_skill_command()
```

这里会构造一条命令：

```json
{
  "type": "command",
  "action_id": "nav-xxxx",
  "skill": "nav",
  "args": {"x": 1, "y": 0, "z": 0}
}
```

然后发到：

```text
ws://127.0.0.1:8765
```

### 3. robot_service 接收命令

`Socket/robot_service.py` 只是入口，实际进入：

- `websocket/robot_service.py:main()`
- `serve()`
- `handle_client()`

`handle_client()` 收到命令后做三件事：

1. `load_status_snapshot()`
   - 先确认 `/tmp/envtest_live_status.json` 已经存在

2. `build_udp_message(skill, args)`
   - 把技能参数转成 `EnvTest` 控制文本

3. `apply_command_text(text, output_paths)`
   - 直接复用旧 `envtest_socket_server.py` 的 `apply_message()`
   - 把文本写进 `/tmp` 控制文件

比如 `nav(1,0,0)` 会被写成：

```text
model_use=4; goal=1,0,0; start=1
```

对应写入：

- `/tmp/model_use.txt`
- `/tmp/envtest_goal_command.txt`
- `/tmp/envtest_start.txt`

### 4. player 执行技能

`NewTools/envtest_model_use_player.py` 每步都会读取这些 `/tmp` 文件。

它看到：

- `model_use=4`
- `goal=1 0 0`
- `start=1`

就会切到导航策略并开始执行。

同时它会持续刷新：

```text
/tmp/envtest_live_status.json
```

这个 JSON 现在至少包含：

- `robot_pose`
- `robot_yaw`
- `box.position`
- `skill`
- `model_use`
- `start`
- `goal`
- `timestamp`

### 5. robot_service 推送 state

`robot_service.py` 在 `_stream_until_feedback()` 里轮询状态 JSON。

每拿到一次新状态，就调用：

- `build_state_payload(raw_status)`

把状态整理成统一协议：

```json
{
  "type": "state",
  "timestamp": 1710000000.123,
  "robot": {"x": 0.8, "y": 0.1, "z": 0.28, "yaw": 0.3},
  "box_relative": {"x": 0.6, "y": -0.2, "z": 0.0},
  "box_world": {"x": 1.42, "y": 0.0, "z": 0.0},
  "skill": "nav",
  "current_skill": "nav",
  "model_use": 4,
  "start": true
}
```

这里：

- `robot` 是机器人世界坐标和偏航角
- `box_world` 是箱子世界坐标
- `box_relative` 是箱子在机器人本体系下的位置

如果 JSON 里只有 `box.position` 和 `robot_yaw`，服务端会自动做坐标转换。

### 6. Planner 接收 state

`Planner/FinalProject/Executor/robot_ws.py` 收到 `state` 后会：

- `Executor.state.update_latest_state(message)`
- `Executor.state.format_latest_state()`

然后 TUI 会实时显示最新状态。

### 7. robot_service 判断动作成功失败

服务端每一轮状态更新后，都会调用：

- `feedback.evaluate_feedback(command, start_state, current_state, elapsed_sec)`

判定规则是：

- `nav`
  - 机器人世界坐标到目标点距离小于 `0.15m`

- `push`
  - 箱子世界坐标到目标点距离小于 `0.08m`

- `walk_skill`
  - 按起始朝向投影位移，至少走出 `0.10m`

- `climb`
  - 机器人高度增量达到目标高度，容差 `0.05m`

如果长时间未达标，就按超时返回失败。

### 8. robot_service 返回 feedback

一旦成功或失败，服务端会先写：

```text
start=0
```

然后回一条 `feedback`：

```json
{
  "type": "feedback",
  "action_id": "nav-xxxx",
  "skill": "nav",
  "signal": "SUCCESS",
  "message": "nav reached target",
  "summary": {
    "target": {"x": 1, "y": 0, "z": 0},
    "final_robot": {"x": 0.98, "y": 0.02, "z": 0.0},
    "distance": 0.03
  }
}
```

### 9. Planner 决定下一步

`Executor.robot_ws._wait_feedback()` 收到 `feedback` 后返回给：

```text
Executor.skills.*
-> Executor.tools.call_tool()
-> Executor.executor.run_plan()
```

然后 `Tui/tui.py` 会把：

- 执行结果
- 最新状态
- 最近反馈

再发回 `LLM`，让它决定下一步动作。

## 协议说明

### command

规划器发给统一服务端：

```json
{
  "type": "command",
  "action_id": "push-xxxx",
  "skill": "push",
  "args": {"x": 1.8, "y": 0.0, "z": 0.1}
}
```

### state

统一服务端持续发给规划器：

```json
{
  "type": "state",
  "timestamp": 1710000000.123,
  "robot": {"x": 0.8, "y": 0.1, "z": 0.28, "yaw": 0.3},
  "box_relative": {"x": 0.6, "y": -0.2, "z": 0.0},
  "box_world": {"x": 1.42, "y": 0.0, "z": 0.0},
  "skill": "push",
  "current_skill": "push",
  "model_use": 3,
  "start": true
}
```

### feedback

统一服务端最终返回：

```json
{
  "type": "feedback",
  "action_id": "push-xxxx",
  "skill": "push",
  "signal": "SUCCESS",
  "message": "push reached target",
  "summary": {
    "target": {"x": 1.8, "y": 0.0, "z": 0.1},
    "final_box_world": {"x": 1.79, "y": 0.01, "z": 0.1},
    "distance": 0.02
  }
}
```

## 为什么现在这样更合适

现在这个目录的目标不是做一个复杂中间层，而是把责任边界固定住：

- `Planner/FinalProject`
  - 只负责规划和发技能

- `Socket/websocket/robot_service.py`
  - 只负责统一接入、统一状态、统一反馈

- `envtest_model_use_player.py`
  - 只负责仿真策略执行

以后如果切到实体机器人，最好也是保留这个边界：

```text
实体机器人侧统一服务端
<-> WebSocket
LLM规划器
```

这样规划器无需知道底层到底是仿真、ROS2 还是实体驱动。
