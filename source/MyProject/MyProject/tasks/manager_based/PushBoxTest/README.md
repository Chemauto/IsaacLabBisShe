# PushBoxTest 配置说明（`push_box_env_cfg.py`）

本文档对应：

- `MyProject/tasks/manager_based/PushBoxTest/push_box_env_cfg.py`
- `MyProject/tasks/manager_based/PushBoxTest/__init__.py`
- `MyProject/tasks/manager_based/PushBoxTest/mdp/`

用于说明 Go2 四足机器人 `push_box` 技能的设计思路、环境配置、训练入口和后续如何接入你的 case 4 场景。

## 1. 任务目标

这个任务只负责训练一个独立技能：

- 机器人在平地上把一个 `0.2 x 0.2 x 0.2 m` 的可移动箱子推到目标点

它不是完整导航任务，也不是“推箱子 + 爬箱子 + 上高台”的端到端任务。  
当前做法是把它作为一个**中层技能**，专门服务于你论文场景里的 case 4：

1. 前方左右高台都超过 `0.3 m`
2. 附近存在一个 `0.2 m` 左右的可推动箱子
3. 高层策略或规则判断后，先调用 `push_box` 技能
4. 箱子到位后，再调用已有 `climb` 技能完成上箱子和上高台

这样拆分的好处是：

- 训练难度明显低于端到端长时程任务
- 技能接口清楚，后续更容易接到 LLM 调度层
- 失败定位更简单，便于调参

## 2. 我的实现思路

这次 `push_box` 的主配置文件沿用了你现有导航任务的组织方式，做法上是：

- 参考 `NaviationTest/naviation_rough_env_cfg.py` 的结构
- 保留“高层策略输出 3 维速度命令，低层调用预训练 walk policy”的分层控制思路
- 单独新建 `PushBoxTest` 任务，不和 walk / climb 混在一起

核心思想：

- **低层策略**：复用已经训练好的 `walk_flat` 策略
- **高层策略**：只学习 `vx, vy, wz`
- **训练目标**：把箱子中心推到目标点
- **训练场景**：平地、固定尺寸箱子、结构化重置

也就是说，这个任务本质上不是重新学走路，而是在“已经会走”的前提下，学习怎么靠近箱子、保持接触并把箱子推到指定位置。

## 3. 文件结构

```text
PushBoxTest/
├── __init__.py
├── README.md
├── push_box_env_cfg.py
├── agents/
│   ├── __init__.py
│   ├── rsl_rl_ppo_cfg.py
│   └── skrl_push_box_ppo_cfg.yaml
└── mdp/
    ├── __init__.py
    ├── commands.py
    ├── observations.py
    ├── rewards.py
    └── terminations.py
```

各文件职责：

- `push_box_env_cfg.py`
  - 主环境配置
  - 场景、命令、动作、观测、奖励、终止、训练参数
- `mdp/commands.py`
  - 箱子目标点命令 `BoxGoalCommand`
- `mdp/observations.py`
  - 箱子相对机器人位置
  - 目标点相对机器人/箱子的相对位置
- `mdp/rewards.py`
  - 推箱子进展奖励、目标距离奖励、成功奖励
- `mdp/terminations.py`
  - 箱子越界终止
- `agents/`
  - PPO 超参数配置

## 4. 分层控制设计

### 4.1 低层控制器

低层直接复用你已有的平地行走策略：

- 文件中使用的策略路径：
  - `ModelBackup/TransPolicy/WalkFlatNewTransfer.pt`

在 `push_box_env_cfg.py` 中，高层动作通过 `PreTrainedPolicyActionCfg` 下发到低层 walk policy：

- 高层动作维度：`3`
  - `vx`
  - `vy`
  - `wz`
- 低层输出：Go2 的关节目标位置

### 4.2 为什么这样做

这样设计的原因很直接：

- `push_box` 不是一个底层稳定性问题，而是一个“怎么接近并把物体推到目标位置”的中层规划问题
- 你已经有可用的 walk 技能，没有必要重新从关节层开始学
- 这也和你前面导航 / climb 的分层架构保持一致

## 5. 场景设置

场景组成：

- 地面：`plane`
- 机器人：`Unitree Go2`
- 箱子：`0.2 x 0.2 x 0.2 m`
- 接触传感器：机器人全身接触力

箱子当前设置：

- 初始位置：`(1.0, 0.0, 0.1)`
- 质量：`4.0 kg`
- 摩擦：
  - `static_friction = 0.8`
  - `dynamic_friction = 0.6`

这个版本先固定箱子尺寸，不做形状和大小随机化，目的是先让技能收敛。

## 6. 命令设计

当前任务不是给机器人一个“终点”，而是给箱子一个“目标位置”。

使用的命令项：

- `box_goal = mdp.BoxGoalCommandCfg(...)`

训练时目标位置采样范围：

- `pos_x = (1.8, 2.4)`
- `pos_y = (-0.35, 0.35)`

这意味着：

- 箱子初始大致在机器人前方 `1.0 m`
- 目标点再往前一些
- 高层策略需要学会先接近箱子，再持续向目标方向施力

## 7. 重置策略

为了让推箱子任务更容易学，重置采用了结构化分布，而不是完全随机。

### 7.1 机器人重置

- `x = (-0.1, 0.1)`
- `y = (-0.2, 0.2)`
- `yaw = (-0.5, 0.5)`

### 7.2 箱子重置

- 在默认位置基础上扰动：
- `x = (-0.2, 0.2)`
- `y = (-0.25, 0.25)`
- `yaw = (-0.3, 0.3)`

这样可以保证大多数 episode 中，机器人和箱子的相对位置是“适合开始推”的，而不是一开始就给出极端无效姿态。

## 8. 观测设计

当前高层策略观测项包括：

- `base_lin_vel`
- `projected_gravity`
- `box_pose`
- `robot_position`
- `goal_command`
- `last_action`

这套观测的含义是：

- 机器人当前在怎么运动
- 箱子在环境坐标系下的位置和姿态
- 机器人在环境坐标系下的位置
- 目标点在环境坐标系下的位置

这里我刻意没有直接上高度图或更复杂点云观测。`box_lin_vel` 也没有放进策略输入，因为这个量在真实系统里通常不容易稳定获得。当前版本优先使用更容易从定位和感知中构造出来的低维相对状态。

## 9. 奖励设计

当前奖励项如下：

- `is_alive`
- `termination_penalty`
- `box_goal_distance`
- `box_goal_progress`
- `robot_box_distance`
- `box_goal_success`
- `flat_orientation`
- `action_rate`

### 9.1 主要奖励

1. `box_goal_distance`

- 奖励箱子接近目标点
- 是主目标奖励

2. `box_goal_progress`

- 奖励箱子相对上一步更接近目标
- 能明显提高训练初期的学习信号密度

3. `box_goal_success`

- 当箱子进入成功区域时给额外奖励
- 当前成功阈值：`distance < 0.08`

### 9.2 辅助奖励与惩罚

- `robot_box_distance`
  - 鼓励机器人保持在能有效推箱子的距离
- `flat_orientation`
  - 限制机身姿态过度倾斜
- `action_rate`
  - 限制高层动作变化过快
- `termination_penalty`
  - 惩罚跌倒或非法终止

## 10. 终止条件

当前终止项：

- `time_out`
- `base_contact`
- `box_out_of_bounds`

其中：

- `base_contact`
  - 机器人机身非法接触地面就终止
- `box_out_of_bounds`
  - 箱子被推太远或掉出地面有效范围时终止

目前没有把“任务成功”作为提前终止条件，而是通过成功奖励体现。  
这么做的原因是避免把成功也一并记入 `termination_penalty`。

## 11. 训练配置

当前主训练环境：

- `num_envs = 2048`
- `env_spacing = 4.0`
- `sim.dt = LOW_LEVEL_ENV_CFG.sim.dt`
- `decimation = LOW_LEVEL_ENV_CFG.decimation * 10`
- `episode_length_s = 12.0`

PPO 配置位于：

- `agents/rsl_rl_ppo_cfg.py`

关键参数：

- `num_steps_per_env = 16`
- `max_iterations = 2000`
- `actor_hidden_dims = [128, 128]`
- `critic_hidden_dims = [128, 128]`
- `learning_rate = 1e-3`

## 12. 已注册环境

当前注册了两个环境：

- `Template-Push-Box-Go2-v0`
- `Template-Push-Box-Go2-Play-v0`

其中：

- `v0` 用于训练
- `Play-v0` 用于可视化与固定场景调试

Play 配置中：

- 固定目标点为 `(2.0, 0.0)`
- 机器人和箱子都使用确定性重置

这更适合观察策略行为是否合理。

## 13. 训练与测试命令

### 13.1 开始训练

```bash
python scripts/rsl_rl/train.py --task Template-Push-Box-Go2-v0 --headless
```

### 13.2 Play 模式

```bash
python scripts/rsl_rl/play.py --task Template-Push-Box-Go2-Play-v0
```

如果后面你保存了专门的 push 策略，也可以按现有工程习惯加载对应 checkpoint 进行可视化。

## 14. 和 case 4 的衔接方式

这个 `push_box` 技能最终是给 case 4 用的，推荐的高层调度方式如下：

```text
用户语音输入目标
    ->
环境感知输出：
  左高台高度
  右高台高度
  箱子位置和是否可推动
    ->
规则或 LLM 选择技能
    ->
push_box(box, target_pose_near_platform)
    ->
climb(box_top_pose)
    ->
climb(platform_top_pose)
    ->
walk(goal_pose)
```

也就是说：

- `push_box` 的职责只到“把箱子推到合适位置”
- 上箱子和上高台仍然交给你已经训练好的 `climb`

## 15. 当前版本的边界

这个版本是第一版可训练技能，还没有包含以下增强项：

- 箱子尺寸随机化
- 箱子质量随机化
- 更复杂的障碍环境
- 推到平台边缘的精确姿态约束
- 与 `climb` 技能的终止状态对齐训练

也就是说，它目前最适合做：

- `push` 技能的单独收敛验证
- 作为 case 4 中“先把箱子推过去”的独立原语

## 16. 后续建议

如果这个版本能稳定收敛，下一步建议按下面顺序增强：

1. 增加箱子质量随机化  
2. 增加箱子尺寸随机化  
3. 把目标点从普通平地改成“高台边缘附近”  
4. 加入和 `climb` 技能串联时的状态分布测试  
5. 最后再接回你的语音 + LLM 调度系统

## 17. 当前状态

本次实现已经完成：

- `PushBoxTest` 任务骨架
- 场景、命令、观测、奖励、终止配置
- Gym 环境注册
- PPO 配置

并且已经通过静态语法检查（`py_compile`）。

尚未完成：

- Isaac Sim 运行时训练验证
- reward 曲线与成功率调参
- 和 case 4 的整链路联调

如果后面训练时出现“只靠近箱子但不持续推”、“把箱子推偏”、“自己绕箱子打转”这三类问题，优先去调：

- `box_goal_progress`
- `robot_box_distance`
- reset 分布范围
