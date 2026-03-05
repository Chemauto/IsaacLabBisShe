# ManualTest -> 论文效果对齐 Todo

目标：以 `ManualTest` 为主线，不追求 1:1 复现论文，但优先解决“当前能导航、但高沟通过率低”的问题，并逐步靠近论文 *Advanced Skills by Learning Locomotion and Local Navigation End-to-End* 的关键思想。

## 当前状态总览（2026-03-04）

- 已完成：
  - `P0` 基线指标记录与固定评估环境（含 `Template-Manual-Rough-Go2-Eval-v0`）
  - `P1.1` 末端任务奖励主导（已接入 `final_position_reward`）
  - `P1.2` `time_to_go` 观测（已接入，观测维度 +1）
  - `P1.3` 早期探索偏置（已接入 `velocity_towards_target_bias`，支持自动关闭）
  - `P2.1` 大部分稳定性惩罚（`torques/acc/action_rate/feet_acc/undesired_contacts` 已接入）
  - `P2.2` 新增 `bad_orientation` 终止项
- 未完成：
  - `P3.1` 有效目标 patch 过滤（ManualTest 仍未迁移 AdvancedPose2dCommand）
  - `P3.2` pit 难度细粒度扩展（仍是 easy/medium/hard）
  - `P4` 课程门控（目前仍是纯 iter，未加成功率冻结/回退）
  - `P5` 动作架构 A/B 决策与路线 B 实施

## 0. 当前差距（先统一认识）

对照论文与现有 `ManualTest`：

1. 任务奖励结构不一致
- 论文核心：末端任务奖励（只在回合末/最后 `Tr` 秒激活）+ 全程惩罚。
- 当前 `ManualTest`：全程位置跟踪奖励（`position_tracking` + `position_tracking_fine_grained`），会强约束“持续追目标”，不利于临沟前减速/蓄势/再发力。

2. 缺少 time-to-go 观测
- 论文把“剩余时间”显式输入策略。
- 当前 `ManualTest` 没有 `time_to_go` 观测，策略不知道“何时该快、何时该稳”。

3. 动作层级仍有速度跟踪偏置
- 当前高层输出 3 维速度命令给低层步态策略，能力上限受低层策略分布影响（尤其是高沟时需要更激进步态）。

4. 目标点采样仍偏理想化
- 未显式做“有效目标 patch 过滤”，在复杂坑洞中可能采样到不可达/高风险点，干扰学习。

5. 课程学习只按 iter 推进
- 目前已有 iter curriculum（这是对的），但缺少“成功率门控/阶段回退”机制，容易出现后期难度上去但策略还没学稳。

---

## 1. P0（必须先做）：建立可比较基线（改成可量化）

### TODO 1.1 定义基线指标（先统一统计口径）
- 状态：`已完成`
- 文件：
  - `manual_rough_env_cfg.py`
  - `mdp/rewards.py`（如需新增统计辅助函数）
- 必须记录的核心指标（按 episode 聚合）：
  - `success_rate`：episode 结束时 `||target_xy|| < 0.5m` 记为成功。
  - `hard_pit_success_rate`：仅在 hard pit 环境子集上统计成功率。
  - `fall_rate`：`base_contact` 或 `bad_orientation` 终止占比。
  - `timeout_rate`：超时结束占比（区分“失败摔倒”与“没到目标”）。
  - `final_distance_mean`：episode 末目标距离均值（越小越好）。
  - `energy_proxy`：`joint_torques_l2 + joint_acc_l2 + action_rate_l2` 的均值。
- 建议补充指标（便于调参）：
  - `curriculum_stage`（当前课程阶段均值）
  - `terrain_level_mean`（平均地形等级）
- 验收：
  - TensorBoard 可看到上述曲线；
  - 每条曲线随 iteration 更新且无 NaN。

### TODO 1.2 固定评估集（和训练解耦）
- 状态：`已完成`
- 文件：
  - `config/terrain.py`（新增评估专用 terrain cfg）
  - `manual_rough_env_cfg.py`（新增 eval/play 配置入口）
- 做法：
  - 新增 `EVAL_PIT_TERRAINS_CFG`，包含固定浅/中/深沟组合；
  - 评估时关闭地形随机课程（固定 `terrain_types` 与 `terrain_levels`）；
  - 固定随机种子（至少 3 个：`0/1/2`）跑同一评估集。
- 验收：
  - 同一 checkpoint 重复评估，指标波动在可接受范围（例如 success_rate 波动 < 3%）。

### TODO 1.3 先跑一次“当前代码基线”
- 状态：`进行中（你已跑通训练与回放，建议补齐 baseline_metrics.md / baseline_video.mp4）`
- 命令（示例）：
  - `python scripts/rsl_rl/train.py --task Template-Manual-Rough-Go2-v0 --headless --num_envs 2048 --max_iterations 300 --run_name p0_baseline`
  - `python scripts/rsl_rl/play.py --task Template-Manual-Rough-Go2-Play-v0 --checkpoint <baseline_ckpt>`
- 产出物：
  - `baseline_metrics.md`（记录上述核心指标）
  - `baseline_video.mp4`（至少含 easy / medium / hard 各一段）
- 验收：
  - 后续任何改动必须和此 baseline 做同口径对比，不允许“只看回报”。

---

## 2. P1（最高优先级）：把任务改成“末端目标 + 时间条件”

> 这是最接近论文、同时最可能直接提升“高沟通过率”的关键改动。

### TODO 2.1 在 ManualTest 引入末端任务奖励
- 状态：`已完成`
- 参考实现：`BiSheTest/mdp/rewards.py::final_position_reward`
- 目标文件：`ManualTest/mdp/rewards.py`、`manual_rough_env_cfg.py`
- 做法：
  - 新增 `final_position_reward(command_name, activate_s, distance_scale)`
  - 在 `RewardsCfg` 里把当前全程位置奖励降权或暂时移除，改为“末端奖励主导 + 惩罚项约束”。
- 验收：策略在临近 episode 末时更主动收敛到目标，且中途轨迹更自由。

### TODO 2.2 加入 time-to-go 观测
- 状态：`已完成`
- 参考实现：`BiSheTest/mdp/observations.py::normalized_time_to_go`
- 目标文件：`ManualTest/mdp/observations.py`（可新建）与 `manual_rough_env_cfg.py`
- 做法：把归一化剩余时间 `[0,1]` 拼进 policy obs。
- 验收：观测维度更新正确；训练稳定性不下降。

### TODO 2.3 增加“早期探索偏置”，后期自动关闭（可选但推荐）
- 状态：`已完成`
- 参考实现：`BiSheTest/mdp/rewards.py::velocity_towards_target_bias`
- 目的：解决纯末端奖励前期探索困难。
- 验收：前期收敛更快，中后期不会被该项锁死（EMA 阈值后自动置零）。

---

## 3. P2（高沟通过率直接相关）：重做惩罚项与终止条件

### TODO 3.1 引入论文同类惩罚
- 状态：`部分完成（核心惩罚已接入，权重仍需实验调参）`
- 目标文件：`manual_rough_env_cfg.py`、`mdp/rewards.py`
- 至少加入：
  - `joint_torques_l2`
  - `joint_acc_l2`
  - `action_rate_l2`
  - `feet_acc_l2`（论文和 BiSheTest 都强调）
  - `undesired_contacts`
- 验收：训练不会学出“暴力蹬跳+高冲击”策略，硬件友好度提升。

### TODO 3.2 调整终止条件，避免过早截断可恢复动作
- 状态：`部分完成（已加 bad_orientation，阈值和 body_names 仍需基于日志细调）`
- 目标文件：`manual_rough_env_cfg.py`
- 做法：
  - 复核 `base_contact` 阈值与 body_names
  - 增加/调试 `bad_orientation`（上限角）
- 验收：既不过早杀死“可恢复动作”，也不过度放任危险姿态。

---

## 4. P3（地形与目标采样）：避免无效任务

### TODO 4.1 引入有效目标 patch 过滤
- 状态：`未完成`
- 参考实现：`BiSheTest/mdp/commands.py::AdvancedPose2dCommand`
- 目标文件：`ManualTest/mdp/commands.py`（建议新增）+ `manual_rough_env_cfg.py`
- 做法：
  - 目标点从可行 patch 采样（限制高度差/距离）
  - 失败时有限次重采样，最后 fallback 并打日志
- 验收：减少“目标在坑底/不可达区域”导致的无效训练样本。

### TODO 4.2 扩展 pit 难度分辨率
- 状态：`未完成`
- 目标文件：`config/terrain.py`
- 做法：把当前 `easy/medium/hard` 拆成更细粒度（如 `pit_l1..pit_l5`），课程更平滑。
- 验收：阶段切换时成功率波动减小。

---

## 5. P4（课程学习升级）：从“纯 iter”到“iter + 成功率门控”

### TODO 5.1 保留 iter 主线，增加阶段门控
- 状态：`未完成`
- 目标文件：`mdp/curriculums.py`
- 做法：
  - 仍按 `iter_stage_boundaries` 前进
  - 若最近窗口成功率低于阈值，冻结或小回退
- 验收：后期 hard pit 不再“突然崩盘”。

### TODO 5.2 课程参数外置化
- 状态：`未完成`
- 目标文件：`config/terrain.py` 或新建 `config/curriculum.py`
- 做法：`terrain_keys / stage_weights / max_level_ratio / thresholds` 统一配置化。
- 验收：换地形类型时几乎不改课程函数主体。

---

## 6. P5（动作架构决策）：两条路线并行 A/B

### 路线 A（短期见效，推荐先走）
- 继续用当前“高层3维命令 -> 低层预训练步态”。
- 先把任务定义、奖励、课程、目标采样做到位。
- 预期：以最小改动提升高沟通过率。

### 路线 B（更接近论文）
- 切到关节位置直接控制（类似 `BiSheTest` 的 `JointPositionActionCfg`）。
- 预期上限更高，但训练不稳定风险更大，工程成本更高。

### TODO 6.1 做明确 A/B 决策门槛
- 状态：`未完成`
- 若路线 A 在固定评估集上达到目标（例如 hard pit 成功率 > 70%），可不切路线 B。
- 若 A 卡住，则启动 B（分阶段、低学习率、强惩罚冷启动）。

---

## 7. P6（实验计划与里程碑）

### Milestone M1（1~2 周）
- 完成 P0 + P1
- 目标：高沟任务成功率显著高于当前 baseline

### Milestone M2（2~4 周）
- 完成 P2 + P3
- 目标：稳定性提升（跌倒率下降、能耗 proxy 降低）

### Milestone M3（4 周+）
- 完成 P4，并决定是否进入 P5 路线 B
- 目标：逼近论文“复杂地形稳健通过”的效果

---

## 8. 具体改动顺序（建议执行顺序）

1. `P0` 基线与评估集
2. `P1` 末端奖励 + time-to-go（最关键）
3. `P2` 惩罚与终止调参
4. `P3` 有效目标采样
5. `P4` 课程门控
6. `P5` 动作架构 A/B 决策

---

## 9. 完成标准（Definition of Done）

在固定评估集上，连续 3 次独立训练（不同 seed）满足：

1. 深沟（hard pit）成功率达到你设定阈值（建议先 60%，再冲 75%）。
2. 失败率（base contact / bad orientation）明显低于当前基线。
3. 无“靠运气过沟”：成功轨迹可复现、动作不过分抖动。
4. 课程推进阶段中无大面积性能崩塌（曲线可解释）。
