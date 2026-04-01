# Push-box Sim2Sim Notes

这份说明对应 MuJoCo 侧新增的：

- `Mujoco/unitree_robots/go2/scene_push_box.xml`
- `Mujoco/simulate_python/config.py`
- `Mujoco/simulate_python/unitree_sdk2py_bridge.py`
- `Mujoco/simulate_python/unitree_mujoco.py`

以及 deploy 侧新增的：

- `deploy/robots/go2_push_box/`

## 现在 MuJoCo 侧已经提供什么

MuJoCo 会额外发布：

- `rt/push_box_obs`

消息类型复用了 `HeightMap_`，但语义上它不是 height map，而是一个 `1 x 16` 的浮点向量。

维度顺序固定为：

1. `base_lin_vel_x`
2. `base_lin_vel_y`
3. `base_lin_vel_z`
4. `projected_gravity_x`
5. `projected_gravity_y`
6. `projected_gravity_z`
7. `box_in_robot_frame_pos_x`
8. `box_in_robot_frame_pos_y`
9. `box_in_robot_frame_pos_z`
10. `box_in_robot_frame_yaw_sin`
11. `box_in_robot_frame_yaw_cos`
12. `goal_in_box_frame_pos_x`
13. `goal_in_box_frame_pos_y`
14. `goal_in_box_frame_pos_z`
15. `goal_in_box_frame_yaw_sin`
16. `goal_in_box_frame_yaw_cos`

## 为什么这里只有 16 维，不是 19 维

PushBox 高层训练观测是：

- `base_lin_vel(3)`
- `projected_gravity(3)`
- `box_in_robot_frame_pos(3)`
- `box_in_robot_frame_yaw(2)`
- `goal_in_box_frame_pos(3)`
- `goal_in_box_frame_yaw(2)`
- `push_actions(3)`

其中前 16 维来自 MuJoCo 外部状态，最后 `push_actions(3)` 是“上一步高层动作”，属于控制器内部记忆。

所以更合理的分工是：

- MuJoCo 只发布外部状态 16 维
- deploy / controller 在本地维护 `last_push_actions`
- 高层 policy 真正输入时再拼成 19 维

## deploy 侧现在怎么接

当前已经单独做了一套并行工程，不改原有 `deploy/robots/go2`：

1. 运行导出脚本，把当前 push 高层和 low-level walk 导成 ONNX：
```bash
cd deploy/robots/go2_push_box
python3 tools/export_push_box_policies.py
```

2. 编译专用控制器：
```bash
cd deploy/robots/go2_push_box/build
cmake ..
make -j4
```

3. 运行控制器：
```bash
cd deploy/robots/go2_push_box/build
./go2_push_box_ctrl --network lo
```

运行时逻辑是：

1. 新增一个 `rt/push_box_obs` 订阅器
2. 本地维护 `last_push_actions`
3. 高层输入按 `16 + 3` 拼成 19 维
4. 高层 policy 输出 3 维命令后，先做和训练一致的裁剪
5. 再把这 3 维命令喂给低层 walk policy 作为 `velocity_commands`
6. 低层继续使用 MuJoCo 发布的 `rt/heightmap`

## heightmap 说明

`config.py` 里已经把 movable box 从 `rt/heightmap` 的射线命中集合里排除了。

这样做是为了对齐 PushBox 训练时 low-level walk 的输入：

- 高层自己通过 `rt/push_box_obs` 感知箱子
- 低层 walk 不把箱子当作 height scan 障碍物
