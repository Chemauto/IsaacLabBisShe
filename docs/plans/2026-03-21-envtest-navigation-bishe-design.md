# EnvTest Navigation BiShe Model Design

**Goal:** Add the `NavigationBishe.pt` policy as a new `model_use=4` option inside the existing EnvTest player, while reusing the current socket/file control flow and `goal_command_file`.

## Scope

- Register `model_use=4` in the player.
- Load `ModelBackup/NaviationPolicy/NavigationBishe.pt` as an RSL-RL actor checkpoint.
- Reuse the existing `goal_command_file` as the world-frame navigation target.
- Convert the world-frame goal into the NavigationTest training command format:
  - `pose_command = [dx_body, dy_body, dz_body, d_yaw]`
- Reuse the rough-walk low-level policy exactly like the NavigationTest hierarchical setup.
- Extend socket-side `model_use` validation to include `4`.

## Training Compatibility

- `NavigationBishe.pt` expects 197-D observations:
  - `base_lin_vel` (3)
  - `projected_gravity` (3)
  - `pose_command` (4)
  - `height_scan` (187)
- The checkpoint actor structure is `197 -> 128 -> 128 -> 3` with `elu`.
- The trained high-level action is a 3-D velocity-style command consumed by the same rough-walk low-level policy already used elsewhere.
- The high-level action should be held for 10 EnvTest steps to match the original NavigationTest hierarchy timing.

## EnvTest Bridge

- Do not modify EnvTest unified observations just to add `pose_command`.
- Instead, build the navigation observation inside the player by concatenating:
  - slices already available from EnvTest
  - a derived `pose_command` tensor computed from the current robot pose and the external goal
- If there is no explicit goal file value, the navigation branch should output zero high-level actions rather than inventing a target.

## External Control

- `model_use=4` should be accepted by:
  - player CLI/file polling
  - socket client
  - socket server
- Existing fields stay unchanged:
  - `goal x y z`
  - optional `yaw` via raw file/text formats already supported by player

## Error Handling

- Missing navigation goal means no navigation command; the robot should remain effectively idle in that mode.
- Reset behavior remains unchanged.
- Status panel should show the new skill name through normal `model_use` mapping.
