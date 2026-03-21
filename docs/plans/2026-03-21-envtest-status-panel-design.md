# EnvTest Status Panel Design

**Goal:** In `NewTools/envtest_model_use_player.py`, show a continuously refreshed terminal panel with the current EnvTest scene state and runtime commands, aligned with the `object_facts` style geometry the downstream project expects.

## Scope

- Render a fixed terminal block that refreshes in place instead of appending logs.
- Show:
  - `model_use`
  - `skill`
  - `scene_id`
  - `start`
  - `pose_command`
  - `vel_command`
  - `robot_pose`
  - `goal`
  - `platform_1`
  - `platform_2`
  - `box`
- For `platform_1`, `platform_2`, and `box`, include both position and size.
- If an asset or command is unavailable, display `None`.

## Data Sources

- Robot position comes from `env.unwrapped.scene["robot"].data.root_pos_w - env.scene.env_origins`.
- Goal comes from the current push goal command when available; otherwise it remains `None`.
- Obstacle and box positions come from `env.unwrapped.scene[...]`.
- Obstacle and box sizes come from the scene config via existing helpers and constants in `scene_layout.py`.
- Runtime command display comes from the already parsed values used by the control loop, not by re-reading files again.

## Rendering

- Use ANSI cursor movement to overwrite the same terminal block every refresh.
- Keep the panel text-only and stable-width enough to remain readable in a normal terminal.
- Throttle refreshes so the panel updates periodically instead of every physics step.
- Preserve existing informational logs; the panel is an additional live view.

## Error Handling

- Missing scene assets are rendered as `None`.
- Missing or inactive goal / pose commands are rendered as `None`.
- Rendering failures must not interrupt simulation.
