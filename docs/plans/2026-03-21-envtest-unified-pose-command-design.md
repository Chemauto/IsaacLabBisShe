# EnvTest Unified Pose Command Design

**Goal:** Add the navigation `pose_command` into EnvTest's unified `policy` observation so all skills, including `model_use=4`, slice their inputs from one shared observation vector.

## Scope

- Extend the EnvTest unified observation with a runtime `pose_command` term.
- Increase the unified observation size from `252` to `256`.
- Update the player to write the navigation `pose_command` into the unified runtime buffers every step.
- Change the navigation branch to slice its `197`-D high-level input directly from the unified observation instead of manually concatenating a custom tensor.

## Unified Observation Layout

- Existing unified observation:
  - low-level walk/climb terms: `235`
  - push-box extra terms: `17`
  - total: `252`
- New unified observation:
  - existing `252`
  - navigation `pose_command`: `4`
  - total: `256`

The navigation slice remains:
- `base_lin_vel` (3)
- `projected_gravity` (3)
- `pose_command` (4)
- `height_scan` (187)
- total `197`

## Runtime Buffers

EnvTest already carries runtime buffers for:
- `velocity_commands`
- `push_goal_command`
- `push_actions`

This change adds:
- `pose_command`

The player computes it from the external world-frame goal using the existing navigation bridge, then writes it into the runtime buffers before observation computation.

## Error Handling

- If no explicit navigation goal is present, the runtime `pose_command` buffer is all zeros.
- Idle, walk, climb, and push-box paths still write deterministic defaults for the new `pose_command` buffer.
- Existing socket/reset/status-panel behavior stays intact except that the unified observation dimension becomes `256`.
