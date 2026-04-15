# EnvTest Lightweight Front Camera Design

**Date:** 2026-04-15

**Goal:** Reduce the runtime cost of `EnvTest` front-camera usage while preserving the existing `--enable_front_camera` CLI flag semantics.

## Context

`NewTools/envtest_model_use_player.py` exposes `--enable_front_camera` and maps it to both:

- keeping `env_cfg.scene.front_camera` enabled
- setting `args_cli.enable_cameras = args_cli.enable_front_camera`

That means the flag does two things at once:

1. it keeps the robot-mounted USD camera sensor in the environment
2. it switches Isaac Lab into camera rendering mode through `AppLauncher`

The current `EnvTest` camera configuration is comparatively heavy for a preview workflow:

- resolution: `640x480`
- outputs: `rgb` and `distance_to_image_plane`
- update period: `0.0` (every sim step)

The player only uses the RGB image and periodically writes it to PNG. It does not consume the depth image.

## Requirements

- Preserve the current meaning of `--enable_front_camera`
- Do not add a new CLI flag
- Make the default front-camera configuration lighter
- Keep RGB snapshot output working in `envtest_model_use_player.py`
- Update docs to match the lighter camera behavior

## Approaches Considered

### Approach A: Keep CLI unchanged and lower the default camera cost

Change the `EnvTest` camera config itself:

- `rgb` only
- `320x240`
- non-zero update period

**Pros**

- Keeps the user-facing interface unchanged
- Minimal code changes
- Directly addresses the unnecessary workload in the current config

**Cons**

- Changes the default camera fidelity for all `EnvTest` front-camera consumers

### Approach B: Add a second preview-only CLI flag

Keep `--enable_front_camera` as the high-cost path and add a new lighter preview flag.

**Pros**

- No behavioral change for the existing flag

**Cons**

- Adds interface complexity
- The user explicitly rejected this direction

### Approach C: Keep the current camera config and only improve failure reporting

Add clearer logging or guardrails when cameras are enabled on unsupported machines.

**Pros**

- No behavior change

**Cons**

- Does not reduce runtime cost
- Does not solve the main complaint that enabling the camera effectively stalls entry

**Recommendation:** Approach A.

## Proposed Configuration

Update `MySceneCfg.front_camera` in `EnvTest` to:

- `update_period=0.2`
- `height=240`
- `width=320`
- `data_types=["rgb"]`

Keep the camera mount pose and optical settings unchanged for now.

## Why These Defaults

- `320x240` is enough for a live preview and cuts image bandwidth substantially versus `640x480`
- `rgb` only matches the actual player usage
- `update_period=0.2` aligns with the player’s default image save interval and avoids per-step rendering

## Impacted Files

- `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env_cfg.py`
- `source/MyProject/MyProject/tasks/manager_based/EnvTest/README.md`
- `tests/test_envtest_hm3d_config.py`

## Testing Strategy

Use a small source-level regression test to confirm the intended camera defaults:

- camera resolution is `320x240`
- camera data types are `["rgb"]`
- camera update period is non-zero and matches the new lightweight default

Also run the existing focused config tests to ensure no unrelated regressions in `EnvTest`.

## Non-Goals

- Changing the `--enable_front_camera` CLI contract
- Adding a separate preview flag
- Making camera quality dynamically configurable from CLI
- Fixing system-level CUDA / renderer availability issues
