# EnvTest HM3D Scene Loading Design

**Date:** 2026-04-15

**Goal:** Extend `EnvTest` so it can optionally load a converted HM3D USD scene by name while preserving the existing `scene_id=0~4` corridor presets.

## Context

`EnvTest` is currently a pure scene environment composed from hand-authored corridor geometry in
`env_test_env_cfg.py`. Scene selection is driven only by `scene_id=0~4`, which maps to obstacle
layouts in `config/layout.py`. The repository now also has a reusable `scripts/convert_hm3d_glb_to_usd.py`
script that outputs HM3D USD assets under `data/hm3d_usd/<scene_name>/<scene_name>.usd`.

The user wants to inspect HM3D scenes through the existing `EnvTest` runtime path instead of creating
a separate task. The requested behavior is:

- keep existing `scene_id=0~4` unchanged
- add `hm3d_scene_name`
- when `hm3d_scene_name` is set, load the matching USD asset directly
- also allow configuring the robot spawn pose for HM3D scenes

## Requirements

- Preserve existing `scene_id=0~4` behavior with no regressions.
- Add optional HM3D mode to the existing `EnvTest` config and scripts.
- HM3D mode must load:
  - `/home/robot/work/IsaacLabBisShe/data/hm3d_usd/<name>/<name>.usd`
- HM3D mode must provide robot spawn configuration:
  - `hm3d_robot_pos`
  - `hm3d_robot_yaw`
- HM3D mode must disable corridor walls, obstacles, and box assets.
- HM3D mode must let scene sensors see the HM3D geometry.
- Reuse the existing `Template-EnvTest-Go2(-Play)-v0` task registration.

## Approaches Considered

### Approach A: Replace one existing `scene_id` with HM3D mode

Use one `scene_id` value as a special HM3D scene.

**Pros**
- Minimal CLI changes

**Cons**
- Destroys one existing fixed case
- Confuses the meaning of `scene_id`
- Makes HM3D scene selection awkward

### Approach B: Add optional HM3D config on top of existing EnvTest

Keep `scene_id=0~4`, and add:

- `hm3d_scene_name`
- `hm3d_robot_pos`
- `hm3d_robot_yaw`

When `hm3d_scene_name` is set, bypass corridor assets and load HM3D USD.

**Pros**
- Keeps all old workflows intact
- Minimal mental overhead
- Easy to debug and extend later

**Cons**
- Adds a conditional path inside scene construction

### Approach C: Create a separate HM3D-only EnvTest variant

Register a new task just for HM3D viewing.

**Pros**
- Clean separation

**Cons**
- Duplicates maintenance surface
- User now has two scene-viewing entry points
- Unnecessary for the current requirement

**Recommendation:** Approach B.

## Proposed Configuration

Add the following fields to `LocomotionEnvTestEnvCfg`:

- `hm3d_scene_name: str | None = None`
- `hm3d_robot_pos: tuple[float, float, float] = (0.0, 0.0, 0.35)`
- `hm3d_robot_yaw: float = 0.0`

Behavior:

- if `hm3d_scene_name is None`:
  - existing `scene_id=0~4` corridor logic remains unchanged
- if `hm3d_scene_name` is not `None`:
  - load HM3D scene USD from `data/hm3d_usd`
  - disable corridor walls/obstacles/box
  - place robot using HM3D-specific spawn pose

## Scene Construction Design

`build_scene_cfg(...)` currently accepts only `scene_id`. It should be extended to also accept:

- `hm3d_scene_name`
- `hm3d_robot_pos`
- `hm3d_robot_yaw`

Implementation flow:

1. Create a fresh `MySceneCfg`
2. If HM3D mode is off:
   - apply current `scene_id` layout pruning
3. If HM3D mode is on:
   - attach a static USD asset at a stable prim path such as `/World/HM3DScene`
   - disable corridor assets by setting them to `None`
   - update robot initial root pose
   - update `height_scanner.mesh_prim_paths` to include HM3D scene prim

## Asset Loading

HM3D scene asset path:

- `REPO_ROOT / "data" / "hm3d_usd" / hm3d_scene_name / f"{hm3d_scene_name}.usd"`

The scene asset should be defined as an `AssetBaseCfg` using `sim_utils.UsdFileCfg`.

If the USD file is missing, scene construction should fail early with a clear error.

## Robot Spawn Design

In HM3D mode the robot should not use the default corridor pose.

Expose:

- `hm3d_robot_pos = (x, y, z)`
- `hm3d_robot_yaw`

Implementation should update the robot init state in `MySceneCfg` so the user can change spawn
without editing scene files.

For this first version:

- yaw-only rotation is enough
- no automatic floor finding is needed

## Sensor Changes

Current `height_scanner` only raycasts against `/World/ground`.

In HM3D mode, `mesh_prim_paths` must include the HM3D prim path as well, for example:

- `/World/ground`
- `/World/HM3DScene`

This is necessary so the robot’s scan and downstream observation code can see indoor geometry.

The front camera can stay unchanged because it is attached to the robot and does not depend on
specific scene asset names.

## Runtime Entry Points

No new task registration is needed.

Existing scripts should gain CLI passthrough arguments:

- `--hm3d_scene_name`
- `--hm3d_robot_pos x y z`
- `--hm3d_robot_yaw`

Target scripts:

- `scripts/zero_agent.py`
- `scripts/random_agent.py`
- `NewTools/envtest_model_use_player.py`

When these args are present and the env config exposes matching attributes, they should overwrite
the parsed `EnvTest` config before environment creation.

## Documentation

Update `source/MyProject/MyProject/tasks/manager_based/EnvTest/README.md` to document:

- HM3D mode fields
- example commands
- the fact that `scene_id` remains active only in corridor mode

## Testing Strategy

Pure Python coverage should focus on the config and path-selection logic where possible.

Practical verification should include:

1. Existing corridor mode still opens with `scene_id=4`
2. HM3D mode opens with:
   - `--hm3d_scene_name TEEsavR23oF`
   - `--hm3d_robot_pos ...`
   - `--hm3d_robot_yaw ...`
3. HM3D scene and robot are visible in GUI
4. No corridor walls/obstacles/box are spawned in HM3D mode

## Non-Goals

- Automatic HM3D goal placement
- Navigation task authoring for HM3D
- Semantic object import from HM3D metadata
- Multiple HM3D scenes loaded at once
