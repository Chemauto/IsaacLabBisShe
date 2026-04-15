# HM3D GLB to USD Converter Design

**Date:** 2026-04-15

**Goal:** Add a reusable repository script that converts a single HM3D `.glb` scene or a directory of HM3D scene folders into USD assets usable from Isaac Lab scene configs.

## Context

The repository currently contains runtime and utility scripts under `scripts/`, but no reusable asset conversion script for external scene meshes. The user already downloaded `hm3d-minival-glb-v0.2`, where each scene directory contains a single `.glb` file. The target workflow is a scriptable `GLB -> USD` conversion path that works with the existing Isaac Lab installation and can be reused for both scene debugging and dataset preprocessing.

## Requirements

- Support converting a single `.glb` file.
- Support converting an HM3D root directory containing multiple scene subdirectories.
- Use Isaac Lab / Isaac Sim runtime so `omni.kit.asset_converter` is available.
- Produce predictable output paths for later use in `sim_utils.UsdFileCfg(...)`.
- Keep CLI small and practical.
- Provide collision configuration suitable for large static indoor scenes.
- Fail clearly on malformed directory structure.

## Approaches Considered

### Approach A: Single-file-only converter

Implement a minimal script that only accepts one `--input_glb`.

**Pros**
- Smallest implementation surface
- Lowest parsing complexity

**Cons**
- Inconvenient for dataset preprocessing
- Encourages one-off shell loops outside the repo

### Approach B: Directory-only batch converter

Implement a batch-oriented script that scans an HM3D dataset root and converts every discovered scene.

**Pros**
- Good for preprocessing entire splits
- Consistent behavior for dataset import

**Cons**
- Annoying for single-scene iteration
- Harder to debug one scene in isolation

### Approach C: Dual-mode converter

Implement one script that accepts either `--input_glb` or `--input_dir`.

**Pros**
- Covers both debugging and preprocessing
- One implementation, one CLI, one naming rule
- Best fit for current user workflow

**Cons**
- Slightly more validation logic

**Recommendation:** Approach C.

## Interface

Script path:

- `scripts/convert_hm3d_glb_to_usd.py`

CLI:

- `--input_glb`: absolute or relative path to one `.glb`
- `--input_dir`: path to HM3D split directory, e.g. `hm3d-minival-glb-v0.2`
- `--output_dir`: root output directory, default `data/hm3d_usd`
- `--force`: force reconversion even if target USD already exists
- `--make_instanceable`: keep Isaac Lab converter instancing enabled when requested
- `--collision_mode`: `none`, `convex_hull`, or `mesh_simplification`
- AppLauncher args passthrough for headless/device/runtime compatibility

Input mode is mutually exclusive: exactly one of `--input_glb` and `--input_dir` must be provided.

## Output Layout

For an input scene:

- `.../00800-TEEsavR23oF/TEEsavR23oF.glb`

The generated USD goes to:

- `data/hm3d_usd/TEEsavR23oF/TEEsavR23oF.usd`

This keeps the output path stable and easy to reference from Isaac Lab configs.

## Internal Design

The script will use four small functions:

1. `parse_args()`
   - Parse CLI arguments
   - Enforce input mode rules

2. `discover_glb_files()`
   - In single-file mode, normalize one input
   - In directory mode, scan immediate child directories
   - Require exactly one `.glb` per scene directory

3. `build_converter_cfg()`
   - Construct `MeshConverterCfg`
   - Map collision mode into Isaac Lab schema config
   - Set output directory and file name

4. `convert_one_scene()`
   - Run conversion
   - Print source and destination
   - Return success/failure for summary reporting

## Data Flow

1. User launches the script through `isaaclab.sh -p`.
2. Script starts Isaac Sim via `AppLauncher`.
3. Script resolves input mode and scene list.
4. For each scene:
   - derive canonical scene name
   - create output directory
   - build `MeshConverterCfg`
   - run `MeshConverter`
5. Print a final summary with successful and failed scenes.
6. Close the simulation app in `finally`.

## Error Handling

- Missing input path: immediate `ValueError`
- Both `--input_glb` and `--input_dir` set: immediate `ValueError`
- Neither set: immediate `ValueError`
- Non-`.glb` single-file input: immediate `ValueError`
- Scene directory with zero or multiple `.glb` files: mark failed and continue batch processing
- Converter runtime failure: catch, report scene name and error, continue in batch mode

## Collision Strategy

Supported modes:

- `none`: no collision schema
- `convex_hull`: cheaper but poor fit for full indoor scenes
- `mesh_simplification`: recommended default for large static interiors

Default:

- `mesh_simplification`

## Testing Strategy

Two layers:

1. Pure Python tests for logic that does not require Isaac Sim:
   - input mode validation
   - HM3D directory scanning
   - output path generation
   - collision mode mapping

2. Manual runtime verification:
   - run one real HM3D `.glb` through `isaaclab.sh -p`
   - verify that the expected `.usd` file is created

## Non-Goals

- Automatically loading the resulting USD into `EnvTest`
- Start/goal generation from HM3D scenes
- GUI import tooling
- Semantic metadata import
