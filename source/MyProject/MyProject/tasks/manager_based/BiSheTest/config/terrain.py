"""Terrain presets for advanced-skills training (gap / pit / obstacle)."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg

BASE_TERRAIN_KEYS = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
)
GAP_TERRAIN_KEYS = ("gap_easy", "gap_hard")
PIT_TERRAIN_KEYS = ("pit_easy", "pit_hard")
OBS_TERRAIN_KEYS = ("obstacle_low", "obstacle_high")

# Step thresholds for automatic stage switching (global env steps).
# With 48 rollout steps/iteration, these thresholds ensure all phases are visited in ~3000 iters.
AUTO_STAGE_STEPS = (0, 12_000, 36_000, 72_000, 108_000)

STAGE_PRESETS = {
    "phase0_walk": {
        "keys": BASE_TERRAIN_KEYS,
        "radius_range": (1.0, 5.0),
        "max_level_ratio": 0.25,
    },
    "phase1_base": {
        "keys": BASE_TERRAIN_KEYS,
        "radius_range": (1.0, 5.0),
        "max_level_ratio": 0.45,
    },
    "phase2_gap": {
        "keys": BASE_TERRAIN_KEYS + GAP_TERRAIN_KEYS,
        "radius_range": (1.0, 5.0),
        "max_level_ratio": 0.65,
    },
    "phase3_pit": {
        "keys": BASE_TERRAIN_KEYS + GAP_TERRAIN_KEYS + PIT_TERRAIN_KEYS,
        "radius_range": (1.0, 5.0),
        "max_level_ratio": 0.85,
    },
    "phase4_obstacle": {
        "keys": BASE_TERRAIN_KEYS + GAP_TERRAIN_KEYS + PIT_TERRAIN_KEYS + OBS_TERRAIN_KEYS,
        "radius_range": (1.0, 5.0),
        "max_level_ratio": 1.0,
    },
}

def _target_patch_sampling() -> dict[str, FlatPatchSamplingCfg]:
    return {
        "target": FlatPatchSamplingCfg(
            num_patches=64,
            patch_radius=[0.20, 0.14],
            x_range=(-5.5, 5.5),
            y_range=(-5.5, 5.5),
            z_range=(-1.2, 1.2),
            max_height_diff=0.05,
        )
    }


def make_advanced_skills_terrains_cfg() -> TerrainGeneratorCfg:
    """Build rough terrain config with dedicated gap / pit / obstacle sub-terrains."""
    return TerrainGeneratorCfg(
        size=(12.0, 12.0),
        border_width=20.0,
        num_rows=10,
        num_cols=24,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            # Base locomotion terrains.
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.14,
                step_height_range=(0.05, 0.23),
                step_width=0.30,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.14,
                step_height_range=(0.05, 0.23),
                step_width=0.30,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.12,
                grid_width=0.45,
                grid_height_range=(0.05, 0.2),
                platform_width=2.0,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.10,
                noise_range=(0.01, 0.08),
                noise_step=0.01,
                border_width=0.25,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            # Gap terrains (jump / stride).
            "gap_easy": terrain_gen.MeshGapTerrainCfg(
                proportion=0.09,
                gap_width_range=(0.08, 0.35),
                platform_width=1.8,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "gap_hard": terrain_gen.MeshGapTerrainCfg(
                proportion=0.06,
                gap_width_range=(0.25, 0.90),
                platform_width=1.4,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            # Pit terrains (climb out with contacts).
            "pit_easy": terrain_gen.MeshPitTerrainCfg(
                proportion=0.08,
                pit_depth_range=(0.08, 0.35),
                platform_width=1.8,
                double_pit=False,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "pit_hard": terrain_gen.MeshPitTerrainCfg(
                proportion=0.05,
                pit_depth_range=(0.25, 0.95),
                platform_width=1.2,
                double_pit=True,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            # Obstacle terrains (encourage detour behavior).
            "obstacle_low": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.06,
                obstacle_height_mode="choice",
                obstacle_width_range=(0.30, 0.80),
                obstacle_height_range=(0.08, 0.35),
                num_obstacles=24,
                platform_width=1.8,
                border_width=0.25,
                flat_patch_sampling=_target_patch_sampling(),
            ),
            "obstacle_high": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.06,
                obstacle_height_mode="choice",
                obstacle_width_range=(0.25, 0.75),
                obstacle_height_range=(0.25, 0.80),
                num_obstacles=36,
                platform_width=1.4,
                border_width=0.25,
                flat_patch_sampling=_target_patch_sampling(),
            ),
        },
    )
