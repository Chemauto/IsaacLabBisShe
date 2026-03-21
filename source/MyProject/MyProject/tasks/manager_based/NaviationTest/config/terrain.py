"""Configuration for navigation and locomotion terrains."""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _make_box(
    size: tuple[float, float, float],
    center: tuple[float, float, float],
) -> trimesh.Trimesh:
    """Create a box mesh centered at the requested position."""

    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))


def two_pit_terrain(
    difficulty: float,
    cfg: "TwoPitTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with two raised rectangular obstacles placed in front of the spawn origin."""

    obstacle_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )
    terrain_height = 1.0
    origin = np.array([cfg.spawn_origin_x, 0.5 * cfg.size[1], 0.0])

    obstacle_specs = (
        (cfg.pit_1_position, cfg.pit_1_size),
        (cfg.pit_2_position, cfg.pit_2_size),
    )
    meshes_list = [
        _make_box(
            (cfg.size[0], cfg.size[1], terrain_height),
            (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * terrain_height),
        )
    ]

    for (pit_x, pit_y), (pit_size_x, pit_size_y) in obstacle_specs:
        center_x = origin[0] + pit_x
        center_y = origin[1] + pit_y

        meshes_list.append(
            _make_box(
                (pit_size_x, pit_size_y, terrain_height + obstacle_height),
                (
                    center_x,
                    center_y,
                    0.5 * (obstacle_height - terrain_height),
                ),
            )
        )

    return meshes_list, origin


@configclass
class TwoPitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain that contains two raised rectangular obstacles."""

    function = two_pit_terrain

    obstacle_height_range: tuple[float, float] = (0.35, 0.35)
    """Height range of the two terrain obstacles."""

    spawn_origin_x: float = 1.0
    """Spawn origin along x inside the sub-terrain."""

    pit_1_position: tuple[float, float] = (1.5, 0.4)
    """Center of pit 1 relative to the spawn origin in the env frame."""

    pit_1_size: tuple[float, float] = (0.5, 1.6)
    """Size of pit 1 in the x/y plane."""

    pit_2_position: tuple[float, float] = (3.3, -0.4)
    """Center of pit 2 relative to the spawn origin in the env frame."""

    pit_2_size: tuple[float, float] = (0.5, 1.6)
    """Size of pit 2 in the x/y plane."""


TWO_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "two_pit": TwoPitTerrainCfg(
            proportion=1.0,
            size=(8.0, 8.0),
            obstacle_height_range=(0.35, 0.35),
            spawn_origin_x=1.0,
            pit_1_position=(1.5, 0.4),
            pit_1_size=(0.5, 1.6),
            pit_2_position=(3.3, -0.4),
            pit_2_size=(0.5, 1.6),
        ),
    },
)

#This file is design to generate terrain
# 创建一个只包含 Box 地形的配置
# 高台/高障碍攀爬地形：中心为单个抬高平台，适合训练 climb skill。
# 这里使用 MeshBoxTerrainCfg，因为它生成的是“中心单个平台”，
# 比连续楼梯或 pit 更接近论文里的 high hurdle / platform climb 语义。
HIGH_PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "high_platform": terrain_gen.MeshBoxTerrainCfg(
            proportion=1.0,
            # difficulty=0 时约 0.10m，difficulty=1 时约 0.26m。
            # 适合作为 easy -> medium -> hard 的高度课程。
            box_height_range=(1.50, 1.50),
            # 顶面尽量做宽，减少机器人从两侧绕开的空间。
            platform_width=0.8,
            # 论文语义更接近“单高台”，因此不使用双层箱体。
            double_box=False,
            size=(0.8, 0.8),
        ),
    },
)

#  障碍地形
#复杂地形
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,  # 50% 正向楼梯
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,  # 50% 反向楼梯
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
