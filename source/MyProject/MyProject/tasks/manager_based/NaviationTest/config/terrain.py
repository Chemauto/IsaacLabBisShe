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


def _biased_sample(
    rng: np.random.Generator,
    value_range: tuple[float, float],
    difficulty: float,
    jitter: float = 0.2,
) -> float:
    """Sample inside a range while keeping higher difficulty tiles slightly harder on average."""

    alpha = np.clip(difficulty + rng.uniform(-jitter, jitter), 0.0, 1.0)
    return value_range[0] + alpha * (value_range[1] - value_range[0])


def _tile_rng(difficulty: float, cfg: "TwoPitTerrainCfg") -> np.random.Generator:
    """Create a deterministic RNG for one terrain tile."""

    seed = getattr(cfg, "seed", None)
    base_seed = 0 if seed is None else int(seed)
    difficulty_seed = int(np.clip(difficulty, 0.0, 1.0) * 1_000_000)
    mixed_seed = (base_seed * 747796405 + difficulty_seed * 2891336453) % (2**32)
    return np.random.default_rng(mixed_seed)


def _edge_attached_center_y(side: float, size_y: float, corridor_half_width: float) -> float:
    """Place an obstacle against the selected side wall while keeping the opposite side open."""

    return side * (corridor_half_width - 0.5 * size_y)


def two_pit_terrain(
    difficulty: float,
    cfg: "TwoPitTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a slalom-like terrain with two randomized raised obstacles."""

    rng = _tile_rng(difficulty, cfg)
    obstacle_height = _biased_sample(rng, cfg.obstacle_height_range, difficulty, jitter=0.1)
    terrain_height = 1.0
    origin = np.array([cfg.spawn_origin_x, 0.5 * cfg.size[1], 0.0], dtype=np.float32)
    corridor_half_width = 0.5 * cfg.navigation_corridor_width

    max_size_y = 2.0 * corridor_half_width - cfg.min_open_gap
    if max_size_y <= 0.0:
        raise ValueError("TwoPitTerrainCfg requires navigation_corridor_width > min_open_gap.")

    pit_1_size_x = _biased_sample(rng, cfg.pit_1_size_x_range, difficulty)
    pit_1_size_y = _biased_sample(
        rng,
        (cfg.pit_1_size_y_range[0], min(cfg.pit_1_size_y_range[1], max_size_y)),
        difficulty,
    )
    pit_2_size_x = _biased_sample(rng, cfg.pit_2_size_x_range, difficulty)
    pit_2_size_y = _biased_sample(
        rng,
        (cfg.pit_2_size_y_range[0], min(cfg.pit_2_size_y_range[1], max_size_y)),
        difficulty,
    )

    pit_1_x = _biased_sample(rng, cfg.pit_1_x_range, difficulty)
    pit_2_x = _biased_sample(rng, cfg.pit_2_x_range, difficulty)
    min_pit_2_x = pit_1_x + 0.5 * (pit_1_size_x + pit_2_size_x) + cfg.min_pit_gap_x
    pit_2_x = max(pit_2_x, min_pit_2_x)
    max_pit_2_x = cfg.size[0] - cfg.spawn_origin_x - 0.5 * pit_2_size_x - cfg.rear_margin
    pit_2_x = min(pit_2_x, max_pit_2_x)

    first_side = rng.choice(np.array([-1.0, 1.0]))
    second_side = -first_side if rng.uniform() < cfg.alternate_sides_probability else first_side

    obstacle_specs = (
        (
            (pit_1_x, _edge_attached_center_y(first_side, pit_1_size_y, corridor_half_width)),
            (pit_1_size_x, pit_1_size_y),
        ),
        (
            (pit_2_x, _edge_attached_center_y(second_side, pit_2_size_y, corridor_half_width)),
            (pit_2_size_x, pit_2_size_y),
        ),
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
    """Configuration for a terrain that contains two randomized raised obstacles."""

    function = two_pit_terrain

    obstacle_height_range: tuple[float, float] = (0.35, 0.35)
    """Height range of the two terrain obstacles."""

    spawn_origin_x: float = 1.0
    """Spawn origin along x inside the sub-terrain."""

    navigation_corridor_width: float = 3.2
    """Usable central corridor width around y=0 where obstacles are placed."""

    min_open_gap: float = 0.8
    """Minimum free lateral gap left on the opposite side of an obstacle."""

    rear_margin: float = 1.0
    """Keep the second obstacle away from the terrain rear border."""

    pit_1_x_range: tuple[float, float] = (1.3, 2.0)
    """Forward position range of the first obstacle relative to the spawn origin."""

    pit_1_size_x_range: tuple[float, float] = (0.45, 0.75)
    """Longitudinal size range of the first obstacle."""

    pit_1_size_y_range: tuple[float, float] = (1.2, 2.2)
    """Lateral size range of the first obstacle."""

    pit_2_x_range: tuple[float, float] = (3.0, 4.5)
    """Forward position range of the second obstacle relative to the spawn origin."""

    pit_2_size_x_range: tuple[float, float] = (0.45, 0.85)
    """Longitudinal size range of the second obstacle."""

    pit_2_size_y_range: tuple[float, float] = (1.2, 2.3)
    """Lateral size range of the second obstacle."""

    min_pit_gap_x: float = 0.7
    """Minimum longitudinal separation between the two obstacles."""

    alternate_sides_probability: float = 0.7
    """Probability that the second obstacle is placed on the opposite side."""


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
            obstacle_height_range=(0.32, 0.48),
            spawn_origin_x=1.0,
            navigation_corridor_width=3.2,
            min_open_gap=0.8,
            rear_margin=1.0,
            pit_1_x_range=(1.3, 2.0),
            pit_1_size_x_range=(0.45, 0.75),
            pit_1_size_y_range=(1.2, 2.2),
            pit_2_x_range=(3.0, 4.5),
            pit_2_size_x_range=(0.45, 0.85),
            pit_2_size_y_range=(1.2, 2.3),
            min_pit_gap_x=0.7,
            alternate_sides_probability=0.7,
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



Climb_TERRAINS_CFG = TerrainGeneratorCfg(
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
            box_height_range=(0.30, 0.30),
            # 顶面尽量做宽，减少机器人从两侧绕开的空间。
            platform_width=1.5,
            # 论文语义更接近“单高台”，因此不使用双层箱体。
            double_box=False,
            size=(8.0, 8.0),
        ),
    },
)
