"""Configuration for box terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
#This file is design to generate terrain
# 创建一个只包含 Box 地形的配置
BOX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),           # 地形尺寸
    border_width=20.0,         # 边界宽度
    num_rows=10,               # 行数
    num_cols=20,               # 列数
    horizontal_scale=0.1,      # 水平缩放
    vertical_scale=0.005,      # 垂直缩放
    slope_threshold=0.75,      # 坡度阈值
    use_cache=False,           # 不使用缓存
    # curriculum_cfg=TerrainGeneratorCfg.TerrainCurriculumCfg(
    #     difficulty_scales=[0.0, 1.0],  # 难度范围
    # ),
    sub_terrains={
        # 只使用 Box 地形
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            proportion=1.0,  # 100% 概率生成 Box 地形
            box_height_range=(0.1, 0.5),  # 箱子高度范围
            platform_width=2.0,  # 中心平台宽度
            double_box=True,  # 双层箱子
            size=(8.0, 8.0),  # 子地形尺寸
        ),
    },
)
"""Box terrains configuration."""

# 毕设攀爬技能专用：box 主导，混入少量楼梯
BISHE_CLIMB_BOX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.6,
            grid_width=0.45,
            grid_height_range=(0.08, 0.25),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

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
            box_height_range=(0.06, 0.34),
            # 顶面尽量做宽，减少机器人从两侧绕开的空间。
            platform_width=3.0,
            # 论文语义更接近“单高台”，因此不使用双层箱体。
            double_box=False,
            size=(8.0, 8.0),
        ),
    },
)

# 固定难度的高台评估地形。
HIGH_PLATFORM_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
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
            # 固定在较高但仍可训练/评估的高度。
            box_height_range=(0.30, 0.30),
            platform_width=3.0,
            double_box=False,
            size=(8.0, 8.0),
        ),
    },
)

HIGH_DOUBLE_PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
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
            box_height_range=(0.06, 0.34),
            # 顶面尽量做宽，减少机器人从两侧绕开的空间。
            platform_width=2.0,
            # 论文语义更接近“单高台”，因此不使用双层箱体。
            double_box=True,
            size=(8.0, 8.0),
        ),
    },
)

# 固定难度的高台评估地形。
HIGH_DOUBLE_PLATFORM_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
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
            # 固定在较高但仍可训练/评估的高度。
            box_height_range=(0.30, 0.30),
            platform_width=2.0,
            double_box=True,
            size=(8.0, 8.0),
        ),
    },
)




# 以坑洞为主的混合地形 - 用于毕业设计的低层动作训练
# 各种地形的训练
BISHE_Test_MIX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Pit-dominant mix: 40% pit + 40% stairs + 10% boxes + 10% rough
        "easy_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15,
            pit_depth_range=(0.05, 0.20),
            platform_width=2.0,
            double_pit=False,
        ),
        "medium_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=(0.20, 0.30),
            platform_width=2.0,
            double_pit=False,
        ),
        "hard_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.05,
            pit_depth_range=(0.25, 0.35),
            platform_width=2.0,
            double_pit=False,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)
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
            step_height_range=(0.05, 0.35),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,  # 50% 反向楼梯
            step_height_range=(0.05, 0.35),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
