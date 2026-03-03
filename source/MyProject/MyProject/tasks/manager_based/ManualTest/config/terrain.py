"""Configuration for box terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
#This file is design to generate terrain

# 统一的坑洞地形尺寸参数
PIT_TERRAIN_COMMON_PARAMS = {
    "size": (10.0, 10.0),
    "border_width": 20.0,
    "num_rows": 10,
    "num_cols": 20,
    "horizontal_scale": 0.1,
    "vertical_scale": 0.005,
    "slope_threshold": 0.75,
    "use_cache": False,
}

# 简单坑洞（浅坑）
EASY_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    **PIT_TERRAIN_COMMON_PARAMS,
    sub_terrains={
        "easy_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=1.0,
            pit_depth_range=(0.1, 0.2),    # 浅坑：10-20cm
            platform_width=2.0,
            double_pit=False,
        ),
    },
)

# 中等坑洞
MEDIUM_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    **PIT_TERRAIN_COMMON_PARAMS,
    sub_terrains={
        "medium_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=1.0,
            pit_depth_range=(0.2, 0.35),   # 中等深度：20-35cm
            platform_width=2.0,
            double_pit=False,
        ),
    },
)

# 困难坑洞（深坑）
HARD_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    **PIT_TERRAIN_COMMON_PARAMS,
    sub_terrains={
        "hard_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=1.0,
            pit_depth_range=(0.35, 0.5),   # 深坑：35-50cm
            platform_width=2.0,
            double_pit=False,
        ),
    },
)

# 课程学习坑洞地形 - 渐进式难度设计
# 通过调整不同难度地形的比例来实现课程学习
CURRICULUM_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    **PIT_TERRAIN_COMMON_PARAMS,
    sub_terrains={
        # 阶段1：主要是简单坑洞
        "easy_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.60,    # 60% 简单坑
            pit_depth_range=(0.1, 0.2),
            platform_width=2.0,
            double_pit=False,
        ),
        # 阶段2：逐渐引入中等坑洞
        "medium_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.30,    # 30% 中等坑
            pit_depth_range=(0.2, 0.35),
            platform_width=2.0,
            double_pit=False,
        ),
        # 阶段3：少量困难坑洞作为挑战
        "hard_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.10,    # 10% 困难坑
            pit_depth_range=(0.35, 0.5),
            platform_width=2.0,
            double_pit=False,
        ),
    },
)


# 分层坑洞地形 - 按难度级别分层排列，配合自定义 curriculum 使用
# 地形网格的每一行对应一个难度级别
# Level 0-1: 简单坑洞 | Level 2-3: 中等坑洞 | Level 4-5: 困难坑洞
LAYERED_PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=12,  # 更多行以支持更细粒度的难度级别
    num_cols=24,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Level 0-2: 简单坑洞（浅坑，10-20cm）
        "easy_pit_shallow": terrain_gen.MeshPitTerrainCfg(
            proportion=0.25,
            pit_depth_range=(0.1, 0.15),  # 很浅
            platform_width=2.0,
            double_pit=False,
        ),
        "easy_pit_medium": terrain_gen.MeshPitTerrainCfg(
            proportion=0.25,
            pit_depth_range=(0.15, 0.2),  # 浅
            platform_width=2.0,
            double_pit=False,
        ),
        # Level 3-5: 中等坑洞（20-35cm）
        "medium_pit_light": terrain_gen.MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=(0.2, 0.27),  # 中浅
            platform_width=2.0,
            double_pit=False,
        ),
        "medium_pit_heavy": terrain_gen.MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=(0.27, 0.35),  # 中深
            platform_width=2.0,
            double_pit=False,
        ),
        # Level 6-8: 困难坑洞（深坑，35-50cm）
        "hard_pit_deep": terrain_gen.MeshPitTerrainCfg(
            proportion=0.10,
            pit_depth_range=(0.35, 0.42),  # 深
            platform_width=2.0,
            double_pit=False,
        ),
        "hard_pit_very_deep": terrain_gen.MeshPitTerrainCfg(
            proportion=0.10,
            pit_depth_range=(0.42, 0.5),  # 很深
            platform_width=2.0,
            double_pit=False,
        ),
    },
)
