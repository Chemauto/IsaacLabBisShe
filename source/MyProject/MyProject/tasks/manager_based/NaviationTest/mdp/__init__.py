# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""

import torch

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_dtype, record_shape

from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .pre_trained_policy_action import *  # noqa: F401, F403


def _record_height_scan_params(output: torch.Tensor, descriptor, **_kwargs) -> None:
    descriptor.params = {"expected_dim": int(output.shape[-1])}


height_scan = generic_io_descriptor(
    dtype=torch.float32,
    observation_type="HeightScan",
    description="Height scan from heightmap around the robot.",
    on_inspect=[record_shape, record_dtype, _record_height_scan_params],
)(height_scan)
