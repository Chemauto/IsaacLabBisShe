# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NavigationTest uses the built-in Isaac Lab observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_dtype, record_shape

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@generic_io_descriptor(
    observation_type="Action",
    description="Processed high-level action after scaling and clipping.",
    on_inspect=[record_shape, record_dtype],
)
def processed_last_action(
    env: ManagerBasedRLEnv,
    action_name: str = "pre_trained_policy_action",
) -> torch.Tensor:
    """Return the processed high-level action after scaling and clipping."""

    action_term = env.action_manager.get_term(action_name)
    return action_term.processed_actions
