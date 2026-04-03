# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NavigationTest uses the built-in Isaac Lab observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def processed_last_action(
    env: ManagerBasedRLEnv,
    action_name: str = "pre_trained_policy_action",
) -> torch.Tensor:
    """Return the processed high-level action after scaling and clipping."""

    action_term = env.action_manager.get_term(action_name)
    return action_term.processed_actions
