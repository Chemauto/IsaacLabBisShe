"""High-level action post-processing for EnvTest skill replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BaseActionProcessor:
    """Common interface for skill-specific action post-processing."""

    def process(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return raw_actions


@dataclass(frozen=True)
class ClampActionProcessor(BaseActionProcessor):
    """Clamp actions to a fixed scalar range."""

    min_value: float
    max_value: float

    def process(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(raw_actions, min=self.min_value, max=self.max_value)


@dataclass(frozen=True)
class ScaledClampActionProcessor(BaseActionProcessor):
    """Scale actions first, then clamp each dimension to a dedicated range."""

    scale: tuple[float, ...]
    clip: tuple[tuple[float, float], ...]

    def process(self, raw_actions: torch.Tensor) -> torch.Tensor:
        action_scale = raw_actions.new_tensor(self.scale).view(1, -1)
        action_clip = raw_actions.new_tensor(self.clip)
        action_clip_min = action_clip[:, 0].view(1, -1)
        action_clip_max = action_clip[:, 1].view(1, -1)
        processed_actions = raw_actions * action_scale
        return torch.maximum(torch.minimum(processed_actions, action_clip_max), action_clip_min)


class PushBoxActionProcessor(ScaledClampActionProcessor):
    """Match `PushBoxTest` high-level action scaling and clipping."""


class NavigationActionProcessor(ClampActionProcessor):
    """Match `NavigationTest` high-level action clipping."""


class NavClimbActionProcessor(ScaledClampActionProcessor):
    """Match `NaviationClimbEnvCfg` high-level action clipping."""


PUSH_BOX_ACTION_PROCESSOR = PushBoxActionProcessor(
    scale=(1.0, 1.0, 1.0),
    clip=((-0.5, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
)
NAVIGATION_ACTION_PROCESSOR = NavigationActionProcessor(min_value=-1.0, max_value=1.0)
NAV_CLIMB_ACTION_PROCESSOR = NavClimbActionProcessor(
    scale=(1.0, 1.0, 1.0),
    clip=((-0.4, 1.0), (-0.4, 0.4), (-0.4, 0.4)),
)


def process_push_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for push-box action post-processing."""

    return PUSH_BOX_ACTION_PROCESSOR.process(raw_actions)


def process_navigation_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for navigation action post-processing."""

    return NAVIGATION_ACTION_PROCESSOR.process(raw_actions)


def process_nav_climb_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for nav-climb action post-processing."""

    return NAV_CLIMB_ACTION_PROCESSOR.process(raw_actions)


__all__ = [
    "BaseActionProcessor",
    "ClampActionProcessor",
    "NAVIGATION_ACTION_PROCESSOR",
    "NAV_CLIMB_ACTION_PROCESSOR",
    "NavClimbActionProcessor",
    "NavigationActionProcessor",
    "PUSH_BOX_ACTION_PROCESSOR",
    "PushBoxActionProcessor",
    "ScaledClampActionProcessor",
    "process_nav_climb_actions",
    "process_navigation_actions",
    "process_push_actions",
]
