"""Observation slicing and alignment helpers for EnvTest skill replay."""

from __future__ import annotations

import math

import torch

from MyProject.tasks.manager_based.EnvTest.observation_schema import (
    NAVIGATION_HIGH_LEVEL_OBS_DIM,
    UNIFIED_POLICY_TERM_DIMS,
)

from .observations import height_scan_without_box
from .skill_specs import (
    CLIMB_LOW_LEVEL_OBS_TERMS,
    NAVIGATION_HIGH_LEVEL_OBS_TERMS,
    PUSH_HIGH_LEVEL_OBS_TERMS,
    WALK_LOW_LEVEL_OBS_TERMS,
)


def build_obs_slices(env, group_name: str = "policy") -> dict[str, slice]:
    """Map each unified observation term to its flat slice."""

    obs_manager = env.observation_manager
    term_names = obs_manager.active_terms[group_name]
    term_dims = obs_manager.group_obs_term_dim[group_name]

    slices: dict[str, slice] = {}
    start = 0
    for term_name, term_dim in zip(term_names, term_dims):
        flat_dim = math.prod(term_dim)
        slices[term_name] = slice(start, start + flat_dim)
        start += flat_dim
    return slices


def build_local_obs_slices(term_names: tuple[str, ...], term_slices: dict[str, slice]) -> dict[str, slice]:
    """Rebuild term slices after extracting a local observation vector."""

    local_slices: dict[str, slice] = {}
    start = 0
    for term_name in term_names:
        term_slice = term_slices[term_name]
        term_dim = term_slice.stop - term_slice.start
        local_slices[term_name] = slice(start, start + term_dim)
        start += term_dim
    return local_slices


def slice_observation(unified_obs: torch.Tensor, term_slices: dict[str, slice], term_names: tuple[str, ...]) -> torch.Tensor:
    """Concatenate selected terms into the observation expected by one skill."""

    parts = [unified_obs[:, term_slices[term_name]] for term_name in term_names]
    return torch.cat(parts, dim=-1)


def validate_required_terms(term_slices: dict[str, slice]) -> None:
    """Ensure EnvTest unified observation covers all required skill terms."""

    required_terms = (
        set(WALK_LOW_LEVEL_OBS_TERMS)
        | set(CLIMB_LOW_LEVEL_OBS_TERMS)
        | set(PUSH_HIGH_LEVEL_OBS_TERMS)
        | set(NAVIGATION_HIGH_LEVEL_OBS_TERMS)
    )
    missing_terms = sorted(required_terms.difference(term_slices))
    if missing_terms:
        raise RuntimeError(f"EnvTest unified observation is missing terms: {missing_terms}")


def print_obs_layout(term_slices: dict[str, slice]) -> None:
    """Print unified observation layout for debugging."""

    print("[INFO] EnvTest unified observation layout:")
    for term_name, term_slice in term_slices.items():
        print(f"  - {term_name:<24} -> [{term_slice.start:>3}, {term_slice.stop:>3})")


def align_low_level_obs_to_play(
    policy_obs: torch.Tensor,
    term_slices: dict[str, slice],
    term_names: tuple[str, ...],
) -> torch.Tensor:
    """Apply Play-mode clipping for walk/climb low-level observations."""

    aligned_obs = policy_obs.clone()
    local_slices = build_local_obs_slices(term_names, term_slices)
    aligned_obs[:, local_slices["height_scan"]] = torch.clamp(
        aligned_obs[:, local_slices["height_scan"]], min=-1.0, max=1.0
    )
    return aligned_obs


def align_low_level_obs_to_training(
    env,
    policy_obs: torch.Tensor,
    term_slices: dict[str, slice],
    term_names: tuple[str, ...],
    last_actions: torch.Tensor,
    include_support_box: bool = True,
) -> torch.Tensor:
    """Match the ObservationManager behavior used when training low-level policies."""

    aligned_obs = policy_obs.clone()
    local_slices = build_local_obs_slices(term_names, term_slices)

    if hasattr(env, "episode_length_buf"):
        last_actions[env.episode_length_buf == 0, :] = 0.0

    aligned_obs[:, local_slices["actions"]] = last_actions
    if include_support_box:
        aligned_obs[:, local_slices["height_scan"]] = policy_obs[:, local_slices["height_scan"]]
    else:
        aligned_obs[:, local_slices["height_scan"]] = height_scan_without_box(env)

    if "base_lin_vel" in local_slices:
        aligned_obs[:, local_slices["base_lin_vel"]] += torch.empty_like(
            aligned_obs[:, local_slices["base_lin_vel"]]
        ).uniform_(-0.1, 0.1)
    if "base_ang_vel" in local_slices:
        aligned_obs[:, local_slices["base_ang_vel"]] += torch.empty_like(
            aligned_obs[:, local_slices["base_ang_vel"]]
        ).uniform_(-0.2, 0.2)
    if "projected_gravity" in local_slices:
        aligned_obs[:, local_slices["projected_gravity"]] += torch.empty_like(
            aligned_obs[:, local_slices["projected_gravity"]]
        ).uniform_(-0.05, 0.05)
    if "joint_pos" in local_slices:
        aligned_obs[:, local_slices["joint_pos"]] += torch.empty_like(
            aligned_obs[:, local_slices["joint_pos"]]
        ).uniform_(-0.01, 0.01)
    if "joint_vel" in local_slices:
        aligned_obs[:, local_slices["joint_vel"]] += torch.empty_like(
            aligned_obs[:, local_slices["joint_vel"]]
        ).uniform_(-1.5, 1.5)

    aligned_obs[:, local_slices["height_scan"]] += torch.empty_like(
        aligned_obs[:, local_slices["height_scan"]]
    ).uniform_(-0.1, 0.1)
    aligned_obs[:, local_slices["height_scan"]] = torch.clamp(
        aligned_obs[:, local_slices["height_scan"]], min=-1.0, max=1.0
    )
    return aligned_obs


def align_push_low_level_obs_to_training(
    env,
    policy_obs: torch.Tensor,
    term_slices: dict[str, slice],
    push_low_level_last_actions: torch.Tensor,
) -> torch.Tensor:
    """Push low-level branch reuses walk obs, but masks support_box in height scan."""

    return align_low_level_obs_to_training(
        env,
        policy_obs,
        term_slices,
        WALK_LOW_LEVEL_OBS_TERMS,
        push_low_level_last_actions,
        include_support_box=False,
    )


def align_navigation_high_level_obs_to_play(policy_obs: torch.Tensor) -> torch.Tensor:
    """Play-time clipping for navigation high-level observations."""

    aligned_obs = policy_obs.clone()
    local_slices = {}
    start = 0
    for term_name in NAVIGATION_HIGH_LEVEL_OBS_TERMS:
        term_dim = UNIFIED_POLICY_TERM_DIMS[term_name]
        local_slices[term_name] = slice(start, start + term_dim)
        start += term_dim
    aligned_obs[:, local_slices["height_scan"]] = torch.clamp(aligned_obs[:, local_slices["height_scan"]], -1.0, 1.0)
    return aligned_obs


__all__ = [
    "align_low_level_obs_to_play",
    "align_low_level_obs_to_training",
    "align_navigation_high_level_obs_to_play",
    "align_push_low_level_obs_to_training",
    "build_local_obs_slices",
    "build_obs_slices",
    "print_obs_layout",
    "slice_observation",
    "validate_required_terms",
]
