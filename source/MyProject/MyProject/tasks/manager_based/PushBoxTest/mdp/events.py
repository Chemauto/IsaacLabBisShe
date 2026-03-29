from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_scaled_box_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Reset box root state while keeping scaled boxes resting on the ground."""
    box: RigidObject = env.scene[asset_cfg.name]
    root_states = box.data.default_root_state[env_ids].clone()

    cache_name = "_push_box_scales"
    if not hasattr(env, cache_name):
        scales = []
        for prim_path in box.root_physx_view.prim_paths:
            scales.append(sim_utils.resolve_prim_scale(sim_utils.get_prim_at_path(prim_path)))
        setattr(env, cache_name, torch.tensor(scales, device="cpu", dtype=torch.float32))
    scales = getattr(env, cache_name)
    env_ids_cpu = env_ids.cpu()
    scale_z = scales[env_ids_cpu, 2].to(device=env.device, dtype=root_states.dtype)

    volume_ratio = scales[env_ids_cpu].prod(dim=1, keepdim=True).to(dtype=box.data.default_mass.dtype)
    masses = box.root_physx_view.get_masses()
    masses[env_ids_cpu] = box.data.default_mass[env_ids_cpu] * volume_ratio
    box.root_physx_view.set_masses(masses, env_ids_cpu)

    inertias = box.root_physx_view.get_inertias()
    inertias[env_ids_cpu] = box.data.default_inertia[env_ids_cpu] * volume_ratio
    box.root_physx_view.set_inertias(inertias, env_ids_cpu)

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=env.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), len(pose_keys)), device=env.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    positions[:, 0] += pose_samples[:, 0]
    positions[:, 1] += pose_samples[:, 1]
    positions[:, 2] = root_states[:, 2] * scale_z + env.scene.env_origins[env_ids, 2] + pose_samples[:, 2]

    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in vel_keys], device=env.device)
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), len(vel_keys)), device=env.device
    )
    velocities = root_states[:, 7:13] + vel_samples

    box.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    box.write_root_velocity_to_sim(velocities, env_ids=env_ids)
