from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "deploy" / "robots" / "go2_nav" / "config" / "exported"
NAVIGATION_CHECKPOINT_PATH = REPO_ROOT / "ModelBackup" / "NaviationPolicy" / "NavigationWalk.pt"
WALK_JIT_PATH = REPO_ROOT / "ModelBackup" / "TransPolicy" / "WalkFlatHighHeightTransfer.pt"

NAVIGATION_ACTIVATION = "elu"
LOW_LEVEL_OBS_DIM = 232


def make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def load_navigation_actor(device: torch.device) -> nn.Module:
    checkpoint = torch.load(NAVIGATION_CHECKPOINT_PATH, map_location=device)
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {NAVIGATION_CHECKPOINT_PATH}")

    actor_state_dict = {
        key[len("actor."):]: value
        for key, value in model_state_dict.items()
        if key.startswith("actor.")
    }
    if not actor_state_dict:
        raise KeyError(f"Checkpoint does not contain actor weights: {NAVIGATION_CHECKPOINT_PATH}")

    linear_layer_indices = sorted({int(key.split(".")[0]) for key in actor_state_dict if key.endswith(".weight")})
    if not linear_layer_indices:
        raise KeyError(f"Checkpoint actor is missing weight tensors: {NAVIGATION_CHECKPOINT_PATH}")

    layer_dims = [actor_state_dict[f"{linear_layer_indices[0]}.weight"].shape[1]]
    layer_dims.extend(actor_state_dict[f"{layer_index}.weight"].shape[0] for layer_index in linear_layer_indices)

    layers: list[nn.Module] = []
    for layer_index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if layer_index < len(layer_dims) - 2:
            layers.append(make_activation(NAVIGATION_ACTIVATION))

    actor = nn.Sequential(*layers)
    actor.load_state_dict(actor_state_dict)
    actor.to(device)
    actor.eval()
    return actor


def export_model(model: nn.Module, dummy_obs: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_obs,
        output_path,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
    )


def main() -> None:
    device = torch.device("cpu")

    navigation_actor = load_navigation_actor(device)
    walk_actor = torch.jit.load(WALK_JIT_PATH, map_location=device).eval()

    export_model(
        navigation_actor,
        torch.zeros(1, navigation_actor[0].in_features, dtype=torch.float32),
        OUTPUT_DIR / "navigation_high_policy.onnx",
    )
    export_model(
        walk_actor,
        torch.zeros(1, LOW_LEVEL_OBS_DIM, dtype=torch.float32),
        OUTPUT_DIR / "walk_low_level_policy.onnx",
    )

    print(f"Exported navigation policies to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
