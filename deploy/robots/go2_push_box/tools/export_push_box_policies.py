from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "deploy" / "robots" / "go2_push_box" / "config" / "exported"
PUSH_CHECKPOINT_PATH = REPO_ROOT / "ModelBackup" / "PushPolicy" / "PushBox.pt"
WALK_JIT_PATH = REPO_ROOT / "ModelBackup" / "TransPolicy" / "WalkFlatLowHeightTransfer.pt"

PUSH_OBS_DIM = 19
PUSH_HIDDEN_DIMS = (512, 256, 128)
PUSH_ACTIVATION = "elu"
LOW_LEVEL_OBS_DIM = 232


def make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def load_push_actor(device: torch.device) -> nn.Module:
    checkpoint = torch.load(PUSH_CHECKPOINT_PATH, map_location=device)
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {PUSH_CHECKPOINT_PATH}")

    actor_state_dict = {
        key[len("actor."):]: value
        for key, value in model_state_dict.items()
        if key.startswith("actor.")
    }
    if not actor_state_dict:
        raise KeyError(f"Checkpoint does not contain actor weights: {PUSH_CHECKPOINT_PATH}")

    bias_layer_indices = sorted(int(key.split(".")[0]) for key in actor_state_dict if key.endswith(".bias"))
    if not bias_layer_indices:
        raise KeyError(f"Checkpoint actor is missing bias tensors: {PUSH_CHECKPOINT_PATH}")

    output_dim = actor_state_dict[f"{bias_layer_indices[-1]}.bias"].shape[0]
    layer_dims = [PUSH_OBS_DIM, *PUSH_HIDDEN_DIMS, output_dim]
    layers: list[nn.Module] = []
    for layer_index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if layer_index < len(layer_dims) - 2:
            layers.append(make_activation(PUSH_ACTIVATION))

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

    push_actor = load_push_actor(device)
    walk_actor = torch.jit.load(WALK_JIT_PATH, map_location=device).eval()

    export_model(
        push_actor,
        torch.zeros(1, PUSH_OBS_DIM, dtype=torch.float32),
        OUTPUT_DIR / "push_high_policy.onnx",
    )
    export_model(
        walk_actor,
        torch.zeros(1, LOW_LEVEL_OBS_DIM, dtype=torch.float32),
        OUTPUT_DIR / "walk_low_level_policy.onnx",
    )

    print(f"Exported push-box policies to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
