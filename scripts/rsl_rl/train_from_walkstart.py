#!/usr/bin/env python3
"""Minimal launcher for scripts/rsl_rl/train.py with fixed defaults."""

from __future__ import annotations

import re
import subprocess
import sys
import os
from pathlib import Path

DEFAULT_TASK = "Template-Velocity-Go2-Walk-BiShe-Pit-v0"
DEFAULT_EXPERIMENT = "go2_walk_bishe"
DEFAULT_RUN_NAME = "42"
DEFAULT_LOAD_RUN = "^test$"
DEFAULT_CHECKPOINT_NAME = "model_3000.pt"
DEFAULT_LOAD_WEIGHTS_ONLY = False
DEFAULT_DEVICE = "cuda:0"
#文件夹形式” rsl_rl/logs/rsl_rl/DEFAULT_EXPERIMENT/DEFAULT_LOAD_RUN/DEFAULT_CHECKPOINT_NAME


def to_exact_regex(text: str) -> str:
    """Convert plain filename to exact regex; keep regex input as-is."""
    if text.startswith("^"):
        return text
    return f"^{re.escape(text)}$"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"
    tmp_dir = script_dir.parents[1] / ".isaaclab_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_script),
        "--task",
        DEFAULT_TASK,
        "--headless",
        "--device",
        DEFAULT_DEVICE,
        "--resume",
        "--experiment_name",
        DEFAULT_EXPERIMENT,
        "--load_run",
        DEFAULT_LOAD_RUN,
        "--checkpoint",
        to_exact_regex(DEFAULT_CHECKPOINT_NAME),
        "--run_name",
        DEFAULT_RUN_NAME,
    ]
    if DEFAULT_LOAD_WEIGHTS_ONLY:
        cmd.append("--load_weights_only")

    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)

    print("[INFO] Running:")
    print(" ".join(cmd))
    print(f"[INFO] IsaacLab temp dir: {tmp_dir}")
    return subprocess.run(cmd, cwd=script_dir, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
