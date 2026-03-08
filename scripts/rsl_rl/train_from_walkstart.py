#!/usr/bin/env python3
"""Minimal launcher for scripts/rsl_rl/train.py with fixed defaults."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DEFAULT_TASK = "Template-Velocity-Go2-Walk-BiShe-Pit-v0"
DEFAULT_EXPERIMENT = "go2_walk_bishe"
DEFAULT_RUN_NAME = "Test2"
DEFAULT_LOAD_RUN = "^bootstrap_from_rough$"
DEFAULT_CHECKPOINT_NAME = "WalkRoughNew.pt"
#文件夹形式” rsl_rl/logs/rsl_rl/DEFAULT_EXPERIMENT/DEFAULT_LOAD_RUN/DEFAULT_CHECKPOINT_NAME


def to_exact_regex(text: str) -> str:
    """Convert plain filename to exact regex; keep regex input as-is."""
    if text.startswith("^"):
        return text
    return f"^{re.escape(text)}$"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--task",
        DEFAULT_TASK,
        "--headless",
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

    print("[INFO] Running:")
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=script_dir).returncode


if __name__ == "__main__":
    raise SystemExit(main())
