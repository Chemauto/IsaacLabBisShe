from __future__ import annotations

import os


def _write_flag(file_path: str, value: str):
    """Write back one-shot control flags."""

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(value.strip() + "\n")


def consume_one_shot_value(
    file_path: str,
    accepted_tokens: tuple[str, ...],
    clear_value: str = "0",
) -> str | None:
    """Consume one-shot control text and clear the file on success."""

    if not file_path or not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            token = file.read().strip().lower()
    except OSError:
        return None

    if token not in accepted_tokens:
        return None

    _write_flag(file_path, clear_value)
    return token


def consume_one_shot_flag(
    file_path: str,
    true_tokens: tuple[str, ...] = ("1", "true", "on", "reset"),
    clear_value: str = "0",
) -> bool:
    """Consume one-shot control flags as a boolean."""

    return consume_one_shot_value(file_path, accepted_tokens=true_tokens, clear_value=clear_value) is not None

