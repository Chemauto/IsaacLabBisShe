from __future__ import annotations

import os


def _write_flag(file_path: str, value: str):
    """写回控制标志文件。"""

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(value.strip() + "\n")


def consume_one_shot_flag(
    file_path: str,
    true_tokens: tuple[str, ...] = ("1", "true", "on", "reset"),
    clear_value: str = "0",
) -> bool:
    """消费一次性控制标志；命中后自动清回默认值。"""

    if not file_path or not os.path.isfile(file_path):
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            token = file.read().strip().lower()
    except OSError:
        return False

    if token not in true_tokens:
        return False

    _write_flag(file_path, clear_value)
    return True
