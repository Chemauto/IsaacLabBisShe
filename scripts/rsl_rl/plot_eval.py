#!/usr/bin/env python3

"""Plot Isaac Lab evaluation CSV outputs into TensorBoard-like static figures."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


PREFERRED_METRICS = [
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Episode_Termination/time_out",
    "Episode_Termination/base_contact",
]


def _resolve_csv_path(input_path: str) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        path = path / "episodes.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    return path


def _parse_optional_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _load_records(csv_path: Path) -> tuple[list[dict[str, float]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                parsed_value = _parse_optional_float(value)
                if parsed_value is not None:
                    parsed[key] = parsed_value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No evaluation rows found in {csv_path}")
    return rows, fieldnames


def _moving_average(values: list[float], window: int) -> list[float]:
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = values[start : index + 1]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed


def _sanitize_filename(metric_name: str) -> str:
    return metric_name.replace("/", "__")


def _choose_metrics(fieldnames: list[str]) -> list[str]:
    excluded = {"step", "num_resets"}
    available = [name for name in fieldnames if name and name not in excluded]
    preferred = [name for name in PREFERRED_METRICS if name in available]
    return preferred or available


def _plot_metric(ax, x_values: list[int], y_values: list[float], metric_name: str, smooth_window: int) -> None:
    ax.plot(x_values, y_values, color="#9ca3af", linewidth=1.0, alpha=0.6, label="raw")
    if smooth_window > 1 and len(y_values) > 1:
        ax.plot(
            x_values,
            _moving_average(y_values, smooth_window),
            color="#1d4ed8",
            linewidth=2.0,
            label=f"ma{smooth_window}",
        )
    else:
        ax.plot(x_values, y_values, color="#1d4ed8", linewidth=2.0, label="value")
    ax.set_title(metric_name)
    ax.set_xlabel("reset event")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def _write_overview_plot(
    records: list[dict[str, float]], metrics: list[str], output_dir: Path, smooth_window: int, max_metrics: int
) -> None:
    chosen_metrics = metrics[:max_metrics]
    cols = min(2, len(chosen_metrics))
    rows = math.ceil(len(chosen_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    x_values = list(range(1, len(records) + 1))

    for axis, metric_name in zip(axes.flatten(), chosen_metrics, strict=False):
        y_values = [record.get(metric_name, float("nan")) for record in records]
        _plot_metric(axis, x_values, y_values, metric_name, smooth_window)

    for axis in axes.flatten()[len(chosen_metrics) :]:
        axis.axis("off")

    fig.suptitle("Evaluation Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_metric_plots(records: list[dict[str, float]], metrics: list[str], output_dir: Path, smooth_window: int) -> None:
    x_values = list(range(1, len(records) + 1))
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for metric_name in metrics:
        y_values = [record.get(metric_name, float("nan")) for record in records]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_metric(ax, x_values, y_values, metric_name, smooth_window)
        fig.tight_layout()
        fig.savefig(metrics_dir / f"{_sanitize_filename(metric_name)}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Isaac Lab evaluation CSV outputs.")
    parser.add_argument("input_path", help="Evaluation directory or the episodes.csv file path.")
    parser.add_argument("--output_dir", default=None, help="Directory to store generated figures.")
    parser.add_argument("--smooth_window", type=int, default=5, help="Moving-average window for plotted curves.")
    parser.add_argument(
        "--max_overview_metrics",
        type=int,
        default=4,
        help="Maximum number of metrics shown in the overview figure.",
    )
    args = parser.parse_args()

    csv_path = _resolve_csv_path(args.input_path)
    records, fieldnames = _load_records(csv_path)
    metrics = _choose_metrics(fieldnames)

    if not metrics:
        raise ValueError(f"No numeric metrics found in {csv_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_overview_plot(records, metrics, output_dir, max(1, args.smooth_window), args.max_overview_metrics)
    _write_metric_plots(records, metrics, output_dir, max(1, args.smooth_window))

    print(f"[INFO] Wrote overview plot: {output_dir / 'overview.png'}")
    print(f"[INFO] Wrote per-metric plots under: {output_dir / 'metrics'}")


if __name__ == "__main__":
    main()
