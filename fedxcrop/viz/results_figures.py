"""Figures for the training and accuracy results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedxcrop.viz.style import METHOD_COLORS, PALETTE, apply_style, save_figure


def val_accuracy_per_round(
    histories: dict[str, dict[float, list[pd.DataFrame]]],
    out_dir: str | Path,
    name: str = "fig2_val_accuracy_per_round",
    centralized_best_val: Optional[float] = None,
    alphas: Optional[list[float]] = None,
) -> list[Path]:
    """Validation accuracy per round, one panel per alpha.

    `histories` maps strategy to alpha to the per seed history frames. The line
    is the mean over seeds and the band is plus or minus one standard
    deviation, so the reader can see directly whether the gap between two
    strategies is larger than the run to run spread.
    """
    apply_style()
    alphas = alphas or sorted({a for strategy in histories.values() for a in strategy})

    fig, axes = plt.subplots(1, len(alphas), figsize=(3.2 * len(alphas), 3.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, alpha in zip(axes, alphas):
        for strategy, by_alpha in histories.items():
            frames = by_alpha.get(alpha)
            if not frames:
                continue
            rounds = frames[0]["round"].to_numpy()
            stacked = np.stack([f["val_accuracy"].to_numpy() for f in frames])
            mean = stacked.mean(axis=0)
            color = METHOD_COLORS.get(strategy, PALETTE[0])
            ax.plot(rounds, 100 * mean, label=f"{strategy} (n={len(frames)})", color=color)
            if len(frames) > 1:
                spread = stacked.std(axis=0, ddof=1)
                ax.fill_between(
                    rounds, 100 * (mean - spread), 100 * (mean + spread),
                    color=color, alpha=0.18, linewidth=0,
                )

        if centralized_best_val is not None:
            ax.axhline(
                100 * centralized_best_val, color=METHOD_COLORS["centralized"],
                linestyle="--", linewidth=1.0, label="centralized best val",
            )

        ax.set_xlabel("round")
        ax.set_title(f"alpha = {alpha:g}")

    axes[0].set_ylabel("validation accuracy (percent)")
    axes[-1].legend(loc="lower right", frameon=False)
    return save_figure(fig, out_dir, name)


def test_metric_bars(
    table: pd.DataFrame,
    out_dir: str | Path,
    name: str = "fig3_test_metrics",
    metrics: tuple[str, str] = ("accuracy", "macro_f1"),
    axis_starts_at_zero: bool = True,
) -> tuple[list[Path], dict]:
    """Final test accuracy and macro F1 with 95 percent intervals.

    The y axis starts at zero by default. The original version of this figure
    started its axis at 97.5 percent, which made differences of well under one
    percentage point look like large effects. If the axis is truncated here,
    the break is drawn and the fact is returned for the figure notes.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.0 * len(table) + 2.4, 3.4), sharex=False)
    axes = np.atleast_1d(axes)

    notes = {"axis_starts_at_zero": axis_starts_at_zero}
    labels = list(table["label"])
    positions = np.arange(len(labels))

    for ax, metric in zip(axes, metrics):
        values = 100 * table[f"{metric}_point"].to_numpy()
        low = 100 * table[f"{metric}_ci_low"].to_numpy()
        high = 100 * table[f"{metric}_ci_high"].to_numpy()
        errors = np.vstack([values - low, high - values])

        colors = [
            METHOD_COLORS.get(str(s).split("_")[0], PALETTE[0]) for s in table["strategy"]
        ]
        ax.bar(positions, values, yerr=errors, capsize=3, color=colors, width=0.7,
               error_kw={"linewidth": 1.0})

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel(f"test {metric.replace('_', ' ')} (percent)")

        if axis_starts_at_zero:
            ax.set_ylim(0, 100)
        else:
            margin = max(1.0, (high.max() - low.min()) * 0.4)
            ax.set_ylim(max(0.0, low.min() - margin), min(100.0, high.max() + margin))
            _draw_axis_break(ax)
            notes[f"{metric}_ylim"] = list(ax.get_ylim())

    return save_figure(fig, out_dir, name), notes


def _draw_axis_break(ax) -> None:
    """Mark a truncated y axis explicitly rather than letting it mislead."""
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=0.9)
    ax.plot([-0.012, 0.012], [0.006, 0.026], **kwargs)
    ax.plot([-0.012, 0.012], [0.016, 0.036], **kwargs)


def accuracy_vs_heterogeneity(
    table: pd.DataFrame,
    out_dir: str | Path,
    name: str = "fig3b_accuracy_vs_alpha",
) -> list[Path]:
    """Test accuracy against Dirichlet alpha, one line per strategy."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    for strategy, group in table[table["alpha"].notna()].groupby("strategy"):
        group = group.sort_values("alpha")
        color = METHOD_COLORS.get(str(strategy), PALETTE[0])
        values = 100 * group["accuracy_point"].to_numpy()
        low = 100 * group["accuracy_ci_low"].to_numpy()
        high = 100 * group["accuracy_ci_high"].to_numpy()
        ax.errorbar(
            group["alpha"], values,
            yerr=np.vstack([values - low, high - values]),
            marker="o", markersize=4, capsize=3, label=str(strategy), color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet alpha (lower is more heterogeneous)")
    ax.set_ylabel("test accuracy (percent)")
    ax.legend(frameon=False)
    return save_figure(fig, out_dir, name)


def training_curves(
    history: pd.DataFrame,
    out_dir: str | Path,
    name: str = "centralized_curves",
) -> list[Path]:
    """Centralized train and validation curves."""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    left.plot(history["epoch"], history["train_loss"], label="train", color=PALETTE[0])
    left.plot(history["epoch"], history["val_loss"], label="validation", color=PALETTE[1])
    left.set_xlabel("epoch")
    left.set_ylabel("loss")
    left.legend(frameon=False)

    right.plot(history["epoch"], 100 * history["train_accuracy"], label="train", color=PALETTE[0])
    right.plot(history["epoch"], 100 * history["val_accuracy"], label="validation", color=PALETTE[1])
    right.set_xlabel("epoch")
    right.set_ylabel("accuracy (percent)")
    right.legend(frameon=False)

    return save_figure(fig, out_dir, name)
