"""Uncertainty and significance.

On PlantVillage every configuration scores somewhere around 99 percent, so the
differences that matter are smaller than one percentage point. A difference
that size is meaningless without an interval around it and a test that accounts
for the two models having been evaluated on the same images. Everything here
exists so that a claim of the form "A beats B" can be checked rather than
asserted.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

from fedxcrop.eval.metrics import compute_metrics


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap interval for a metric, resampling test images.

    Images are resampled with replacement, which reflects the uncertainty from
    having evaluated on this particular test set rather than another draw from
    the same distribution. The point estimate is the metric on the real sample,
    not the bootstrap mean.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true, {len(y_pred)} predicted")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        values[i] = metric_fn(y_true[indices], y_pred[indices])

    alpha = 1.0 - confidence
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "bootstrap_seed": seed,
        "bootstrap_std": float(values.std(ddof=1)) if n_resamples > 1 else 0.0,
    }


def bootstrap_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    num_classes: int = 38,
    metrics: Sequence[str] = ("accuracy", "macro_f1", "macro_precision", "macro_recall"),
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap intervals for several metrics at once, sharing the resamples.

    Sharing one set of resampled index vectors across metrics keeps the
    intervals mutually consistent and costs one pass instead of several.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    draws: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample = compute_metrics(y_true[indices], y_pred[indices], num_classes=num_classes)
        for name in metrics:
            draws[name].append(sample[name])

    point = compute_metrics(y_true, y_pred, num_classes=num_classes)
    alpha = 1.0 - confidence
    out = {}
    for name in metrics:
        values = np.asarray(draws[name])
        low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = {
            "point": float(point[name]),
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_std": float(values.std(ddof=1)),
        }
    out["n_resamples"] = n_resamples
    out["confidence"] = confidence
    out["bootstrap_seed"] = seed
    out["n_correct"] = point["n_correct"]
    out["n_total"] = point["n_total"]
    return out


def mcnemar(
    y_true: Sequence,
    y_pred_a: Sequence,
    y_pred_b: Sequence,
    exact_threshold: int = 25,
) -> dict:
    """McNemar's test for two models evaluated on the same images.

    Only the discordant pairs carry information: images that model A got right
    and B got wrong (b01), and the reverse (b10). Images both models agree on,
    right or wrong, say nothing about which is better. The exact binomial test
    is used when there are few discordant pairs, where the chi square
    approximation is unreliable.
    """
    y_true = np.asarray(y_true)
    correct_a = y_true == np.asarray(y_pred_a)
    correct_b = y_true == np.asarray(y_pred_b)

    both_correct = int((correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b

    if discordant == 0:
        return {
            "test": "none",
            "statistic": 0.0,
            "p_value": 1.0,
            "only_a_correct": 0,
            "only_b_correct": 0,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "n_discordant": 0,
            "note": "the two models make identical predictions, so there is nothing to test",
        }

    if discordant < exact_threshold:
        test = "exact binomial"
        result = scipy_stats.binomtest(only_a, discordant, 0.5)
        p_value = float(result.pvalue)
        statistic = float(min(only_a, only_b))
    else:
        test = "chi square with continuity correction"
        statistic = float((abs(only_a - only_b) - 1) ** 2 / discordant)
        p_value = float(scipy_stats.chi2.sf(statistic, df=1))

    return {
        "test": test,
        "statistic": statistic,
        "p_value": p_value,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "n_discordant": discordant,
    }


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired Wilcoxon signed rank test, used for per image XAI metrics.

    Attribution scores are bounded and skewed, so a rank based paired test is
    safer than a t test. Pairs with a zero difference are dropped, which is the
    standard treatment and is reported in `n_pairs_nonzero`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} and {len(b)}")

    differences = a - b
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return {
            "test": "wilcoxon signed rank",
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": int(len(a)),
            "n_pairs_nonzero": 0,
            "median_difference": 0.0,
            "note": "all pairs are identical, so there is nothing to test",
        }

    result = scipy_stats.wilcoxon(a, b, zero_method="wilcox")
    return {
        "test": "wilcoxon signed rank",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": int(len(a)),
        "n_pairs_nonzero": int(len(nonzero)),
        "median_difference": float(np.median(differences)),
    }


def aggregate_seeds(values: Sequence[float]) -> dict:
    """Mean and standard deviation across seeds.

    The standard deviation uses the sample convention (ddof=1) and is reported
    as the spread over however many seeds were actually run, which is stated in
    `n_seeds` so a reader is never left guessing whether it came from three
    runs or thirty.
    """
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        raise ValueError("cannot aggregate an empty list of seeds")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n_seeds": int(len(array)),
        "values": [float(v) for v in array],
        "min": float(array.min()),
        "max": float(array.max()),
    }


def format_mean_std(aggregate: dict, digits: int = 2, scale: float = 100.0) -> str:
    """Render an aggregate as "mean plus or minus std" for a table cell."""
    mean = aggregate["mean"] * scale
    std = aggregate["std"] * scale
    if aggregate["n_seeds"] == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def format_ci(entry: dict, digits: int = 2, scale: float = 100.0) -> str:
    """Render a bootstrap entry as "point [low, high]" for a table cell."""
    return (
        f"{entry['point'] * scale:.{digits}f} "
        f"[{entry['ci_low'] * scale:.{digits}f}, {entry['ci_high'] * scale:.{digits}f}]"
    )
