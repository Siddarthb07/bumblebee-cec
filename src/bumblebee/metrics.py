from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = corr[mask].mean()
        avg_conf = conf[mask].mean()
        ece += mask.mean() * abs(acc - avg_conf)
    return float(ece)


def risk_coverage_curve(
    risks: Sequence[float], confidences: Sequence[float], steps: int = 20
) -> List[Tuple[float, float]]:
    """
    risks: 1 for error / bad answer, 0 for correct (refusals on OOD can be risk 0 if correct refuse)
    Sort by confidence descending; coverage = fraction answered (or retained).
    Here we treat higher confidence as more willing to answer.
    """
    if not risks:
        return [(0.0, 0.0)]
    order = np.argsort(-np.asarray(confidences))
    r = np.asarray(risks, dtype=np.float64)[order]
    curve = []
    for frac in np.linspace(0.05, 1.0, steps):
        k = max(1, int(round(frac * len(r))))
        curve.append((float(k / len(r)), float(r[:k].mean())))
    return curve


def aurc(risks: Sequence[float], confidences: Sequence[float]) -> float:
    curve = risk_coverage_curve(risks, confidences, steps=50)
    cov = [c for c, _ in curve]
    risk = [r for _, r in curve]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(risk, cov))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Optional[Tuple[float, float, float]]:
    """
    Return (mean, lo, hi) for a 1-alpha CI via percentile bootstrap.
    None if fewer than 2 values.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        if arr.size == 1:
            v = float(arr[0])
            return v, v, v
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return float(arr.mean()), lo, hi
