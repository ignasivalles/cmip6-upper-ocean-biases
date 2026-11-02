"""Dominance analysis, bootstrap confidence intervals and significance."""

from itertools import combinations
from math import comb

import numpy as np
from scipy import stats


def r_squared(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return 1 - resid @ resid / np.sum((y - y.mean()) ** 2)


def dominance(df, target, predictors):
    """General dominance weights (Azen & Budescu, 2003).

    For each predictor, the increase in R^2 when it is added to a subset of
    the other predictors, averaged within each subset size and then across
    sizes. The weights sum to the full-model R^2.
    Returns (weights dict, total R^2).
    """
    sub = df[list(predictors) + [target]].dropna()
    y = sub[target].to_numpy(float)
    k = len(predictors)

    r2 = {(): 0.0}
    for size in range(1, k + 1):
        for s in combinations(range(k), size):
            r2[s] = r_squared(sub.iloc[:, list(s)].to_numpy(float), y)

    weights = {}
    for i, p in enumerate(predictors):
        others = [j for j in range(k) if j != i]
        level_means = []
        for size in range(k):
            gains = [r2[tuple(sorted(s + (i,)))] - r2[s] for s in combinations(others, size)]
            level_means.append(np.mean(gains))
        weights[p] = float(np.mean(level_means))
    return weights, r2[tuple(range(k))]


def bootstrap_dominance(df, target, predictors, n_boot=1000, seed=42, ci=(2.5, 97.5),
                        min_models=6):
    """Percentile CIs of dominance weights and total R^2, resampling models."""
    sub = df[list(predictors) + [target]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draws = {p: [] for p in list(predictors) + ["total_r2"]}
    for _ in range(n_boot):
        boot = sub.iloc[rng.integers(0, len(sub), len(sub))]
        if boot.drop_duplicates().shape[0] < min_models:
            continue
        w, total = dominance(boot, target, predictors)
        for p in predictors:
            draws[p].append(w[p])
        draws["total_r2"].append(total)
    return {k: tuple(np.percentile(v, ci)) for k, v in draws.items()}


def critical_r2(n, k=3, alpha=0.05):
    """R^2 above which a k-predictor regression on n samples is significant (F-test)."""
    df2 = n - k - 1
    if df2 < 1:
        return np.nan
    f = stats.f.ppf(1 - alpha, k, df2)
    return f * k / (df2 + f * k)
