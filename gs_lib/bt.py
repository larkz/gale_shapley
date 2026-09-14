"""
bt.py
Bradley-Terry estimation, confidence intervals, and ranking utilities.

Convention used throughout:
    For an unordered pair {b1, b2}, let (c1, c2) = _canonical(b1, b2).
    An observation x = 1 means c1 won the comparison, x = 0 means c2 won.
    This convention is uniform across PairCounts, _sample_comparison,
    observe(), and bt_mle_mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Tuple

import math


# ============================================================================
# Data structures
# ============================================================================

PairKey = Tuple[Hashable, Hashable]


def _canonical(b1: Hashable, b2: Hashable) -> PairKey:
    """Canonicalise an unordered pair by hash order (stable within a run)."""
    return (b1, b2) if hash(b1) <= hash(b2) else (b2, b1)


@dataclass
class PairCounts:
    """Accumulated comparison counts for one agent over its partner set.

    wins[(c1, c2)]  = number of comparisons in which c1 beat c2
    total[(c1, c2)] = total number of comparisons of c1 vs c2
    where (c1, c2) = _canonical(b1, b2).
    """
    wins:  Dict[PairKey, int] = field(default_factory=dict)
    total: Dict[PairKey, int] = field(default_factory=dict)

    def record(self, b1: Hashable, b2: Hashable, x: int) -> None:
        """Record one comparison.

        x = 1 means the canonical-first of (b1, b2) won.
        x = 0 means the canonical-second won.
        """
        key = _canonical(b1, b2)
        self.wins[key] = self.wins.get(key, 0) + x
        self.total[key] = self.total.get(key, 0) + 1

    def n_comparisons(self) -> int:
        return sum(self.total.values())


# ============================================================================
# Bradley-Terry MM algorithm
# ============================================================================

def bt_mle_mm(
    items: List[Hashable],
    wins: Dict[PairKey, int],
    totals: Dict[PairKey, int],
    max_iter: int = 500,
    tol: float = 1e-9,
    theta_init: Dict[Hashable, float] | None = None,
) -> Dict[Hashable, float]:
    """Compute the Bradley-Terry MLE via the MM algorithm of Hunter (2004).

    Returns a dict item -> estimated theta, all positive, normalised to sum 1.
    """
    K = len(items)
    if K == 0:
        return {}
    if K == 1:
        return {items[0]: 1.0}

    if theta_init is not None and all(i in theta_init for i in items):
        theta = {i: float(theta_init[i]) for i in items}
    else:
        theta = {i: 1.0 / K for i in items}

    w_i: Dict[Hashable, int] = {i: 0 for i in items}

    for (b1, b2), t in totals.items():
        w12 = wins.get((b1, b2), 0)
        w21 = t - w12
        w_i[b1] += w12
        w_i[b2] += w21

    for _ in range(max_iter):
        denom: Dict[Hashable, float] = {i: 0.0 for i in items}
        for (b1, b2), t in totals.items():
            s = theta[b1] + theta[b2]
            if s <= 0:
                s = 1e-12
            denom[b1] += t / s
            denom[b2] += t / s

        new_theta: Dict[Hashable, float] = {}
        for i in items:
            if denom[i] <= 0 or w_i[i] <= 0:
                new_theta[i] = max(theta[i], 1e-12)
            else:
                new_theta[i] = w_i[i] / denom[i]

        Z = sum(new_theta.values())
        if Z <= 0:
            break
        new_theta = {i: v / Z for i, v in new_theta.items()}

        delta = sum(abs(new_theta[i] - theta[i]) for i in items)
        theta = new_theta
        if delta < tol:
            break

    return theta


# ============================================================================
# Confidence intervals
# ============================================================================

def bt_ci_width(n_pair: int, t_global: int, constant: float = 0.1) -> float:
    """Half-width of the pairwise CI:
        w = sqrt( constant * log(t_global) / n_pair )
    """
    if n_pair <= 0:
        return float("inf")
    t = max(t_global, 2)
    return math.sqrt(constant * math.log(t) / n_pair)


def bt_confidence_intervals(
    items: List[Hashable],
    theta_hat: Dict[Hashable, float],
    counts: PairCounts,
    t_global: int,
    constant: float = 0.1,
) -> Dict[Tuple[Hashable, Hashable], Tuple[float, float]]:
    """Per-pair CI on the empirical preference p_hat = wins / n_pair.

    Returns a dict keyed by the canonical unordered pair.
    """
    intervals: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = {}
    for key, n in counts.total.items():
        w = bt_ci_width(n, t_global, constant)
        wins = counts.wins.get(key, 0)
        p_hat = wins / n if n > 0 else 0.5
        lo = max(p_hat - w, 0.0)
        hi = min(p_hat + w, 1.0)
        intervals[key] = (lo, hi)
    return intervals


# ============================================================================
# Ranking
# ============================================================================

def bt_ranking(theta_hat: Dict[Hashable, float],
               tie_break: str = "id") -> List[Hashable]:
    """Sort items by descending theta. Ties broken alphabetically by id."""
    items = list(theta_hat.keys())
    if tie_break == "id":
        items.sort(key=lambda i: (-theta_hat[i], str(i)))
    else:
        items.sort(key=lambda i: -theta_hat[i])
    return items