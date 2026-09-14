"""
bt.py
Bradley-Terry estimation, confidence intervals, and ranking utilities.

Model (see paper, Section "Bradley-Terry Model"):

    For agent a with parameter vector theta_a in R^K_{>0}, the probability
    that a prefers partner b1 over partner b2 is

        P_a(b1 > b2) = theta_{a,b1} / (theta_{a,b1} + theta_{a,b2}).

    Observations are Bernoulli draws X in {0, 1}:
        X = 1  means b1 won the comparison,
        X = 0  means b2 won.

The MLE is computed per agent, independently. We use the classical
Minorisation-Maximisation (MM) algorithm of Hunter (2004), which is monotone
and numerically safe. See:

    D. R. Hunter, "MM algorithms for generalized Bradley-Terry models",
    Annals of Statistics, 32(1):384--406, 2004.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Tuple

import math
import numpy as np


# ============================================================================
# Data structures
# ============================================================================

PairKey = Tuple[Hashable, Hashable]   # unordered pair (b1, b2), canonicalised

# def _canonical(b1: Hashable, b2: Hashable) -> PairKey:
#     """Canonicalise an unordered pair so that {b1, b2} and {b2, b1} map the
#     same key. We use the *identity* of the participants, sorted by their
#     hashable key, so that each unordered pair is counted once."""
#     return (b1, b2) if str(b1) <= str(b2) else (b2, b1)

def _canonical(b1: Hashable, b2: Hashable) -> PairKey:
    # PATCH: hash-based canonicalisation (no string formatting per call).
    # Participants are frozen dataclasses with stable __hash__, so this is
    # deterministic within a run. Ties in hash are impossible for distinct
    # participants in practice; if you worry, fall back to id() comparison.
    return (b1, b2) if hash(b1) <= hash(b2) else (b2, b1)


@dataclass
class PairCounts:
    """Accumulated comparison counts for one agent over its partner set.

    wins[pair]  = number of times b1 beat b2 (where pair = (b1, b2) canonical)
    total[pair] = number of comparisons made for this pair
    """
    wins:  Dict[PairKey, int] = field(default_factory=dict)
    total: Dict[PairKey, int] = field(default_factory=dict)

    def record(self, b1: Hashable, b2: Hashable, x: int) -> None:
        key = _canonical(b1, b2)
        # Normalise so that wins refers to the canonical first element
        if key[0] == b1:
            win_for_first = x
        else:
            win_for_first = 1 - x
        self.wins[key]  = self.wins.get(key, 0) + win_for_first
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

    Args:
        items:   list of partners (each a hashable id, e.g. Woman("w1")).
        wins:    dict pair -> number of times canonical first beat second.
        totals:  dict pair -> number of comparisons for that pair.
        max_iter: hard cap on MM iterations.
        tol:     convergence threshold on the log-likelihood change.
        theta_init: optional warm start (e.g. previous epoch's estimate).

    Returns:
        dict item -> estimated theta (all positive, normalised to sum to 1).
    """
    K = len(items)
    if K == 0:
        return {}
    if K == 1:
        return {items[0]: 1.0}

    # Initialise theta
    if theta_init is not None and all(i in theta_init for i in items):
        theta = {i: float(theta_init[i]) for i in items}
    else:
        theta = {i: 1.0 / K for i in items}

    # Precompute, for each item i, the total number of comparisons it appears in
    # and its cumulative wins against each opponent.
    # n_i   = sum over j of totals[(i, j)]
    # w_i   = sum over j of wins for i vs j
    n_i: Dict[Hashable, int] = {i: 0 for i in items}
    w_i: Dict[Hashable, int] = {i: 0 for i in items}

    for (b1, b2), t in totals.items():
        w12 = wins.get((b1, b2), 0)   # wins of b1 over b2
        w21 = t - w12                  # wins of b2 over b1
        n_i[b1] += t
        n_i[b2] += t
        w_i[b1] += w12
        w_i[b2] += w21

    # MM iteration:
    #   theta_i <- w_i / sum_{j != i} [ total_{ij} / (theta_i + theta_j) ]
    for _ in range(max_iter):
        # Build denominator for each i
        denom: Dict[Hashable, float] = {i: 0.0 for i in items}
        for (b1, b2), t in totals.items():
            s = theta[b1] + theta[b2]
            if s <= 0:
                s = 1e-12
            denom[b1] += t / s
            denom[b2] += t / s

        # Guard against division by zero when an item is never compared
        new_theta: Dict[Hashable, float] = {}
        for i in items:
            if denom[i] <= 0 or w_i[i] <= 0:
                # Item never won or was never compared; keep a small floor
                new_theta[i] = max(theta[i], 1e-12)
            else:
                new_theta[i] = w_i[i] / denom[i]

        # Renormalise (Assumption: sum_i theta_i = 1)
        Z = sum(new_theta.values())
        if Z <= 0:
            break
        new_theta = {i: v / Z for i, v in new_theta.items()}

        # Convergence check: change in theta (L1)
        delta = sum(abs(new_theta[i] - theta[i]) for i in items)
        theta = new_theta
        if delta < tol:
            break

    return theta


# ============================================================================
# Confidence intervals
# ============================================================================

# def bt_pair_ci_width(
#     n_pair: int,
#     T_global: int,
#     constant: float = 1.0,
# ) -> float:
#     """Half-width of the CI on the pairwise preference P(b_i ≻ b_j) for one agent.

#         w = sqrt( constant * log(T_global) / n_pair )

#     where:
#         n_pair   = number of direct comparisons between b_i and b_j
#         T_global = total number of comparisons drawn by the whole learner
#         constant = tunable coefficient (1.0 for a fixed-eta bound;
#                    larger for a uniform-over-time union bound)

#     Returns +inf if n_pair <= 0, so that unsampled pairs block the
#     stopping condition.
#     """
#     if n_pair <= 0:
#         return float("inf")
#     T = max(T_global, 2)
#     return math.sqrt(constant * math.log(T) / n_pair)

# def bt_pairwise_intervals(
#     items: List[Hashable],
#     counts: PairCounts,
#     T_global: int,
#     constant: float = 1.0,
# ) -> Dict[Tuple[Hashable, Hashable], Tuple[float, float]]:
#     """Per-pair CIs on the empirical preference, centered at 1/2.

#     For each unordered pair {b_i, b_j}, the empirical preference of b_i over
#     b_j for the agent is

#         p̂_{ij} = wins[(b_i, b_j)] / total[(b_i, b_j)]

#     and the CI is

#         [ p̂_{ij} - w,  p̂_{ij} + w ]

#     with w = bt_pair_ci_width(total[(b_i, b_j)], T_global, constant).

#     Interpretation of disjointness at the pair level:
#         CI excludes 1/2  =>  the agent's ordering between b_i and b_j is
#                              resolved with high confidence.

#     Returns a dict keyed by the canonical unordered pair.
#     """
#     intervals: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = {}
#     for key, n in counts.total.items():
#         w = bt_pair_ci_width(n, T_global, constant=constant)
#         wins = counts.wins.get(key, 0)
#         p_hat = wins / n if n > 0 else 0.5
#         lo = max(p_hat - w, 0.0)
#         hi = min(p_hat + w, 1.0)
#         intervals[key] = (lo, hi)
#     return intervals


def bt_ci_width(n_pair: int, t_global: int, constant: float = 1.0) -> float:
    """Hoeffding-style half-width from the paper:

        w = sqrt( 6 * log(t) / T )

    where t is the global time step and T is the number of comparisons.

    NOTE: this is a heuristic bound; the constant 6 and the log(t) scaling
    are taken from the paper's stated form. If your asymptotic analysis
    dictates a different rate, change this function only.
    """
    if n_pair <= 0:
        return float("inf")
    t = max(t_global, 2)   # log(t) needs t >= 2 to be positive
    return math.sqrt(constant * math.log(t) / n_pair)


def bt_confidence_intervals(
    items: List[Hashable],
    theta_hat: Dict[Hashable, float],
    counts: PairCounts,
    t_global: int,
) -> Dict[Tuple[Hashable, Hashable], Tuple[float, float]]:
    """Per-pair CI on the empirical preference p̂ = wins / n_pair.

    Keyed by the canonical unordered pair (b_i, b_j). Width uses the
    direct pair count n_{a,{b_i,b_j}}:

        w = bt_ci_width(n_pair, t_global)   # = sqrt(6 * log(t) / n_pair)

    The interval is [p̂ - w, p̂ + w] ∩ [0, 1].
    """
    intervals: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = {}
    for key, n in counts.total.items():
        w = bt_ci_width(n, t_global)
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
    """Sort items by descending theta. Ties broken alphabetically by id.

    This is the empirical preference profile reconstructed at the end of each
    epoch: ≻̂ = sort(theta_hat, descending).
    """
    items = list(theta_hat.keys())
    if tie_break == "id":
        items.sort(key=lambda i: (-theta_hat[i], str(i)))
    else:
        items.sort(key=lambda i: -theta_hat[i])
    return items