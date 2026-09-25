"""metrics.py

Evaluation metrics for the LLM-task stable matching experiment.

* Exact matching recovery        1{H_hat == H*_train}
* Stability under hidden TRAIN preferences (StabilityVerifier)
* Blocking pair count (explicit, all pairs)
* Pairwise resolution fraction   #CIs excluding 1/2 / total arms
* Test welfare                   sum_d U_d^test(H(d))
* Normalized welfare             (W - W_rand) / (W_hung - W_rand)
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Tuple

import pandas as pd

from gs_lib.gs_tools import Matching, PreferenceList, StabilityVerifier


# ============================================================================
# Matching comparison
# ============================================================================

def matching_equal(m1: Matching, m2: Matching) -> bool:
    return set(m1.pairs) == set(m2.pairs)


def exact_oracle_match(estimate: Matching, oracle: Matching) -> bool:
    return matching_equal(estimate, oracle)


def matching_to_dict(matching: Matching) -> Dict[str, str]:
    return {pair.woman.id: pair.man.id for pair in matching.pairs}


def matching_str(matching: Matching) -> str:
    pairs = sorted(matching.pairs, key=lambda p: (p.woman.id, p.man.id))
    return "|".join(f"{p.woman.id}={p.man.id}" for p in pairs)


# ============================================================================
# Stability
# ============================================================================

def is_stable(prefs: PreferenceList, matching: Matching) -> bool:
    ok, _, _ = StabilityVerifier(prefs).is_stable(matching)
    return ok


def count_blocking_pairs(prefs: PreferenceList, matching: Matching) -> int:
    """Explicit count of ALL blocking pairs (not just the first found).

    A pair (m, w) not matched to each other is blocking iff m strictly
    prefers w to his current partner and w strictly prefers m to her
    current partner. Unmatched agents prefer any acceptable partner.
    """
    all_men = set(p.man for p in matching.pairs) | set(matching.unmatched_men)
    all_women = set(p.woman for p in matching.pairs) | set(matching.unmatched_women)

    count = 0
    for man in all_men:
        partner_m = matching.get_partner(man)
        for woman in all_women:
            if woman == partner_m:
                continue
            partner_w = matching.get_partner(woman)
            if prefs.prefers(man, woman, partner_m) and prefs.prefers(
                woman, man, partner_w
            ):
                count += 1
    return count


# ============================================================================
# Pairwise resolution
# ============================================================================

def resolved_fraction(agent_states: Dict[Hashable, object]) -> float:
    """Fraction of pairwise arms whose CI excludes 1/2.

    agent_states: P2ETG/PrefLID-style mapping agent -> AgentState with
    .ci (dict pair -> (lo, hi)) and .partners.
    """
    total_arms = 0
    resolved = 0
    for state in agent_states.values():
        n = len(state.partners)
        total_arms += n * (n - 1) // 2
        for lo, hi in state.ci.values():
            if lo > 0.5 or hi < 0.5:
                resolved += 1
    if total_arms == 0:
        return 0.0
    return resolved / total_arms


# ============================================================================
# Welfare
# ============================================================================

def test_welfare(
    matching: Matching,
    task_util: pd.DataFrame,
) -> float:
    """W(H) = sum_d U_d(H(d)) under the given task-side utilities."""
    total = 0.0
    for pair in matching.pairs:
        total += float(task_util.loc[pair.woman.id, pair.man.id])
    return total


def mean_test_score(
    matching: Matching,
    task_util: pd.DataFrame,
) -> float:
    n = max(len(matching.pairs), 1)
    return test_welfare(matching, task_util) / n


def normalized_welfare(
    welfare: float,
    welfare_random: float,
    welfare_hungarian: float,
) -> float:
    """(W - W_rand) / (W_hung - W_rand), safe against zero denominator."""
    denom = welfare_hungarian - welfare_random
    if abs(denom) < 1e-12:
        return 0.0
    return (welfare - welfare_random) / denom


# ============================================================================
# Preference consistency across splits
# ============================================================================

def spearman_rank_correlations(
    util_a: pd.DataFrame, util_b: pd.DataFrame
) -> pd.DataFrame:
    """Per-agent Spearman correlation between two utility matrices."""
    from scipy.stats import spearmanr

    rows = []
    for agent in util_a.index:
        ranks_a = util_a.loc[agent].rank(ascending=False)
        ranks_b = util_b.loc[agent].rank(ascending=False)
        rho, _ = spearmanr(ranks_a.values, ranks_b.values)
        rows.append(
            {
                "agent": str(agent),
                "spearman_rho": float(rho) if pd.notna(rho) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
