"""
experiment_preflid.py
Sweep harness for PrefLID vs P2ETG.

Regret is measured against the SET of stable matchings under the true
Bradley-Terry ground truth, not against any single extreme (man-optimal
or woman-optimal). A round incurs loss 0 iff the matching played is
stable under the true preferences; loss 1 otherwise.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gs_lib.gs_tools import (
    Man, Woman, PreferenceList, GaleShapley, StabilityVerifier,
    StableMatchingLattice,
)
from p2etg import P2ETG
from preflid import PrefLID


RUNS_DIR  = Path("runs_preflid")
PLOTS_DIR = Path("plots_preflid")
RUNS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Cap on the number of true stable matchings enumerated per seed.
MAX_TRUE_MATCHINGS = 5000


def random_theta(participants, partners, rng, alpha=0.5):
    K = len(partners)
    return {
        p: {q: float(w) for q, w in zip(partners, rng.dirichlet([alpha] * K))}
        for p in participants
    }


def build_true_preferences(men, women, ttm, ttw):
    return PreferenceList({
        **{m: sorted(women, key=lambda w: -ttm[m][w]) for m in men},
        **{w: sorted(men,   key=lambda m: -ttw[w][m]) for w in women},
    })


def gs_on_true_preferences(men, women, ttm, ttw):
    prefs = build_true_preferences(men, women, ttm, ttw)
    return GaleShapley(prefs).find_stable_matching("men")


def enumerate_true_stable_matchings(
    prefs_true: PreferenceList,
    max_matchings: int = MAX_TRUE_MATCHINGS,
) -> Tuple[Optional[List], Optional[int], Optional[str]]:
    """Return (list_of_matchings, count, error_string).

    If the true lattice exceeds `max_matchings`, the list is None and
    count is the (partial) observed size. On exception, both list and
    count are None and error is a string.
    """
    try:
        lattice = StableMatchingLattice(prefs_true)
        lattice.build_lattice()
        all_matchings = lattice.get_all_matchings()
    except Exception as e:
        return None, None, str(e)

    if len(all_matchings) > max_matchings:
        return None, len(all_matchings), "exceeds_cap"

    return list(all_matchings), len(all_matchings), None


def collect_ground_truth(men, women, ttm, ttw, prefs_true):
    """Compute the ground-truth block to attach to each summary."""
    h_star_men = GaleShapley(prefs_true).find_stable_matching("men")
    h_star_women = GaleShapley(prefs_true).find_stable_matching("women")

    matchings_list, matchings_count, matchings_error = (
        enumerate_true_stable_matchings(prefs_true)
    )

    return {
        "gt_true_theta_men": {
            str(m): {str(w): float(v) for w, v in ttm[m].items()}
            for m in men
        },
        "gt_true_theta_women": {
            str(w): {str(m): float(v) for m, v in ttw[w].items()}
            for w in women
        },
        "gt_prefs_men": {
            str(m): [str(w) for w in prefs_true.get_preference(m)]
            for m in men
        },
        "gt_prefs_women": {
            str(w): [str(m) for m in prefs_true.get_preference(w)]
            for w in women
        },
        "gt_h_star_men_str": str(h_star_men),
        "gt_h_star_men_pairs": [
            [str(p.man), str(p.woman)] for p in h_star_men.pairs
        ],
        "gt_h_star_women_str": str(h_star_women),
        "gt_h_star_women_pairs": [
            [str(p.man), str(p.woman)] for p in h_star_women.pairs
        ],
        "gt_stable_matchings_count": matchings_count,
        "gt_stable_matchings_list": (
            [str(m) for m in matchings_list] if matchings_list is not None
            else None
        ),
        "gt_stable_matchings_error": matchings_error,
    }


def run_single(
    N: int, K: int, alpha: float, seed: int,
    budget: int, constant: float,
    max_iterations: int,
    horizon: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """Run PrefLID and P2ETG on the same instance. Return (rounds, summary)."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men   = [Man(f"m{i}")   for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    ttm = random_theta(men,   women, np_rng, alpha)
    ttw = random_theta(women, men,   np_rng, alpha)

    prefs_true = build_true_preferences(men, women, ttm, ttw)
    h_star_men = GaleShapley(prefs_true).find_stable_matching("men")

    # ------------------------------------------------------------------
    # Enumerate the TRUE stable set M*_true.
    # Regret is measured against membership in this set, not against a
    # single extreme.
    # ------------------------------------------------------------------
    true_matchings_list, true_matchings_count, true_matchings_error = (
        enumerate_true_stable_matchings(prefs_true)
    )
    if true_matchings_list is not None:
        true_stable_set = set(true_matchings_list)
        true_stable_strs = {str(m) for m in true_matchings_list}
        true_stable_enum_ok = True
    else:
        # Fallback: use the singleton {H_*_men}.
        true_stable_set = {h_star_men}
        true_stable_strs = {str(h_star_men)}
        true_stable_enum_ok = False

    # Ground-truth block to attach to the summary.
    gt = collect_ground_truth(men, women, ttm, ttw, prefs_true)
    gt["gt_stable_enum_ok"] = true_stable_enum_ok

    # ---------------- PrefLID ----------------
    preflid = PrefLID(
        men, women,
        true_theta_men=ttm,
        true_theta_women=ttw,
        rng=random.Random(seed),
        constant=constant,
        budget=budget,
    )
    raw_rounds: List[Dict] = []
    try:
        pl_result = preflid.run_until_stop(
            max_iterations=max_iterations, rounds=raw_rounds, verbose=False,
        )
    except Exception as e:
        summary = {
            "N": N, "K": K, "seed": seed, "budget": budget,
            "constant": constant,
            "preflid_error": str(e),
            "preflid_stopped": False,
            "preflid_T_stop": None,
            "preflid_iterations": None,
            "preflid_islands_final": None,
            "preflid_correct": 0,
            "preflid_in_mstar_true": 0,
            "preflid_is_hstar_men": 0,
            "preflid_stable_truth": 0,
            "preflid_reason": "exception",
            "p2etg_stopped": None,
            "p2etg_T_stop": None,
            "p2etg_correct": None,
            "T_stop_ratio": None,
            "horizon": horizon,
        }
        summary.update(gt)
        return [], summary

    # Stability under truth (individual rationality + no blocking pairs).
    pl_ok, _, _ = StabilityVerifier(prefs_true).is_stable(pl_result["matching"])

    # Regret-relevant correctness: membership in M*_true.
    pl_in_mstar = int(pl_result["matching"] in true_stable_set)
    # Secondary diagnostic: is it the man-optimal extreme?
    pl_is_hstar_men = int(pl_result["matching"] == h_star_men)

    # ------------------------------------------------------------------
    # Build per-round timeline.
    # ------------------------------------------------------------------
    T_stop = pl_result["T_stop"]
    T_end = horizon if (horizon is not None and horizon > T_stop) else T_stop

    rows: List[Dict] = []
    prev_t = 0
    cumulative_regret = 0

    if not raw_rounds:
        for tt in range(1, T_end + 1):
            cumulative_regret += 1
            rows.append({
                "t": tt,
                "matching_str": "None",
                "disjoint": False,
                "correct": 0,
                "regret": cumulative_regret,
                "n_islands": None,
                "n_lattices_largest_island": None,
                "support_max": None,
                "support_min": None,
                "support_mean": None,
                "status": "insufficient_iterations",
                "reason": "insufficient_iterations",
            })
    else:
        for r in raw_rounds:
            t_end = r["t"]
            if r.get("H_star_str"):
                matching_str = r["H_star_str"]
                disjoint = (r.get("status") in ("ok", "committed"))
                # Correct iff the played matching is in M*_true.
                correct = int(matching_str in true_stable_strs)
            else:
                matching_str = "None"
                disjoint = False
                correct = 0
            for tt in range(prev_t + 1, t_end + 1):
                cumulative_regret += (1 - correct)
                rows.append({
                    "t": tt,
                    "matching_str": matching_str,
                    "disjoint": disjoint,
                    "correct": correct,
                    "regret": cumulative_regret,
                    "n_islands": r.get("n_islands"),
                    "n_lattices_largest_island": r.get("n_lattices_largest_island"),
                    "support_max": r.get("support_max"),
                    "support_min": r.get("support_min"),
                    "support_mean": r.get("support_mean"),
                    "status": r.get("status"),
                    "reason": r.get("reason"),
                })
            prev_t = t_end

        # Extend to horizon with the committed matching (frozen).
        if prev_t < T_end:
            committed = pl_result["matching"]
            matching_str = str(committed)
            # Correct iff the committed matching is in M*_true.
            correct = int(matching_str in true_stable_strs)
            for tt in range(prev_t + 1, T_end + 1):
                cumulative_regret += (1 - correct)
                rows.append({
                    "t": tt,
                    "matching_str": matching_str,
                    "disjoint": True,
                    "correct": correct,
                    "regret": cumulative_regret,
                    "n_islands": 1 if pl_result["stopped"] else None,
                    "n_lattices_largest_island": None,
                    "support_max": None,
                    "support_min": None,
                    "support_mean": None,
                    "status": "committed" if pl_result["stopped"] else "fallback",
                    "reason": pl_result.get(
                        "reason",
                        "certified" if pl_result["stopped"] else "max_iterations",
                    ),
                })

    # ---------------- P2ETG baseline ----------------
    p2 = P2ETG(
        men, women,
        true_theta_men=ttm,
        true_theta_women=ttw,
        rng=random.Random(seed),
        constant=constant,
    )
    p2_result = p2.run_until_stop(
        adaptive=True, check_every=25, max_samples=2000, verbose=False,
    )
    p2_in_mstar = int(p2_result["matching"] in true_stable_set)

    summary = {
        "N": N, "K": K, "alpha": alpha, "seed": seed,
        "budget": budget, "constant": constant,
        "horizon": horizon,
        # PrefLID
        "preflid_stopped": bool(pl_result["stopped"]),
        "preflid_T_stop": pl_result["T_stop"],
        "preflid_iterations": pl_result["iterations"],
        "preflid_islands_final": pl_result["n_islands"],
        "preflid_in_mstar_true": pl_in_mstar,
        "preflid_is_hstar_men": pl_is_hstar_men,
        "preflid_correct": pl_in_mstar,
        "preflid_stable_truth": int(pl_ok),
        "preflid_reason": pl_result.get(
            "reason",
            "certified" if pl_result["stopped"] else "unknown",
        ),
        # P2ETG
        "p2etg_stopped": bool(p2_result["stopped"]),
        "p2etg_T_stop": p2_result["T_stop"],
        "p2etg_correct": p2_in_mstar,
        # Derived
        "T_stop_ratio": (
            pl_result["T_stop"] / p2_result["T_stop"]
            if p2_result["T_stop"] > 0 else None
        ),
    }
    summary.update(gt)

    return rows, summary


def generate_data(
    Ns: List[int] = [3, 4],
    Ks: List[int] = [3, 4],
    seeds: List[int] = list(range(3)),
    budgets: List[int] = [50, 100, 500],
    constant: float = 0.1,
    max_iterations: int = 100,
    horizon: Optional[int] = None,
    run_id: str = "",
) -> pd.DataFrame:
    all_rounds: List[Dict] = []
    all_summaries: List[Dict] = []

    for N in Ns:
        for K in Ks:
            for budget in budgets:
                for seed in seeds:
                    tag = f"N{N}_K{K}_b{budget}_seed{seed}"
                    if run_id:
                        tag = f"{run_id}_{tag}"
                    run_dir = RUNS_DIR / tag
                    run_dir.mkdir(exist_ok=True)

                    print(f"  running {tag} ...", end=" ", flush=True)
                    rounds, summary = run_single(
                        N, K, alpha=0.5, seed=seed,
                        budget=budget, constant=constant,
                        max_iterations=max_iterations,
                        horizon=horizon,
                    )

                    pd.DataFrame(rounds).to_csv(
                        run_dir / "rounds.csv", index=False
                    )
                    (run_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2, default=str)
                    )

                    for r in rounds:
                        r.update({"N": N, "K": K, "budget": budget, "seed": seed})
                    all_rounds.extend(rounds)
                    all_summaries.append(summary)

                    print(f"T_stop={summary['preflid_T_stop']} "
                          f"in_mstar={summary['preflid_in_mstar_true']}")

    prefix = f"{run_id}_" if run_id else ""

    df_rounds = pd.DataFrame(all_rounds)
    df_rounds.to_csv(RUNS_DIR / f"{prefix}all_rounds.csv", index=False)

    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(RUNS_DIR / f"{prefix}all_summaries.csv", index=False)

    return df_sum


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("Generating data...")
    df_sum = generate_data(
        Ns=[5],
        Ks=[5],
        seeds=list(range(10)),
        budgets=[20000],
        constant=0.1,
        max_iterations=20000,
        horizon=200_000,
        run_id="ver3_PrefLID",
    )

    print(f"\nDone. Data in {RUNS_DIR.resolve()}/")
    print(f"Total runs: {len(df_sum)}")
    print(f"In M*_true (correct): {df_sum['preflid_in_mstar_true'].sum()}")
    print(f"Is H_* (men optimal): {df_sum['preflid_is_hstar_men'].sum()}")
    print(f"Stopped: {df_sum['preflid_stopped'].sum()}")
    if "preflid_reason" in df_sum.columns:
        print(f"Stopping reasons:")
        print(df_sum["preflid_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()