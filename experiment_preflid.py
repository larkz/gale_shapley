"""
experiment_preflid.py
Sweep harness for PrefLID vs P2ETG. Saves per-run summaries and produces
plots comparing stopping times, island trajectories, and correctness.
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
)
from p2etg import P2ETG
from preflid import PrefLID


RUNS_DIR  = Path("runs_preflid")
PLOTS_DIR = Path("plots_preflid")
RUNS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def random_theta(participants, partners, rng, alpha=0.5):
    K = len(partners)
    return {
        p: {q: float(w) for q, w in zip(partners, rng.dirichlet([alpha] * K))}
        for p in participants
    }


def gs_on_true_preferences(men, women, ttm, ttw):
    prefs = {
        **{m: sorted(women, key=lambda w: -ttm[m][w]) for m in men},
        **{w: sorted(men, key=lambda m: -ttw[w][m]) for w in women},
    }
    return GaleShapley(PreferenceList(prefs)).find_stable_matching("men")


def run_single(
    N: int, K: int, alpha: float, seed: int,
    budget: int, constant: float,
    max_iterations: int,
    horizon: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """Run PrefLID and P2ETG on the same instance. Return (rounds, summary).

    If `horizon` is specified, the per-round timeline is extended past
    T_stop up to `t = horizon`, with the committed matching repeated.
    """
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men   = [Man(f"m{i}")   for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    ttm = random_theta(men,   women, np_rng, alpha)
    ttw = random_theta(women, men,   np_rng, alpha)

    h_star = gs_on_true_preferences(men, women, ttm, ttw)

    prefs_true = PreferenceList({
        **{m: sorted(women, key=lambda w: -ttm[m][w]) for m in men},
        **{w: sorted(men,   key=lambda m: -ttw[w][m]) for w in women},
    })

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
        return [], {
            "N": N, "K": K, "seed": seed, "budget": budget,
            "constant": constant,
            "preflid_error": str(e),
            "preflid_stopped": False,
            "preflid_T_stop": None,
            "preflid_iterations": None,
            "preflid_islands_final": None,
            "preflid_correct": 0,
            "preflid_stable_truth": 0,
            "p2etg_stopped": None,
            "p2etg_T_stop": None,
            "p2etg_correct": None,
            "T_stop_ratio": None,
            "horizon": horizon,
        }

    pl_ok, _, _ = StabilityVerifier(prefs_true).is_stable(pl_result["matching"])
    pl_correct = int(pl_result["matching"] == h_star)

    # ------------------------------------------------------------------
    # Build per-round timeline.
    #
    # Each raw round covers the interval (prev_t, t]. The matching played
    # during that interval is the H* of the largest island at that round
    # (or the final committed matching, if a round doesn't carry H*).
    #
    # After T_stop, if a horizon is specified and > T_stop, the committed
    # matching is repeated up to horizon.
    # ------------------------------------------------------------------
    T_stop = pl_result["T_stop"]
    T_end = horizon if (horizon is not None and horizon > T_stop) else T_stop

    rows: List[Dict] = []
    prev_t = 0
    cumulative_regret = 0

    if not raw_rounds:
        # Algorithm never built a lattice — no H* was ever computed.
        # Fill everything with "None" and mark as unstable.
        for tt in range(1, T_end + 1):
            cumulative_regret += 1
            rows.append({
                "t": tt,
                "matching_str": "None",
                "disjoint": False,
                "correct": 0,
                "regret": cumulative_regret,
            })
    else:
        for r in raw_rounds:
            t_end = r["t"]
            if r.get("H_star_str"):
                matching_str = r["H_star_str"]
                disjoint = (r.get("status") in ("ok", "committed"))
                correct = int(matching_str == str(h_star))
            else:
                # No H* was computed at this iteration — mark as unknown.
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
                })
            prev_t = t_end

        # Extend to horizon with the committed matching.
        if prev_t < T_end:
            committed = pl_result["matching"]
            matching_str = str(committed)
            correct = int(committed == h_star)
            for tt in range(prev_t + 1, T_end + 1):
                cumulative_regret += (1 - correct)
                rows.append({
                    "t": tt,
                    "matching_str": matching_str,
                    "disjoint": True,
                    "correct": correct,
                    "regret": cumulative_regret,
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
    p2_correct = int(p2_result["matching"] == h_star)

    summary = {
        "N": N, "K": K, "alpha": alpha, "seed": seed,
        "budget": budget, "constant": constant,
        "horizon": horizon,
        # PrefLID
        "preflid_stopped": bool(pl_result["stopped"]),
        "preflid_T_stop": pl_result["T_stop"],
        "preflid_iterations": pl_result["iterations"],
        "preflid_islands_final": pl_result["n_islands"],
        "preflid_correct": pl_correct,
        "preflid_stable_truth": int(pl_ok),
        # P2ETG
        "p2etg_stopped": bool(p2_result["stopped"]),
        "p2etg_T_stop": p2_result["T_stop"],
        "p2etg_correct": p2_correct,
        # Derived
        "T_stop_ratio": (
            pl_result["T_stop"] / p2_result["T_stop"]
            if p2_result["T_stop"] > 0 else None
        ),
    }

    return rows, summary


def generate_data(
    Ns: List[int] = [3, 4],
    Ks: List[int] = [3, 4],
    seeds: List[int] = list(range(3)),
    budgets: List[int] = [50, 100, 500],
    constant: float = 0.1,
    max_iterations: int = 100,
    horizon: Optional[int] = None,
) -> pd.DataFrame:
    all_rounds: List[Dict] = []
    all_summaries: List[Dict] = []

    for N in Ns:
        for K in Ks:
            for budget in budgets:
                for seed in seeds:
                    tag = f"N{N}_K{K}_b{budget}_seed{seed}"
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
                          f"correct={summary['preflid_correct']}")

    df_rounds = pd.DataFrame(all_rounds)
    df_rounds.to_csv(RUNS_DIR / "all_rounds.csv", index=False)

    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(RUNS_DIR / "all_summaries.csv", index=False)

    return df_sum


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("Generating data...")
    df_sum = generate_data(
        Ns=[3],
        Ks=[3],
        seeds=list(range(100)),
        budgets=[300000],
        constant=0.005,
        max_iterations=10000,
        horizon=10000,       # <-- extend the timeline past T_stop
    )

    print(f"\nDone. Data in {RUNS_DIR.resolve()}/")
    print(f"Total runs: {len(df_sum)}")
    print(f"Successful (correct): {df_sum['preflid_correct'].sum()}")
    print(f"Stopped: {df_sum['preflid_stopped'].sum()}")


if __name__ == "__main__":
    main()
