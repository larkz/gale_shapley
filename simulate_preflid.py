"""
simulate_preflid.py
Single-instance driver for PrefLID, with a side-by-side comparison
against P2ETG's stopping time.
"""

from __future__ import annotations

import random

import numpy as np

from gs_lib.gs_tools import (
    Man, Woman, PreferenceList, GaleShapley, StabilityVerifier,
)
from p2etg import P2ETG
from preflid import PrefLID


def random_theta(participants, partners, rng, alpha: float = 0.5):
    K = len(partners)
    return {
        p: {q: float(w) for q, w in zip(partners, rng.dirichlet([alpha] * K))}
        for p in participants
    }


def gs_on_true_preferences(men, women, true_theta_men, true_theta_women):
    prefs = {
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men, key=lambda m: -true_theta_women[w][m]) for w in women},
    }
    return GaleShapley(PreferenceList(prefs)).find_stable_matching("men")


def main(N: int = 4, K: int = 4, seed: int = 0,
         budget: int = 100, constant: float = 0.1):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men = [Man(f"m{i}") for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    true_theta_men = random_theta(men, women, np_rng)
    true_theta_women = random_theta(women, men, np_rng)

    h_star = gs_on_true_preferences(men, women, true_theta_men, true_theta_women)
    print(f"Oracle H_*: {h_star}")

    # ---------------- PrefLID ----------------
    print("\n--- PrefLID ---")
    preflid = PrefLID(
        men, women,
        true_theta_men=true_theta_men,
        true_theta_women=true_theta_women,
        rng=random.Random(seed),
        constant=constant,
        budget=budget,
    )
    rounds: list = []
    result = preflid.run_until_stop(
        max_iterations=200, rounds=rounds, verbose=False,
    )
    print(f"\nStopped: {result['stopped']}")
    print(f"T_stop (PrefLID): {result['T_stop']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final |islands|: {result['n_islands']}")
    print(f"Committed: {result['matching']}")
    print(f"Committed == H_*: {result['matching'] == h_star}")

    # Verify stability under truth.
    prefs_true = PreferenceList({
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men, key=lambda m: -true_theta_women[w][m]) for w in women},
    })
    ok, reason, _ = StabilityVerifier(prefs_true).is_stable(result["matching"])
    print(f"Stable under truth: {ok} ({reason})")

    # ---------------- P2ETG baseline ----------------
    print("\n--- P2ETG baseline ---")
    p2 = P2ETG(
        men, women,
        true_theta_men=true_theta_men,
        true_theta_women=true_theta_women,
        rng=random.Random(seed),
        constant=constant,
    )
    p2_result = p2.run_until_stop(
        adaptive=True, check_every=25, max_samples=200_000, verbose=False,
    )
    print(f"Stopped: {p2_result['stopped']}")
    print(f"T_stop (P2ETG): {p2_result['T_stop']}")
    print(f"Committed == H_*: {p2_result['matching'] == h_star}")

    if p2_result["T_stop"] > 0:
        ratio = result["T_stop"] / p2_result["T_stop"]
        print(f"\nT_stop ratio PrefLID/P2ETG: {ratio:.3f}")

    # Print PrefLID round log.
    print("\nPrefLID iteration log:")
    for r in rounds:
        print(f"  iter={r['iteration']:>3} t={r['t']:>6} "
              f"|Omega|={r['omega_size']:>8} "
              f"islands={r['n_islands']} "
              f"status={r['status']}")


if __name__ == "__main__":
    main()