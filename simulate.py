"""
simulate.py
Synthetic experiment driver for P2ETG.
"""

from __future__ import annotations

import random

import numpy as np

from gs_lib.gs_tools import (
    Man, Woman, PreferenceList, GaleShapley, StabilityVerifier,
)
from p2etg import P2ETG


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


def main(N: int = 4, K: int = 4, seed: int = 0, max_epochs: int = 20):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men = [Man(f"m{i}") for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    true_theta_men = random_theta(men, women, np_rng)
    true_theta_women = random_theta(women, men, np_rng)

    h_star = gs_on_true_preferences(men, women, true_theta_men, true_theta_women)
    print(f"Oracle H_*: {h_star}")

    learner = P2ETG(
        men, women,
        true_theta_men=true_theta_men,
        true_theta_women=true_theta_women,
        rng=rng,
    )
    result = learner.run_until_stop(
        max_epochs=max_epochs,
        adaptive=True,
        check_every=250,
        max_samples=200_000,
        verbose=True,
    )

    print(f"\nStopped: {result['stopped']}")
    print(f"T_stop:  {result['T_stop']}")
    print(f"Committed matching: {result['matching']}")

    prefs_true = PreferenceList({
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men, key=lambda m: -true_theta_women[w][m]) for w in women},
    })
    ok_true, reason_true, _ = StabilityVerifier(prefs_true).is_stable(result["matching"])
    print(f"Stable under truth: {ok_true} ({reason_true})")
    print(f"Committed matching == H_*: {result['matching'] == h_star}")


if __name__ == "__main__":
    main()