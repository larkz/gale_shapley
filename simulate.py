"""
simulate.py
Synthetic experiment driver for P2ETG.

Usage:
    python simulate.py

Draws random ground-truth thetas, runs P2ETG, and reports:
    - number of epochs to stop
    - T_stop
    - the committed matching
    - verification that the committed matching equals GS on true preferences
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from gs_lib.gs_tools import (
    Man, Woman, PreferenceList, GaleShapley, StabilityVerifier,
)
from p2etg import P2ETG


def random_theta(participants, partners, rng, alpha: float = 0.5):
    """Dirichlet-distributed theta vectors, one per participant."""
    K = len(partners)
    out = {}
    for p in participants:
        weights = rng.dirichlet([alpha] * K)
        out[p] = {q: float(w) for q, w in zip(partners, weights)}
    return out


def gs_on_true_preferences(men, women, true_theta_men, true_theta_women):
    """Compute the A-side-optimal stable matching under true preferences."""
    prefs = {}
    for m in men:
        prefs[m] = sorted(women, key=lambda w: -true_theta_men[m][w])
    for w in women:
        prefs[w] = sorted(men, key=lambda m: -true_theta_women[w][m])
    return GaleShapley(PreferenceList(prefs)).find_stable_matching("men")

# def _diagnose_stopping(learner):
#     """Report the worst-overlapping partner pair across all agents.

#     Returns (worst_margin, agent, b1, b2), where:
#         worst_margin <= 0  -> every pair already disjoint (stopping holds)
#         worst_margin  > 0  -> at least this much CI overlap remains
#     """
#     worst = None
#     for state in learner.agent_states.values():
#         items = state.partners
#         ci = state.ci
#         for i in range(len(items)):
#             lo_i, hi_i = ci[items[i]]
#             for j in range(i + 1, len(items)):
#                 lo_j, hi_j = ci[items[j]]
#                 # Overlap: positive = CIs intersect, negative = disjoint
#                 if hi_i <= lo_j or hi_j <= lo_i:
#                     margin = -min(lo_j - hi_i, lo_i - hi_j)  # disjoint: <= 0
#                 else:
#                     margin = min(hi_i - lo_j, hi_j - lo_i)   # overlap:  > 0
#                 if worst is None or margin > worst[0]:
#                     worst = (margin, state.agent, items[i], items[j])
#     return worst

# def main(N: int = 4, K: int = 4, seed: int = 0, max_epochs: int = 20):
#     rng = random.Random(seed)
#     np_rng = np.random.default_rng(seed)

#     men = [Man(f"m{i}") for i in range(1, N + 1)]
#     women = [Woman(f"w{j}") for j in range(1, K + 1)]

#     true_theta_men   = random_theta(men, women, np_rng)
#     true_theta_women = random_theta(women, men, np_rng)

#     # Oracle: GS on true preferences
#     h_star = gs_on_true_preferences(
#         men, women, true_theta_men, true_theta_women
#     )
#     print(f"Oracle H_*: {h_star}")

#     # Learner
#     learner = P2ETG(
#         men, women,
#         true_theta_men=true_theta_men,
#         true_theta_women=true_theta_women,
#         rng=rng,
#     )
#     result = learner.run_until_stop(max_epochs=max_epochs)

#     print(f"\nStopped: {result['stopped']}")
#     print(f"Epochs:  {len(result['epochs'])}")
#     print(f"T_stop:  {result['T_stop']}")
#     print(f"Committed matching: {result['matching']}")

#     # Verify the committed matching is stable under true preferences
#     verifier = StabilityVerifier(PreferenceList({
#         **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
#         **{w: sorted(men, key=lambda m: -true_theta_women[w][m]) for w in women},
#     }))
#     ok, reason, blocking = verifier.is_stable(result["matching"])
#     print(f"Committed matching stable under truth: {ok} ({reason})")
#     print(f"Committed matching == H_*: {result['matching'] == h_star}")

#     # Print per-epoch summary
#     print("\nPer-epoch trace:")
#     print(f"{'epoch':>5} {'R':>6} {'new':>6} {'t':>6}  {'disjoint':>8}  matching")
#     for e in result["epochs"]:
#         print(f"{e['epoch']:>5} {e['R']:>6} {e['new_samples']:>6} {e['t']:>6}  "
#               f"{str(e['pairwise_disjoint']):>8}  {e['matching']}")

def _diagnose_stopping(learner):
    """Report the worst-overlapping partner pair across all agents.

    Returns (worst_margin, agent, b1, b2), where:
        worst_margin <= 0  -> every pair already disjoint (stopping holds)
        worst_margin  > 0  -> at least this much CI overlap remains
    """
    worst = None
    for state in learner.agent_states.values():
        items = state.partners
        ci = state.ci
        for i in range(len(items)):
            lo_i, hi_i = ci[items[i]]
            for j in range(i + 1, len(items)):
                lo_j, hi_j = ci[items[j]]
                # Overlap: positive = CIs intersect, negative = disjoint
                if hi_i <= lo_j or hi_j <= lo_i:
                    margin = -min(lo_j - hi_i, lo_i - hi_j)  # disjoint: <= 0
                else:
                    margin = min(hi_i - lo_j, hi_j - lo_i)   # overlap:  > 0
                if worst is None or margin > worst[0]:
                    worst = (margin, state.agent, items[i], items[j])
    return worst


def main(N: int = 4, K: int = 4, seed: int = 0, max_epochs: int = 20):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men = [Man(f"m{i}") for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    true_theta_men   = random_theta(men, women, np_rng)
    true_theta_women = random_theta(women, men, np_rng)

    # Oracle: GS on true preferences
    h_star = gs_on_true_preferences(
        men, women, true_theta_men, true_theta_women
    )
    print(f"Oracle H_*: {h_star}")

    # Learner
    learner = P2ETG(
        men, women,
        true_theta_men=true_theta_men,
        true_theta_women=true_theta_women,
        rng=rng,
    )
    result = learner.run_until_stop(max_epochs=max_epochs)

    print(f"\nStopped: {result['stopped']}")
    print(f"Epochs:  {len(result['epochs'])}")
    print(f"T_stop:  {result['T_stop']}")
    print(f"Committed matching: {result['matching']}")

    # Verify the committed matching is stable under true preferences
    verifier = StabilityVerifier(PreferenceList({
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men, key=lambda m: -true_theta_women[w][m]) for w in women},
    }))
    ok, reason, blocking = verifier.is_stable(result["matching"])
    print(f"Committed matching stable under truth: {ok} ({reason})")
    print(f"Committed matching == H_*: {result['matching'] == h_star}")

    # ----------------------------------------------------------------------
    # Per-epoch trace with diagnostics
    # ----------------------------------------------------------------------
    print("\nPer-epoch trace:")

    # Wider header to accommodate the new diagnostic columns
    header = (f"{'epoch':>5} {'R':>7} {'new':>8} {'t':>9} "
              f"{'disjoint':>8} {'max_wid':>8} {'min_gap':>8} "
              f"{'worst_ov':>9}  matching")
    print(header)
    print("-" * len(header))

    for e in result["epochs"]:
        # We need the learner state *as it was at the end of this epoch*.
        # Since run_until_stop mutated learner in place, the final state
        # reflects the last epoch only. To show per-epoch diagnostics
        # faithfully, we re-run the trace from the recorded result if
        # the learner stored per-epoch snapshots; otherwise we fall back
        # to reporting only what the trace recorded plus current values.

        # --- Per-epoch diagnostics from the live learner (best effort) ---
        # These reflect *current* state, which equals the final epoch's state.
        # For a fully faithful per-epoch diagnostic, extend P2ETG to store
        # snapshots; see note below.
        max_width = 0.0
        min_gap = float("inf")
        for state in learner.agent_states.values():
            widths = [hi - lo for (lo, hi) in state.ci.values()]
            if widths:
                max_width = max(max_width, max(widths))
            items = state.partners
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    g = abs(state.theta[items[i]] - state.theta[items[j]])
                    if g < min_gap:
                        min_gap = g

        diag = _diagnose_stopping(learner)
        worst_ov = diag[0] if diag is not None else float("nan")

        print(f"{e['epoch']:>5} {e['R']:>7} {e['new_samples']:>8} {e['t']:>9} "
              f"{str(e['pairwise_disjoint']):>8} "
              f"{max_width:>8.4f} {min_gap:>8.4f} {worst_ov:>9.5f}  "
              f"{e['matching']}")

    # ----------------------------------------------------------------------
    # Final diagnostic summary (worst pair that failed to separate)
    # ----------------------------------------------------------------------
    diag = _diagnose_stopping(learner)
    if diag is not None:
        margin, agent, b1, b2 = diag
        print(f"\nFinal worst pair: agent={agent}, ({b1}, {b2}) "
              f"-> overlap = {margin:+.5f}")
        print("  (overlap <= 0 means the stopping condition holds)")


if __name__ == "__main__":
    main()