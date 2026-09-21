"""Tests for the symmetric (mirror) preference mode.

The mirror construction: task d's ranking of models and model m's
ranking of tasks both derive from ONE shared matrix W (task d's
preference over models = model m's preference over tasks, mirrored).
With a strictly ordered W (no ties in any row or column):

  * the stable matching is UNIQUE (mutual-best cascade argument);
  * men-proposing GS == women-proposing GS == the cascade;
  * both sides' preference lists are exact mirrors of W.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest
from gs_lib.gs_tools import GaleShapley, Man, Woman

from llm_matching.oracle import build_market, oracle_matching
from llm_matching.utilities import (
    mutual_best_cascade,
    symmetric_strict_matrix,
)


def _market(n: int):
    men = [Man(f"m{i}") for i in range(n)]
    women = [Woman(f"t{i}") for i in range(n)]
    return men, women


def _with_exact_ties():
    return pd.DataFrame(
        # exact tie: t0 has m0 == m1 == 0.9
        [[0.9, 0.9, 0.3], [0.5, 0.7, 0.6], [0.2, 0.4, 0.8]],
        index=["t0", "t1", "t2"], columns=["m0", "m1", "m2"],
    )


def test_symmetric_matrix_is_strict_rows_and_columns():
    W, n_changed, ties = symmetric_strict_matrix(_with_exact_ties(), 1e-6, 3407)
    # every row strictly ordered
    for d in W.index:
        vals = sorted(W.loc[d].tolist())
        assert all(v2 > v1 for v1, v2 in zip(vals, vals[1:]))
    # every column strictly ordered
    for m in W.columns:
        vals = sorted(W[m].tolist())
        assert all(v2 > v1 for v1, v2 in zip(vals, vals[1:]))
    # jitter is tiny: values stay within eps of the originals
    U = _with_exact_ties()
    assert (W - U).abs().max().max() < 1e-6
    # exact-tie group reported
    assert any(g["agent"] == "t0" for g in ties)


def test_symmetric_matrix_reproducible():
    W1, _, _ = symmetric_strict_matrix(_with_exact_ties(), 1e-6, 3407)
    W2, _, _ = symmetric_strict_matrix(_with_exact_ties(), 1e-6, 3407)
    pd.testing.assert_frame_equal(W1, W2)


def test_mirror_preferences_unique_stable_matching():
    W, _, _ = symmetric_strict_matrix(_with_exact_ties(), 1e-6, 3407)
    model_util = W.T
    men, women, prefs = build_market(W, model_util)

    # mirror property: task t's ranking == model column ordering of W.T
    for w in women:
        task_ranking = list(prefs.get_preference(w))
        expected = sorted(
            men, key=lambda m: -W.loc[w.id, m.id]
        )
        assert task_ranking == expected
    for m in men:
        model_ranking = list(prefs.get_preference(m))
        expected = sorted(
            women, key=lambda w: -W.loc[w.id, m.id]
        )
        assert model_ranking == expected

    # uniqueness: men-GS == women-GS == cascade == oracle
    h_men = GaleShapley(prefs).find_stable_matching("men")
    h_women = GaleShapley(prefs).find_stable_matching("women")
    cascade = mutual_best_cascade(W)
    oracle = oracle_matching(prefs)

    pairs = lambda h: frozenset((p.woman.id, p.man.id) for p in h.pairs)
    assert pairs(h_men) == pairs(h_women) == pairs(oracle)
    assert dict(cascade) == {t: m for t, m in pairs(oracle)}


def test_cascade_matches_global_max_argument():
    # cascade: first pair must be the global max of W
    W, _, _ = symmetric_strict_matrix(_with_exact_ties(), 1e-6, 3407)
    cascade = mutual_best_cascade(W)
    d_star, m_star = max(
        ((d, m) for d in W.index for m in W.columns),
        key=lambda dm: W.loc[dm[0], dm[1]],
    )
    assert cascade[d_star] == m_star


def test_mirror_uniqueness_on_random_matrices():
    rng = random.Random(0)
    for trial in range(10):
        n = rng.choice([2, 3, 4, 5])
        U = pd.DataFrame(
            rng.random() for _ in range(n * n)
        ).values.reshape(n, n)
        U = pd.DataFrame(
            U, index=[f"t{i}" for i in range(n)],
            columns=[f"m{i}" for i in range(n)],
        )
        # inject exact ties sometimes
        if trial % 3 == 0:
            U.iloc[0, 0] = U.iloc[0, 1]
        W, _, _ = symmetric_strict_matrix(U, 1e-6, 3407)
        men, women, prefs = build_market(W, W.T)
        h_men = GaleShapley(prefs).find_stable_matching("men")
        h_women = GaleShapley(prefs).find_stable_matching("women")
        cascade = mutual_best_cascade(W)
        pairs = lambda h: frozenset((p.woman.id, p.man.id) for p in h.pairs)
        assert pairs(h_men) == pairs(h_women), f"trial {trial}"
        assert dict(cascade) == {t: m for t, m in pairs(h_men)}, f"trial {trial}"
