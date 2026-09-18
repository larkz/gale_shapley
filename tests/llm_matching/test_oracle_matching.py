"""Oracle preference construction and GS matching tests."""

from __future__ import annotations

import pandas as pd
import pytest
from gs_lib.gs_tools import Man, Woman

from llm_matching.metrics import count_blocking_pairs, is_stable
from llm_matching.oracle import build_market, matching_equal, oracle_matching


def _utils():
    """3 models x 3 tasks with non-trivial cross preferences."""
    task_util = pd.DataFrame(
        # rows: tasks, cols: models
        [
            {"m1": 0.9, "m2": 0.5, "m3": 0.4},   # t1: m1 > m2 > m3
            {"m1": 0.3, "m2": 0.8, "m3": 0.35},  # t2: m2 > m3 > m1
            {"m1": 0.2, "m2": 0.25, "m3": 0.9},  # t3: m3 > m2 > m1
        ],
        index=["t1", "t2", "t3"],
    )
    # V_m(d) = U_d(m) - mean over OTHER models (explicit, row-aligned).
    rows = {}
    models = list(task_util.columns)
    for m in models:
        others = [x for x in models if x != m]
        rows[m] = {
            d: task_util.loc[d, m] - sum(task_util.loc[d, x] for x in others) / len(others)
            for d in task_util.index
        }
    model_util = pd.DataFrame(rows).T
    model_util.index.name = "model"
    return task_util, model_util


def test_build_market_sides_and_rankings():
    task_util, model_util = _utils()
    men, women, prefs = build_market(task_util, model_util)

    assert [m.id for m in men] == ["m1", "m2", "m3"]
    assert [w.id for w in women] == ["t1", "t2", "t3"]
    assert [p.id for p in prefs.get_preference(Woman("t1"))] == ["m1", "m2", "m3"]
    assert [p.id for p in prefs.get_preference(Woman("t2"))] == ["m2", "m3", "m1"]
    # m3's comparative advantage: t3 highest
    assert [p.id for p in prefs.get_preference(Man("m3"))][0] == "t3"


def test_oracle_matching_is_stable_and_perfect():
    task_util, model_util = _utils()
    men, women, prefs = build_market(task_util, model_util)
    matching = oracle_matching(prefs)

    assert is_stable(prefs, matching)
    assert count_blocking_pairs(prefs, matching) == 0
    assert len(matching.pairs) == 3
    assert not matching.unmatched_men and not matching.unmatched_women
    partners = {p.woman.id: p.man.id for p in matching.pairs}
    assert set(partners.keys()) == {"t1", "t2", "t3"}
    assert set(partners.values()) == {"m1", "m2", "m3"}


def test_oracle_matching_model_optimal():
    """Men-proposing GS gives each man his best stable partner."""
    task_util, model_util = _utils()
    men, women, prefs = build_market(task_util, model_util)
    matching = oracle_matching(prefs)

    woman_optimal = prefs  # sanity: run women-proposing too
    from gs_lib.gs_tools import GaleShapley

    m_opt = GaleShapley(prefs).find_stable_matching("men")
    w_opt = GaleShapley(prefs).find_stable_matching("women")
    assert matching_equal(matching, m_opt)
    # Men are (weakly) better off under men-proposing.
    for man in men:
        rank_m = prefs.get_rank(man, m_opt.get_partner(man))
        rank_w = prefs.get_rank(man, w_opt.get_partner(man))
        assert rank_m <= rank_w


def test_blocking_pair_counter_detects_instability():
    task_util, model_util = _utils()
    men, women, prefs = build_market(task_util, model_util)
    stable = oracle_matching(prefs)
    assert count_blocking_pairs(prefs, stable) == 0

    # Swap two men's partners -> count at least one blocking pair.
    from gs_lib.gs_tools import Matching

    swapped = {p.man: p.woman for p in stable.pairs}
    keys = sorted(swapped.keys(), key=lambda m: m.id)
    swapped[keys[0]], swapped[keys[1]] = swapped[keys[1]], swapped[keys[0]]
    bad = Matching.from_dict(swapped, set(men), set(women))
    assert count_blocking_pairs(prefs, bad) >= 1
    assert not is_stable(prefs, bad)
