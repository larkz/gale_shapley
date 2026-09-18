"""Utility construction tests: U_d(m) and comparative advantage V_m(d)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_matching.utilities import (
    deterministic_jitter,
    median_positive_gap,
    model_utility,
    strictify_utilities,
    task_utility,
)

from conftest import make_records

MODELS = ["modelA", "modelB", "modelC"]
DATASETS = ["ds1", "ds2"]


def _build():
    scores = {
        "ds1": {
            "modelA": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0},
            "modelB": {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0},
            "modelC": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0},
        },
        "ds2": {
            "modelA": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            "modelB": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0},
            "modelC": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 0.0},
        },
    }
    records = make_records(scores, MODELS)
    splits = pd.DataFrame(
        [
            (d, i, "train" if i < 3 else ("val" if i == 3 else "test"))
            for d in DATASETS
            for i in range(5)
        ],
        columns=["dataset", "record_index", "split"],
    )
    return records, splits


def test_task_utility_is_mean_score():
    records, splits = _build()
    util = task_utility(records, splits, "train", DATASETS, MODELS)
    # ds1 train records: {0,1,2}
    # modelA: 1,1,0 -> 2/3 ; modelB: 0,1,1 -> 2/3 ; modelC: 1,0,1 -> 2/3
    assert util.loc["ds1", "modelA"] == pytest.approx(2 / 3)
    assert util.loc["ds1", "modelB"] == pytest.approx(2 / 3)
    assert util.loc["ds1", "modelC"] == pytest.approx(2 / 3)
    # ds2 train: modelA 0,0,0 -> 0
    assert util.loc["ds2", "modelA"] == pytest.approx(0.0)


def test_comparative_advantage_formula():
    records, splits = _build()
    util = task_utility(records, splits, "train", DATASETS, MODELS)
    v = model_utility(util)

    assert list(v.index) == MODELS
    assert list(v.columns) == DATASETS

    for model in MODELS:
        for dataset in DATASETS:
            others = [m for m in MODELS if m != model]
            expected = util.loc[dataset, model] - float(
                np.mean([util.loc[dataset, m] for m in others])
            )
            assert v.loc[model, dataset] == pytest.approx(expected)


def test_comparative_advantage_is_zero_sum_within_task():
    """sum_m V_m(d) = 0 for every task d."""
    records, splits = _build()
    util = task_utility(records, splits, "train", DATASETS, MODELS)
    v = model_utility(util)
    for dataset in DATASETS:
        assert v[dataset].sum() == pytest.approx(0.0, abs=1e-12)


def test_strictify_only_touches_ties():
    util = pd.DataFrame(
        [[0.5, 0.5, 0.9], [0.1, 0.7, 0.7]],
        index=["agent1", "agent2"],
        columns=["p1", "p2", "p3"],
    )
    strict, n_changed, report = strictify_utilities(
        util, tie_epsilon=1e-6, split_seed=3407, agent_kind="test"
    )
    # agent1: p1,p2 tied -> both changed; p3 untouched.
    assert n_changed == 4  # 2 ties for agent1 + 2 for agent2
    assert len(report) == 2
    assert strict.loc["agent1", "p3"] == pytest.approx(0.9)
    assert strict.loc["agent2", "p1"] == pytest.approx(0.1)
    # Ties are strictly ordered after jitter.
    assert strict.loc["agent1", "p1"] != strict.loc["agent1", "p2"]
    assert strict.loc["agent2", "p2"] != strict.loc["agent2", "p3"]
    # The tie-group order is reproducible.
    strict2, _, _ = strictify_utilities(
        util, tie_epsilon=1e-6, split_seed=3407, agent_kind="test"
    )
    pd.testing.assert_frame_equal(strict, strict2)


def test_strictify_preserves_non_tied_ordering():
    util = pd.DataFrame(
        [[0.10, 0.50, 0.90]], index=["a"], columns=["p1", "p2", "p3"]
    )
    strict, n_changed, _ = strictify_utilities(
        util, tie_epsilon=1e-6, split_seed=3407, agent_kind="test"
    )
    assert n_changed == 0
    assert list(strict.iloc[0]) == [0.10, 0.50, 0.90]


def test_deterministic_jitter_stable():
    a = deterministic_jitter("agent", "partner", 3407)
    b = deterministic_jitter("agent", "partner", 3407)
    c = deterministic_jitter("agent", "other", 3407)
    assert a == b
    assert -1.0 <= a < 1.0
    assert a != c


def test_median_positive_gap():
    util = pd.DataFrame(
        [[0.0, 0.02, 0.04, 0.10]], index=["a"], columns=["p1", "p2", "p3", "p4"]
    )
    # gaps: |p1-p2|=.02 |p1-p3|=.04 |p1-p4|=.10 |p2-p3|=.02 |p2-p4|=.08 |p3-p4|=.06
    # sorted: [.02, .02, .04, .06, .08, .10] -> median = (.04 + .06) / 2 = .05
    med = median_positive_gap(util, tie_epsilon=1e-9)
    assert med == pytest.approx(0.05)


def test_median_positive_gap_excludes_ties():
    util = pd.DataFrame(
        [[0.0, 0.0, 0.02, 0.10]], index=["a"], columns=["p1", "p2", "p3", "p4"]
    )
    # gaps > eps: |p1-p3|=.02 |p1-p4|=.10 |p2-p3|=.02 |p2-p4|=.10 |p3-p4|=.08
    # sorted [.02, .02, .08, .10, .10] -> median .08
    med = median_positive_gap(util, tie_epsilon=1e-6)
    assert med == pytest.approx(0.08)
