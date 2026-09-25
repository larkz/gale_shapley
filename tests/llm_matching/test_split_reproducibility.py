"""Deterministic split tests."""

from __future__ import annotations

import pandas as pd

from llm_matching.splits import make_splits

from conftest import make_records

MODELS = ["modelA", "modelB"]


def _records(n: int = 100) -> pd.DataFrame:
    scores = {
        "ds1": {
            "modelA": {i: float(i % 2) for i in range(n)},
            "modelB": {i: float(i % 3 == 0) for i in range(n)},
        }
    }
    return make_records(scores, MODELS)


def test_no_record_in_multiple_splits():
    splits = make_splits(_records(), split_seed=3407)
    counts = splits.groupby(["dataset", "record_index"])["split"].nunique()
    assert (counts == 1).all()


def test_every_record_assigned():
    records = _records()
    splits = make_splits(records, split_seed=3407)
    assert set(splits["record_index"]) == set(range(100))
    assert splits["split"].isin(["train", "val", "test"]).all()


def test_same_record_same_split_across_models():
    """The split is per (dataset, record), never per model."""
    records = _records()
    splits = make_splits(records, split_seed=3407)
    # splits table has one row per (dataset, record); merging back onto
    # the long records must give each model's row the same split.
    merged = records.merge(splits, on=["dataset", "record_index"])
    per_record = merged.groupby(["dataset", "record_index"])["split"].nunique()
    assert (per_record == 1).all()


def test_reproducible_under_same_seed():
    records = _records()
    a = make_splits(records, split_seed=3407)
    b = make_splits(records, split_seed=3407)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_split():
    records = _records()
    a = make_splits(records, split_seed=3407)
    b = make_splits(records, split_seed=1234)
    merged = a.merge(b, on=["dataset", "record_index"], suffixes=("_a", "_b"))
    assert (merged["split_a"] != merged["split_b"]).any()


def test_approximate_fractions():
    records = _records()
    splits = make_splits(records, split_seed=3407)
    sizes = splits["split"].value_counts(normalize=True)
    assert abs(sizes["train"] - 0.60) < 0.05
    assert abs(sizes["val"] - 0.20) < 0.05
    assert abs(sizes["test"] - 0.20) < 0.05


def test_multiple_datasets_independent():
    scores = {
        "ds1": {
            "modelA": {i: 1.0 for i in range(50)},
            "modelB": {i: 0.0 for i in range(50)},
        },
        "ds2": {
            "modelA": {i: 1.0 for i in range(80)},
            "modelB": {i: 0.0 for i in range(80)},
        },
    }
    records = make_records(scores, MODELS)
    splits = make_splits(records, split_seed=3407)
    for dataset in ("ds1", "ds2"):
        sub = splits[splits["dataset"] == dataset]
        assert len(sub) == (50 if dataset == "ds1" else 80)
