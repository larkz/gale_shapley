"""Alignment tests for RouterBenchLoader."""

from __future__ import annotations

import json

import pytest

from llm_matching.routerbench_loader import RouterBenchDataError, RouterBenchLoader

from conftest import write_fake_bench

MODELS = ["modelA", "modelB", "modelC"]


def _scores_full():
    return {
        "ds1": {
            "modelA": {i: 1.0 for i in range(10)},
            "modelB": {i: 0.0 for i in range(10)},
            "modelC": {i: 0.5 for i in range(10)},
        },
        "ds2": {
            "modelA": {i: float(i % 2) for i in range(20)},
            "modelB": {i: float(i % 3 == 0) for i in range(20)},
            "modelC": {i: float(i % 4 == 0) for i in range(20)},
        },
    }


def test_every_retained_record_has_all_models(tmp_path):
    write_fake_bench(tmp_path, _scores_full())
    loader = RouterBenchLoader(tmp_path, ["ds1", "ds2"], MODELS, min_aligned_records=1)
    table = loader.load()

    assert set(table.records["dataset"]) == {"ds1", "ds2"}
    for (dataset, idx), group in table.records.groupby(["dataset", "record_index"]):
        assert set(group["model"]) == set(MODELS), (
            f"retained record ({dataset}, {idx}) lacks a model score"
        )


def test_intersection_alignment_drops_non_shared_records(tmp_path):
    scores = _scores_full()
    # modelB misses records 8 and 9 on ds1; alignment must drop them.
    scores["ds1"]["modelB"] = {i: 0.0 for i in range(8)}
    write_fake_bench(tmp_path, scores)

    loader = RouterBenchLoader(tmp_path, ["ds1"], MODELS, min_aligned_records=1)
    table = loader.load()

    ds1 = table.records[table.records["dataset"] == "ds1"]
    assert sorted(ds1["record_index"].unique()) == list(range(8))
    stats = {s.dataset: s for s in table.stats}
    assert stats["ds1"].aligned_records == 8
    assert stats["ds1"].raw_min_records == 8
    assert stats["ds1"].raw_max_records == 10


def test_missing_model_raises_with_clear_message(tmp_path):
    scores = _scores_full()
    del scores["ds2"]["modelC"]
    write_fake_bench(tmp_path, scores)

    loader = RouterBenchLoader(tmp_path, ["ds1", "ds2"], MODELS)
    with pytest.raises(RouterBenchDataError, match="modelC"):
        loader.load()


def test_missing_dataset_raises(tmp_path):
    write_fake_bench(tmp_path, {"ds1": _scores_full()["ds1"]})
    loader = RouterBenchLoader(tmp_path, ["nope"], MODELS)
    with pytest.raises(RouterBenchDataError, match="nope"):
        loader.load()


def test_missing_bench_root_raises(tmp_path):
    loader = RouterBenchLoader(tmp_path, ["ds1"], MODELS)
    with pytest.raises(RouterBenchDataError, match="bench-release"):
        loader.load()


def test_min_aligned_records_threshold(tmp_path):
    scores = _scores_full()
    scores["ds1"] = {
        "modelA": {i: 1.0 for i in range(5)},
        "modelB": {i: 0.0 for i in range(5)},
        "modelC": {i: 0.5 for i in range(5)},
    }
    write_fake_bench(tmp_path, scores)
    loader = RouterBenchLoader(tmp_path, ["ds1"], MODELS, min_aligned_records=10)
    with pytest.raises(RouterBenchDataError, match="min_aligned_records"):
        loader.load()


def test_latest_timestamp_wins(tmp_path):
    from llm_matching.routerbench_loader import _latest_json

    write_fake_bench(tmp_path, _scores_full(), timestamp="20250101_000000")
    # A newer file for modelA on ds1 with different scores.
    newer = {
        "ds1": {
            "modelA": {i: 0.0 for i in range(10)},
        }
    }
    write_fake_bench(tmp_path, newer, timestamp="20250601_120000")

    model_dir = tmp_path / "results" / "bench" / "ds1" / "test" / "modelA"
    latest = _latest_json(model_dir)
    assert latest is not None and "20250601" in latest.name

    loader = RouterBenchLoader(tmp_path, ["ds1"], MODELS, min_aligned_records=1)
    table = loader.load()
    ds1_a = table.records[
        (table.records["dataset"] == "ds1") & (table.records["model"] == "modelA")
    ]
    assert (ds1_a["score"] == 0.0).all()


def test_demo_files_skipped(tmp_path):
    write_fake_bench(tmp_path, _scores_full(), demo=True)
    loader = RouterBenchLoader(tmp_path, ["ds1"], MODELS)
    with pytest.raises(RouterBenchDataError):
        loader.load()


def test_split_resolution_prefers_complete_split(tmp_path):
    """Model present only in one split -> that split is chosen."""
    scores = _scores_full()
    write_fake_bench(tmp_path, scores, split="test")
    # Also write a smaller qualifying split.
    small = {
        "ds1": {
            m: {i: 1.0 for i in range(3)}
            for m in MODELS
        }
    }
    write_fake_bench(tmp_path, small, split="valid", timestamp="20250202_000000")

    loader = RouterBenchLoader(tmp_path, ["ds1"], MODELS)
    specs = loader.resolve_specs()
    # test has 10 records > valid has 3 -> test chosen
    assert specs["ds1"].split == "test"
