"""Tests for the online matching-bandit experiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest
from conftest import write_fake_bench

from llm_matching.bandit import (
    build_bandit_context,
    run_bandit_seed,
    run_matching_bandit,
)
from llm_matching.runner import DEFAULT_CONFIG, deep_merge
import yaml


def _bandit_config(tmp_path, datasets, models):
    write_fake_bench(tmp_path, _fake_scores(datasets, models))
    config = {
        "routerbench_root": str(tmp_path),
        "datasets": datasets,
        "models": models,
        "min_aligned_records": 1,
        "split": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 3407},
        "feedback": {"mode": "bt", "target_median_win_probability": 0.7},
        "ties": {"policy": "deterministic_jitter", "epsilon": 1e-6},
        "preferences": {"mode": "symmetric"},
        "p2etg": {
            "adaptive": True, "check_every": 6, "max_samples": 600,
            "constant": 0.1,
        },
        "preflid": {
            "budget": 100, "max_iterations": 100, "constant": 0.1,
            "max_lattice_vertices": 5000, "min_samples_per_pair": 1,
            "min_sample_ratio": 0.5,
        },
        "experiment": {"seeds": 2, "random_baseline_seeds": 50},
        "output_dir": str(tmp_path / "bandit_out"),
    }
    return deep_merge(DEFAULT_CONFIG, config)


def _fake_scores(datasets, models):
    """Nested dict: scores[dataset][model] = {record_index: score}."""
    scores = {}
    for d_idx, d in enumerate(datasets):
        scores[d] = {}
        for m_idx, m in enumerate(models):
            base = 0.2 + 0.15 * m_idx + 0.1 * d_idx
            scores[d][m] = {
                i: base + 0.02 * ((i + m_idx + d_idx) % 5)
                for i in range(10)
            }
    return scores


def test_bandit_context_unique_stable_matching(tmp_path):
    config = _bandit_config(
        tmp_path, ["d1", "d2", "d3"], ["m1", "m2", "m3"]
    )
    ctx = build_bandit_context(config)
    # unique oracle: dict task -> model, one per model
    assert set(ctx.oracle.values()) == {"m1", "m2", "m3"}
    assert len(ctx.oracle) == 3
    # w_star at least as good as random, and finite
    assert ctx.w_star > ctx.w_star - ctx.regret_random
    assert ctx.w_hungarian >= ctx.w_star - 1e-9


def test_bandit_context_requires_symmetric_mode(tmp_path):
    config = _bandit_config(
        tmp_path, ["d1", "d2", "d3"], ["m1", "m2", "m3"]
    )
    config["preferences"]["mode"] = "comparative"
    with pytest.raises(ValueError, match="symmetric"):
        build_bandit_context(config)


def test_bandit_p2etg_seed_produces_regret_curve(tmp_path):
    config = _bandit_config(
        tmp_path, ["d1", "d2", "d3"], ["m1", "m2", "m3"]
    )
    ctx = build_bandit_context(config)
    rows, summary = run_bandit_seed(ctx, "p2etg", 0, budget=600)
    assert len(rows) >= 1
    # clipped convention: per-round regret >= 0 and cumulative regret
    # is non-decreasing (beating H* in welfare earns no credit)
    assert (rows["regret"] >= 0).all()
    assert rows["cum_regret"].notna().all()
    assert (rows["cum_regret"].diff().fillna(0) >= -1e-9).all()
    assert rows["t"].is_monotonic_increasing
    assert "final_exact" in summary
    assert summary["T_stop"] <= 600


def test_bandit_preflid_seed_records_per_iteration(tmp_path):
    config = _bandit_config(
        tmp_path, ["d1", "d2", "d3"], ["m1", "m2", "m3"]
    )
    ctx = build_bandit_context(config)
    rows, summary = run_bandit_seed(ctx, "preflid", 0, budget=600)
    # one row per RRT iteration (or until certification)
    assert len(rows) >= 10
    assert rows["t"].is_monotonic_increasing


def test_bandit_full_study_end_to_end(tmp_path):
    config = _bandit_config(
        tmp_path, ["d1", "d2", "d3"], ["m1", "m2", "m3"]
    )
    summary = run_matching_bandit(
        config, algorithms=["p2etg", "preflid"], seeds=[0, 1], budget=600
    )
    assert len(summary) == 4  # 2 algorithms x 2 seeds
    out = Path(config["output_dir"]) / "bandit"
    assert (out / "regret_traces.csv").exists()
    assert (out / "per_seed_summary.csv").exists()
    assert (out / "cumulative_regret.png").exists()
    assert (out / "bandit_report.md").exists()
