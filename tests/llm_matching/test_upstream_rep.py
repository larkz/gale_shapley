"""Tests for the upstream replication harness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest
from conftest import write_fake_bench

from llm_matching.runner import DEFAULT_CONFIG, deep_merge
from llm_matching.upstream_rep import (
    build_upstream_context,
    run_single_p2etg,
    run_single_preflid,
)


def _config(tmp_path):
    datasets = ["d1", "d2", "d3"]
    models = ["m1", "m2", "m3"]
    scores = {}
    for d_idx, d in enumerate(datasets):
        scores[d] = {}
        for m_idx, m in enumerate(models):
            base = 0.2 + 0.15 * m_idx + 0.1 * d_idx
            # (d, m) interaction keeps V_m(d) non-degenerate on the
            # model side (a purely additive score matrix makes the
            # comparative advantage constant across tasks)
            interact = 0.05 * ((m_idx * 2 + d_idx) % 3)
            scores[d][m] = {
                i: base + interact + 0.02 * ((i + m_idx + d_idx) % 5)
                for i in range(10)
            }
    write_fake_bench(tmp_path, scores)
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "routerbench_root": str(tmp_path),
            "datasets": datasets,
            "models": models,
            "min_aligned_records": 1,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 3407},
            "feedback": {
                "mode": "bt", "target_median_win_probability": 0.7,
            },
            "ties": {"policy": "deterministic_jitter", "epsilon": 1e-6},
            "output_dir": str(tmp_path / "uprep_out"),
        },
    )
    return config


def test_context_builds_oracle_and_thetas(tmp_path):
    ctx = build_upstream_context(_config(tmp_path))
    assert ctx["N"] == 3 and ctx["K"] == 3
    assert len(ctx["h_star"].pairs) == 3
    assert set(ctx["theta_task"]) == {"d1", "d2", "d3"}
    assert set(ctx["theta_model"]) == {"m1", "m2", "m3"}


def test_p2etg_replication_dense_timeline_and_regret(tmp_path):
    ctx = build_upstream_context(_config(tmp_path))
    rows, summary = run_single_p2etg(
        ctx, seed=0, adaptive=True, check_every=6, max_samples=300,
        constant=0.25,
    )
    df = pd.DataFrame(rows)
    # dense timeline: consecutive t from 1 to the last check
    assert df["t"].tolist() == list(range(1, len(df) + 1))
    # 0/1 regret is the cumulative count of incorrect rounds
    expected = list((1 - df["correct"]).cumsum())
    assert df["regret"].tolist() == expected
    assert summary["T_stop"] <= 300
    assert isinstance(summary["final_regret"], int)
    # stability fields present (upstream summary schema)
    for key in ("correct_at_stop", "stable_under_truth", "stable_under_hat"):
        assert key in summary


def test_p2etg_replication_post_stop_tail(tmp_path):
    ctx = build_upstream_context(_config(tmp_path))
    # easy synthetic market + tiny constant resolves quickly
    rows, summary = run_single_p2etg(
        ctx, seed=0, adaptive=True, check_every=6, max_samples=3_000,
        constant=0.05,
    )
    if summary["stopped"]:
        t_stop = summary["T_stop"]
        tail = [r["t"] for r in rows if r["t"] > t_stop]
        assert len(tail) == 200  # POST_STOP_TAIL
        # tail rows carry the committed matching's correctness
        tail_correct = {r["correct"] for r in rows if r["t"] > t_stop}
        assert tail_correct == {summary["correct_at_stop"]}


def test_preflid_replication_runs_and_fills_horizon(tmp_path):
    ctx = build_upstream_context(_config(tmp_path))
    rows, summary = run_single_preflid(
        ctx, seed=0, budget=100_000, constant=0.05,
        max_iterations=200, horizon=3_000,
    )
    df = pd.DataFrame(rows)
    assert df["t"].max() == 3_000  # filled to the horizon
    assert df["regret"].tolist() == list((1 - df["correct"]).cumsum())
    assert "preflid_T_stop" in summary
