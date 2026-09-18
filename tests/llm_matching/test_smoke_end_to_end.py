"""End-to-end smoke test on the 4x4 configuration.

Requires the real LLMRouterBench results to be downloaded (skipped
otherwise). The run is stochastic; we only assert structural
correctness, not statistical recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llm_matching.runner import DEFAULT_CONFIG, deep_merge, run_experiment

from conftest import BENCH_DIR, requires_routerbench_data

GALE_SHAPLEY_ROOT = Path(__file__).resolve().parents[2]


def _smoke_config(tmp_path: Path) -> dict:
    user_cfg = {
        "routerbench_root": str(GALE_SHAPLEY_ROOT.parent / "LLMRouterBench"),
        "datasets": ["math500", "livecodebench", "finqa", "medqa"],
        "models": [
            "DeepSeek-R1-0528-Qwen3-8B",
            "Fin-R1",
            "Llama-3.1-8B-UltraMedical",
            "Qwen2.5-Coder-7B-Instruct",
        ],
        "p2etg": {
            "adaptive": True,
            "check_every": 24,
            "max_samples": 4000,
            "constant": 0.1,
        },
        "experiment": {"seeds": 2, "random_baseline_seeds": 50},
        "output_dir": str(tmp_path / "smoke_out"),
    }
    return deep_merge(DEFAULT_CONFIG, user_cfg)


@requires_routerbench_data
def test_smoke_end_to_end_bt(tmp_path):
    config = _smoke_config(tmp_path)
    per_seed = run_experiment(
        config, algorithm="p2etg", feedback="bt", seeds_override=[0, 1]
    )

    out = Path(config["output_dir"])

    # --- data + splits ---
    alignment = pd.read_csv(out / "data_alignment.csv")
    assert len(alignment) == 4
    assert (alignment["aligned_records"] >= 100).all()
    splits = pd.read_csv(out / "splits.csv")
    assert splits["split"].isin(["train", "val", "test"]).all()

    # --- utilities ---
    for split in ("train", "val", "test"):
        tu = pd.read_csv(out / f"task_utility_{split}.csv", index_col=0)
        mu = pd.read_csv(out / f"model_utility_{split}.csv", index_col=0)
        assert tu.shape == (4, 4)
        assert mu.shape == (4, 4)
        assert not tu.isna().any().any()

    # --- oracle ---
    with open(out / "oracle_matching_train.json") as fh:
        oracle = json.load(fh)
    assert len(oracle) == 4
    assert set(oracle.values()) == set(config["models"])
    with open(out / "oracle_preferences_train.json") as fh:
        prefs = json.load(fh)
    assert len(prefs) == 8  # 4 models + 4 tasks
    with open(out / "bt_calibration.json") as fh:
        cal = json.load(fh)
    assert cal["eta_task"] > 0 and cal["eta_model"] > 0
    assert cal["target_median_win_probability"] == pytest.approx(0.70)

    # --- traces + summaries ---
    for seed in (0, 1):
        trace = pd.read_csv(out / "traces" / f"seed_{seed:03d}.csv")
        expected_cols = {
            "seed", "feedback_mode", "algorithm", "t", "matching",
            "exact_oracle_match", "train_stable", "train_blocking_pairs",
            "test_welfare", "test_mean_score",
            "pairwise_resolved_fraction", "stopped",
        }
        assert expected_cols <= set(trace.columns)
        assert len(trace) >= 1
        assert (trace["t"].diff().dropna() > 0).all()

    assert len(per_seed) == 2
    for _, row in per_seed.iterrows():
        assert isinstance(row["test_welfare"], float)
        assert 0.0 <= row["resolved_fraction_at_stop"] <= 1.0
    assert (per_seed["test_welfare"] >= 0).all()

    summary = pd.read_csv(out / "per_seed_summary.csv")
    assert set(summary["seed"]) == {0, 1}
    agg = pd.read_csv(out / "aggregate_summary.csv")
    assert len(agg) > 0

    # --- plots + report ---
    plots_dir = out / "plots"
    for name in (
        "matching_accuracy_vs_queries.png",
        "test_welfare_vs_queries.png",
        "resolved_fraction_vs_queries.png",
        "stopping_time_distribution.png",
        "model_task_train_heatmap.png",
        "comparative_advantage_heatmap.png",
    ):
        assert (plots_dir / name).exists(), name
    report = (out / "matching_report.md").read_text()
    assert "Oracle model-proposing stable matching" in report
    assert "Hungarian" in report


@requires_routerbench_data
def test_smoke_end_to_end_replay(tmp_path):
    """Milestone-5 provider: empirical replay must also run cleanly."""
    config = _smoke_config(tmp_path)
    config["output_dir"] = str(tmp_path / "smoke_out_replay")
    per_seed = run_experiment(
        config, algorithm="p2etg", feedback="replay", seeds_override=[0]
    )
    assert len(per_seed) == 1
    trace = pd.read_csv(
        Path(config["output_dir"]) / "traces" / "seed_000.csv"
    )
    assert (trace["feedback_mode"] == "replay").all()
    assert len(trace) >= 1
