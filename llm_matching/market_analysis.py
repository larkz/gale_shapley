"""market_analysis.py

Diagnostic analysis of the FULL available LLMRouterBench performance
matrix (the official ~20-model x ~15-dataset baseline pool, discovered
from the LLMRouterBench checkout).

This is a DIAGNOSTIC only: it does not redefine the main 8x8 market.

Reports:
  * per-dataset best model
  * number of distinct top-1 models
  * utility variance per dataset
  * mean pairwise Spearman correlation between task rankings
  * number of unique task-side top models / model-side top tasks
  * heatmaps: full performance matrix, task-rank correlation matrix
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from llm_matching.routerbench_loader import (
    RouterBenchDataError,
    RouterBenchLoader,
    _latest_json,
    _read_records,
)

logger = logging.getLogger(__name__)


def _baseline_pool(root: Path) -> Tuple[List[str], List[str]]:
    """Read the official baseline model/dataset pool from the
    LLMRouterBench baseline config (read-only data dependency)."""
    cfg_path = root / "config" / "baseline_config.yaml"
    if cfg_path.exists():
        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            filters = (cfg.get("baseline", {}) or {}).get("filters", {}) or {}
            datasets = filters.get("datasets") or []
            models = filters.get("models") or []
            if datasets and models:
                return list(models), list(datasets)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse baseline_config.yaml: %s", exc)

    # Fallback: discover everything under results/bench.
    bench = root / "results" / "bench"
    if not bench.is_dir():
        raise RouterBenchDataError(
            f"No benchmark results under {bench}; download the "
            "bench-release.tar.gz first."
        )
    datasets = sorted(d.name for d in bench.iterdir() if d.is_dir())
    models = set()
    for d in datasets:
        for split in (bench / d).iterdir():
            if split.is_dir():
                models.update(m.name for m in split.iterdir() if m.is_dir())
    return sorted(models), datasets


def load_full_matrix(root: Path) -> pd.DataFrame:
    """Mean score per (dataset, model) over the full baseline pool.

    Returns a DataFrame index=dataset, columns=model, with NaN where a
    (dataset, model) result is unavailable.
    """
    models, datasets = _baseline_pool(root)
    bench = root / "results" / "bench"
    if not bench.is_dir():
        raise RouterBenchDataError(
            f"No benchmark results under {bench}."
        )

    rows: Dict[str, Dict[str, float]] = {}
    available_dirs = {d.name.lower(): d.name for d in bench.iterdir() if d.is_dir()}
    for dataset in datasets:
        key = dataset.strip().lower()
        ds_dir_name = available_dirs.get(key)
        if ds_dir_name is None:
            logger.warning("Dataset %s not on disk; skipped", dataset)
            continue
        ds_dir = bench / ds_dir_name
        rows[dataset] = {}
        for model in models:
            # find the first split that contains this model
            for split_dir in sorted(ds_dir.iterdir()):
                model_dir = split_dir / model
                if not model_dir.is_dir():
                    continue
                latest = _latest_json(model_dir)
                if latest is None:
                    continue
                records = _read_records(
                    latest, ds_dir_name, split_dir.name, model
                )
                if records:
                    scores = [v["score"] for v in records.values()]
                    rows[dataset][model] = float(np.mean(scores))
                break
    return pd.DataFrame(rows).T.reindex(columns=models)


def analyze_market(matrix: pd.DataFrame) -> Dict[str, object]:
    """Diversity metrics for the full performance matrix."""
    from scipy.stats import spearmanr

    complete = matrix.dropna(axis=0, how="any")
    report: Dict[str, object] = {
        "n_datasets": int(matrix.shape[0]),
        "n_models": int(matrix.shape[1]),
        "n_complete_datasets": int(complete.shape[0]),
    }

    # per-dataset best model (over models with data)
    best = matrix.idxmax(axis=1)
    report["per_dataset_best"] = {
        ds: str(m) for ds, m in best.items() if pd.notna(m)
    }
    report["n_distinct_top1_models"] = int(best.dropna().nunique())

    # utility variance per dataset (across models)
    report["utility_variance_per_dataset"] = {
        ds: float(np.nanvar(matrix.loc[ds].to_numpy()))
        for ds in matrix.index
    }

    # task-ranking correlations (pairwise Spearman between dataset
    # columns of model scores) on the complete submatrix
    if complete.shape[0] >= 2:
        corr = pd.DataFrame(
            np.eye(len(complete.index)),
            index=complete.index,
            columns=complete.index,
        )
        ds_list = list(complete.index)
        for i in range(len(ds_list)):
            for j in range(i + 1, len(ds_list)):
                rho, _ = spearmanr(
                    complete.loc[ds_list[i]].to_numpy(),
                    complete.loc[ds_list[j]].to_numpy(),
                )
                if pd.isna(rho):
                    rho = 0.0
                corr.loc[ds_list[i], ds_list[j]] = rho
                corr.loc[ds_list[j], ds_list[i]] = rho
        iu = np.triu_indices(len(corr), k=1)
        report["mean_pairwise_task_rank_correlation"] = float(
            corr.to_numpy()[iu].mean()
        )
        report["task_rank_correlation"] = corr
    else:
        report["mean_pairwise_task_rank_correlation"] = float("nan")
        report["task_rank_correlation"] = None

    # model-side comparative advantage (within the complete submatrix)
    if complete.shape[1] >= 2 and complete.shape[0] >= 2:
        row_sum = complete.sum(axis=1)
        other_mean = complete.rsub(row_sum, axis="index") / (
            complete.shape[1] - 1
        )
        adv = (complete - other_mean).T  # index=model, columns=dataset
        top_task = adv.idxmax(axis=1)
        report["n_unique_model_side_top_tasks"] = int(top_task.nunique())
        report["model_side_top_task"] = {
            str(m): str(t) for m, t in top_task.items()
        }
    else:
        report["n_unique_model_side_top_tasks"] = 0
        report["model_side_top_task"] = {}

    return report


def run_market_analysis(config: Dict) -> pd.DataFrame:
    """Load the full matrix, analyze, save outputs + plots + report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(config["routerbench_root"])
    out_dir = Path(config["output_dir"]) / "market_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_full_matrix(root)
    matrix.to_csv(out_dir / "market_matrix.csv")

    report = analyze_market(matrix)

    # heatmap: full performance matrix
    fig, ax = plt.subplots(
        figsize=(1.2 + 0.55 * matrix.shape[1], 1.2 + 0.5 * matrix.shape[0])
    )
    im = ax.imshow(
        matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto",
        vmin=0, vmax=1,
    )
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(list(matrix.columns), rotation=90, fontsize=6)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(list(matrix.index), fontsize=7)
    ax.set_title("Full LLMRouterBench performance matrix (mean score)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "full20x15_task_model_heatmap.png", dpi=170)
    plt.close(fig)

    # task rank correlation heatmap
    corr = report.get("task_rank_correlation")
    if corr is not None:
        fig, ax = plt.subplots(
            figsize=(1.2 + 0.55 * corr.shape[1], 1.2 + 0.5 * corr.shape[0])
        )
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_xticklabels(list(corr.columns), rotation=90, fontsize=7)
        ax.set_yticks(range(corr.shape[0]))
        ax.set_yticklabels(list(corr.index), fontsize=7)
        ax.set_title("Spearman correlation between task rankings", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / "task_rank_correlation_heatmap.png", dpi=170)
        plt.close(fig)
        corr.to_csv(out_dir / "task_rank_correlation.csv")

    # report
    lines: List[str] = []
    lines.append("# Full Market Analysis (diagnostic)")
    lines.append("")
    lines.append(
        f"* matrix: {report['n_datasets']} datasets x {report['n_models']} "
        f"models; complete rows: {report['n_complete_datasets']}"
    )
    lines.append(
        f"* distinct per-dataset top-1 models: "
        f"{report['n_distinct_top1_models']}"
    )
    lines.append(
        f"* mean pairwise task-rank Spearman correlation: "
        f"{report['mean_pairwise_task_rank_correlation']:.3f}"
    )
    lines.append(
        f"* unique model-side top tasks: "
        f"{report['n_unique_model_side_top_tasks']}"
    )
    lines.append("")
    lines.append("## Per-dataset best model")
    lines.append("")
    lines.append("| dataset | best model | utility variance |")
    lines.append("|---|---|---|")
    for ds, m in report["per_dataset_best"].items():
        lines.append(
            f"| {ds} | {m} "
            f"| {report['utility_variance_per_dataset'].get(ds, float('nan')):.4f} |"
        )
    lines.append("")
    (out_dir / "market_analysis_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    logger.info("Market analysis written to %s", out_dir)
    return matrix
