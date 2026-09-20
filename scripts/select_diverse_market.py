#!/usr/bin/env python3
"""select_diverse_market.py

Data-driven selection of a high-separation 8x8 market from the full
LLMRouterBench 20-model x 15-dataset pool.

Rules:
  1. A dataset's WINNER is its deterministic top-1 model (idxmax over
     the full pool).
  2. One dataset per winner model, maximising the winner's full-pool
     margin (top1 - top2), with the record-count constraint first and
     a margin-threshold preference cascade:
       (a) margin >= 0.005 and records >= 300
       (b) margin >= 0.001 and records >= 300
       (c) margin >= 0.001 and records >= 150
       (d) any records >= 150 (tie-at-top fallback, e.g. humaneval)
     Ties at the full-pool level usually disappear in the TRAIN split
     means, and the stable-matching structure (the tied rival has a
     stronger win elsewhere) can disambiguate them.
  3. The market's 8 models are the 8 winners.
  4. Preview: utilities, gap stats, distinct top-1 counts on both
     sides, rank correlations, exact-tie scan.

Usage:
    python scripts/select_diverse_market.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_matching.market_analysis import _baseline_pool, _latest_json, _read_records

ROOT = Path(__file__).resolve().parents[2] / "LLMRouterBench"


def record_counts(bench: Path, datasets: list[str]) -> dict[str, int]:
    counts = {}
    for dataset in datasets:
        ds_dir = bench / dataset
        n = None
        for split_dir in sorted(ds_dir.iterdir()):
            for model_dir in sorted(split_dir.iterdir()):
                latest = _latest_json(model_dir)
                if latest is None:
                    continue
                n = len(_read_records(latest, dataset, split_dir.name, model_dir.name))
                break
            if n is not None:
                break
        counts[dataset] = n or 0
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default="outputs/llm_matching_8x8/market_analysis/market_matrix.csv"
    )
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix, index_col=0)
    _, datasets_pool = _baseline_pool(ROOT)
    bench = ROOT / "results" / "bench"
    counts = record_counts(bench, datasets_pool)

    # deterministic winner + margin per dataset
    info = []
    for ds in matrix.index:
        row = matrix.loc[ds]
        winner = row.idxmax()
        ordered = row.sort_values(ascending=False)
        margin = float(ordered.iloc[0] - ordered.iloc[1])
        info.append(
            {
                "dataset": ds, "winner": winner,
                "margin": margin, "records": counts.get(ds, 0),
            }
        )
    info_df = pd.DataFrame(info)

    # one dataset per winner, threshold cascade
    chosen: dict[str, dict] = {}
    for winner, group in info_df.groupby("winner"):
        picked = None
        for min_margin, min_records in (
            (0.005, 300), (0.001, 300), (0.001, 150), (0.0, 150)
        ):
            eligible = group[
                (group["records"] >= min_records) & (group["margin"] >= min_margin)
            ]
            if not eligible.empty:
                picked = eligible.sort_values("margin", ascending=False).iloc[0]
                break
        if picked is None:
            continue
        chosen[winner] = picked

    tasks = sorted(chosen[w]["dataset"] for w in chosen)
    models = sorted(chosen.keys())

    # ---- preview ----
    sub = matrix.loc[tasks, models]
    row_sum = sub.sum(axis=1)
    other_mean = sub.rsub(row_sum, axis="index") / (len(models) - 1)
    adv = (sub - other_mean).T

    top1_task = sub.idxmax(axis=1)
    top1_model_adv = adv.idxmax(axis=1)

    task_gaps = []
    for ds in tasks:
        ordered = sub.loc[ds].sort_values(ascending=False)
        task_gaps.append(float(ordered.iloc[0] - ordered.iloc[1]))
    model_gaps = []
    for m in models:
        ordered = adv.loc[m].sort_values(ascending=False)
        model_gaps.append(float(ordered.iloc[0] - ordered.iloc[1]))

    from scipy.stats import spearmanr

    corrs = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            rho, _ = spearmanr(sub.loc[tasks[i]], sub.loc[tasks[j]])
            corrs.append(rho)

    tie_rows = [
        ds for ds in tasks if sub.loc[ds].duplicated().any()
    ]

    lines = []
    lines.append("# Diverse-Market Selection (from the full 20x15 pool)")
    lines.append("")
    lines.append("## Chosen (winner -> dataset, by margin, threshold cascade)")
    lines.append("")
    lines.append("| winner model | dataset | full-pool margin | records |")
    lines.append("|---|---|---|---|")
    for w in sorted(chosen):
        c = chosen[w]
        lines.append(
            f"| {w} | {c['dataset']} | {c['margin']:.4f} | {c['records']} |"
        )
    lines.append("")
    lines.append("## Resulting 8x8 submarket preview (full-pool means)")
    lines.append("")
    lines.append(f"* distinct task-side top-1 models: {top1_task.nunique()}/8")
    lines.append(f"* distinct model-side top tasks (V): {top1_model_adv.nunique()}/8")
    lines.append(
        f"* task-side top-2 gaps: min {min(task_gaps):.4f}, "
        f"median {float(np.median(task_gaps)):.4f}"
    )
    lines.append(
        f"* model-side top-2 gaps: min {min(model_gaps):.4f}, "
        f"median {float(np.median(model_gaps)):.4f}"
    )
    lines.append(
        f"* mean pairwise task-rank Spearman: {np.mean(corrs):.3f} "
        f"(original 8x8 market: 0.55; full pool: 0.403)"
    )
    lines.append(f"* tasks with any exact utility tie: {len(tie_rows)}/8 {tie_rows}")
    lines.append("")
    lines.append("## Task utility matrix")
    lines.append("")
    lines.append("| dataset | " + " | ".join(m[:14] for m in models) + " |")
    lines.append("|---|" + "---|" * len(models))
    for ds in tasks:
        lines.append(
            f"| {ds} | " + " | ".join(f"{sub.loc[ds, m]:.3f}" for m in models) + " |"
        )
    lines.append("")
    lines.append("## Model comparative advantage V_m(d)")
    lines.append("")
    lines.append("| model | top task | V gap (top1-top2) |")
    lines.append("|---|---|---|")
    for m in models:
        lines.append(f"| {m} | {top1_model_adv[m]} | {adv.loc[m].max() - adv.loc[m].nlargest(2).iloc[1]:.4f} |")
    lines.append("")
    lines.append("## YAML lists")
    lines.append("")
    lines.append("datasets:")
    lines.extend(f"  - {t}" for t in tasks)
    lines.append("models:")
    lines.extend(f"  - {m}" for m in models)

    report = "\n".join(lines)
    out = Path("outputs/llm_matching_8x8_diverse")
    out.mkdir(parents=True, exist_ok=True)
    (out / "market_selection.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
