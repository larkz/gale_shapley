"""bootstrap.py

Bootstrap stability analysis of the oracle stable matching.

Protocol (per replicate b):
  1. For every task dataset d, independently resample TRAIN instances
     WITH replacement (same resampled indices for all models on that
     task, so utility differences remain within-instance comparable).
  2. Recompute U_d^(b)(m) on the resample, then V_m^(b)(d).
  3. Strictify exact ties with the existing deterministic SHA256 tie
     convention (same tie_epsilon and split_seed as the base market).
  4. H*^(b) = model-proposing Gale-Shapley on the strictified profile.

Every replicate is independently reproducible from
(bootstrap_seed, replicate index).

The bootstrap answers: is the empirical oracle matching H*_train a
statistical artefact of the particular TRAIN sample, or is it robust to
resampling noise in the instance-level scores?
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from llm_matching.oracle import build_market, oracle_matching
from llm_matching.utilities import strictify_utilities

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    num_bootstrap: int
    bootstrap_seed: int
    # (replicate, task, model) long frames of resampled utilities
    task_utility_replicates: pd.DataFrame
    model_utility_replicates: pd.DataFrame
    # (replicate, task, model) long frame of bootstrap assignments
    assignments: pd.DataFrame
    assignment_frequencies: pd.DataFrame
    exact_matching_frequencies: pd.DataFrame
    utility_uncertainty_task: pd.DataFrame
    utility_uncertainty_model: pd.DataFrame
    bootstrap_matchings: List[Dict[str, str]]
    base_oracle: Dict[str, str]
    p_equal_train_oracle: float
    n_distinct_matchings: int
    # preference-reversal probability per (side, agent, p1, p2)
    preference_flips: pd.DataFrame
    elapsed_seconds: float = 0.0


# ============================================================================
# Core bootstrap
# ============================================================================

def _resample_indices(
    n: int, bootstrap_seed: int, replicate: int, dataset: str
) -> np.ndarray:
    """Deterministic per-dataset bootstrap resample of range(n).

    Each (replicate, dataset) pair gets its own independent stream,
    seeded via a stable SHA256 tag (never Python hash()).
    """
    ds_tag = int.from_bytes(
        hashlib.sha256(dataset.encode("utf-8")).digest()[:4], "big"
    )
    rng = np.random.default_rng([bootstrap_seed, replicate, ds_tag])
    return rng.integers(0, n, size=n)


def run_bootstrap(
    ctx,
    num_bootstrap: int = 500,
    bootstrap_seed: int = 20260918,
) -> BootstrapResult:
    """Run the bootstrap stability analysis from an ExperimentContext.

    `ctx` must provide: aligned, splits, task_util (train raw),
    model_util (train raw), oracle['train'], config (ties, split),
    datasets/models ordering.
    """
    t0 = time.monotonic()

    datasets = list(ctx.config["datasets"])
    models = list(ctx.config["models"])
    tie_epsilon = float(ctx.config["ties"]["epsilon"])
    split_seed = int(ctx.config["split"]["seed"])
    pref_mode = str(ctx.config.get("preferences", {}).get("mode", "comparative"))

    # Per-dataset TRAIN score matrices [n_instances x n_models].
    merged = ctx.aligned.records.merge(
        ctx.splits, on=["dataset", "record_index"], how="inner"
    )
    train = merged[merged["split"] == "train"]

    score_matrices: Dict[str, np.ndarray] = {}
    for dataset in datasets:
        sub = train[train["dataset"] == dataset]
        wide = sub.pivot(index="record_index", columns="model", values="score")
        wide = wide.reindex(columns=models).sort_index()
        score_matrices[dataset] = wide.to_numpy(dtype=float)

    base_task_util = ctx.task_util["train"]  # raw, index=dataset
    base_model_util = ctx.model_util["train"]  # raw, index=model

    task_rows: List[dict] = []
    model_rows: List[dict] = []
    assign_rows: List[dict] = []
    bootstrap_matchings: List[Dict[str, str]] = []

    task_util_df = pd.DataFrame(index=datasets, columns=models, dtype=float)
    model_util_df = pd.DataFrame(index=models, columns=datasets, dtype=float)

    # Raw signs for flip computation (agent-wise utility vectors).
    # task side: base sign per (dataset, model_i, model_j)
    # model side: base sign per (model, dataset_i, dataset_j)

    for b in range(num_bootstrap):
        # --- resampled utilities ---
        for dataset in datasets:
            mat = score_matrices[dataset]
            idx = _resample_indices(mat.shape[0], bootstrap_seed, b, dataset)
            task_util_df.loc[dataset] = mat[idx].mean(axis=0)
        if pref_mode == "symmetric":
            # mirror mode: model side ranks by the SAME resampled U
            model_util_df = task_util_df.T
        else:
            # V_m^(b) from U^(b) (same formula as utilities.model_utility)
            row_sum = task_util_df.sum(axis=1)
            other_mean = task_util_df.rsub(row_sum, axis="index") / (
                len(models) - 1
            )
            model_util_df = (task_util_df - other_mean).T

        for dataset in datasets:
            for model in models:
                task_rows.append(
                    {
                        "replicate": b,
                        "dataset": dataset,
                        "model": model,
                        "utility": float(task_util_df.loc[dataset, model]),
                    }
                )
        for model in models:
            for dataset in datasets:
                model_rows.append(
                    {
                        "replicate": b,
                        "model": model,
                        "dataset": dataset,
                        "utility": float(model_util_df.loc[model, dataset]),
                    }
                )

        # --- strictify exact ties (same convention as the base market) ---
        tu_strict, _, _ = strictify_utilities(
            task_util_df, tie_epsilon, split_seed, agent_kind="task"
        )
        mu_strict, _, _ = strictify_utilities(
            model_util_df, tie_epsilon, split_seed, agent_kind="model"
        )

        # --- model-proposing GS ---
        _, _, prefs = build_market(tu_strict, mu_strict)
        h_b = oracle_matching(prefs)
        assignment = {p.woman.id: p.man.id for p in h_b.pairs}
        bootstrap_matchings.append(assignment)
        for dataset, model in assignment.items():
            assign_rows.append(
                {"replicate": b, "task": dataset, "model": model}
            )

    task_rep = pd.DataFrame(task_rows)
    model_rep = pd.DataFrame(model_rows)
    assignments = pd.DataFrame(assign_rows)

    # --- assignment frequencies ---
    base_oracle = {
        p.woman.id: p.man.id for p in ctx.oracle["train"].pairs
    }
    counts = (
        assignments.groupby(["task", "model"]).size().reset_index(name="count")
    )
    counts["frequency"] = counts["count"] / num_bootstrap
    counts["oracle_train_assignment"] = counts["task"].map(base_oracle)
    counts["is_oracle_assignment"] = (
        counts["model"] == counts["oracle_train_assignment"]
    )
    assignment_frequencies = counts.sort_values(
        ["task", "frequency"], ascending=[True, False]
    ).reset_index(drop=True)

    # --- exact matching frequencies ---
    matching_strs = [
        "|".join(f"{t}={m}" for t, m in sorted(a.items()))
        for a in bootstrap_matchings
    ]
    base_str = "|".join(f"{t}={m}" for t, m in sorted(base_oracle.items()))
    mc = pd.Series(matching_strs).value_counts().reset_index()
    mc.columns = ["matching", "count"]
    mc["frequency"] = mc["count"] / num_bootstrap
    mc["is_train_oracle"] = mc["matching"] == base_str
    exact_matching_frequencies = mc.sort_values(
        "count", ascending=False
    ).reset_index(drop=True)

    p_equal = float(
        exact_matching_frequencies.loc[
            exact_matching_frequencies["is_train_oracle"], "count"
        ].sum()
        / num_bootstrap
    )

    # --- utility uncertainty ---
    def _uncertainty(rep: pd.DataFrame, key1: str, key2: str) -> pd.DataFrame:
        g = rep.groupby([key1, key2])["utility"]
        out = g.agg(
            mean="mean", std="std",
            p2_5=lambda s: s.quantile(0.025),
            p50=lambda s: s.quantile(0.50),
            p97_5=lambda s: s.quantile(0.975),
        ).reset_index()
        return out

    utility_uncertainty_task = _uncertainty(task_rep, "dataset", "model")
    utility_uncertainty_model = _uncertainty(model_rep, "model", "dataset")

    # --- preference reversal probabilities (Deliverable B input) ---
    flips = _preference_flip_probabilities(
        task_rep, model_rep, base_task_util, base_model_util,
        datasets, models,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Bootstrap done: %d replicates in %.1fs; P(H_b == H_train)=%.3f; "
        "%d distinct matchings",
        num_bootstrap, elapsed, p_equal,
        len(exact_matching_frequencies),
    )

    return BootstrapResult(
        num_bootstrap=num_bootstrap,
        bootstrap_seed=bootstrap_seed,
        task_utility_replicates=task_rep,
        model_utility_replicates=model_rep,
        assignments=assignments,
        assignment_frequencies=assignment_frequencies,
        exact_matching_frequencies=exact_matching_frequencies,
        utility_uncertainty_task=utility_uncertainty_task,
        utility_uncertainty_model=utility_uncertainty_model,
        bootstrap_matchings=bootstrap_matchings,
        base_oracle=base_oracle,
        p_equal_train_oracle=p_equal,
        n_distinct_matchings=len(exact_matching_frequencies),
        preference_flips=flips,
        elapsed_seconds=elapsed,
    )


def _preference_flip_probabilities(
    task_rep: pd.DataFrame,
    model_rep: pd.DataFrame,
    base_task_util: pd.DataFrame,
    base_model_util: pd.DataFrame,
    datasets: List[str],
    models: List[str],
) -> pd.DataFrame:
    """P_b( sign(u_b(i) - u_b(j)) != sign(u(i) - u(j)) ) per arm.

    Uses the RAW base utilities; a base exact tie (sign 0) counts as
    flipped whenever the replicate has any strict order, which matches
    the interpretation 'the preference direction is not stable'.
    """
    rows: List[dict] = []
    n_b = int(task_rep["replicate"].nunique())

    # Task side: agent = dataset, partners = models
    for dataset in datasets:
        sub = task_rep[task_rep["dataset"] == dataset]
        wide = sub.pivot(index="replicate", columns="model", values="utility")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                base = float(base_task_util.loc[dataset, m1]) - float(
                    base_task_util.loc[dataset, m2]
                )
                diff = wide[m1] - wide[m2]
                if base == 0.0:
                    flips = int((diff != 0).sum())
                else:
                    flips = int((np.sign(diff) != np.sign(base)).sum())
                rows.append(
                    {
                        "side": "task",
                        "agent": dataset,
                        "partner_1": m1,
                        "partner_2": m2,
                        "base_gap": abs(base),
                        "bootstrap_flip_probability": flips / n_b,
                    }
                )

    # Model side: agent = model, partners = datasets
    for model in models:
        sub = model_rep[model_rep["model"] == model]
        wide = sub.pivot(index="replicate", columns="dataset", values="utility")
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                d1, d2 = datasets[i], datasets[j]
                base = float(base_model_util.loc[model, d1]) - float(
                    base_model_util.loc[model, d2]
                )
                diff = wide[d1] - wide[d2]
                if base == 0.0:
                    flips = int((diff != 0).sum())
                else:
                    flips = int((np.sign(diff) != np.sign(base)).sum())
                rows.append(
                    {
                        "side": "model",
                        "agent": model,
                        "partner_1": d1,
                        "partner_2": d2,
                        "base_gap": abs(base),
                        "bootstrap_flip_probability": flips / n_b,
                    }
                )
    return pd.DataFrame(rows)


# ============================================================================
# Outputs
# ============================================================================

def save_bootstrap_outputs(out_dir: Path, result: BootstrapResult) -> None:
    bdir = out_dir / "bootstrap"
    bdir.mkdir(parents=True, exist_ok=True)

    result.assignments.to_csv(bdir / "bootstrap_matchings.csv", index=False)
    result.assignment_frequencies.to_csv(
        bdir / "assignment_frequencies.csv", index=False
    )
    result.exact_matching_frequencies.to_csv(
        bdir / "exact_matching_frequencies.csv", index=False
    )
    result.utility_uncertainty_task.to_csv(
        bdir / "utility_uncertainty_task.csv", index=False
    )
    result.utility_uncertainty_model.to_csv(
        bdir / "utility_uncertainty_model.csv", index=False
    )
    result.task_utility_replicates.to_csv(
        bdir / "utility_replicates_task.csv", index=False
    )
    result.model_utility_replicates.to_csv(
        bdir / "utility_replicates_model.csv", index=False
    )
    result.preference_flips.to_csv(
        bdir / "preference_flips.csv", index=False
    )

    _write_bootstrap_report(bdir, result)
    _write_bootstrap_plots(bdir, result)


def _write_bootstrap_report(bdir: Path, result: BootstrapResult) -> None:
    af = result.assignment_frequencies
    em = result.exact_matching_frequencies
    lines: List[str] = []
    lines.append("# Bootstrap Oracle Stability Report")
    lines.append("")
    lines.append(
        f"* replicates: {result.num_bootstrap}, seed: "
        f"{result.bootstrap_seed}, elapsed: {result.elapsed_seconds:.1f}s"
    )
    lines.append("")

    # A. overall stability
    lines.append("## A. Overall matching stability")
    lines.append("")
    lines.append(
        f"* **P(H_bootstrap == H_train) = {result.p_equal_train_oracle:.3f}**"
    )
    lines.append(
        f"* distinct bootstrap stable matchings: "
        f"{result.n_distinct_matchings}"
    )
    lines.append("")
    lines.append("Top-5 most frequent bootstrap matchings:")
    lines.append("")
    lines.append("| rank | matching | count | frequency | is train oracle |")
    lines.append("|---|---|---|---|---|")
    for i, row in em.head(5).iterrows():
        lines.append(
            f"| {i + 1} | {row['matching']} | {row['count']} "
            f"| {row['frequency']:.3f} | {row['is_train_oracle']} |"
        )
    lines.append("")

    # B/C. per-task stability
    lines.append("## B/C. Per-task assignment stability")
    lines.append("")
    lines.append("| task | most frequent model | frequency | oracle assignment | max frequency |")
    lines.append("|---|---|---|---|---|")
    unstable_tasks: List[str] = []
    for task, group in af.groupby("task"):
        top = group.iloc[0]
        if float(top["frequency"]) < 0.7:
            unstable_tasks.append(task)
        lines.append(
            f"| {task} | {top['model']} | {top['frequency']:.3f} "
            f"| {top['oracle_train_assignment']} "
            f"| {group['frequency'].max():.3f} |"
        )
    lines.append("")
    stable_assignments = af[(af["is_oracle_assignment"]) & (af["frequency"] >= 0.9)]
    lines.append(
        f"* assignments with frequency >= 0.9: {len(stable_assignments)} "
        f"(of {len(af['task'].unique())} tasks)"
    )
    lines.append(
        f"* ambiguous tasks (max assignment frequency < 0.7): "
        f"{unstable_tasks if unstable_tasks else 'none'}"
    )
    lines.append("")

    # D. which tasks explain full-matching instability
    lines.append("## D. Which tasks drive full-matching instability?")
    lines.append("")
    mismatch = []
    n = len(result.bootstrap_matchings)
    for task in sorted(result.base_oracle.keys()):
        different = sum(
            1
            for a in result.bootstrap_matchings
            if a.get(task) != result.base_oracle[task]
        )
        mismatch.append((task, different / n))
    mismatch.sort(key=lambda x: -x[1])
    lines.append("| task | P_b(assignment differs from H_train) |")
    lines.append("|---|---|")
    for task, rate in mismatch:
        lines.append(f"| {task} | {rate:.3f} |")
    lines.append("")

    # E. preference reversals
    lines.append("## E. Pairwise preference reversal rates")
    lines.append("")
    pf = result.preference_flips
    lines.append(
        f"* median flip probability: {pf['bootstrap_flip_probability'].median():.3f}"
    )
    for thr in (0.1, 0.25, 0.5):
        n_above = int((pf["bootstrap_flip_probability"] > thr).sum())
        lines.append(
            f"* arms with flip probability > {thr}: {n_above} / {len(pf)}"
        )
    top_flips = pf.sort_values(
        "bootstrap_flip_probability", ascending=False
    ).head(10)
    lines.append("")
    lines.append("Top-10 most fragile preference pairs:")
    lines.append("")
    lines.append(
        "| side | agent | partner_1 | partner_2 | base_gap | flip prob |"
    )
    lines.append("|---|---|---|---|---|---|")
    for _, r in top_flips.iterrows():
        lines.append(
            f"| {r['side']} | {r['agent']} | {r['partner_1']} "
            f"| {r['partner_2']} | {r['base_gap']:.4f} "
            f"| {r['bootstrap_flip_probability']:.3f} |"
        )
    lines.append("")

    (bdir / "bootstrap_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    logger.info("Wrote %s", bdir / "bootstrap_report.md")


def _write_bootstrap_plots(bdir: Path, result: BootstrapResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = bdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Assignment frequency heatmap (task x model)
    tasks = sorted(result.assignment_frequencies["task"].unique())
    models = sorted(result.assignment_frequencies["model"].unique())
    freq_matrix = pd.DataFrame(0.0, index=tasks, columns=models)
    for _, r in result.assignment_frequencies.iterrows():
        freq_matrix.loc[r["task"], r["model"]] = r["frequency"]

    fig, ax = plt.subplots(
        figsize=(1.2 + 0.75 * len(models), 1.2 + 0.55 * len(tasks))
    )
    im = ax.imshow(freq_matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    for i in range(len(tasks)):
        for j in range(len(models)):
            v = freq_matrix.iloc[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    ax.set_title("Bootstrap assignment frequency (task x model)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(plots_dir / "bootstrap_assignment_frequency_heatmap.png", dpi=160)
    plt.close(fig)

    # Utility std heatmaps
    for name, unc, key1, key2 in (
        ("task", result.utility_uncertainty_task, "dataset", "model"),
        ("model", result.utility_uncertainty_model, "model", "dataset"),
    ):
        rows = sorted(unc[key1].unique())
        cols = sorted(unc[key2].unique())
        std_matrix = pd.DataFrame(0.0, index=rows, columns=cols)
        for _, r in unc.iterrows():
            std_matrix.loc[r[key1], r[key2]] = r["std"]
        fig, ax = plt.subplots(
            figsize=(1.2 + 0.75 * len(cols), 1.2 + 0.55 * len(rows))
        )
        im = ax.imshow(std_matrix.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8)
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, f"{std_matrix.iloc[i, j]:.3f}", ha="center",
                        va="center", fontsize=6, color="white")
        ax.set_title(f"Bootstrap utility std — {name} side", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(plots_dir / f"bootstrap_utility_std_{name}.png", dpi=160)
        plt.close(fig)

    logger.info("Wrote bootstrap plots to %s", plots_dir)
