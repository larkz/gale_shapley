"""plots.py

Visual outputs and the human-readable matching report.

All plots are written under <output_dir>/plots/. Matplotlib uses the
non-interactive Agg backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_matching.metrics import matching_to_dict, test_welfare
from llm_matching.oracle import matching_equal

logger = logging.getLogger(__name__)


# ============================================================================
# Accuracy / welfare / resolution vs. queries
# ============================================================================

def _series_per_seed(
    traces: List[pd.DataFrame], column: str
) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for trace in traces:
        if trace.empty or column not in trace.columns:
            continue
        seed = int(trace["seed"].iloc[0])
        sub = trace[["t", column, "stopped"]].copy()
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
        out[seed] = sub
    return out


def _metric_vs_queries(
    traces: List[pd.DataFrame],
    column: str,
    ylabel: str,
    title: str,
    out_path: Path,
    reference_lines: Optional[Dict[str, float]] = None,
) -> None:
    series = _series_per_seed(traces, column)
    if not series:
        logger.warning("No trace data for %s; skipping plot", column)
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    t_max = max(sub["t"].max() for sub in series.values())
    grid = np.arange(0, t_max + 1)

    # Per-seed forward-filled curves within their valid range. A seed
    # that stopped keeps its committed value (frozen); a seed that hit
    # max_samples without stopping simply ends there.
    stacked: Dict[int, np.ndarray] = {}
    for seed, sub in series.items():
        sub = sub.sort_values("t").drop_duplicates("t", keep="last")
        stopped = bool(sub["stopped"].iloc[-1])
        t_end = int(sub["t"].max())
        values = np.interp(
            grid, sub["t"].to_numpy(), sub[column].to_numpy(),
            left=np.nan, right=np.nan,
        )
        # fill within observed range
        values[grid <= t_end] = pd.Series(
            values[grid <= t_end]
        ).ffill().bfill().to_numpy()
        if stopped:
            values[grid > t_end] = sub[column].iloc[-1]
        stacked[seed] = values

        color = "tab:blue" if stopped else "tab:red"
        label = None
        ax.plot(
            grid[grid <= t_end], values[grid <= t_end],
            color=color, alpha=0.35, linewidth=1.0, label=label,
        )

    matrix = np.vstack(list(stacked.values()))
    mean = np.nanmean(matrix, axis=0)
    ax.plot(grid, mean, color="black", linewidth=2.2, label="mean over seeds")

    if reference_lines:
        for name, value in reference_lines.items():
            ax.axhline(value, linestyle="--", linewidth=1.2, alpha=0.8, label=name)

    n_not_stopped = sum(
        1 for sub in series.values() if not bool(sub["stopped"].iloc[-1])
    )
    n_stopped = len(series) - n_not_stopped
    ax.set_title(
        f"{title}\n({n_stopped} stopped / {n_not_stopped} hit max_samples)",
        fontsize=11,
    )
    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def matching_accuracy_vs_queries(
    ctx, traces: List[pd.DataFrame], out_path: Path
) -> None:
    _metric_vs_queries(
        traces,
        column="exact_oracle_match",
        ylabel="P(matching == H*_train)",
        title="Exact oracle matching recovery vs. queries",
        out_path=out_path,
    )


def test_welfare_vs_queries(
    ctx, traces: List[pd.DataFrame], out_path: Path
) -> None:
    oracle_w = test_welfare(ctx.oracle["train"], ctx.task_util["test"])
    hung_w = test_welfare(ctx.hungarian, ctx.task_util["test"])
    rand_mean, _ = ctx.random_welfare_test
    _metric_vs_queries(
        traces,
        column="test_welfare",
        ylabel="test welfare  sum_d U_d^test(H(d))",
        title="Test welfare vs. queries",
        out_path=out_path,
        reference_lines={
            "oracle GS (H*_train)": oracle_w,
            "Hungarian (train util)": hung_w,
            "random mean": rand_mean,
        },
    )


def resolved_fraction_vs_queries(
    ctx, traces: List[pd.DataFrame], out_path: Path
) -> None:
    _metric_vs_queries(
        traces,
        column="pairwise_resolved_fraction",
        ylabel="fraction of pairwise arms resolved (CI excludes 1/2)",
        title="Pairwise resolution fraction vs. queries",
        out_path=out_path,
    )


def stopping_time_distribution(
    ctx, per_seed: pd.DataFrame, out_path: Path
) -> None:
    stopped = per_seed[pd.to_numeric(per_seed["stopped"]) == 1]
    not_stopped = len(per_seed) - len(stopped)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(stopped):
        t_stop = pd.to_numeric(stopped["T_stop"])
        ax.hist(t_stop, bins=min(30, max(10, len(stopped))), color="tab:blue", alpha=0.8)
    ax.set_xlabel("T_stop (pairwise observations)")
    ax.set_ylabel("number of seeds")
    ax.set_title(
        f"Stopping time distribution "
        f"({len(stopped)} stopped, {not_stopped} not stopped)",
        fontsize=11,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ============================================================================
# Heatmaps
# ============================================================================

def _heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_path: Path,
    annotate: bool = True,
    fmt: str = "{:.3f}",
) -> None:
    fig, ax = plt.subplots(
        figsize=(1.2 + 0.75 * matrix.shape[1], 1.2 + 0.55 * matrix.shape[0])
    )
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(list(matrix.columns), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(list(matrix.index), fontsize=8)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, fmt.format(matrix.iloc[i, j]),
                    ha="center", va="center", fontsize=7, color="white",
                )
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def model_task_train_heatmap(ctx, out_path: Path) -> None:
    _heatmap(
        ctx.task_util["train"],
        "Task utility U_d(m) — TRAIN (rows: tasks, cols: models)",
        out_path,
    )


def comparative_advantage_heatmap(ctx, out_path: Path) -> None:
    _heatmap(
        ctx.model_util["train"],
        "Model comparative advantage V_m(d) — TRAIN (rows: models, cols: tasks)",
        out_path,
    )


# ============================================================================
# Entry point for all plots
# ============================================================================

def make_all_plots(
    ctx,
    traces: List[pd.DataFrame],
    per_seed: pd.DataFrame,
) -> None:
    plots_dir = ctx.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    matching_accuracy_vs_queries(ctx, traces, plots_dir / "matching_accuracy_vs_queries.png")
    test_welfare_vs_queries(ctx, traces, plots_dir / "test_welfare_vs_queries.png")
    resolved_fraction_vs_queries(ctx, traces, plots_dir / "resolved_fraction_vs_queries.png")
    stopping_time_distribution(ctx, per_seed, plots_dir / "stopping_time_distribution.png")
    model_task_train_heatmap(ctx, plots_dir / "model_task_train_heatmap.png")
    comparative_advantage_heatmap(ctx, plots_dir / "comparative_advantage_heatmap.png")


# ============================================================================
# 5-seed diagnostic plots (Phase 2)
# ============================================================================

def _load_traces(mode_dir: Path) -> List[pd.DataFrame]:
    traces = []
    tdir = mode_dir / "traces"
    if not tdir.is_dir():
        return traces
    for p in sorted(tdir.glob("seed_*.csv")):
        if p.name.startswith("arm_"):
            continue
        df = pd.read_csv(p)
        if not df.empty:
            traces.append(df)
    return traces


def _mean_series(traces: List[pd.DataFrame], column: str):
    """Union grid + per-seed forward fill within observed range."""
    grid = np.arange(0, max(df["t"].max() for df in traces) + 1)
    stacked = []
    for df in traces:
        sub = df.sort_values("t").drop_duplicates("t", keep="last")
        values = np.interp(
            grid, sub["t"].to_numpy(),
            pd.to_numeric(sub[column], errors="coerce").to_numpy(),
            left=np.nan, right=np.nan,
        )
        t_end = int(sub["t"].max())
        values[grid <= t_end] = (
            pd.Series(values[grid <= t_end]).ffill().bfill().to_numpy()
        )
        stopped = bool(pd.to_numeric(sub["stopped"]).iloc[-1])
        if stopped:
            values[grid > t_end] = sub[column].iloc[-1]
        stacked.append(values)
    return grid, np.vstack(stacked)


def _plot_metric_5seed(
    traces: List[pd.DataFrame],
    column: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    grid, matrix = _mean_series(traces, column)
    for i, df in enumerate(traces):
        sub = df.sort_values("t")
        ax.plot(sub["t"], pd.to_numeric(sub[column], errors="coerce"),
                color="tab:blue", alpha=0.25, linewidth=1.0)
    mean = np.nanmean(matrix, axis=0)
    ax.plot(grid, mean, color="black", linewidth=2.2, label="mean")
    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_occupancy_bars(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seeds = summary["seed"].tolist()
    occ = pd.to_numeric(summary["oracle_occupancy"]).tolist()
    colors = ["tab:green" if s else "tab:gray"
              for s in pd.to_numeric(summary["stopped"])]
    ax.bar([str(s) for s in seeds], occ, color=colors, alpha=0.85)
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("seed")
    ax.set_ylabel("oracle occupancy")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_matching_changes(traces: List[pd.DataFrame], out_path: Path) -> None:
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for df in traces:
        df = df.sort_values("t")
        matching = df["matching"].astype(str).to_numpy()
        changes = np.zeros(len(df))
        for i in range(1, len(matching)):
            changes[i] = changes[i - 1] + (matching[i] != matching[i - 1])
        ax.plot(df["t"], changes, alpha=0.6, linewidth=1.2,
                label=f"seed {df['seed'].iloc[0]}")
    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel("cumulative matching changes")
    ax.set_title("Matching changes vs. queries", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_critical_unresolved(
    traces: List[pd.DataFrame], out_path: Path, title: str
) -> None:
    if not traces or "unresolved_critical" not in traces[0].columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    grid, crit = _mean_series(traces, "unresolved_critical")
    _, noncrit = _mean_series(traces, "unresolved_noncritical")
    ax.plot(grid, np.nanmean(crit, axis=0), color="tab:red", linewidth=2.0,
            label="unresolved critical arms")
    ax.plot(grid, np.nanmean(noncrit, axis=0), color="tab:blue", linewidth=2.0,
            label="unresolved non-critical arms")
    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel("number of unresolved arms")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def write_diagnostics_summary_and_plots(
    diag_dir: Path,
    combined: pd.DataFrame,
    feedbacks: List[str],
    seeds: List[int],
) -> None:
    """Write diagnostic_5seed_summary.csv and the section-18 plots."""
    diag_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = diag_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(diag_dir / "diagnostic_5seed_summary.csv", index=False)

    for feedback in feedbacks:
        mode_dir = diag_dir / feedback
        traces = _load_traces(mode_dir)
        sub = combined[combined["feedback"] == feedback]
        if not traces:
            continue
        _plot_metric_5seed(
            traces, "exact_oracle_match", "P(matching == H*_train)",
            f"Exact oracle recovery vs. queries — {feedback.upper()} (5 seeds)",
            plots_dir / f"matching_accuracy_vs_queries_5seed_{feedback}.png",
        )
        _plot_metric_5seed(
            traces, "pairwise_resolved_fraction", "resolved fraction",
            f"Pairwise resolution vs. queries — {feedback.upper()} (5 seeds)",
            plots_dir / f"resolved_fraction_vs_queries_5seed_{feedback}.png",
        )
        _plot_occupancy_bars(
            sub, plots_dir / f"oracle_occupancy_by_seed_{feedback}.png",
            f"Oracle occupancy by seed — {feedback.upper()}",
        )
        _plot_matching_changes(
            traces, plots_dir / f"matching_changes_vs_queries_{feedback}.png"
        )
        _plot_critical_unresolved(
            traces,
            plots_dir / f"critical_unresolved_vs_queries_{feedback}.png",
            f"Unresolved critical vs non-critical arms — {feedback.upper()}",
        )

    # Combined BT vs replay
    if len(feedbacks) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for feedback, color in zip(feedbacks, ("tab:blue", "tab:orange")):
            traces = _load_traces(diag_dir / feedback)
            if not traces:
                continue
            grid, acc = _mean_series(traces, "exact_oracle_match")
            axes[0].plot(grid, np.nanmean(acc, axis=0), color=color,
                         linewidth=2.0, label=feedback)
            _, res = _mean_series(traces, "pairwise_resolved_fraction")
            axes[1].plot(grid, np.nanmean(res, axis=0), color=color,
                         linewidth=2.0, label=feedback)
        axes[0].set_title("Oracle recovery: BT vs replay", fontsize=11)
        axes[0].set_ylabel("P(matching == H*_train)")
        axes[1].set_title("Resolved fraction: BT vs replay", fontsize=11)
        axes[1].set_ylabel("resolved fraction")
        for ax in axes:
            ax.set_xlabel("number of pairwise observations (t)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "combined_bt_vs_replay.png", dpi=160)
        plt.close(fig)

        # Core diagnostic: unresolved critical vs non-critical (both modes)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for feedback in feedbacks:
            traces = _load_traces(diag_dir / feedback)
            if not traces or "unresolved_critical" not in traces[0].columns:
                continue
            grid, crit = _mean_series(traces, "unresolved_critical")
            _, noncrit = _mean_series(traces, "unresolved_noncritical")
            ax.plot(grid, np.nanmean(crit, axis=0), linewidth=2.0,
                    label=f"{feedback}: critical")
            ax.plot(grid, np.nanmean(noncrit, axis=0), linewidth=2.0,
                    linestyle="--", label=f"{feedback}: non-critical")
        ax.set_xlabel("number of pairwise observations (t)")
        ax.set_ylabel("number of unresolved arms")
        ax.set_title(
            "Are remaining unresolved arms irrelevant to the matching?",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "unresolved_vs_matching_critical.png", dpi=160)
        plt.close(fig)
        logger.info("Wrote %s", plots_dir / "unresolved_vs_matching_critical.png")


# ============================================================================
# Matching-ID comparison plots (Phase 2)
# ============================================================================

def write_matching_id_comparison(
    mid_dir: Path, per_seed: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate + comparison plots for the 4x4 Matching-ID study."""
    import numpy as np

    plots_dir = mid_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "T_stop", "exact_oracle_match", "false_certification",
        "resolved_fraction_at_stop", "test_welfare",
        "certify_seconds_total",
    ]
    rows = []
    for (algo, fb), group in per_seed.groupby(["algorithm", "feedback_mode"]):
        row = {"algorithm": algo, "feedback": fb, "n_seeds": len(group)}  # noqa
        for metric in metrics:
            if metric in group.columns:
                series = pd.to_numeric(
                    group[metric], errors="coerce"
                ).astype("float64")
                row[f"{metric}_mean"] = float(series.mean())
                row[f"{metric}_median"] = float(series.median())
                row[f"{metric}_std"] = float(
                    series.std(ddof=1)
                ) if len(series) > 1 else 0.0
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(mid_dir / "aggregate_summary.csv", index=False)

    # T_stop comparison
    fig, ax = plt.subplots(figsize=(8.5, 5))
    combos = list(
        zip(per_seed["algorithm"], per_seed["feedback_mode"])
    )
    labels = sorted({f"{a}\n({f})" for a, f in combos})
    data, colors = [], []
    palette = {
        ("p2etg", "bt"): "tab:blue", ("p2etg", "replay"): "tab:cyan",
        ("matching_id", "bt"): "tab:red", ("matching_id", "replay"): "tab:orange",
    }
    for algo in ("p2etg", "matching_id"):
        for fb in ("bt", "replay"):
            group = per_seed[
                (per_seed["algorithm"] == algo) & (per_seed["feedback_mode"] == fb)
            ]
            data.append(pd.to_numeric(group["T_stop"], errors="coerce").values)
            colors.append(palette[(algo, fb)])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("T_stop (pairwise observations)")
    ax.set_title("Stopping time: full-preference vs Matching-ID (4x4)", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "t_stop_comparison.png", dpi=160)
    plt.close(fig)

    # resolved fraction at stop
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = []
    for algo in ("p2etg", "matching_id"):
        for fb in ("bt", "replay"):
            group = per_seed[
                (per_seed["algorithm"] == algo) & (per_seed["feedback_mode"] == fb)
            ]
            data.append(
                pd.to_numeric(
                    group["resolved_fraction_at_stop"], errors="coerce"
                ).values
            )
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("resolved fraction at stop")
    ax.set_title("Preference resolution at stop (4x4)", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "resolved_fraction_at_stop.png", dpi=160)
    plt.close(fig)

    logger.info("Wrote Matching-ID comparison plots to %s", plots_dir)
    return agg

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (no tabulate dependency)."""
    if df.empty:
        return "_(empty)_"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_matching_report(
    ctx,
    per_seed: pd.DataFrame,
    algorithm: str,
    feedback: str,
) -> Path:
    from llm_matching.baselines import datasetwise_best_ignoring_capacity

    out_path = ctx.out_dir / "matching_report.md"
    lines: List[str] = []

    tu_train = ctx.task_util["train"]
    tu_test = ctx.task_util["test"]
    mu_train = ctx.model_util["train"]
    prefs = ctx.prefs["train"]

    oracle_dict = matching_to_dict(ctx.oracle["train"])
    hungarian_dict = matching_to_dict(ctx.hungarian)
    best_model = datasetwise_best_ignoring_capacity(tu_train)

    lines.append("# LLM-Task Stable Matching Report")
    lines.append("")
    lines.append(f"* algorithm: `{algorithm}`, feedback mode: `{feedback}`")
    lines.append(f"* datasets: {len(ctx.women)}, models: {len(ctx.men)}")
    cal = ctx.calibration
    lines.append(
        f"* BT calibration: target median win prob = "
        f"{cal['target_median_win_probability']}, median task gap = "
        f"{cal['median_task_gap']:.6f}, median model gap = "
        f"{cal['median_model_gap']:.6f}"
    )
    lines.append(
        f"* eta_task = {cal['eta_task']:.4f}, eta_model = {cal['eta_model']:.4f}"
    )
    tie_train = ctx.tie_report["train"]
    lines.append(
        f"* ties strictified (train): task side "
        f"{tie_train['task_side_changed_entries']} entries, model side "
        f"{tie_train['model_side_changed_entries']} entries"
    )
    lines.append("")

    lines.append("## Oracle model-proposing stable matching (H*_train)")
    lines.append("")
    lines.append(
        "| Task | Assigned model | task-rank of model | model-rank of task "
        "| train utility U | test utility U | comparative adv. V (train) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for task, model in oracle_dict.items():
        task_rank = (
            prefs.get_rank(
                next(w for w in ctx.women if w.id == task),
                next(m for m in ctx.men if m.id == model),
            )
        )
        model_rank = (
            prefs.get_rank(
                next(m for m in ctx.men if m.id == model),
                next(w for w in ctx.women if w.id == task),
            )
        )
        lines.append(
            f"| {task} | {model} | {task_rank + 1} | {model_rank + 1} "
            f"| {tu_train.loc[task, model]:.4f} "
            f"| {tu_test.loc[task, model]:.4f} "
            f"| {mu_train.loc[model, task]:+.4f} |"
        )
    lines.append("")

    lines.append("## Hungarian maximum-welfare matching (train utility)")
    lines.append("")
    lines.append("| Task | Hungarian model | Oracle model | same? |")
    lines.append("|---|---|---|---|")
    for task in oracle_dict:
        hm = hungarian_dict.get(task, "-")
        lines.append(
            f"| {task} | {hm} | {oracle_dict[task]} "
            f"| {'yes' if hm == oracle_dict[task] else 'no'} |"
        )
    lines.append("")

    oracle_w = test_welfare(ctx.oracle["train"], tu_test)
    hung_w = test_welfare(ctx.hungarian, tu_test)
    rand_mean, rand_std = ctx.random_welfare_test
    lines.append("## Welfare comparison (test utilities)")
    lines.append("")
    lines.append("| Matching | Test welfare |")
    lines.append("|---|---|")
    lines.append(f"| Oracle GS (H*_train) | {oracle_w:.4f} |")
    lines.append(f"| Hungarian (train util) | {hung_w:.4f} |")
    lines.append(f"| Random (mean over seeds) | {rand_mean:.4f} (std {rand_std:.4f}) |")
    lines.append("")

    lines.append("## Preference generalization across splits")
    lines.append("")
    lines.append(
        f"* H*_train == H*_val: "
        f"{matching_equal(ctx.oracle['train'], ctx.oracle['val'])}"
    )
    lines.append(
        f"* H*_train == H*_test: "
        f"{matching_equal(ctx.oracle['train'], ctx.oracle['test'])}"
    )
    from llm_matching.metrics import spearman_rank_correlations

    for side, util_train, util_test in (
        ("task", tu_train, ctx.task_util["test"]),
        ("model", mu_train, ctx.model_util["test"]),
    ):
        corr = spearman_rank_correlations(util_train, util_test)
        mean_rho = corr["spearman_rho"].mean()
        lines.append(
            f"* mean Spearman rank correlation ({side} side, train vs test): "
            f"{mean_rho:.4f}"
        )
    lines.append("")

    lines.append("## Sanity checks")
    lines.append("")
    distinct_best = {m for m in best_model.values()}
    lines.append(
        f"* Dataset-wise best model ignoring capacity: "
        f"{len(distinct_best)} distinct model(s) chosen across "
        f"{len(best_model)} tasks -> "
        + (
            "specialization exists (multiple distinct best models)."
            if len(distinct_best) > 1
            else "weak specialization (a single model dominates every task)."
        )
    )
    lines.append(
        f"| Task | Best model (ignoring capacity) | Oracle model |"
    )
    lines.append("|---|---|---|")
    for task in oracle_dict:
        lines.append(f"| {task} | {best_model[task]} | {oracle_dict[task]} |")
    gs_is_hung = matching_equal(ctx.oracle["train"], ctx.hungarian)
    lines.append("")
    lines.append(
        f"* Oracle stable matching {'equals' if gs_is_hung else 'differs from'} "
        "the Hungarian welfare-optimal matching."
    )
    lines.append("")

    if per_seed is not None and len(per_seed):
        lines.append(f"## {algorithm} ({feedback}) per-seed results")
        lines.append("")
        cols = [
            "seed", "stopped", "T_stop", "exact_oracle_match", "stable_train",
            "blocking_pairs_train", "resolved_fraction_at_stop",
            "test_welfare", "normalized_test_welfare",
        ]
        cols = [c for c in cols if c in per_seed.columns]
        lines.append(_df_to_markdown(per_seed[cols]))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return out_path
