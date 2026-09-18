"""regret.py

Welfare regret curves for learner matchings.

Instantaneous regret (welfare units on the TEST split):

    regret_ref(t) = W_ref - W_test(H_t)

References:
  * H*_train      — the oracle stable matching (identification target);
  * Hungarian(tr) — welfare-optimal train-legal matching;
  * Hungarian(te) — test-side welfare ceiling (analysis only, uses
                    test utilities);
  * random mean   — random-bijection baseline.

Notes:
  * Regret can be NEGATIVE when the learner's current matching has
    higher TEST welfare than the reference (train/test utilities
    differ; the learner is not penalised for that).
  * For runs that STOPPED (e.g. Matching-ID certification), the
    post-stop welfare is the COMMITTED matching's welfare (taken from
    the per-seed summary), not the last empirical trace row.

Everything is computed from already-saved outputs; no reruns needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# References
# ============================================================================

def welfare_of_matching_dict(
    matching: Dict[str, str], task_util: pd.DataFrame
) -> float:
    """W(matching) = sum_d U_d^test(matching(d))."""
    return float(
        sum(task_util.loc[task, model] for task, model in matching.items())
    )


def test_welfare_ceiling(task_util: pd.DataFrame) -> float:
    """Welfare of the Hungarian matching computed ON test utilities
    (upper bound for any one-to-one assignment; analysis only)."""
    from scipy.optimize import linear_sum_assignment

    cost = -task_util.to_numpy(dtype=float)
    rows, cols = linear_sum_assignment(cost)
    return float(-cost[rows, cols].sum())


def load_references(run_dir: Path) -> Dict[str, float]:
    task_util_test = pd.read_csv(run_dir / "task_utility_test.csv", index_col=0)
    oracle = json.load(open(run_dir / "oracle_matching_train.json"))
    hungarian = json.load(open(run_dir / "hungarian_matching.json"))
    random_info = json.load(open(run_dir / "random_baseline.json"))

    w_oracle = welfare_of_matching_dict(oracle, task_util_test)
    w_hungarian = welfare_of_matching_dict(hungarian, task_util_test)
    return {
        "oracle": w_oracle,
        "hungarian_train": w_hungarian,
        "ceiling_test": test_welfare_ceiling(task_util_test),
        "random_mean": float(random_info["test_welfare_mean"]),
        "task_util_test": task_util_test,  # type: ignore[dict-item]
    }


# ============================================================================
# Trace discovery
# ============================================================================

def _read_traces(traces_dir: Path, pattern: str = "seed_*.csv") -> List[pd.DataFrame]:
    out = []
    for p in sorted(traces_dir.glob(pattern)):
        if p.name.startswith("arm_"):
            continue
        df = pd.read_csv(p)
        if not df.empty and "test_welfare" in df.columns:
            out.append(df)
    return out


def collect_trace_groups(
    run_dir: Path,
) -> Dict[str, Dict[str, object]]:
    """Find all run groups under a run directory.

    Returns {label: {"traces": [...], "per_seed": DataFrame|None}}.
    Groups:
      * diagnostics_5seed/<feedback>/      -> label p2etg_<feedback>
      * matching_id/traces/<algo>_<fb>_..  -> label <algo>_<feedback>
      * traces/ (plain single run)         -> label from run_meta.json
    """
    groups: Dict[str, Dict[str, object]] = {}

    diag = run_dir / "diagnostics_5seed"
    if diag.is_dir():
        for fb_dir in sorted(diag.iterdir()):
            if not fb_dir.is_dir():
                continue
            traces = _read_traces(fb_dir / "traces")
            if traces:
                per_seed_path = fb_dir / "per_seed_summary.csv"
                per_seed = (
                    pd.read_csv(per_seed_path) if per_seed_path.exists() else None
                )
                groups[f"p2etg_{fb_dir.name}"] = {
                    "traces": traces, "per_seed": per_seed,
                }

    mid_traces = run_dir / "matching_id" / "traces"
    if mid_traces.is_dir():
        by_label: Dict[str, List[pd.DataFrame]] = {}
        for p in sorted(mid_traces.glob("*_seed_*.csv")):
            label = p.name.rsplit("_seed_", 1)[0]
            df = pd.read_csv(p)
            if not df.empty and "test_welfare" in df.columns:
                by_label.setdefault(label, []).append(df)
        for label, traces in by_label.items():
            # per-seed summary lives in matching_id/<label>/ when the
            # label matches a combo directory
            combo_dir = run_dir / "matching_id" / label
            per_seed_path = combo_dir / "per_seed_summary.csv"
            per_seed = (
                pd.read_csv(per_seed_path) if per_seed_path.exists() else None
            )
            groups[label] = {"traces": traces, "per_seed": per_seed}

    plain = run_dir / "traces"
    if plain.is_dir():
        traces = _read_traces(plain)
        if traces:
            label = "run"
            meta_path = run_dir / "run_meta.json"
            if meta_path.exists():
                meta = json.load(open(meta_path))
                label = (
                    f"{meta.get('algorithm', 'p2etg')}_"
                    f"{meta.get('feedback_mode', 'bt')}_single"
                )
            per_seed_path = run_dir / "per_seed_summary.csv"
            per_seed = (
                pd.read_csv(per_seed_path) if per_seed_path.exists() else None
            )
            groups[label] = {"traces": traces, "per_seed": per_seed}

    return groups


# ============================================================================
# Regret curves
# ============================================================================

def _regret_curves(
    traces: List[pd.DataFrame],
    reference_welfare: float,
    committed_welfare_by_seed: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(grid, regret_matrix[n_seeds x len(grid)]) vs. one reference.

    Stopped seeds are forward-filled with their COMMITTED welfare when
    known (Matching-ID certified matchings), else with their last
    empirical value.
    """
    t_max = max(df["t"].max() for df in traces)
    grid = np.arange(0, int(t_max) + 1)
    rows = []
    for df in traces:
        df = df.sort_values("t").drop_duplicates("t", keep="last")
        seed = int(df["seed"].iloc[0])
        welfare = pd.to_numeric(df["test_welfare"], errors="coerce")
        values = np.interp(
            grid, df["t"].to_numpy(), welfare.to_numpy(),
            left=np.nan, right=np.nan,
        )
        t_end = int(df["t"].max())
        values[grid <= t_end] = (
            pd.Series(values[grid <= t_end]).ffill().bfill().to_numpy()
        )
        stopped = bool(pd.to_numeric(df["stopped"]).iloc[-1]) if "stopped" in df else False
        if stopped:
            fill = None
            if committed_welfare_by_seed and seed in committed_welfare_by_seed:
                fill = committed_welfare_by_seed[seed]
            else:
                fill = float(welfare.iloc[-1])
            values[grid > t_end] = fill
        rows.append(reference_welfare - values)
    return grid, np.vstack(rows)


def _committed_welfare_map(per_seed: Optional[pd.DataFrame]) -> Dict[int, float]:
    if per_seed is None or "test_welfare" not in per_seed.columns:
        return {}
    out = {}
    for _, r in per_seed.iterrows():
        try:
            out[int(r["seed"])] = float(r["test_welfare"])
        except (TypeError, ValueError):
            continue
    return out


# ============================================================================
# Cumulative regret
# ============================================================================

def _cumulative_trapezoid(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Trapezoid cumulative integral of `values` over `grid` (step-1 grid).

    Returns array of same length; CR[0] = 0. NaNs are treated as 0
    contribution (they only occur before the first observation, where
    values are backfilled anyway).
    """
    v = np.nan_to_num(values, nan=0.0)
    dt = np.diff(grid)
    incr = (v[1:] + v[:-1]) / 2.0 * dt
    return np.concatenate([[0.0], np.cumsum(incr)])


def _cumulative_curves(
    traces: List[pd.DataFrame],
    reference_welfare: float,
    committed_welfare_by_seed: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grid, CR_matrix[n_seeds x len(grid)], t_end_per_seed).

    t_end_per_seed[i] is the last observed check time of seed i (the
    run's own horizon: T_stop for stopped runs, max_samples otherwise).
    """
    t_max = max(df["t"].max() for df in traces)
    grid = np.arange(0, int(t_max) + 1)
    rows = []
    t_ends = []
    for df in traces:
        df = df.sort_values("t").drop_duplicates("t", keep="last")
        seed = int(df["seed"].iloc[0])
        welfare = pd.to_numeric(df["test_welfare"], errors="coerce")
        values = np.interp(
            grid, df["t"].to_numpy(), welfare.to_numpy(),
            left=np.nan, right=np.nan,
        )
        t_end = int(df["t"].max())
        t_ends.append(t_end)
        values[grid <= t_end] = (
            pd.Series(values[grid <= t_end]).ffill().bfill().to_numpy()
        )
        stopped = (
            bool(pd.to_numeric(df["stopped"]).iloc[-1])
            if "stopped" in df else False
        )
        if stopped:
            fill = None
            if committed_welfare_by_seed and seed in committed_welfare_by_seed:
                fill = committed_welfare_by_seed[seed]
            else:
                fill = float(welfare.iloc[-1])
            values[grid > t_end] = fill
        rows.append(_cumulative_trapezoid(grid, reference_welfare - values))
    return grid, np.vstack(rows), np.array(t_ends, dtype=int)


# ============================================================================
# Entry point
# ============================================================================

def write_regret_plots(run_dir: Path) -> pd.DataFrame:
    """Compute and plot regret curves for every run group under run_dir."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refs = load_references(run_dir)
    task_util_test: pd.DataFrame = refs["task_util_test"]  # type: ignore[assignment]
    w_oracle = refs["oracle"]

    groups = collect_trace_groups(run_dir)
    if not groups:
        raise FileNotFoundError(
            f"No traces with test_welfare found under {run_dir}"
        )

    out_dir = run_dir / "regret"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- main figure: regret vs H*_train ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [
        "tab:blue", "tab:orange", "tab:red", "tab:green",
        "tab:purple", "tab:brown",
    ]
    summary_rows: List[dict] = []
    cumulative_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for i, (label, group) in enumerate(sorted(groups.items())):
        traces: List[pd.DataFrame] = group["traces"]
        committed = _committed_welfare_map(group["per_seed"])
        color = colors[i % len(colors)]

        grid, regret = _regret_curves(traces, w_oracle, committed)
        # faint individual curves when few seeds
        if len(traces) <= 6:
            for j in range(regret.shape[0]):
                ax.plot(grid, regret[j], color=color, alpha=0.22, linewidth=0.9)
        mean = np.nanmean(regret, axis=0)
        ax.plot(grid, mean, color=color, linewidth=2.2, label=label)

        # cumulative regret (same reference, same post-stop fill)
        cr_grid, cr_matrix, cr_t_ends = _cumulative_curves(
            traces, w_oracle, committed
        )
        cumulative_data[label] = (cr_grid, cr_matrix, cr_t_ends)

        # summary stats
        per_seed = group["per_seed"]
        final_regret = float(np.nanmean(regret[:, -1]))
        best_regret = float(np.nanmin(regret, axis=1).mean())
        row = {
            "group": label,
            "n_seeds": len(traces),
            "reference": "oracle_train",
            "final_regret_mean": final_regret,
            "final_regret_std": float(np.nanstd(regret[:, -1], ddof=1))
            if len(traces) > 1 else 0.0,
            "best_regret_mean": best_regret,
        }
        # cumulative regret at each run's OWN horizon (T_stop when
        # stopped): area under the regret curve while the run lived.
        cr_at_end = np.array(
            [
                cr_matrix[j, int(cr_t_ends[j])]
                for j in range(cr_matrix.shape[0])
            ]
        )
        row["cumulative_regret_end_mean"] = float(cr_at_end.mean())
        row["cumulative_regret_end_std"] = (
            float(cr_at_end.std(ddof=1)) if len(cr_at_end) > 1 else 0.0
        )
        if per_seed is not None and "test_welfare" in per_seed.columns:
            committed_regret = w_oracle - pd.to_numeric(
                per_seed["test_welfare"], errors="coerce"
            )
            stopped = pd.to_numeric(
                per_seed.get("stopped", pd.Series([0] * len(per_seed)))
            )
            row["committed_regret_mean"] = float(committed_regret.mean())
            row["committed_regret_std"] = (
                float(committed_regret.std(ddof=1)) if len(per_seed) > 1 else 0.0
            )
            row["n_stopped"] = int(stopped.sum())
            if "T_stop" in per_seed.columns and int(stopped.sum()) > 0:
                stopped_t = pd.to_numeric(
                    per_seed.loc[stopped == 1, "T_stop"], errors="coerce"
                )
                row["T_stop_median_stopped"] = float(stopped_t.median())
        summary_rows.append(row)

    # reference levels (regret of a matching equals these values)
    ax.axhline(0.0, color="black", linewidth=1.4, alpha=0.8)
    ax.axhline(
        w_oracle - refs["hungarian_train"], color="tab:gray", linestyle="--",
        linewidth=1.2, alpha=0.9,
        label=f"Hungarian(train): {w_oracle - refs['hungarian_train']:+.4f}",
    )
    ax.axhline(
        w_oracle - refs["ceiling_test"], color="black", linestyle=":",
        linewidth=1.2, alpha=0.9,
        label=f"test-welfare ceiling: {w_oracle - refs['ceiling_test']:+.4f}",
    )
    ax.axhline(
        w_oracle - refs["random_mean"], color="tab:gray", linestyle=":",
        linewidth=1.2, alpha=0.9,
        label=f"random mean: {w_oracle - refs['random_mean']:+.4f}",
    )

    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel("regret  W(H*_train) − W_test(H_t)")
    ax.set_title(
        "Welfare regret vs. oracle stable matching "
        f"({run_dir.name}, mean over seeds)", fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "regret_vs_queries.png", dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_dir / "regret_vs_queries.png")

    # ---- secondary figure: regret vs Hungarian(train) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, group) in enumerate(sorted(groups.items())):
        traces = group["traces"]
        committed = _committed_welfare_map(group["per_seed"])
        color = colors[i % len(colors)]
        grid, regret = _regret_curves(traces, refs["hungarian_train"], committed)
        if len(traces) <= 6:
            for j in range(regret.shape[0]):
                ax.plot(grid, regret[j], color=color, alpha=0.22, linewidth=0.9)
        ax.plot(
            grid, np.nanmean(regret, axis=0), color=color, linewidth=2.2,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=1.4, alpha=0.8)
    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel("regret  W(Hungarian_train) − W_test(H_t)")
    ax.set_title(
        "Welfare regret vs. Hungarian (train-legal welfare optimum), "
        f"({run_dir.name})", fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "regret_vs_hungarian.png", dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_dir / "regret_vs_hungarian.png")

    # ---- cumulative regret figure (vs H*_train) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, (cr_grid, cr_matrix, cr_t_ends)) in enumerate(
        sorted(cumulative_data.items())
    ):
        color = colors[i % len(colors)]
        # faint individual cumulative curves when few seeds
        if cr_matrix.shape[0] <= 6:
            for j in range(cr_matrix.shape[0]):
                ax.plot(
                    cr_grid, cr_matrix[j], color=color, alpha=0.22,
                    linewidth=0.9,
                )
        ax.plot(
            cr_grid, np.nanmean(cr_matrix, axis=0), color=color,
            linewidth=2.2, label=label,
        )
        # mark stopped runs' own horizons (cumulation stops there)
        stopped_idx = [
            j for j, df in enumerate(groups[label]["traces"])
            if "stopped" in df and bool(pd.to_numeric(df["stopped"]).iloc[-1])
        ]
        for j in stopped_idx:
            t_end = int(cr_t_ends[j])
            ax.plot(
                [t_end], [cr_matrix[j, t_end]], marker="x", color=color,
                markersize=7, markeredgewidth=1.6, alpha=0.9,
            )

    # slope references: cumulative regret of fixed policies
    t_ref = np.array([0, cr_grid[-1]])
    ax.plot(
        t_ref, (w_oracle - refs["random_mean"]) * t_ref,
        linestyle="--", color="tab:gray", linewidth=1.3, alpha=0.9,
        label=f"random-always slope ({w_oracle - refs['random_mean']:+.3f}/t)",
    )
    ax.plot(
        t_ref, (w_oracle - refs["hungarian_train"]) * t_ref,
        linestyle=":", color="black", linewidth=1.3, alpha=0.9,
        label=f"Hungarian(train) slope ({w_oracle - refs['hungarian_train']:+.3f}/t)",
    )

    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel("cumulative regret  ∫ [W(H*_train) − W_test(H_t)] dτ")
    ax.set_title(
        f"Cumulative welfare regret vs. oracle ({run_dir.name}, "
        "mean over seeds; × = certified stop)",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_regret.png", dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_dir / "cumulative_regret.png")

    # ---- average-regret convergence figure: CR(T)/T ----
    # Convergence reading: CR(T)/T -> the asymptotic instantaneous
    # regret of the run's fixed point (0 for certified-correct stops;
    # a positive plateau if the run settles on a suboptimal matching;
    # a constant for fixed policies like random).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    avg_by_label: Dict[str, np.ndarray] = {}
    for i, (label, (cr_grid, cr_matrix, cr_t_ends)) in enumerate(
        sorted(cumulative_data.items())
    ):
        color = colors[i % len(colors)]
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = cr_matrix / cr_grid[None, :]
        avg[:, 0] = np.nan
        avg_by_label[label] = avg

        if avg.shape[0] <= 6:
            for j in range(avg.shape[0]):
                ax1.plot(cr_grid, avg[j], color=color, alpha=0.22, linewidth=0.9)
        ax1.plot(
            cr_grid, np.nanmean(avg, axis=0), color=color, linewidth=2.2,
            label=label,
        )
        # log-log: decay rate of the early average regret
        mask = cr_grid > 0
        if avg.shape[0] <= 6:
            for j in range(avg.shape[0]):
                ax2.loglog(
                    cr_grid[mask], avg[j][mask], color=color, alpha=0.22,
                    linewidth=0.9,
                )
        ax2.loglog(
            cr_grid[mask], np.nanmean(avg, axis=0)[mask], color=color,
            linewidth=2.2,
        )

    # fixed-policy average regret = constant
    ax1.axhline(
        w_oracle - refs["random_mean"], linestyle="--", color="tab:gray",
        linewidth=1.3, alpha=0.9,
        label=f"random always ({w_oracle - refs['random_mean']:+.3f})",
    )
    ax1.axhline(
        w_oracle - refs["hungarian_train"], linestyle=":", color="black",
        linewidth=1.3, alpha=0.9,
        label=f"Hungarian(train) ({w_oracle - refs['hungarian_train']:+.3f})",
    )
    ax1.axhline(0.0, color="black", linewidth=1.1, alpha=0.7)
    ax1.set_xlabel("number of pairwise observations (t)")
    ax1.set_ylabel("average regret  CR(t)/t")
    ax1.set_title(
        f"Average regret convergence ({run_dir.name})", fontsize=11,
    )
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(alpha=0.25)

    # slope guides on log-log (anchored at the first check time where
    # the mean average regret is finite)
    t0 = None
    for idx in range(1, len(cr_grid)):
        vals = [avg[j, idx] for avg in avg_by_label.values() for j in range(avg.shape[0])]
        if np.any(np.isfinite(vals)):
            t0 = int(cr_grid[idx])
            avg_ref_scale = float(np.nanmean(vals))
            break
    if t0 is None or not np.isfinite(avg_ref_scale) or avg_ref_scale <= 0:
        t0, avg_ref_scale = 1, 1.0
    t_guide = np.array([t0, cr_grid[-1]], dtype=float)
    for alpha_exp, style in ((-0.5, "--"), (-1.0, ":")):
        ax2.plot(
            t_guide, avg_ref_scale * (t_guide / t0) ** alpha_exp,
            linestyle=style, color="black", linewidth=1.1, alpha=0.8,
            label=f"t^{alpha_exp:+.1f}",
        )
    ax2.set_xlabel("t (log scale)")
    ax2.set_ylabel("average regret CR(t)/t (log scale)")
    ax2.set_title("Average regret decay rate (log-log)", fontsize=11)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / "average_regret_convergence.png", dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_dir / "average_regret_convergence.png")

    # ---- convergence summary columns ----
    for row, label in zip(
        summary_rows, [r["group"] for r in summary_rows]
    ):
        cr_grid, cr_matrix, cr_t_ends = cumulative_data[label]
        avg = avg_by_label[label]
        for thr in (1_000, 10_000, 25_000, 50_000, 100_000):
            if cr_grid[-1] < thr:
                continue
            idx = int(np.searchsorted(cr_grid, thr))
            vals = [
                avg[j, min(idx, int(cr_t_ends[j]))] for j in range(avg.shape[0])
            ]
            row[f"avg_regret_at_{thr // 1000}k"] = float(np.nanmean(vals))
        # average regret at each run's own horizon
        end_vals = [avg[j, int(cr_t_ends[j])] for j in range(avg.shape[0])]
        row["avg_regret_at_end"] = float(np.nanmean(end_vals))
        # asymptotic instantaneous regret: mean over last 10% of horizon
        asymp = []
        for j in range(cr_matrix.shape[0]):
            t_end = int(cr_t_ends[j])
            lo = int(t_end * 0.9)
            tail = cr_matrix[j, lo:t_end + 1]
            if t_end > lo:
                asymp.append(float((tail[-1] - tail[0]) / (t_end - lo)))
        row["asymptotic_regret_mean"] = (
            float(np.mean(asymp)) if asymp else float("nan")
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "regret_summary.csv", index=False)

    # reference table
    with open(out_dir / "regret_references.json", "w") as fh:
        json.dump(
            {
                "W_test_oracle_train": w_oracle,
                "W_test_hungarian_train": refs["hungarian_train"],
                "W_test_ceiling_test": refs["ceiling_test"],
                "W_test_random_mean": refs["random_mean"],
            },
            fh, indent=2,
        )
    return summary
