"""bandit.py — online matching-bandit experiment.

Setting: ONLINE LEARNING. There is NO train/test split: the
environment is defined by the FULL-data utility matrix, and
preferences are built in symmetric (mirror) mode so the stable
matching H* is provably UNIQUE (verified at build time: men-GS ==
women-GS == mutual-best cascade).

The learner never sees the utilities; it only observes noisy pairwise
BT feedback. At every algorithm checkpoint the learner's current
matching M_t is evaluated on the TRUE utilities:

    instantaneous regret  r(t) = max(0, W(H*) - W(M_t))
                          (clipped at 0 per round: holding a
                          welfare-better-than-H* matching earns no
                          credit, so cumulative regret is
                          non-decreasing by construction)
    cumulative regret     R(T) = integral_0^T r(t) dt  (trapezoid)

Post-stop semantics: once an algorithm certifies and commits, its
matching is frozen for the remaining horizon (regret accrues at the
committed matching's rate), so all algorithms are compared over the
same horizon.

Algorithms share the SAME estimator (MLE-GS ranking on current
counts); they differ only in the SAMPLING policy:

    P2ETG   — uniform round-robin over all pairwise arms
    PrefLID — RRT center-agent rounds (one full round-robin per
              iteration for the center, least-sampled pairs first)

Outputs under <output_dir>/bandit/:
  per_seed_summary.csv   one row per (algorithm, seed)
  regret_traces.csv      long format: algorithm, seed, t, regret, cum_regret
  cumulative_regret.png  mean curves + faint per-seed + random slope
  cumulative_regret_loglog.png  decay-rate view with t^0.5 / t guides
  bandit_report.md
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gs_lib.gs_tools import GaleShapley, Man, Matching, Woman

from llm_matching.oracle import build_market, oracle_matching
from llm_matching.providers import (
    RouterBenchBTProvider,
    build_bt_thetas,
    calibrate_eta,
)
from llm_matching.routerbench_loader import RouterBenchLoader
from llm_matching.utilities import (
    median_positive_gap,
    mutual_best_cascade,
    positive_gaps,
    symmetric_strict_matrix,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Online (full-data) environment
# ============================================================================

@dataclass
class BanditContext:
    config: Dict
    U_raw: pd.DataFrame          # full-data raw utilities (index=dataset)
    W: pd.DataFrame               # tie-free shared preference matrix
    theta_task: Dict
    theta_model: Dict
    men: List[Man]
    women: List[Woman]
    w_star: float                 # W(H*) welfare of the unique stable matching
    oracle: Dict[str, str]        # task -> model
    w_hungarian: float
    regret_random: float          # W(H*) - E W(random matching)
    eta_task: float
    eta_model: float
    out_dir: Path


def build_bandit_context(config: Dict) -> BanditContext:
    """Full-data online environment (no splits, symmetric preferences)."""
    pref_mode = str(config.get("preferences", {}).get("mode", "comparative"))
    if pref_mode != "symmetric":
        raise ValueError(
            "the matching-bandit experiment requires "
            "preferences.mode: symmetric (unique stable matching)"
        )

    out_dir = Path(config["output_dir"]) / "bandit"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(config["datasets"])
    models = list(config["models"])
    loader = RouterBenchLoader(
        root=Path(config["routerbench_root"]),
        datasets=datasets,
        models=models,
        min_aligned_records=int(config["min_aligned_records"]),
    )
    aligned = loader.load()

    # full-data utilities: mean score over ALL aligned records
    U = (
        aligned.records.groupby(["dataset", "model"])["score"]
        .mean()
        .unstack("model")
        .reindex(index=datasets, columns=models)
    )
    if U.isna().any().any():
        raise ValueError("full-data utility matrix has missing entries")

    tie_epsilon = float(config["ties"]["epsilon"])
    split_seed = int(config["split"]["seed"])
    W, _, _ = symmetric_strict_matrix(U, tie_epsilon, split_seed)

    # unique stable matching: GS both ways + cascade cross-check
    men, women, prefs = build_market(W, W.T)
    h_men = GaleShapley(prefs).find_stable_matching("men")
    h_women = GaleShapley(prefs).find_stable_matching("women")
    cascade = mutual_best_cascade(W)
    pairs = lambda h: frozenset((p.woman.id, p.man.id) for p in h.pairs)
    if not (pairs(h_men) == pairs(h_women)):
        raise AssertionError(
            "symmetric full-data market does not have a unique stable "
            "matching — cannot define the bandit regret target"
        )
    if dict(cascade) != {t: m for t, m in pairs(h_men)}:
        raise AssertionError("cascade != GS; mirror construction broken")
    oracle_dict = dict(cascade)

    w_star = float(sum(W.loc[d, m] for d, m in oracle_dict.items()))

    # baselines
    from scipy.optimize import linear_sum_assignment

    cost = -W.to_numpy(dtype=float)
    rows, cols = linear_sum_assignment(cost)
    w_hungarian = float(-cost[rows, cols].sum())
    n = len(datasets)
    regret_random = w_star - float(W.to_numpy().mean() * n)

    # BT thetas calibrated on full-data gaps (both sides share W)
    feedback_cfg = config["feedback"]
    target = float(feedback_cfg.get("target_median_win_probability", 0.70))
    eta_task = calibrate_eta(median_positive_gap(W, tie_epsilon), target)
    eta_model = calibrate_eta(
        median_positive_gap(W.T, tie_epsilon), target
    )
    theta_task, theta_model = build_bt_thetas(W, W.T, eta_task, eta_model)

    logger.info(
        "Bandit context: W* = %.4f (Hungarian %.4f, random regret %.4f); "
        "eta_task %.4f, eta_model %.4f",
        w_star, w_hungarian, regret_random, eta_task, eta_model,
    )
    return BanditContext(
        config=config, U_raw=U, W=W, theta_task=theta_task,
        theta_model=theta_model, men=men, women=women, w_star=w_star,
        oracle=oracle_dict, w_hungarian=w_hungarian,
        regret_random=regret_random, eta_task=eta_task,
        eta_model=eta_model, out_dir=out_dir,
    )


# ============================================================================
# Per-seed bandit runs
# ============================================================================

def _matching_welfare(matching: Matching, W: pd.DataFrame) -> float:
    return float(sum(W.loc[p.woman.id, p.man.id] for p in matching.pairs))


def _welfare_of(m, W):
    return float(sum(W.loc[p.woman.id, p.man.id] for p in m.pairs))


class _RecordingPrefLID:
    """PrefLID wrapper that records (t, estimated matching) after every
    RRT iteration (same MLE-GS estimator as P2ETG, different sampling)."""

    def __init__(self, base, records_out: List, W: pd.DataFrame):
        original_rrt = base.rrt_round

        def rrt_round_with_record(center):
            n = original_rrt(center)
            base._refresh_estimates()
            records_out.append((base.t, base._current_gs_matching()))
            return n

        base.rrt_round = rrt_round_with_record


def run_bandit_seed(
    ctx: BanditContext, algorithm: str, seed: int, budget: int
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """One online run; returns (per-check regret rows, summary)."""
    t0 = time.monotonic()
    config = ctx.config
    records: List[Tuple[int, Matching]] = []  # (t, estimated matching)
    stopped = False
    t_stop = budget
    committed = None

    provider = RouterBenchBTProvider(
        ctx.theta_task, ctx.theta_model,
        rng=random.Random(f"bandit::bt::{seed}"),
        probability_floor=float(
            ctx.config.get("feedback", {}).get("probability_floor", 0.0)
        ),
    )

    if algorithm == "p2etg":
        from p2etg import P2ETG

        cfg = config["p2etg"]
        learner = P2ETG(
            men=ctx.men, women=ctx.women, provider=provider,
            rng=random.Random(f"bandit::p2etg::{seed}"),
            constant=float(cfg.get("constant", 0.1)),
        )

        class _Recorder(list):
            def append(self, item):
                t, matching, disjoint = item
                records.append((t, matching))

        result = learner.run_until_stop(
            adaptive=bool(cfg.get("adaptive", True)),
            check_every=int(cfg.get("check_every", 448)),
            max_samples=budget,
            verbose=False,
            rounds=_Recorder(),
        )
        stopped = bool(result["stopped"])
        t_stop = int(result["T_stop"])
        committed = result["matching"] if stopped else None
    elif algorithm == "preflid":
        from preflid import PrefLID

        cfg = config["preflid"]
        learner = PrefLID(
            men=ctx.men, women=ctx.women, provider=provider,
            rng=random.Random(f"bandit::preflid::{seed}"),
            constant=float(cfg.get("constant", 0.1)),
            budget=int(cfg.get("budget", 100)),
            max_lattice_vertices=int(cfg.get("max_lattice_vertices", 5000)),
            min_samples_per_pair=int(cfg.get("min_samples_per_pair", 10)),
            min_sample_ratio=float(cfg.get("min_sample_ratio", 0.5)),
        )
        wrapper = _RecordingPrefLID(learner, records, ctx.W)
        result = learner.run_until_stop(
            max_iterations=int(cfg.get("max_iterations", 7143)),
            rounds=[],
            verbose=False,
        )
        stopped = bool(result["stopped"])
        t_stop = int(result["T_stop"])
        committed = result["matching"] if stopped else None
    else:
        raise ValueError(f"unknown bandit algorithm {algorithm!r}")

    # per-check regret rows (frozen committed welfare after stop)
    W = ctx.W
    rows = []
    prev_t, area = 0, 0.0
    prev_regret = (
        max(0.0, ctx.w_star - _welfare_of(records[0][1], W))
        if records else 0.0
    )
    for t, m in records:
        w = _welfare_of(m, W)
        r = max(0.0, ctx.w_star - w)  # clipped: no credit for beating H*
        dt = t - prev_t
        area += (prev_regret + r) / 2.0 * dt
        rows.append(
            {
                "algorithm": algorithm, "seed": seed, "t": int(t),
                "welfare": w, "regret": r, "cum_regret": area,
            }
        )
        prev_t, prev_regret = t, r
    # freeze after stop (committed matching plays the remaining horizon)
    if stopped and t_stop < budget and rows:
        r_frozen = max(0.0, ctx.w_star - _welfare_of(committed, W))
        dt = budget - t_stop
        area += (prev_regret + r_frozen) / 2.0 * dt
        rows.append(
            {
                "algorithm": algorithm, "seed": seed, "t": budget,
                "welfare": _welfare_of(committed, W), "regret": r_frozen,
                "cum_regret": area,
            }
        )
        final_matching = committed
    else:
        final_matching = records[-1][1] if records else None

    final_exact = (
        frozenset((p.woman.id, p.man.id) for p in final_matching.pairs)
        == frozenset(ctx.oracle.items())
        if final_matching is not None else False
    )
    inst = [r_["regret"] for r_ in rows]
    summary = {
        "algorithm": algorithm, "seed": seed, "stopped": stopped,
        "T_stop": t_stop, "final_exact": final_exact,
        "final_regret": inst[-1] if inst else float("nan"),
        "best_regret": min(inst) if inst else float("nan"),
        "final_cum_regret": rows[-1]["cum_regret"] if rows else float("nan"),
        "n_checks": len(rows),
        "elapsed_seconds": time.monotonic() - t0,
    }
    return pd.DataFrame(rows), summary


# ============================================================================
# Study driver
# ============================================================================

def run_matching_bandit(
    config: Dict,
    algorithms: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    budget: Optional[int] = None,
) -> pd.DataFrame:
    """Run the online matching-bandit study; writes outputs + plots."""
    if algorithms is None:
        algorithms = ["p2etg", "preflid"]
    if seeds is None:
        seeds = list(range(10))
    if budget is None:
        budget = int(config["p2etg"].get("max_samples", 200000))

    ctx = build_bandit_context(config)

    all_rows: List[pd.DataFrame] = []
    summaries = []
    for algorithm in algorithms:
        for seed in seeds:
            logger.info("bandit: %s seed=%d ...", algorithm, seed)
            rows, summary = run_bandit_seed(ctx, algorithm, seed, budget)
            all_rows.append(rows)
            summaries.append(summary)
            logger.info(
                "bandit %s seed=%d: cum_regret=%.1f final=%.4f exact=%s "
                "(%.1fs)",
                algorithm, seed, summary["final_cum_regret"],
                summary["final_regret"], summary["final_exact"],
                summary["elapsed_seconds"],
            )

    traces = pd.concat(all_rows, ignore_index=True)
    traces.to_csv(ctx.out_dir / "regret_traces.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(ctx.out_dir / "per_seed_summary.csv", index=False)

    _bandit_plots(ctx, traces, summary_df, algorithms, budget)
    _bandit_report(ctx, summary_df, algorithms)
    return summary_df


def _bandit_plots(ctx, traces, summary_df, algorithms, budget) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"p2etg": "tab:blue", "preflid": "tab:red"}
    grid = np.arange(0, budget + 1, max(budget // 500, 1))

    def _mean_curve(algorithm, column):
        curves = []
        for seed, sub in traces[traces["algorithm"] == algorithm].groupby(
            "seed"
        ):
            sub = sub.sort_values("t")
            values = np.interp(
                grid, sub["t"].to_numpy(),
                pd.to_numeric(sub[column], errors="coerce").to_numpy(),
                left=np.nan, right=np.nan,
            )
            t_end = int(sub["t"].max())
            values[grid <= t_end] = (
                pd.Series(values[grid <= t_end]).ffill().bfill().to_numpy()
            )
            curves.append(values)
        return grid, np.vstack(curves)

    # linear cumulative regret
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algorithm in algorithms:
        sub = traces[traces["algorithm"] == algorithm]
        color = colors.get(algorithm, "tab:green")
        for seed, s in sub.groupby("seed"):
            ax.plot(
                s["t"], pd.to_numeric(s["cum_regret"], errors="coerce"),
                color=color, alpha=0.18, linewidth=0.9,
            )
        grid_, matrix = _mean_curve(algorithm, "cum_regret")
        ax.plot(
            grid_, np.nanmean(matrix, axis=0), color=color, linewidth=2.4,
            label=f"{algorithm} (mean, n={matrix.shape[0]})",
        )
    ax.set_xlabel("number of pairwise observations (t)")
    ax.set_ylabel("cumulative regret  R(t)")
    ax.set_title(
        f"Online matching bandit — cumulative regret vs unique stable "
        f"matching H*\n({ctx.out_dir.parent.name}, BT feedback)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ctx.out_dir / "cumulative_regret.png", dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", ctx.out_dir / "cumulative_regret.png")

    # log-log decay view
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algorithm in algorithms:
        grid_, matrix = _mean_curve(algorithm, "regret")
        mean = np.nanmean(matrix, axis=0)
        mask = (grid_ > 0) & np.isfinite(mean)
        ax.loglog(grid_[mask], np.clip(mean[mask], 1e-6, None),
                  linewidth=2.2, color=colors.get(algorithm, "tab:green"),
                  label=algorithm)
    t_guide = np.array([grid[1], grid[-1]], dtype=float)
    anchor = np.nanmean(
        [np.nanmean(_mean_curve(a, "regret")[1][:, 5]) for a in algorithms]
    )
    for alpha_exp, style in ((-0.5, "--"), (-1.0, ":")):
        ax.plot(t_guide, anchor * (t_guide / grid[1]) ** alpha_exp,
                linestyle=style, color="black", linewidth=1.1, alpha=0.8,
                label=f"t^{alpha_exp:+.1f}")
    ax.set_xlabel("t (log)")
    ax.set_ylabel("instantaneous regret (log)")
    ax.set_title("Instantaneous regret decay (log-log)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(ctx.out_dir / "regret_decay_loglog.png", dpi=160)
    plt.close(fig)


def _bandit_report(ctx, summary_df, algorithms) -> None:
    lines = [
        "# Online Matching-Bandit Experiment", "",
        "* setting: online learning, NO train/test split; utilities from "
        "the FULL data; symmetric (mirror) preferences -> UNIQUE stable "
        "matching H* (men-GS == women-GS == cascade, verified)",
        f"* W(H*) = {ctx.w_star:.4f}; Hungarian = "
        f"{ctx.w_hungarian:.4f}",
        f"* BT feedback, eta_task={ctx.eta_task:.4f}, "
        f"eta_model={ctx.eta_model:.4f}",
        "* same estimator (MLE-GS) for both algorithms; only the SAMPLING "
        "policy differs (P2ETG uniform round-robin vs PrefLID RRT rounds)",
        "",
    ]
    for algorithm in algorithms:
        s = summary_df[summary_df["algorithm"] == algorithm]
        num = lambda c: pd.to_numeric(s[c], errors="coerce")
        lines.append(
            f"## {algorithm} ({len(s)} seeds)"
        )
        lines.append("")
        lines.append(
            f"* stopped (certified): {int(pd.to_numeric(s['stopped']).sum())}"
            f"/{len(s)}; final exact: "
            f"{int(pd.to_numeric(s['final_exact']).sum())}/{len(s)}"
        )
        lines.append(
            f"* final cumulative regret: mean "
            f"{num('final_cum_regret').mean():.1f} "
            f"(std {num('final_cum_regret').std():.1f}); median "
            f"{num('final_cum_regret').median():.1f}"
        )
        lines.append(
            f"* final instantaneous regret: mean "
            f"{num('final_regret').mean():.4f}; best-seen mean "
            f"{num('best_regret').mean():.4f}"
        )
        lines.append(
            f"* elapsed: mean {num('elapsed_seconds').mean():.1f}s/seed"
        )
        lines.append("")
    (ctx.out_dir / "bandit_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
