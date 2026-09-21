"""upstream_rep.py — replication of the upstream gale_shapley experiments
(origin/main: experiment.py + experiment_preflid.py) with the market and
reward sourced from REAL LLM benchmark data (LLMRouterBench), everything
else kept as close to the upstream pipeline as the real-data setting
allows.

Upstream pipeline being replicated
----------------------------------
1. Market: per-side theta vectors; preference = ranking of theta;
   oracle H* = men-proposing GS on true theta rankings.
   [UPSTREAM: theta ~ Dirichlet([alpha]*K) independently per side]
   [HERE:     theta = exp(eta * U) resp. exp(eta * V) from the REAL
              full-data utility matrix (U = task-side mean scores,
              V = model-side comparative advantage). Preference =
              ranking of U resp. V — the SAME object the BT reward is
              built from, exactly like upstream where the preference
              IS the theta. No train/test split (upstream has none).]
2. Feedback: Bradley-Terry with the true theta; the learner sees only
   binary comparison outcomes. [Same here: eta calibrated so the
   median positive gap maps to win prob 0.70.]
3. Learners + stopping: P2ETG (adaptive, all-CIs-exclude-0.5 stop)
   and PrefLID (RRT sampling, island certificate).
4. Metrics (upstream semantics):
   * dense per-round timeline: the matching recorded at check time t
     applies to rounds (prev_t, t]; post-stop, the committed matching
     fills POST_STOP_TAIL=200 extra rounds;
   * DEVIATION: dense rows drop upstream's matching_str column (disk:
     ~1.7 GB -> ~25 MB at 50 seeds x 100K rounds); oracle/committed
     strings stay in summary.json;
   * DEVIATION: dense rows drop upstream's matching_str column (disk:
     ~1.7 GB -> ~25 MB at 50 seeds x 100K rounds); oracle/committed
     strings stay in summary.json;
   * `correct` = (current matching == H*) per round;
   * cumulative 0/1 regret = #rounds with correct == 0 so far
     (IDENTIFICATION regret, not welfare regret);
   * summary: T_stop, correct_at_stop, final_regret,
     stable_under_truth / stable_under_hat.
5. Harness: per-run dir runs_upstream/<algo>/N{N}_K{K}_seed{s}/ with
   rounds.csv + summary.json + config.json; aggregated
   all_rounds.csv / all_summaries.csv; upstream-style plots
   (regret mean±std band over seeds, T_stop distribution).

Deviations (documented, all forced by real data):
   * market size 8x8 real (upstream defaults: 10x10 for P2ETG, 3x3
     for PrefLID);
   * theta from calibrated real utilities instead of Dirichlet(alpha)
     — eta plays the role of 1/alpha (gap sharpness);
   * exact utility ties broken by deterministic jitter (upstream's
     Dirichlet draws have no ties);
   * PYTHONHASHSEED=0 required for reproducibility.
"""

from __future__ import annotations

import json
import logging
import random
import time
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
    model_utility,
    strictify_utilities,
    task_utility,
)

logger = logging.getLogger(__name__)

POST_STOP_TAIL = 200  # upstream experiment.py


# ============================================================================
# Real-data environment in upstream semantics
# ============================================================================

def build_upstream_context(config: Dict) -> Dict:
    """Full-data market: U (task side) and V (model side), no split.

    Preference on each side = ranking of that side's utility; the BT
    reward theta = exp(eta * utility) — the same object — matching
    upstream's `random_theta` + `BradleyTerryProvider(true_theta, ...)`
    structure exactly, with real utilities in place of Dirichlet draws.
    """
    out_root = Path(config["output_dir"]) / "runs_upstream"
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = list(config["datasets"])
    models = list(config["models"])
    loader = RouterBenchLoader(
        root=Path(config["routerbench_root"]),
        datasets=datasets,
        models=models,
        min_aligned_records=int(config["min_aligned_records"]),
    )
    aligned = loader.load()

    # full-data utilities (no split)
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
    V = model_utility(U)
    U_s, _, _ = strictify_utilities(U, tie_epsilon, split_seed, "task")
    V_s, _, _ = strictify_utilities(V, tie_epsilon, split_seed, "model")

    men, women, prefs = build_market(U_s, V_s)
    h_star = oracle_matching(prefs)

    feedback_cfg = config["feedback"]
    target = float(feedback_cfg.get("target_median_win_probability", 0.70))
    eta_task = calibrate_eta(median_positive_gap(U_s, tie_epsilon), target)
    eta_model = calibrate_eta(median_positive_gap(V_s, tie_epsilon), target)
    theta_task, theta_model = build_bt_thetas(U_s, V_s, eta_task, eta_model)

    ctx = {
        "config": config, "out_root": out_root,
        "men": men, "women": women, "prefs": prefs, "h_star": h_star,
        "theta_task": theta_task, "theta_model": theta_model,
        "eta_task": eta_task, "eta_model": eta_model,
        "N": len(models), "K": len(datasets),
    }
    logger.info(
        "upstream-rep context: N=%d K=%d eta_task=%.4f eta_model=%.4f",
        ctx["N"], ctx["K"], eta_task, eta_model,
    )
    return ctx


def _pairs(matching: Matching) -> frozenset:
    return frozenset((p.woman.id, p.man.id) for p in matching.pairs)


# ============================================================================
# P2ETG replication (upstream experiment.py::run_single)
# ============================================================================

def run_single_p2etg(
    ctx: Dict, seed: int,
    adaptive: bool = True, check_every: int = 25,
    max_samples: int = 100_000, constant: float = 0.25,
) -> Tuple[List[Dict], Dict]:
    """One P2ETG run in upstream experiment.py semantics."""
    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"uprep::bt::{seed}"),
    )
    from p2etg import P2ETG

    learner = P2ETG(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=constant,
    )

    rounds: List[Tuple[int, Matching, bool]] = []
    result = learner.run_until_stop(
        adaptive=adaptive, check_every=check_every,
        max_samples=max_samples, verbose=False,
        rounds=rounds,
    )

    h_star_pairs = _pairs(ctx["h_star"])
    committed = result["matching"]

    # dense timeline: matching at check t applies to (prev_t, t]
    rows: List[Dict] = []
    prev_t = 0
    for (t, matching, disjoint) in rounds:
        correct = int(_pairs(matching) == h_star_pairs)
        for tt in range(prev_t + 1, t + 1):
            rows.append({"t": tt, "disjoint": disjoint, "correct": correct})
        prev_t = t

    if result["stopped"]:
        correct = int(_pairs(committed) == h_star_pairs)
        for tt in range(prev_t + 1, prev_t + POST_STOP_TAIL + 1):
            rows.append({"t": tt, "disjoint": True, "correct": correct})

    cumulative = 0
    for r in rows:
        cumulative += 1 - r["correct"]
        r["regret"] = cumulative

    from gs_lib.gs_tools import StabilityVerifier

    ok_true, reason_true, _ = StabilityVerifier(ctx["prefs"]).is_stable(
        committed
    )
    prefs_hat = learner._build_preference_lists()
    ok_hat, reason_hat, _ = StabilityVerifier(prefs_hat).is_stable(committed)

    summary = {
        "N": ctx["N"], "K": ctx["K"], "seed": seed,
        "stopped": result["stopped"], "T_stop": result["T_stop"],
        "correct_at_stop": int(_pairs(committed) == h_star_pairs),
        "oracle_str": str(ctx["h_star"]),
        "committed_str": str(committed),
        "final_regret": rows[-1]["regret"] if rows else None,
        "stable_under_truth": int(ok_true),
        "stable_under_hat": int(ok_hat),
        "reason_truth": reason_true, "reason_hat": reason_hat,
        "constant": constant,
    }
    return rows, summary


# ============================================================================
# PrefLID replication (upstream experiment_preflid.py::run_single)
# ============================================================================

def run_single_preflid(
    ctx: Dict, seed: int,
    budget: int = 300_000, constant: float = 0.005,
    max_iterations: int = 10_000, horizon: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """One PrefLID run in upstream experiment_preflid.py semantics.

    Dense timeline to `horizon` (default: the t reached when the run
    ends); after a certified stop the committed matching fills the
    rest of the horizon.
    """
    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"uprep::bt-preflid::{seed}"),
    )
    from preflid import PrefLID

    learner = PrefLID(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=constant, budget=budget,
    )

    trace: List[Tuple[int, Optional[Matching]]] = []
    original_rrt = learner.rrt_round

    def rrt_round_with_record(center):
        n = original_rrt(center)
        learner._refresh_estimates()
        matching = learner._current_gs_matching()
        trace.append((learner.t, matching))
        return n

    learner.rrt_round = rrt_round_with_record
    result = learner.run_until_stop(
        max_iterations=max_iterations, rounds=[], verbose=False,
    )

    h_star_pairs = _pairs(ctx["h_star"])
    stopped = bool(result["stopped"])
    committed = result["matching"]
    t_end = int(result["T_stop"])
    if horizon is None:
        horizon = t_end + (POST_STOP_TAIL if stopped else 0)

    rows: List[Dict] = []
    prev_t = 0
    for (t, matching) in trace:
        if matching is None or t > horizon:
            continue
        correct = int(_pairs(matching) == h_star_pairs)
        for tt in range(prev_t + 1, t + 1):
            rows.append({"t": tt, "correct": correct})
        prev_t = t
    if stopped and committed is not None and t_end < horizon:
        correct = int(_pairs(committed) == h_star_pairs)
        for tt in range(prev_t + 1, horizon + 1):
            rows.append({"t": tt, "correct": correct})
    elif prev_t < horizon:
        # not stopped: the last observed matching fills to the horizon
        last_correct = rows[-1]["correct"] if rows else 0
        for tt in range(prev_t + 1, horizon + 1):
            rows.append({"t": tt, "correct": last_correct})

    cumulative = 0
    for r in rows:
        cumulative += 1 - r["correct"]
        r["regret"] = cumulative

    summary = {
        "N": ctx["N"], "K": ctx["K"], "seed": seed,
        "preflid_stopped": stopped,
        "preflid_T_stop": t_end,
        "preflid_correct": int(_pairs(committed) == h_star_pairs),
        "horizon": horizon,
        "final_regret": rows[-1]["regret"] if rows else None,
        "constant": constant, "budget": budget,
    }
    return rows, summary


# ============================================================================
# Sweep harness (upstream generate_data)
# ============================================================================

def generate_data(
    config: Dict, algorithm: str, seeds: List[int],
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    ctx = build_upstream_context(config)
    algo_root = ctx["out_root"] / algorithm
    algo_root.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = {}
    if algorithm == "p2etg":
        defaults = dict(
            adaptive=True, check_every=25, max_samples=100_000,
            constant=0.25,
        )
    elif algorithm == "preflid":
        defaults = dict(
            budget=300_000, constant=0.005, max_iterations=10_000,
            horizon=None,
        )
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    defaults.update(params)
    params = defaults

    all_rows: List[Dict] = []
    all_summaries: List[Dict] = []
    t0 = time.monotonic()
    for seed in seeds:
        if algorithm == "p2etg":
            rows, summary = run_single_p2etg(ctx, seed, **params)
        else:
            rows, summary = run_single_preflid(ctx, seed, **params)
        run_dir = algo_root / f"N{ctx['N']}_K{ctx['K']}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(run_dir / "rounds.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        (run_dir / "config.json").write_text(
            json.dumps(
                {"algorithm": algorithm, **params,
                 "market": {"N": ctx["N"], "K": ctx["K"],
                            "source": "LLMRouterBench full data"}},
                indent=2,
            )
        )
        for r in rows:
            r.update({"seed": seed})
        all_rows.extend(rows)
        all_summaries.append(summary)
        logger.info(
            "upstream-rep %s seed=%d: T_stop=%s correct=%s regret=%s "
            "(%.1fs)", algorithm, seed, summary.get("T_stop",
            summary.get("preflid_T_stop")),
            summary.get("correct_at_stop", summary.get("preflid_correct")),
            summary.get("final_regret"), time.monotonic() - t0,
        )

    df_rounds = pd.DataFrame(all_rows)
    df_rounds.to_csv(algo_root / "all_rounds.csv", index=False)
    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(algo_root / "all_summaries.csv", index=False)

    _plots(algo_root, df_rounds, df_sum, algorithm)
    return df_sum


def _plots(algo_root: Path, df_rounds: pd.DataFrame,
           df_sum: pd.DataFrame, algorithm: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = algo_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    # regret with uncertainty band (upstream plot_regret_with_uncertainty)
    grouped = (
        df_rounds.groupby("t")["regret"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(grouped["t"], grouped["mean"], color="tab:blue",
            linewidth=2.0, label="mean cumulative 0/1 regret")
    ax.fill_between(
        grouped["t"], grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"], alpha=0.25, color="tab:blue",
        label="±1 std",
    )
    ax.set_xlabel("t (rounds)")
    ax.set_ylabel("cumulative 0/1 regret")
    ax.set_title(
        f"{algorithm}: identification regret vs time "
        f"({df_sum.shape[0]} seeds, real LLM market)", fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "regret_with_uncertainty.png", dpi=160)
    plt.close(fig)

    # T_stop distribution
    t_col = "T_stop" if "T_stop" in df_sum.columns else "preflid_T_stop"
    fig, ax = plt.subplots(figsize=(8, 5))
    stopped = df_sum[pd.to_numeric(df_sum.get("stopped",
             df_sum.get("preflid_stopped")), errors="coerce") == 1]
    if len(stopped):
        ax.hist(pd.to_numeric(stopped[t_col], errors="coerce"), bins=30,
                color="tab:orange", alpha=0.8, edgecolor="white")
    ax.set_xlabel("T_stop (pairwise observations)")
    ax.set_ylabel("#seeds")
    ax.set_title(
        f"{algorithm}: T_stop distribution "
        f"({len(stopped)}/{len(df_sum)} stopped)", fontsize=11,
    )
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "tstop_distribution.png", dpi=160)
    plt.close(fig)
    logger.info("Wrote plots to %s", plots_dir)
