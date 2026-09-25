"""diagnostics.py

Hindsight diagnostics for P2ETG / Matching-ID runs, and orchestration
of the 5-seed diagnostic study.

IMPORTANT: every metric in `compute_run_diagnostics` is a HINDSIGHT
diagnostic. None of them may be used as a stopping rule (section 19 of
the Phase-2 spec); they only describe what happened along a run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OCCUPANCY_THRESHOLDS = (10_000, 25_000, 50_000, 100_000)


# ============================================================================
# Per-run hindsight diagnostics (pure functions of the trace)
# ============================================================================

def compute_run_diagnostics(trace_df: pd.DataFrame) -> Dict[str, object]:
    """First-hit / occupancy / streak / change metrics from a trace.

    The trace must contain columns: t, matching, exact_oracle_match,
    train_stable, test_welfare.
    """
    out: Dict[str, object] = {
        "T_first_oracle_hit": float("nan"),
        "oracle_occupancy": float("nan"),
        "longest_oracle_streak": 0,
        "num_matching_changes": 0,
        "T_first_train_stable": float("nan"),
        "best_test_welfare": float("nan"),
    }
    for thr in OCCUPANCY_THRESHOLDS:
        out[f"oracle_occupancy_after_{thr // 1000}k"] = float("nan")

    if trace_df is None or trace_df.empty:
        return out

    df = trace_df.sort_values("t").reset_index(drop=True)
    exact = pd.to_numeric(df["exact_oracle_match"]).astype(bool)
    t = pd.to_numeric(df["t"])

    hits = df.loc[exact, "t"]
    if len(hits):
        out["T_first_oracle_hit"] = int(hits.iloc[0])

    out["oracle_occupancy"] = float(exact.mean())

    for thr in OCCUPANCY_THRESHOLDS:
        mask = t >= thr
        if mask.any():
            out[f"oracle_occupancy_after_{thr // 1000}k"] = float(
                exact[mask].mean()
            )

    # longest streak of consecutive exact-match checks
    longest = 0
    current = 0
    for flag in exact:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    out["longest_oracle_streak"] = longest

    # number of matching changes between consecutive checks
    matching = df["matching"].astype(str)
    if len(matching) > 1:
        out["num_matching_changes"] = int(
            (matching.iloc[1:].values != matching.iloc[:-1].values).sum()
        )

    stable = pd.to_numeric(df["train_stable"]).astype(bool)
    st = df.loc[stable, "t"]
    if len(st):
        out["T_first_train_stable"] = int(st.iloc[0])

    if "test_welfare" in df.columns:
        out["best_test_welfare"] = float(pd.to_numeric(df["test_welfare"]).max())

    return out


# ============================================================================
# Per-arm state snapshots
# ============================================================================

def arm_state_rows(learner, seed: int, t: int) -> pd.DataFrame:
    """One row per pairwise arm with the learner's current CI state.

    partner_1 is the CANONICAL-FIRST partner, so p_hat reads as the
    win probability of partner_1.
    """
    rows: List[dict] = []
    for state in learner.agent_states.values():
        agent = state.agent
        side = "task" if type(agent).__name__ == "Woman" else "model"
        for (c1, c2), (lo, hi) in state.ci.items():
            n = state.counts.total.get((c1, c2), 0)
            wins = state.counts.wins.get((c1, c2), 0)
            rows.append(
                {
                    "seed": seed,
                    "t": t,
                    "agent_side": side,
                    "agent": str(agent.id),
                    "partner_1": str(c1.id if hasattr(c1, "id") else c1),
                    "partner_2": str(c2.id if hasattr(c2, "id") else c2),
                    "n_samples": n,
                    "p_hat": (wins / n) if n > 0 else float("nan"),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "resolved": bool(lo > 0.5 or hi < 0.5),
                }
            )
    return pd.DataFrame(rows)


def annotation_lookup(annotations: Optional[pd.DataFrame]) -> Optional[dict]:
    """Fast lookup: (side, agent, sorted(p1, p2)) -> annotation row."""
    if annotations is None or annotations.empty:
        return None
    lookup = {}
    for _, r in annotations.iterrows():
        key = (
            r["side"],
            r["agent"],
            tuple(sorted((r["partner_1"], r["partner_2"]))),
        )
        lookup[key] = r
    return lookup


def arm_row_with_annotations(
    row: dict, lookup: Optional[dict]
) -> dict:
    """Join a CI-state arm row with its static annotation columns."""
    out = dict(row)
    if lookup is None:
        for col in (
            "utility_gap",
            "local_matching_critical",
            "bootstrap_flip_probability",
            "bt_true_probability",
            "adjacent_in_oracle",
            "pair_swap_changes_matching",
        ):
            out[col] = None
        return out
    key = (
        row["agent_side"],
        row["agent"],
        tuple(sorted((row["partner_1"], row["partner_2"]))),
    )
    ann = lookup.get(key)
    if ann is None:
        for col in (
            "utility_gap",
            "local_matching_critical",
            "bootstrap_flip_probability",
            "bt_true_probability",
            "adjacent_in_oracle",
            "pair_swap_changes_matching",
        ):
            out[col] = None
        return out
    out["utility_gap"] = float(ann["utility_gap"])
    # primary criticality: adjacent minimal reversal
    out["local_matching_critical"] = bool(
        ann["adjacent_reversal_changes_matching"]
    ) if pd.notna(ann["adjacent_reversal_changes_matching"]) else False
    out["bootstrap_flip_probability"] = float(
        ann["bootstrap_flip_probability"]
    )
    # bt_true_probability in the sensitivity table is P(ITS partner_1
    # wins), where its partner_1 is the true-preferred partner. Re-express
    # relative to THIS row's canonical-first partner.
    ann_p1 = str(ann["partner_1"])
    p_true = float(ann["bt_true_probability"])
    out["bt_true_probability"] = (
        p_true if ann_p1 == row["partner_1"] else 1.0 - p_true
    )
    out["adjacent_in_oracle"] = bool(ann["adjacent_in_oracle"])
    out["pair_swap_changes_matching"] = bool(
        ann["pair_swap_changes_matching"]
    )
    return out


def count_unresolved_critical(
    arm_rows: pd.DataFrame,
) -> Tuple[int, int]:
    """(# unresolved critical arms, # unresolved non-critical arms)."""
    if arm_rows is None or arm_rows.empty:
        return (0, 0)
    unresolved = arm_rows[~arm_rows["resolved"].astype(bool)]
    critical = unresolved["local_matching_critical"]
    critical = pd.Series(critical).fillna(False).astype(bool)
    n_critical = int(critical.sum())
    return (n_critical, int(len(unresolved) - n_critical))


# ============================================================================
# Orchestration: 5-seed diagnostic study
# ============================================================================

def run_diagnostics_study(
    config: Dict,
    seeds: Optional[List[int]] = None,
    feedbacks: Optional[List[str]] = None,
    num_bootstrap: int = 500,
    bootstrap_seed: int = 20260918,
) -> pd.DataFrame:
    """Run bootstrap -> sensitivity -> 5+5 P2ETG seeds -> summary+plots."""
    from llm_matching.bootstrap import run_bootstrap, save_bootstrap_outputs
    from llm_matching.runner import build_context, run_experiment
    from llm_matching.sensitivity import (
        compute_arm_sensitivity,
        save_sensitivity_outputs,
    )

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if feedbacks is None:
        feedbacks = ["bt", "replay"]

    ctx = build_context(config)

    # 1. bootstrap (writes outputs/<run>/bootstrap/)
    boot = run_bootstrap(
        ctx, num_bootstrap=num_bootstrap, bootstrap_seed=bootstrap_seed
    )
    save_bootstrap_outputs(ctx.out_dir, boot)

    # 2. sensitivity (writes outputs/<run>/sensitivity/)
    arms = compute_arm_sensitivity(ctx, bootstrap_result=boot)
    save_sensitivity_outputs(ctx.out_dir, arms)

    # 3. run seeds per feedback mode into diagnostics_5seed/<mode>/
    diag_dir = ctx.out_dir / "diagnostics_5seed"
    diag_dir.mkdir(parents=True, exist_ok=True)

    per_seed_frames: List[pd.DataFrame] = []
    for feedback in feedbacks:
        mode_config = dict(config)
        mode_config["output_dir"] = str(diag_dir / feedback)
        mode_config = _ensure_output_dir(mode_config, str(diag_dir / feedback))
        run_experiment(
            mode_config,
            algorithm="p2etg",
            feedback=feedback,
            seeds_override=list(seeds),
            arm_annotations=arms,
            make_plots=False,
            make_report=False,
        )
        mode_dir = Path(mode_config["output_dir"])
        summary = pd.read_csv(mode_dir / "per_seed_summary.csv")
        summary["feedback"] = feedback

        # final arm counts per seed
        for seed in seeds:
            arm_final_path = mode_dir / "traces" / f"arm_final_seed_{seed:03d}.csv"
            if arm_final_path.exists():
                arm_final = pd.read_csv(arm_final_path)
                n_crit, n_noncrit = count_unresolved_critical(arm_final)
                summary.loc[summary["seed"] == seed, "unresolved_critical_final"] = n_crit
                summary.loc[
                    summary["seed"] == seed, "unresolved_noncritical_final"
                ] = n_noncrit
        per_seed_frames.append(summary)

    combined = pd.concat(per_seed_frames, ignore_index=True)

    # 4. write combined summary
    from llm_matching.plots import write_diagnostics_summary_and_plots

    write_diagnostics_summary_and_plots(
        diag_dir, combined, feedbacks, seeds
    )
    return combined


def _ensure_output_dir(config: Dict, out_dir: str) -> Dict:
    """Deep-copy-ish config with a replaced output_dir."""
    new_config = dict(config)
    for key in ("split", "feedback", "ties", "p2etg", "preflid", "experiment"):
        if key in config:
            new_config[key] = dict(config[key])
    new_config["output_dir"] = out_dir
    return new_config


# ============================================================================
# 4x4 Matching-ID vs P2ETG comparison study
# ============================================================================

def run_matching_id_study(
    config: Dict,
    seeds: Optional[List[int]] = None,
    feedbacks: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    num_bootstrap: int = 500,
    bootstrap_seed: int = 20260918,
) -> pd.DataFrame:
    """Run P2ETG vs Matching-ID under BT and replay on the configured
    (smoke 4x4) market; write matching_id/ outputs."""
    from llm_matching.bootstrap import run_bootstrap
    from llm_matching.plots import write_matching_id_comparison
    from llm_matching.runner import build_context, run_experiment
    from llm_matching.sensitivity import compute_arm_sensitivity

    if seeds is None:
        seeds = list(range(20))
    if feedbacks is None:
        feedbacks = ["bt", "replay"]
    if algorithms is None:
        algorithms = ["p2etg", "matching_id", "preflid"]

    ctx = build_context(config)

    # Arm annotations (bootstrap + sensitivity) for this market, needed
    # for the "unresolved critical arms at stop" metric.
    boot = run_bootstrap(
        ctx, num_bootstrap=num_bootstrap, bootstrap_seed=bootstrap_seed
    )
    arms = compute_arm_sensitivity(ctx, bootstrap_result=boot)

    mid_dir = ctx.out_dir / "matching_id"
    mid_dir.mkdir(parents=True, exist_ok=True)
    mid_traces = mid_dir / "traces"
    mid_traces.mkdir(parents=True, exist_ok=True)

    per_seed_frames = []
    for algorithm in algorithms:
        for feedback in feedbacks:
            combo_dir = mid_dir / f"{algorithm}_{feedback}"
            combo_config = _ensure_output_dir(config, str(combo_dir))
            run_experiment(
                combo_config,
                algorithm=algorithm,
                feedback=feedback,
                seeds_override=list(seeds),
                arm_annotations=arms,
                make_plots=False,
                make_report=False,
            )
            summary = pd.read_csv(combo_dir / "per_seed_summary.csv")
            per_seed_frames.append(summary)
            # Copy traces under a combo-prefixed name.
            import shutil

            for seed in seeds:
                src = combo_dir / "traces" / f"seed_{seed:03d}.csv"
                if src.exists():
                    shutil.copy(
                        src,
                        mid_traces / f"{algorithm}_{feedback}_seed_{seed:03d}.csv",
                    )

    combined = pd.concat(per_seed_frames, ignore_index=True)
    combined.to_csv(mid_dir / "per_seed_summary.csv", index=False)
    write_matching_id_comparison(mid_dir, combined)
    return combined
