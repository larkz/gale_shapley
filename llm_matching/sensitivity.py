"""sensitivity.py

Per-arm preference criticality analysis for the 448-arm 8x8 market
(works for any NxN market).

For every pairwise preference arm (agent, partner_i, partner_j):

  1. utility_gap      |U_a(i) - U_a(j)| (raw TRAIN utilities)
  2. bootstrap_flip_probability
                      P_b(sign(u_b(i)-u_b(j)) != sign(u(i)-u(j)))
                      from the bootstrap replicates
  3. local GS sensitivity:
     - adjacent_reversal_changes_matching: swap two partners that are
       ADJACENT in the agent's true oracle ranking (a minimal single
       pairwise preference reversal), rerun model-proposing GS, and
       check whether the matching changes. Primary criticality metric.
     - pair_swap_changes_matching: for ALL pairs, exchange the two
       partners' positions in the total ranking. For non-adjacent
       pairs this is NOT a minimal preference reversal (other implied
       pairwise relations also change); it is reported separately and
       must not be confused with the adjacent metric.
  4. bt_true_probability: the true BT win probability of partner_1
     over partner_2 under the calibrated thetas.

Outputs under outputs/<run>/sensitivity/:
  arm_sensitivity.csv                (canonical order)
  arm_sensitivity_by_flip.csv        (bootstrap_flip_probability DESC)
  arm_sensitivity_by_gap.csv         (utility_gap ASC)
  plots/gap_criticality_scatter.png
  plots/bootstrap_flip_vs_gap.png
  plots/critical_arm_gap_distribution.png
  sensitivity_report.md
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gs_lib.gs_tools import GaleShapley, Man, Matching, PreferenceList, Woman

from llm_matching.bootstrap import BootstrapResult, run_bootstrap
from llm_matching.metrics import matching_to_dict
from llm_matching.oracle import build_market

logger = logging.getLogger(__name__)


# ============================================================================
# Preference ranking helpers
# ============================================================================

def _agent_ranking(
    prefs: PreferenceList, agent
) -> List:
    return list(prefs.get_preference(agent))


def _swap_positions(ranking: List, i: int, j: int) -> List:
    """Exchange the entries at positions i and j (0-based)."""
    out = list(ranking)
    out[i], out[j] = out[j], out[i]
    return out


def _matching_signature(matching: Matching) -> frozenset:
    return frozenset((p.man.id, p.woman.id) for p in matching.pairs)


def _num_assignment_changes(m1: Matching, m2: Matching) -> int:
    """Number of matched pairs that differ (both directions counted once
    per broken pair)."""
    return len(_matching_signature(m1) ^ _matching_signature(m2)) // 2


# ============================================================================
# Core analysis
# ============================================================================

def compute_arm_sensitivity(
    ctx,
    bootstrap_result: Optional[BootstrapResult] = None,
    num_bootstrap: int = 500,
    bootstrap_seed: int = 20260918,
) -> pd.DataFrame:
    """Build the per-arm sensitivity table for the TRAIN oracle market."""
    t0 = time.monotonic()

    if bootstrap_result is None:
        bootstrap_result = run_bootstrap(
            ctx, num_bootstrap=num_bootstrap, bootstrap_seed=bootstrap_seed
        )

    prefs_train = ctx.prefs["train"]
    oracle_sig = _matching_signature(ctx.oracle["train"])

    task_util = ctx.task_util["train"]  # raw
    model_util = ctx.model_util["train"]  # raw
    datasets = list(ctx.config["datasets"])
    models = list(ctx.config["models"])
    men = ctx.men
    women = ctx.women

    flip_lookup = {
        (r["side"], r["agent"], r["partner_1"], r["partner_2"]): r[
            "bootstrap_flip_probability"
        ]
        for _, r in bootstrap_result.preference_flips.iterrows()
    }

    def flip_prob(side: str, agent_id: str, p1: str, p2: str) -> float:
        key = (side, agent_id, p1, p2)
        if key in flip_lookup:
            return float(flip_lookup[key])
        return float(flip_lookup[(side, agent_id, p2, p1)])

    theta_task = ctx.theta_task  # agent_id (task) -> {model -> theta}
    theta_model = ctx.theta_model

    def bt_true_probability(side: str, agent_id: str, p1: str, p2: str) -> float:
        table = theta_task[agent_id] if side == "task" else theta_model[agent_id]
        t1, t2 = table[p1], table[p2]
        return t1 / (t1 + t2)

    rows: List[dict] = []
    all_agents = [(w, "task", women) for w in women] + [
        (m, "model", men) for m in men
    ]

    for agent, side, partners_same_side in all_agents:
        agent_id = agent.id
        ranking = _agent_ranking(prefs_train, agent)
        rank_of = {p: i for i, p in enumerate(ranking)}
        util = (
            task_util.loc[agent_id] if side == "task"
            else model_util.loc[agent_id]
        )

        partners = list(ranking)
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                p1, p2 = partners[i], partners[j]
                r1, r2 = rank_of[p1], rank_of[p2]
                adjacent = abs(r1 - r2) == 1

                # pair-swap perturbation (positions r1, r2 exchanged)
                swapped_ranking = _swap_positions(ranking, r1, r2)
                perturbed_prefs = _replace_ranking(
                    prefs_train, agent, swapped_ranking
                )
                h_swap = GaleShapley(perturbed_prefs).find_stable_matching("men")
                swap_sig = _matching_signature(h_swap)
                pair_swap_changes = swap_sig != oracle_sig
                n_changes = _num_assignment_changes(h_swap, ctx.oracle["train"])

                # adjacent minimal reversal (only defined for adjacent
                # pairs; identical to the pair swap there)
                adjacent_changes: Optional[bool] = (
                    pair_swap_changes if adjacent else None
                )

                u1 = float(util[p1.id])
                u2 = float(util[p2.id])

                rows.append(
                    {
                        "side": side,
                        "agent": agent_id,
                        "partner_1": p1.id,
                        "partner_2": p2.id,
                        "utility_1": u1,
                        "utility_2": u2,
                        "utility_gap": abs(u1 - u2),
                        "rank_1": r1 + 1,
                        "rank_2": r2 + 1,
                        "adjacent_in_oracle": adjacent,
                        "adjacent_reversal_changes_matching": adjacent_changes,
                        "pair_swap_changes_matching": pair_swap_changes,
                        "num_assignment_changes_if_swapped": n_changes,
                        "bootstrap_flip_probability": flip_prob(
                            side, agent_id, p1.id, p2.id
                        ),
                        "bt_true_probability": bt_true_probability(
                            side, agent_id, p1.id, p2.id
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    logger.info(
        "Arm sensitivity computed in %.1fs (%d arms)", time.monotonic() - t0, len(df)
    )
    return df


def _replace_ranking(
    prefs: PreferenceList, agent, new_ranking: List
) -> PreferenceList:
    """PreferenceList with one agent's ranking replaced."""
    new_preferences = dict(prefs.preferences)
    new_preferences[agent] = list(new_ranking)
    return PreferenceList(new_preferences)


# ============================================================================
# Outputs
# ============================================================================

def save_sensitivity_outputs(out_dir: Path, arms: pd.DataFrame) -> None:
    sdir = out_dir / "sensitivity"
    sdir.mkdir(parents=True, exist_ok=True)

    arms.to_csv(sdir / "arm_sensitivity.csv", index=False)
    arms.sort_values(
        "bootstrap_flip_probability", ascending=False
    ).to_csv(sdir / "arm_sensitivity_by_flip.csv", index=False)
    arms.sort_values("utility_gap").to_csv(
        sdir / "arm_sensitivity_by_gap.csv", index=False
    )

    _write_sensitivity_plots(sdir, arms)
    _write_sensitivity_report(sdir, arms)


def _write_sensitivity_plots(sdir: Path, arms: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = sdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # gap vs criticality scatter (pair-swap for all arms; adjacent
    # highlighted)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    noncrit = arms[~arms["pair_swap_changes_matching"]]
    crit = arms[arms["pair_swap_changes_matching"]]
    ax.scatter(
        noncrit["utility_gap"], noncrit["bootstrap_flip_probability"],
        s=18, alpha=0.5, color="tab:blue", label="non-critical (pair-swap)",
    )
    ax.scatter(
        crit["utility_gap"], crit["bootstrap_flip_probability"],
        s=30, alpha=0.8, color="tab:red", marker="X",
        label="changes matching (pair-swap)",
    )
    adj = arms[arms["adjacent_in_oracle"]]
    ax.scatter(
        adj["utility_gap"], adj["bootstrap_flip_probability"],
        facecolors="none", edgecolors="black", s=60, linewidths=0.8,
        label="adjacent in oracle ranking",
    )
    ax.set_xlabel("utility gap")
    ax.set_ylabel("bootstrap flip probability")
    ax.set_title("Preference-arm criticality vs utility gap", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "gap_criticality_scatter.png", dpi=160)
    plt.close(fig)

    # bootstrap flip vs gap (log-x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        arms["utility_gap"].clip(lower=1e-5),
        arms["bootstrap_flip_probability"],
        s=16, alpha=0.5, color="tab:purple",
    )
    ax.set_xscale("log")
    ax.set_xlabel("utility gap (log scale, clipped at 1e-5)")
    ax.set_ylabel("bootstrap flip probability")
    ax.set_title("Bootstrap preference-reversal rate vs utility gap", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(plots / "bootstrap_flip_vs_gap.png", dpi=160)
    plt.close(fig)

    # critical arm gap distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    adj_arms = arms[arms["adjacent_in_oracle"]]
    adj_crit = adj_arms[
        adj_arms["adjacent_reversal_changes_matching"] == True  # noqa: E712
    ]["utility_gap"]
    adj_noncrit = adj_arms[
        adj_arms["adjacent_reversal_changes_matching"] == False  # noqa: E712
    ]["utility_gap"]
    bins = np.linspace(0, max(adj_arms["utility_gap"].max(), 0.05), 30)
    ax.hist(
        adj_noncrit.dropna(), bins=bins, alpha=0.65, color="tab:blue",
        label=f"adjacent, non-critical (n={len(adj_noncrit)})",
    )
    ax.hist(
        adj_crit.dropna(), bins=bins, alpha=0.65, color="tab:red",
        label=f"adjacent, matching-critical (n={len(adj_crit)})",
    )
    ax.set_xlabel("utility gap")
    ax.set_ylabel("number of arms")
    ax.set_title("Adjacent-arm gap distribution by criticality", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "critical_arm_gap_distribution.png", dpi=160)
    plt.close(fig)

    logger.info("Wrote sensitivity plots to %s", plots)


def _write_sensitivity_report(sdir: Path, arms: pd.DataFrame) -> None:
    adjacent = arms[arms["adjacent_in_oracle"]]
    n_adjacent = len(adjacent)
    n_adjacent_critical = int(
        adjacent["adjacent_reversal_changes_matching"].fillna(False).sum()
    )
    n_adjacent_noncritical = n_adjacent - n_adjacent_critical

    crit_gaps = adjacent.loc[
        adjacent["adjacent_reversal_changes_matching"] == True,  # noqa: E712
        "utility_gap",
    ]
    noncrit_gaps = adjacent.loc[
        adjacent["adjacent_reversal_changes_matching"] == False,  # noqa: E712
        "utility_gap",
    ]

    lines: List[str] = []
    lines.append("# Preference-Arm Sensitivity Report")
    lines.append("")
    lines.append(f"* total arms: {len(arms)}")
    lines.append(
        f"* adjacent-in-oracle arms: {n_adjacent} "
        f"(task side {int((adjacent['side'] == 'task').sum())}, "
        f"model side {int((adjacent['side'] == 'model').sum())})"
    )
    lines.append(
        f"* adjacent arms whose minimal reversal changes the stable "
        f"matching: **{n_adjacent_critical}** "
        f"({n_adjacent_critical / max(n_adjacent, 1):.1%} of adjacent)"
    )
    lines.append(f"* adjacent non-critical: {n_adjacent_noncritical}")
    lines.append("")
    lines.append("## Critical vs non-critical adjacent-arm gap statistics")
    lines.append("")
    lines.append("| group | n | min | median | mean | max |")
    lines.append("|---|---|---|---|---|---|")
    for name, series in (
        ("critical", crit_gaps), ("non-critical", noncrit_gaps)
    ):
        if len(series):
            lines.append(
                f"| {name} | {len(series)} | {series.min():.5f} "
                f"| {series.median():.5f} | {series.mean():.5f} "
                f"| {series.max():.5f} |"
            )
        else:
            lines.append(f"| {name} | 0 | - | - | - | - |")
    lines.append("")
    lines.append("## Pair-swap sensitivity (all arms, not minimal for non-adjacent)")
    lines.append("")
    n_swap = int(arms["pair_swap_changes_matching"].sum())
    lines.append(
        f"* arms whose position-swap changes the matching: {n_swap} / "
        f"{len(arms)}"
    )
    lines.append("")
    lines.append("## Critical arms (adjacent, minimal reversal)")
    lines.append("")
    crit_rows = adjacent[
        adjacent["adjacent_reversal_changes_matching"] == True  # noqa: E712
    ].sort_values("utility_gap")
    lines.append(
        "| side | agent | partner_1 | partner_2 | gap | flip prob | "
        "bt_true_prob | rank_1 | rank_2 | changes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in crit_rows.iterrows():
        lines.append(
            f"| {r['side']} | {r['agent']} | {r['partner_1']} "
            f"| {r['partner_2']} | {r['utility_gap']:.5f} "
            f"| {r['bootstrap_flip_probability']:.3f} "
            f"| {r['bt_true_probability']:.3f} | {r['rank_1']} "
            f"| {r['rank_2']} | {r['num_assignment_changes_if_swapped']} |"
        )
    lines.append("")

    (sdir / "sensitivity_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    logger.info("Wrote %s", sdir / "sensitivity_report.md")
