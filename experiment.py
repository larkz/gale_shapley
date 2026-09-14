"""
experiment.py
Data generation and plotting harness for P2ETG.

Runs P2ETG across seeds and configurations, records per-round regret,
saves results as CSV + JSON, and produces seaborn visualizations.

Directory layout:
    runs/
        N{N}_K{K}_alpha{alpha}_seed{seed}/
            config.json
            rounds.csv        # t, matching_str, disjoint
            summary.json      # T_stop, stopped, regret_at_T, etc.

    plots/
        regret_curve.png
        tstop_distribution.png
        ci_convergence.png
        regret_by_N.png
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gs_lib.gs_tools import (
    Man, Woman, PreferenceList, GaleShapley, StabilityVerifier, Matching,
)
from p2etg import P2ETG


# ============================================================================
# Directory setup
# ============================================================================

RUNS_DIR  = Path("runs")
PLOTS_DIR = Path("plots")
RUNS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# Helpers (kept local so this file is self-contained)
# ============================================================================

def random_theta(participants, partners, rng, alpha: float = 0.5):
    """Dirichlet-distributed theta vectors, one per participant."""
    K = len(partners)
    return {
        p: {q: float(w) for q, w in zip(partners, rng.dirichlet([alpha] * K))}
        for p in participants
    }


def gs_on_true_preferences(men, women, true_theta_men, true_theta_women):
    prefs = {
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men,   key=lambda m: -true_theta_women[w][m]) for w in women},
    }
    return GaleShapley(PreferenceList(prefs)).find_stable_matching("men")


def run_single(
    N: int, K: int, alpha: float, seed: int,
    max_epochs: int, adaptive: bool, check_every: int, max_samples: int,
) -> Tuple[List[Dict], Dict]:
    """Run one experiment. Returns (per-round rows, summary dict)."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men   = [Man(f"m{i}")   for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    true_theta_men   = random_theta(men,   women, np_rng, alpha)
    true_theta_women = random_theta(women, men,   np_rng, alpha)

    h_star = gs_on_true_preferences(men, women, true_theta_men, true_theta_women)

    learner = P2ETG(
        men, women,
        true_theta_men=true_theta_men,
        true_theta_women=true_theta_women,
        rng=rng,
    )
    result = learner.run_with_trace(
        max_epochs=max_epochs,
        adaptive=adaptive,
        check_every=check_every,
        max_samples=max_samples,
        verbose=False,
    )

    # Reconstruct per-round matching: P2ETG plays H_t = GS(hat_theta_t)
    # throughout exploration, and H_{T_stop} thereafter.
    # Because the trace only records the matching at each check, we
    # interpolate: each recorded matching is played until the next record.
    rounds = result.get("rounds", [])
    rows: List[Dict] = []
    committed = result["matching"]

    # Build a timeline: each (t, matching) applies from previous t+1 to t
    prev_t = 0
    for (t, matching, disjoint) in rounds:
        for tt in range(prev_t + 1, t + 1):
            rows.append({
                "t": tt,
                "matching_str": str(matching),
                "disjoint": disjoint,
                "correct": int(matching == h_star),
            })
        prev_t = t

    # After T_stop, play committed matching forever (within the horizon)
    horizon = max_samples if adaptive else (2 ** max_epochs)
    for tt in range(prev_t + 1, horizon + 1):
        rows.append({
            "t": tt,
            "matching_str": str(committed),
            "disjoint": result["stopped"],
            "correct": int(committed == h_star),
        })

    # Cumulative 0/1 regret: sum over t of (1 - correct_t)
    cumulative = 0
    for r in rows:
        cumulative += (1 - r["correct"])
        r["regret"] = cumulative

    # ------------------------------------------------------------------
    # Stability checks (under truth and under the learner's estimate)
    # ------------------------------------------------------------------
    prefs_true = PreferenceList({
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men,   key=lambda m: -true_theta_women[w][m]) for w in women},
    })
    ok_true, reason_true, _ = StabilityVerifier(prefs_true).is_stable(committed)

    prefs_hat = learner._build_preference_lists()
    ok_hat, reason_hat, _ = StabilityVerifier(prefs_hat).is_stable(committed)

    summary = {
        "N": N, "K": K, "alpha": alpha, "seed": seed,
        "stopped": result["stopped"],
        "T_stop": result["T_stop"],
        "n_epochs": len(result["epochs"]),
        "correct_at_stop": int(committed == h_star),
        "oracle_str": str(h_star),
        "committed_str": str(committed),
        "final_regret": rows[-1]["regret"] if rows else None,
        # NEW:
        "stable_under_truth": int(ok_true),
        "stable_under_hat":   int(ok_hat),
        "reason_truth":       reason_true,
        "reason_hat":         reason_hat,
    }
    return rows, summary

# ============================================================================
# Data generation driver
# ============================================================================

def generate_data(
    Ns: List[int] = [4, 6, 8],
    Ks: List[int] = [4, 6, 8],
    alphas: List[float] = [0.5],
    seeds: List[int] = list(range(3)),
    max_epochs: int = 20,
    adaptive: bool = True,
    check_every: int = 50,
    max_samples: int = 200_000_000,
) -> pd.DataFrame:
    """Run all configurations, save per-run CSVs and summaries, and return
    a long-format DataFrame for plotting."""

    all_rows: List[Dict] = []
    all_summaries: List[Dict] = []

    for N in Ns:
        for K in Ks:
            for alpha in alphas:
                for seed in seeds:
                    tag = f"N{N}_K{K}_a{alpha}_seed{seed}"
                    run_dir = RUNS_DIR / tag
                    run_dir.mkdir(exist_ok=True)

                    rows, summary = run_single(
                        N, K, alpha, seed,
                        max_epochs=max_epochs,
                        adaptive=adaptive,
                        check_every=check_every,
                        max_samples=max_samples,
                    )

                    df = pd.DataFrame(rows)
                    df.to_csv(run_dir / "rounds.csv", index=False)
                    (run_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2)
                    )
                    (run_dir / "config.json").write_text(
                        json.dumps({
                            "N": N, "K": K, "alpha": alpha, "seed": seed,
                            "adaptive": adaptive,
                            "check_every": check_every,
                            "max_samples": max_samples,
                        }, indent=2)
                    )

                    for r in rows:
                        r.update({"N": N, "K": K, "alpha": alpha, "seed": seed})
                    all_rows.extend(rows)
                    all_summaries.append(summary)

                    print(f"  [{tag}] stopped={summary['stopped']} "
                          f"T_stop={summary['T_stop']} "
                          f"correct={summary['correct_at_stop']}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(RUNS_DIR / "all_rounds.csv", index=False)

    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(RUNS_DIR / "all_summaries.csv", index=False)

    return df_all


# ============================================================================
# Plots
# ============================================================================

def plot_regret_with_uncertainty(df_all: pd.DataFrame):
    """Regret curves with 95% band across seeds, one curve per (N,K)."""
    sns.set_theme(style="whitegrid", context="talk")

    # Aggregate regret at each (N, K, t): mean and 95% CI across seeds
    grouped = (df_all
               .groupby(["N", "K", "t"])
               .agg(regret_mean=("regret", "mean"),
                    regret_std=("regret", "std"),
                    n=("regret", "count"))
               .reset_index())
    grouped["regret_se"] = grouped["regret_std"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["lo"] = grouped["regret_mean"] - 1.96 * grouped["regret_se"]
    grouped["hi"] = grouped["regret_mean"] + 1.96 * grouped["regret_se"]

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("viridis", n_colors=grouped["K"].nunique())
    k_to_color = dict(zip(sorted(grouped["K"].unique()), palette))

    for (N, K), sub in grouped.groupby(["N", "K"]):
        sub = sub.sort_values("t")
        color = k_to_color[K]
        ax.plot(sub["t"], sub["regret_mean"],
                label=f"N={N}, K={K}", color=color, lw=2)
        ax.fill_between(sub["t"], sub["lo"], sub["hi"],
                        color=color, alpha=0.25, linewidth=0)

    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cumulative 0/1 regret")
    ax.set_title("Expected regret with 95% band across seeds")
    ax.legend(title="Configuration", frameon=True, fontsize=10, ncol=2)
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regret_curve.png", dpi=160)
    plt.close(fig)


def plot_tstop_distribution(df_sum: pd.DataFrame):
    """Box + strip plot of T_stop across (N, K)."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))

    # Only show configurations where the algorithm actually stopped
    df_plot = df_sum[df_sum["stopped"]].copy()
    # df_plot["config"] = df_plot.apply(lambda r: f"N{int(r.N)}K{int(r.K)}", axis=1)
    df_plot["config"] = (
        "N" + df_plot["N"].astype(int).astype(str)
        + "K" + df_plot["K"].astype(int).astype(str)
    )

    sns.boxplot(data=df_plot, x="config", y="T_stop", ax=ax,
                color="lightsteelblue", width=0.5, fliersize=0)
    sns.stripplot(data=df_plot, x="config", y="T_stop", ax=ax,
                  color="navy", size=4, alpha=0.6, jitter=0.15)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("$T_{stop}$")
    ax.set_title("Stopping time distribution across seeds")
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tstop_distribution.png", dpi=160)
    plt.close(fig)


def plot_ci_convergence(df_all: pd.DataFrame):
    """Mean CI width vs t, one line per (N,K)."""
    # We don't currently log max_wid per round in rounds.csv — add it if you
    # extend run_with_trace to record it. Fallback: use regret as proxy.
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))

    grouped = (df_all
               .groupby(["N", "K", "t"])
               .agg(correct_rate=("correct", "mean"))
               .reset_index())

    for (N, K), sub in grouped.groupby(["N", "K"]):
        sub = sub.sort_values("t")
        ax.plot(sub["t"], sub["correct_rate"],
                label=f"N={N}, K={K}", lw=2)

    ax.set_xlabel("Round $t$")
    ax.set_ylabel("P(H_t = H_*)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Probability of correct matching over time")
    ax.legend(title="Configuration", frameon=True, fontsize=10, ncol=2)
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ci_convergence.png", dpi=160)
    plt.close(fig)


def plot_regret_by_N(df_sum: pd.DataFrame):
    """Final regret as a function of N, with hue by K."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.boxplot(data=df_sum, x="N", y="final_regret", hue="K", ax=ax,
                palette="viridis", fliersize=0)
    sns.stripplot(data=df_sum, x="N", y="final_regret", hue="K", ax=ax,
                  dodge=True, color="black", size=3, alpha=0.5)

    ax.set_xlabel("$N$ (A-side size)")
    ax.set_ylabel("Final regret")
    ax.set_title("Final regret vs problem size")
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:df_sum["K"].nunique()],
              labels[:df_sum["K"].nunique()],
              title="K", frameon=True, fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regret_by_N.png", dpi=160)
    plt.close(fig)


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("Generating data...")
    df_all = generate_data(
        Ns=[3],
        Ks=[3],
        alphas=[0.5],
        seeds=list(range(2)),
        max_epochs=3,
        adaptive=True,
        check_every=25,
        max_samples=200_000,
    )

    print("\nLoading summaries...")
    df_sum = pd.read_csv(RUNS_DIR / "all_summaries.csv")

    print("Producing plots...")
    plot_regret_with_uncertainty(df_all)
    plot_tstop_distribution(df_sum)
    plot_ci_convergence(df_all)
    plot_regret_by_N(df_sum)

    print(f"\nDone. Plots saved to {PLOTS_DIR.resolve()}/")
    print(f"     Data saved to {RUNS_DIR.resolve()}/")


if __name__ == "__main__":
    main()