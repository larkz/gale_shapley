"""
experiment.py
Data generation and plotting harness for P2ETG.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gs_lib.gs_tools import (
    Man, Woman, PreferenceList, GaleShapley, StabilityVerifier,
)
from p2etg import P2ETG
from providers import BradleyTerryProvider


RUNS_DIR  = Path("runs")
PLOTS_DIR = Path("plots")
RUNS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Number of post-stop rounds recorded in rounds.csv.
POST_STOP_TAIL = 200


# ============================================================================
# Helpers
# ============================================================================

def random_theta(participants, partners, rng, alpha: float = 0.5):
    K = len(partners)
    return {
        p: {q: float(w) for q, w in zip(partners, rng.dirichlet([alpha] * K))}
        for p in participants
    }


def gs_on_true_preferences(men, women, true_theta_men, true_theta_women):
    prefs = {
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men, key=lambda m: -true_theta_women[w][m]) for w in women},
    }
    return GaleShapley(PreferenceList(prefs)).find_stable_matching("men")


def run_single(
    N: int, K: int, alpha: float, seed: int,
    max_epochs: int, adaptive: bool, check_every: int, max_samples: int,
    constant: float = 0.1,
) -> Tuple[List[Dict], Dict]:
    """Run one experiment. Returns (per-round rows, summary dict)."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men   = [Man(f"m{i}")   for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    true_theta_men   = random_theta(men,   women, np_rng, alpha)
    true_theta_women = random_theta(women, men,   np_rng, alpha)

    h_star = gs_on_true_preferences(men, women, true_theta_men, true_theta_women)

    # Signal provider: BT model with the ground-truth θ.
    # The learner never sees θ — only binary comparison outcomes.
    provider = BradleyTerryProvider(
        true_theta_men, true_theta_women, rng=rng,
    )

    learner = P2ETG(
        men, women,
        provider=provider,
        rng=rng,
        constant=constant,
    )
    result = learner.run_with_trace(
        max_epochs=max_epochs,
        adaptive=adaptive,
        check_every=check_every,
        max_samples=max_samples,
        verbose=False,
    )

    rounds = result.get("rounds", [])
    committed = result["matching"]

    # ------------------------------------------------------------------
    # Build the dense per-round timeline.
    #
    # `rounds` contains one entry per check: (t, matching, disjoint).
    # The matching recorded at time t applies to rounds (prev_t, t],
    # because P2ETG only recomputes the matching at check time.
    # ------------------------------------------------------------------
    rows: List[Dict] = []
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

    # If the algorithm stopped, fill a small tail with the committed matching.
    # If it didn't stop, `prev_t` should already equal `max_samples`; in that
    # case there is nothing to fill.
    if result["stopped"]:
        for tt in range(prev_t + 1, prev_t + POST_STOP_TAIL + 1):
            rows.append({
                "t": tt,
                "matching_str": str(committed),
                "disjoint": True,
                "correct": int(committed == h_star),
            })

    # Cumulative 0/1 regret.
    cumulative = 0
    for r in rows:
        cumulative += (1 - r["correct"])
        r["regret"] = cumulative

    # ------------------------------------------------------------------
    # Stability checks under true and estimated preferences.
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
        "stable_under_truth": int(ok_true),
        "stable_under_hat":   int(ok_hat),
        "reason_truth":       reason_true,
        "reason_hat":         reason_hat,
        "constant": constant,
    }
    return rows, summary


# ============================================================================
# Data generation
# ============================================================================

def generate_data(
    Ns: List[int] = [3, 5],
    Ks: List[int] = [3, 5],
    alphas: List[float] = [0.5],
    seeds: List[int] = list(range(2)),
    max_epochs: int = 20,
    adaptive: bool = True,
    check_every: int = 25,
    max_samples: int = 200_000,
    constant: float = 0.1,
) -> pd.DataFrame:
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
                        constant=constant,
                    )

                    pd.DataFrame(rows).to_csv(run_dir / "rounds.csv", index=False)
                    (run_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2)
                    )
                    (run_dir / "config.json").write_text(
                        json.dumps({
                            "N": N, "K": K, "alpha": alpha, "seed": seed,
                            "adaptive": adaptive,
                            "check_every": check_every,
                            "max_samples": max_samples,
                            "constant": constant,
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
    sns.set_theme(style="whitegrid", context="talk")

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
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))

    df_plot = df_sum[df_sum["stopped"]].copy()
    if len(df_plot) == 0:
        plt.close(fig)
        return

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
    ax.set_ylabel("P($H_t = H_*$)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Probability of correct matching over time")
    ax.legend(title="Configuration", frameon=True, fontsize=10, ncol=2)
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ci_convergence.png", dpi=160)
    plt.close(fig)


def plot_regret_by_N(df_sum: pd.DataFrame):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.boxplot(data=df_sum, x="N", y="final_regret", hue="K", ax=ax,
                palette="viridis", fliersize=0)
    sns.stripplot(data=df_sum, x="N", y="final_regret", hue="K", ax=ax,
                  dodge=True, color="black", size=3, alpha=0.5)

    ax.set_xlabel("$N$ (A-side size)")
    ax.set_ylabel("Final regret")
    ax.set_title("Final regret vs problem size")

    handles, labels = ax.get_legend_handles_labels()
    k_vals = sorted(df_sum["K"].unique())
    ax.legend(handles[:len(k_vals)], labels[:len(k_vals)],
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
        Ns=[10],
        Ks=[10],
        alphas=[2.0],
        seeds=list(range(50)),
        max_epochs=20,
        adaptive=True,
        check_every=25,
        max_samples=100_000,
        constant=0.25,
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