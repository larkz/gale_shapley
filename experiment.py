"""
experiment.py
Data generation and plotting harness for P2ETG.

`generate_data` takes a `baseline` flag ("p2etg" or "etc_uniform") and a
`T0` value. Everything else in the pipeline is unchanged.
"""

from __future__ import annotations

import itertools
import json
import random
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
from baseline_etc_uniform import ETCUniform


# ============================================================================
# Directory setup
# ============================================================================

RUNS_DIR  = Path("runs")
PLOTS_DIR = Path("plots")
RUNS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

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
        **{w: sorted(men,   key=lambda m: -true_theta_women[w][m]) for w in women},
    }
    return GaleShapley(PreferenceList(prefs)).find_stable_matching("men")


def compute_min_bt_gap(learner) -> float:
    min_gap = float("inf")
    for state in learner.agent_states.values():
        theta = state.theta
        items = state.partners
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                g = abs(theta[items[i]] - theta[items[j]])
                if g < min_gap:
                    min_gap = g
    return float(min_gap) if min_gap != float("inf") else float("nan")


def extract_bt_params(learner, men, women) -> Dict[str, Dict]:
    men_params = {}
    for m in men:
        state = learner.agent_states[m]
        men_params[str(m)] = {str(w): float(state.theta[w]) for w in women}
    women_params = {}
    for w in women:
        state = learner.agent_states[w]
        women_params[str(w)] = {str(m): float(state.theta[m]) for m in men}
    return {"men": men_params, "women": women_params}


def count_stable_matchings(
    men: List[Man], women: List[Woman],
    true_theta_men: Dict, true_theta_women: Dict,
) -> int:
    prefs_true = PreferenceList({
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men,   key=lambda m: -true_theta_women[w][m]) for w in women},
    })
    verifier = StabilityVerifier(prefs_true)
    men_set = set(men)
    women_set = set(women)

    n_men = len(men)
    n_women = len(women)

    count = 0
    if n_men <= n_women:
        for perm in itertools.permutations(women, n_men):
            matching_dict = {m: perm[i] for i, m in enumerate(men)}
            m = Matching.from_dict(matching_dict, men_set, women_set)
            ok, _, _ = verifier.is_stable(m)
            if ok:
                count += 1
    else:
        for perm in itertools.permutations(men, n_women):
            matching_dict = {perm[i]: women[i] for i in range(n_women)}
            m = Matching.from_dict(matching_dict, men_set, women_set)
            ok, _, _ = verifier.is_stable(m)
            if ok:
                count += 1
    return count


# ============================================================================
# run_single — dispatch by baseline
# ============================================================================

def run_single(
    N: int, K: int, alpha: float, seed: int,
    max_epochs: int, adaptive: bool, check_every: int, max_samples: int,
    constant: float = 0.1,
    run_id: str = "",
    baseline: str = "p2etg",
    T0: int = 100,
) -> Tuple[List[Dict], Dict, Dict]:
    """Run one experiment. `baseline` selects the learner."""

    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men   = [Man(f"m{i}")   for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    true_theta_men   = random_theta(men,   women, np_rng, alpha)
    true_theta_women = random_theta(women, men,   np_rng, alpha)

    h_star = gs_on_true_preferences(men, women, true_theta_men, true_theta_women)

    # ---------------- dispatch ----------------
    if baseline == "etc_uniform":
        learner = ETCUniform(
            men, women,
            true_theta_men=true_theta_men,
            true_theta_women=true_theta_women,
            rng=rng,
            T0=T0,
            constant=constant,
        )
    else:
        learner = P2ETG(
            men, women,
            true_theta_men=true_theta_men,
            true_theta_women=true_theta_women,
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

    rounds    = result.get("rounds", [])
    committed = result["matching"]

    # ---------------- dense timeline ----------------
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

    if result["stopped"]:
        for tt in range(prev_t + 1, prev_t + POST_STOP_TAIL + 1):
            rows.append({
                "t": tt,
                "matching_str": str(committed),
                "disjoint": True,
                "correct": int(committed == h_star),
            })

    if not rows:
        rows.append({
            "t": result["T_stop"],
            "matching_str": str(committed),
            "disjoint": result["stopped"],
            "correct": int(committed == h_star),
        })

    cumulative = 0
    for r in rows:
        cumulative += (1 - r["correct"])
        r["regret"] = cumulative

    # ---------------- stability checks ----------------
    prefs_true = PreferenceList({
        **{m: sorted(women, key=lambda w: -true_theta_men[m][w]) for m in men},
        **{w: sorted(men,   key=lambda m: -true_theta_women[w][m]) for w in women},
    })
    ok_true, reason_true, _ = StabilityVerifier(prefs_true).is_stable(committed)

    prefs_hat = learner._build_preference_lists()
    ok_hat, reason_hat, _ = StabilityVerifier(prefs_hat).is_stable(committed)

    # ---------------- diagnostics ----------------
    min_bt_gap = compute_min_bt_gap(learner)
    bt_params  = extract_bt_params(learner, men, women)
    n_stable   = count_stable_matchings(
        men, women, true_theta_men, true_theta_women
    )

    summary = {
        "N": N, "K": K, "alpha": alpha, "seed": seed,
        "run_id": run_id,
        "baseline": baseline,
        "T0": T0,
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
        "min_bt_gap": float(min_bt_gap),
        "n_stable_matchings": int(n_stable),
    }
    return rows, summary, bt_params


# ============================================================================
# Data generation — one new flag: baseline
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
    run_id: str = "",
    baseline: str = "p2etg",     # <-- NEW: "p2etg" | "etc_uniform"
    T0: int = 100,               # <-- NEW: only used by etc_uniform
) -> pd.DataFrame:

    all_rows: List[Dict] = []
    all_summaries: List[Dict] = []

    for N in Ns:
        for K in Ks:
            for alpha in alphas:
                for seed in seeds:
                    prefix = f"{run_id}_" if run_id else ""
                    tag = f"{prefix}N{N}_K{K}_a{alpha}_seed{seed}"
                    run_dir = RUNS_DIR / tag
                    run_dir.mkdir(exist_ok=True)

                    rows, summary, bt_params = run_single(
                        N, K, alpha, seed,
                        max_epochs=max_epochs,
                        adaptive=adaptive,
                        check_every=check_every,
                        max_samples=max_samples,
                        constant=constant,
                        run_id=run_id,
                        baseline=baseline,
                        T0=T0,
                    )

                    pd.DataFrame(rows).to_csv(run_dir / "rounds.csv", index=False)
                    (run_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2)
                    )
                    (run_dir / "bt_params.json").write_text(
                        json.dumps(bt_params, indent=2)
                    )
                    (run_dir / "config.json").write_text(
                        json.dumps({
                            "run_id": run_id,
                            "baseline": baseline,
                            "T0": T0,
                            "N": N, "K": K, "alpha": alpha, "seed": seed,
                            "adaptive": adaptive,
                            "check_every": check_every,
                            "max_samples": max_samples,
                            "constant": constant,
                        }, indent=2)
                    )

                    for r in rows:
                        r.update({
                            "N": N, "K": K, "alpha": alpha, "seed": seed,
                            "run_id": run_id, "baseline": baseline, "T0": T0,
                        })
                    all_rows.extend(rows)
                    all_summaries.append(summary)

                    print(f"  [{tag}] baseline={baseline} "
                          f"stopped={summary['stopped']} "
                          f"T_stop={summary['T_stop']} "
                          f"correct={summary['correct_at_stop']} "
                          f"min_gap={summary['min_bt_gap']:.4f} "
                          f"n_stable={summary['n_stable_matchings']}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(RUNS_DIR / "all_rounds.csv", index=False)

    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(RUNS_DIR / "all_summaries.csv", index=False)

    return df_all


# ============================================================================
# Plots (unchanged)
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
    df_plot = df_sum[df_sum["stopped"]].copy()
    if len(df_plot) == 0:
        return
    df_plot["config"] = (
        "N" + df_plot["N"].astype(int).astype(str)
        + "K" + df_plot["K"].astype(int).astype(str)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
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
# Entry point — unchanged from before
# ============================================================================

def main():
    print("Generating data...")
    df_all = generate_data(
        Ns=[8],
        Ks=[8],
        alphas=[2.0],
        seeds=list(range(10)),
        max_epochs=20,
        adaptive=True,
        check_every=250,
        max_samples=20_000,
        constant=0.01,
        run_id="001_baseline",
        baseline="etc_uniform",     # default
        T0=3000,
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