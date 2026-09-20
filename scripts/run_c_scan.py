#!/usr/bin/env python3
"""run_c_scan.py

CI-constant soundness-delay tradeoff scan for certified stopping.

Runs Matching-ID (the exact certificate) on the 4x4 smoke market under
BT feedback for c in {0.1, 0.2, 0.5, 1.0}, 20 seeds each, budget
100,000 (10x the original 10K so larger c has room to certify).

Controlled design: the RNG seeds (learner + BT provider) are identical
across c values, so every run observes the SAME sample stream; c only
changes the CI half-width sqrt(c ln t / n) -> which edges are certified
-> when (and whether) certification fires. Differences across c are
therefore attributable to the certificate strictness alone.

Outputs under outputs/llm_matching_smoke/c_scan/:
  scan_per_seed.csv      combined per-seed summaries (one row per c x seed)
  scan_summary.csv       per-c aggregate
  t_stop_vs_c.png        delay-vs-soundness figure
  scan_report.md         findings

Run: PYTHONHASHSEED=0 python scripts/run_c_scan.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_matching.runner import DEFAULT_CONFIG, deep_merge, run_experiment

C_VALUES = [0.1, 0.2, 0.5, 1.0]
N_SEEDS = 20
BUDGET = 100_000
BASE_CONFIG = "configs/llm_matching_smoke.yaml"
OUT_ROOT = Path("outputs/llm_matching_smoke/c_scan")


def main() -> int:
    with open(BASE_CONFIG, "r", encoding="utf-8") as fh:
        base = deep_merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})

    rows = []
    for c in C_VALUES:
        config = copy.deepcopy(base)
        config["p2etg"]["max_samples"] = BUDGET
        config["p2etg"]["constant"] = c
        config.setdefault("matching_id", {})
        config["matching_id"]["timeout_seconds"] = 1.0
        out_dir = OUT_ROOT / f"c_{c}"
        config["output_dir"] = str(out_dir)

        run_experiment(
            config,
            algorithm="matching_id",
            feedback="bt",
            seeds_override=list(range(N_SEEDS)),
            make_plots=False,
            make_report=False,
        )
        s = pd.read_csv(out_dir / "per_seed_summary.csv")
        s["c"] = c
        rows.append(s)
        stopped = pd.to_numeric(s["stopped"])
        exact = pd.to_numeric(s["exact_oracle_match"])
        print(
            f"[c={c}] stopped {int(stopped.sum())}/{N_SEEDS}, "
            f"cert-correct {(stopped & (exact == 1)).sum()}, "
            f"false-cert {(stopped & (exact == 0)).sum()}"
        )

    combined = pd.concat(rows, ignore_index=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_ROOT / "scan_per_seed.csv", index=False)

    # ---- aggregate ----
    summary_rows = []
    for c in C_VALUES:
        s = combined[combined["c"] == c]
        num = lambda col: pd.to_numeric(s[col], errors="coerce")
        stopped = pd.to_numeric(s["stopped"]) == 1
        exact = pd.to_numeric(s["exact_oracle_match"])
        ts = pd.to_numeric(s.loc[stopped, "T_stop"], errors="coerce")
        # per-pair CI failure probability implied at t=10K: 2 t^(-2c)
        delta_10k = 2 * (10_000 ** (-2 * c))
        summary_rows.append(
            {
                "c": c,
                "stopped": int(stopped.sum()),
                "cert_correct": int((stopped & (exact == 1)).sum()),
                "false_cert": int((stopped & (exact == 0)).sum()),
                "T_stop_mean": float(ts.mean()) if stopped.any() else float("nan"),
                "T_stop_median": float(ts.median()) if stopped.any() else float("nan"),
                "T_stop_min": float(ts.min()) if stopped.any() else float("nan"),
                "T_stop_max": float(ts.max()) if stopped.any() else float("nan"),
                "stopped_before_10k": int(
                    (pd.to_numeric(s["T_stop"], errors="coerce") <= 10_000).sum()
                ),
                "resolved_at_stop": float(num("resolved_fraction_at_stop").mean()),
                "final_exact": int(exact.sum()),
                "final_welfare": float(num("test_welfare").mean()),
                "pair_CI_fail_prob@10K": delta_10k,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "scan_summary.csv", index=False)

    _plot(summary)
    _report(summary)
    print(summary.to_string(index=False))
    return 0


def _plot(summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    per_seed = pd.read_csv(OUT_ROOT / "scan_per_seed.csv")

    # left: T_stop scatter per c (stopped seeds only)
    for c in C_VALUES:
        s = per_seed[(per_seed["c"] == c) & (pd.to_numeric(per_seed["stopped"]) == 1)]
        ts = pd.to_numeric(s["T_stop"], errors="coerce")
        exact = pd.to_numeric(s["exact_oracle_match"])
        ax1.scatter(
            [c] * len(ts), ts, s=42,
            c=["tab:green" if e == 1 else "tab:red" for e in exact],
            marker="o" if (exact == 1).all() else "o",
            alpha=0.75, edgecolors="black", linewidths=0.5,
        )
    ax1.set_xlabel("CI constant c (width sqrt(c ln t / n))")
    ax1.set_ylabel("certified T_stop")
    ax1.set_yscale("log")
    ax1.set_title(
        "Delay: certification time vs c (green=correct, red=false cert)",
        fontsize=10,
    )
    ax1.axhline(10_000, linestyle="--", color="gray", linewidth=1, alpha=0.8)
    ax1.text(0.1, 11_000, "original 10K budget", fontsize=8, color="gray")
    ax1.grid(alpha=0.25)

    # right: stop rate + false-cert vs c
    ax2.plot(summary["c"], summary["stopped"], "o-", color="tab:blue",
             label="certified stops /20")
    ax2.plot(summary["c"], summary["false_cert"], "s-", color="tab:red",
             label="false certifications")
    ax2.plot(summary["c"], summary["cert_correct"], "^-", color="tab:green",
             label="correct certifications")
    ax2.set_xlabel("CI constant c")
    ax2.set_ylabel("count (of 20 seeds)")
    ax2.set_title("Soundness vs delay", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_ROOT / "t_stop_vs_c.png", dpi=160)
    plt.close(fig)


def _report(summary: pd.DataFrame) -> None:
    def md_table(df):
        header = "| " + " | ".join(str(col) for col in df.columns) + " |"
        sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
        lines = [header, sep]
        for _, r in df.iterrows():
            cells = []
            for v in r:
                if isinstance(v, float):
                    cells.append("NaN" if pd.isna(v) else f"{v:.4f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# CI-Constant Soundness-Delay Scan (Matching-ID, 4x4 BT, 20 seeds)",
        "",
        f"Budget {BUDGET:,} (10x the original), PYTHONHASHSEED=0, identical",
        "sample streams across c (RNG seeds fixed per seed index).",
        "",
        md_table(summary),
        "",
        "Reading: larger c widens the CI sqrt(c ln t / n) — resolving an",
        "arm with |p-1/2| = g requires n ~ c ln t / g^2, so certification",
        "delay scales ~ linearly in c while the per-pair CI failure",
        "probability 2 t^(-2c) collapses exponentially.",
        "",
        "Outputs: scan_per_seed.csv, scan_summary.csv, t_stop_vs_c.png",
    ]
    (OUT_ROOT / "scan_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
