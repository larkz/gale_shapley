#!/usr/bin/env python3
"""export_data_share.py — export our real-LLM-market runs in the
gale_shapley_data_share format (ver4_P2ETG_* runs).

Target format (per run directory, see gale_shapley_data_share/):
  <run_id>_N{N}_K{K}_a{alpha}_seed{seed}/
    config.json    {run_id, baseline, T0, N, K, alpha, seed, adaptive,
                    check_every, max_samples, constant, + source extras}
    bt_params.json {men: {Man: {Woman: theta}}, women: {...}}  thetas
                   NORMALISED per agent to the simplex (sum=1; BT
                   probabilities are ratio-invariant, so this matches
                   the Dirichlet-simplex scale of the synthetic runs)
    rounds.csv     t, matching_str, disjoint, correct, regret
                   (dense per-round timeline; matching recorded at a
                   check applies to (prev_t, t]; post-stop the
                   committed matching fills POST_STOP_TAIL rounds;
                   regret = cumulative 0/1 identification regret)
    summary.json   {N, K, alpha, seed, run_id, baseline, T0, stopped,
                    T_stop, n_epochs, correct_at_stop, oracle_str,
                    committed_str, final_regret, stable_under_truth,
                    stable_under_hat, reason_truth, reason_hat,
                    constant, min_bt_gap, n_stable_matchings}

Source market: the REAL original 8x8 LLMRouterBench market
(comparative preferences, full-data utilities, eta-calibrated BT, no
floor) — the closest real analog of the synthetic N8 K8 alpha=2.0
market. Learner params mirror the ver4 config: adaptive=True,
check_every=500, max_samples=200_000, constant=0.1. T0=100 is
recorded for schema compatibility (our first check fires at t=500).

Usage:
    PYTHONHASHSEED=0 python scripts/export_data_share.py \
        [--seeds 0 1 2 ...] [--out ../gale_shapley_llm_routing/gale_shapley_data_share]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import permutations
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs_lib.gs_tools import StabilityVerifier

from llm_matching.providers import RouterBenchBTProvider
from llm_matching.upstream_rep import build_upstream_context, _pairs

RUN_ID = "llm_P2ETG"
ALPHA = None  # real market: no Dirichlet alpha
POST_STOP_TAIL = 200


# ---------------------------------------------------------------------------
# Extras for the ver4 summary schema
# ---------------------------------------------------------------------------

def simplex_thetas(theta_task, theta_model) -> Dict:
    """bt_params.json with per-agent simplex-normalised thetas."""
    def _norm(table):
        out = {}
        for agent, partners in table.items():
            s = sum(partners.values())
            out[str(agent)] = {str(p): v / s for p, v in partners.items()}
        return out
    return {
        "men": _norm(theta_model),
        "women": _norm(theta_task),
    }


def min_bt_gap(bt_params: Dict) -> float:
    """Minimum adjacent theta gap on the simplex (their min_bt_gap)."""
    gaps = []
    for side in ("men", "women"):
        for _agent, partners in bt_params[side].items():
            vals = sorted(partners.values())
            gaps.extend(b - a for a, b in zip(vals, vals[1:]))
    return min(gaps)


def count_stable_matchings(prefs, men, women) -> int:
    """Number of stable matchings of the TRUE (strict) market.

    Brute force over all N! bijections with an O(N^2) blocking-pair
    check (fine at N=8: 40,320 x 64 lookups).
    """
    rank_m = {m: {w: i for i, w in enumerate(prefs.get_preference(m))}
              for m in men}
    rank_w = {w: {m: i for i, m in enumerate(prefs.get_preference(w))}
              for w in women}

    n = 0
    for perm in permutations(women):  # perm[i] = woman assigned to men[i]
        wife_of = dict(zip(men, perm))
        husband_of = dict(zip(perm, men))
        stable = True
        for m in men:
            wm = wife_of[m]
            rm = rank_m[m][wm]
            for w in women:
                if w != wm and rank_m[m][w] < rm and \
                        rank_w[w][m] < rank_w[w][husband_of[w]]:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            n += 1
    return n


# ---------------------------------------------------------------------------
# One run, ver4 format
# ---------------------------------------------------------------------------

def export_run(ctx, seed: int, out_root: Path, params: Dict) -> Dict:
    from p2etg import P2ETG

    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"datashare::bt::{seed}"),
        probability_floor=float(
            ctx["config"].get("feedback", {}).get("probability_floor", 0.0)
        ),
    )
    learner = P2ETG(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=params["constant"],
    )
    rounds: List = []
    result = learner.run_until_stop(
        adaptive=params["adaptive"], check_every=params["check_every"],
        max_samples=params["max_samples"], verbose=False, rounds=rounds,
    )

    stopped = bool(result["stopped"])
    committed = result["matching"]
    h_star_pairs = _pairs(ctx["h_star"])

    # dense timeline (their exact columns, matching_str included)
    rows = []
    prev_t = 0
    for (t, matching, disjoint) in rounds:
        correct = int(_pairs(matching) == h_star_pairs)
        ms = str(matching)
        for tt in range(prev_t + 1, t + 1):
            rows.append((tt, ms, disjoint, correct))
        prev_t = t
    if stopped:
        correct = int(_pairs(committed) == h_star_pairs)
        ms = str(committed)
        for tt in range(prev_t + 1, prev_t + POST_STOP_TAIL + 1):
            rows.append((tt, ms, True, correct))

    cumulative = 0
    csv_rows = []
    for tt, ms, disjoint, correct in rows:
        cumulative += 1 - correct
        csv_rows.append((tt, ms, disjoint, correct, cumulative))

    run_dir = out_root / f"{RUN_ID}_N{ctx['N']}_K{ctx['K']}_real_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        csv_rows, columns=["t", "matching_str", "disjoint", "correct", "regret"]
    ).to_csv(run_dir / "rounds.csv", index=False)

    bt = simplex_thetas(ctx["theta_task"], ctx["theta_model"])
    (run_dir / "bt_params.json").write_text(json.dumps(bt, indent=2))

    ok_true, reason_true, _ = StabilityVerifier(ctx["prefs"]).is_stable(committed)
    prefs_hat = learner._build_preference_lists()
    ok_hat, reason_hat, _ = StabilityVerifier(prefs_hat).is_stable(committed)

    config = {
        "run_id": RUN_ID, "baseline": "p2etg", "T0": 100,
        "N": ctx["N"], "K": ctx["K"], "alpha": ALPHA, "seed": seed,
        "adaptive": params["adaptive"], "check_every": params["check_every"],
        "max_samples": params["max_samples"], "constant": params["constant"],
        # source extras (additive; loaders keyed on the fields above)
        "market_source": "LLMRouterBench original 8x8 (full data, no split)",
        "feedback": "BT, eta-calibrated (target 0.70), no floor",
        "preferences": "comparative (U task-side / V model-side)",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary = {
        "N": ctx["N"], "K": ctx["K"], "alpha": ALPHA, "seed": seed,
        "run_id": RUN_ID, "baseline": "p2etg", "T0": 100,
        "stopped": stopped, "T_stop": result["T_stop"],
        "n_epochs": len(rounds),
        "correct_at_stop": int(_pairs(committed) == h_star_pairs),
        "oracle_str": str(ctx["h_star"]),
        "committed_str": str(committed),
        "final_regret": csv_rows[-1][4] if csv_rows else None,
        "stable_under_truth": int(ok_true),
        "stable_under_hat": int(ok_hat),
        "reason_truth": reason_true, "reason_hat": reason_hat,
        "constant": params["constant"],
        "min_bt_gap": min_bt_gap(bt),
        "n_stable_matchings": count_stable_matchings(
            ctx["prefs"], ctx["men"], ctx["women"]
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/llm_matching_8x8.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    parser.add_argument(
        "--out",
        default="../gale_shapley_llm_routing/gale_shapley_data_share",
    )
    args = parser.parse_args()

    import yaml

    from llm_matching.runner import DEFAULT_CONFIG, deep_merge

    with open(args.config, "r", encoding="utf-8") as fh:
        config = deep_merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})
    ctx = build_upstream_context(config)

    params = {
        "adaptive": True, "check_every": 500,
        "max_samples": 200_000, "constant": 0.1,
    }
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        s = export_run(ctx, seed, out_root, params)
        print(
            f"[seed {seed}] stopped={s['stopped']} T_stop={s['T_stop']} "
            f"correct_at_stop={s['correct_at_stop']} "
            f"final_regret={s['final_regret']} "
            f"n_stable_matchings={s['n_stable_matchings']}"
        )
    print(f"Exported {len(args.seeds)} runs to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
