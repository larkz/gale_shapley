#!/usr/bin/env python3
"""run_ver_scaling.py — ver6/ver11 parameters on REAL LLM markets (3/5/10).

Faithful port of the larkin/preflid_tuning experiment settings to the
real LLMRouterBench scaling markets:

  ver6_P2ETG:  Ns=[3,5,10], alphas=2.0 -> REAL market, seeds 0..4,
               max_epochs=200, adaptive=True, check_every=10,
               max_samples=200_000, constant=0.01
  ver11_PrefLID: seeds 0..24, budgets=[3100], constant=0.01,
               max_iterations=10_000, horizon=200_000, plus the ver11
               embedded P2ETG baseline (check_every=25,
               max_samples=2000).

Market construction: the greedy distinct-winner scaling markets
(configs/bandit_scaling_{n}x{n}.yaml datasets/models), comparative
preferences (U task-side / V model-side), FULL-data utilities, no
split, eta-calibrated BT (target 0.70), NO floor — the closest real
analog of the synthetic independent-theta markets. Output: the exact
data-share format (config/bt_params/rounds.csv/summary.json per run),
ver6 + ver11 schemas.

Deviations (documented): no max_epochs cap in our P2ETG port (with
constant=0.01 arms resolve ~10x earlier, so runs stop long before any
epoch cap would bind); n_stable_matchings brute-forced at 3x3/5x5,
skipped (null) at 10x10.

Run from the gale_shapley working repo:
  PYTHONHASHSEED=0 python scripts/run_ver_scaling.py --sizes 3 5 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs_lib.gs_tools import StabilityVerifier

from llm_matching.providers import RouterBenchBTProvider
from llm_matching.upstream_rep import _pairs, build_upstream_context
from scripts.export_data_share import (
    count_stable_matchings,
    min_bt_gap,
    simplex_thetas,
)

VER6 = dict(adaptive=True, check_every=10, max_samples=200_000, constant=0.01)
VER11 = dict(budget=3100, constant=0.01, max_iterations=10_000, horizon=200_000)
P2_BASELINE = dict(check_every=25, max_samples=2000)  # ver11 embedded
POST_STOP_TAIL = 200


def _dense_rows(records, committed, stopped, h_star_pairs, t_end):
    """(t, matching_str, disjoint, correct) dense rows to t_end."""
    rows = []
    prev_t = 0
    for (t, matching) in records:
        if t > t_end:
            break
        correct = int(_pairs(matching) == h_star_pairs)
        ms = str(matching)
        for tt in range(prev_t + 1, t + 1):
            rows.append((tt, ms, True, correct))
        prev_t = t
    if prev_t < t_end:
        if stopped and committed is not None:
            correct = int(_pairs(committed) == h_star_pairs)
            ms = str(committed)
        else:
            correct = rows[-1][3] if rows else 0
            ms = rows[-1][1] if rows else "None"
        for tt in range(prev_t + 1, t_end + 1):
            rows.append((tt, ms, True, correct))
    cumulative = 0
    out = []
    for tt, ms, dj, c in rows:
        cumulative += 1 - c
        out.append((tt, ms, dj, c, cumulative))
    return out


def run_ver6(ctx, seed, out_root) -> Dict:
    """P2ETG, ver6 schema."""
    from p2etg import P2ETG

    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"ver6::bt::{seed}"),
    )
    learner = P2ETG(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=VER6["constant"],
    )
    records: List = []

    class _Rec(list):
        def append(self, item):
            t, matching, disjoint = item
            records.append((t, matching))

    result = learner.run_until_stop(
        adaptive=VER6["adaptive"], check_every=VER6["check_every"],
        max_samples=VER6["max_samples"], verbose=False, rounds=_Rec(),
    )
    stopped = bool(result["stopped"])
    committed = result["matching"]
    h_star_pairs = _pairs(ctx["h_star"])
    rows = _dense_rows(records, committed, stopped, h_star_pairs,
                       result["T_stop"] + (POST_STOP_TAIL if stopped else 0))

    ok_true, reason_true, _ = StabilityVerifier(ctx["prefs"]).is_stable(committed)
    prefs_hat = learner._build_preference_lists()
    ok_hat, reason_hat, _ = StabilityVerifier(prefs_hat).is_stable(committed)

    n = ctx["N"]
    bt = simplex_thetas(ctx["theta_task"], ctx["theta_model"])
    run_dir = out_root / f"ver6_P2ETG_N{n}_K{n}_real_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "bt_params.json").write_text(json.dumps(bt, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "ver6_P2ETG", "baseline": "p2etg", "T0": 100,
        "N": n, "K": n, "alpha": None, "seed": seed, **VER6,
        "market_source": "LLMRouterBench greedy distinct-winner "
                         f"scaling market {n}x{n} (full data, no split)",
        "feedback": "BT, eta-calibrated (target 0.70), no floor",
        "preferences": "comparative (U task-side / V model-side)",
    }, indent=2))
    summary = {
        "N": n, "K": n, "alpha": None, "seed": seed,
        "run_id": "ver6_P2ETG", "baseline": "p2etg", "T0": 100,
        "stopped": stopped, "T_stop": result["T_stop"],
        "n_epochs": len(records),
        "correct_at_stop": int(_pairs(committed) == h_star_pairs),
        "oracle_str": str(ctx["h_star"]),
        "committed_str": str(committed),
        "final_regret": rows[-1][4] if rows else None,
        "stable_under_truth": int(ok_true),
        "stable_under_hat": int(ok_hat),
        "reason_truth": reason_true, "reason_hat": reason_hat,
        "constant": VER6["constant"],
        "min_bt_gap": min_bt_gap(bt),
        "n_stable_matchings": (count_stable_matchings(
            ctx["prefs"], ctx["men"], ctx["women"]) if n <= 5 else None),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def run_ver11(ctx, seed, out_root) -> Dict:
    """PrefLID + embedded P2ETG baseline, ver11 schema."""
    from p2etg import P2ETG
    from preflid import PrefLID

    # ---------------- PrefLID ----------------
    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"ver11::bt-preflid::{seed}"),
    )
    learner = PrefLID(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=VER11["constant"],
        budget=VER11["budget"],
    )
    records: List = []
    original_rrt = learner.rrt_round

    def rrt_round_with_record(center):
        n_samples = original_rrt(center)
        learner._refresh_estimates()
        records.append((learner.t, learner._current_gs_matching()))
        return n_samples

    learner.rrt_round = rrt_round_with_record
    pl = learner.run_until_stop(max_iterations=VER11["max_iterations"],
                               verbose=False)
    pl_stopped = bool(pl["stopped"])
    pl_committed = pl["matching"]
    pl_t_stop = int(pl["T_stop"])

    # ---------------- embedded P2ETG baseline (their design) ----------
    provider2 = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"ver11::bt-p2::{seed}"),
    )
    p2 = P2ETG(men=ctx["men"], women=ctx["women"], provider=provider2,
               rng=random.Random(seed), constant=VER11["constant"])
    p2_result = p2.run_until_stop(
        adaptive=True, check_every=P2_BASELINE["check_every"],
        max_samples=P2_BASELINE["max_samples"], verbose=False,
    )

    h_star_pairs = _pairs(ctx["h_star"])
    horizon = VER11["horizon"]
    t_end = horizon if horizon > pl_t_stop else pl_t_stop
    rows = _dense_rows(records, pl_committed, pl_stopped, h_star_pairs, t_end)

    n = ctx["N"]
    run_dir = out_root / f"ver11_PrefLID_N{n}_K{n}_real_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "bt_params.json").write_text(
        json.dumps(simplex_thetas(ctx["theta_task"], ctx["theta_model"]),
                   indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "ver11_PrefLID", "N": n, "K": n, "alpha": None,
        "seed": seed, **VER11,
        "p2etg_baseline": P2_BASELINE,
        "market_source": "LLMRouterBench greedy distinct-winner "
                         f"scaling market {n}x{n} (full data, no split)",
        "feedback": "BT, eta-calibrated (target 0.70), no floor",
        "preferences": "comparative (U task-side / V model-side)",
    }, indent=2))
    summary = {
        "N": n, "K": n, "alpha": None, "seed": seed,
        "budget": VER11["budget"], "constant": VER11["constant"],
        "horizon": horizon,
        "preflid_stopped": pl_stopped,
        "preflid_T_stop": pl_t_stop,
        "preflid_iterations": pl.get("iterations"),
        "preflid_islands_final": pl.get("n_islands"),
        "preflid_correct": int(_pairs(pl_committed) == h_star_pairs),
        "preflid_stable_truth": int(
            StabilityVerifier(ctx["prefs"]).is_stable(pl_committed)[0]
        ),
        "p2etg_stopped": bool(p2_result["stopped"]),
        "p2etg_T_stop": int(p2_result["T_stop"]),
        "p2etg_correct": int(
            _pairs(p2_result["matching"]) == h_star_pairs
        ),
        "T_stop_ratio": (
            pl_t_stop / p2_result["T_stop"]
            if p2_result["T_stop"] else None
        ),
        "final_regret": rows[-1][4] if rows else None,
        "market_source": f"LLMRouterBench real {n}x{n} scaling market",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[3, 5, 10])
    parser.add_argument("--which", choices=["ver6", "ver11", "both"],
                        default="both")
    parser.add_argument("--out",
                        default="../gale_shapley_llm_routing/"
                                "gale_shapley_data_share")
    args = parser.parse_args()

    import yaml

    from llm_matching.runner import DEFAULT_CONFIG, deep_merge

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for n in args.sizes:
        with open(f"configs/bandit_scaling_{n}x{n}.yaml") as fh:
            config = deep_merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})
        # real-market analog of the synthetic independent-theta market:
        # comparative preferences, no floor, full data
        config["preferences"] = {"mode": "comparative"}
        config["feedback"].pop("probability_floor", None)
        ctx = build_upstream_context(config)

        if args.which in ("ver6", "both"):
            for seed in range(5):
                s = run_ver6(ctx, seed, out_root)
                print(f"[{n}x{n} ver6 s{seed}] stopped={s['stopped']} "
                      f"T_stop={s['T_stop']} correct={s['correct_at_stop']} "
                      f"regret={s['final_regret']}")
        if args.which in ("ver11", "both"):
            for seed in range(25):
                s = run_ver11(ctx, seed, out_root)
                print(f"[{n}x{n} ver11 s{seed}] pl_stopped="
                      f"{s['preflid_stopped']} pl_T={s['preflid_T_stop']} "
                      f"pl_correct={s['preflid_correct']} "
                      f"p2_T={s['p2etg_T_stop']} "
                      f"p2_correct={s['p2etg_correct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
