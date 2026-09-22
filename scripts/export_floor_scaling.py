#!/usr/bin/env python3
"""export_floor_scaling.py — the LEARNABLE-config scaling series in the
gale_shapley_data_share format.

Series: `floor15_P2ETG_N{n}_K{n}_real_seed{s}` and
`floor15_PrefLID_N{n}_K{n}_real_seed{s}` for n in {3, 5, 10}.

This is the 10x10 learnability scan's winning configuration
(learn10x10_D) extended to all three sizes — the settings under which
the REAL markets are fully learnable within a 200K horizon:

  market:  greedy distinct-winner scaling markets, symmetric (mirror)
           preferences -> unique stable matching H*, full-data
           utilities, no split
  reward:  BT, eta-calibrated at target 0.90 (sharpened), with the
           no-tie guarantee probability_floor = 0.15
  P2ETG:   adaptive, check_every = one pass over all arms
           (2*N*C(N,2)), max_samples = 200K, constant = 0.1
  PrefLID: round_robin centers, budget = 20000, constant = 0.1,
           min_samples_per_pair = 10, min_sample_ratio = 0.5,
           max_iterations = 200K // C(N,2)
  seeds:   0-9 for both algorithms

Output: the exact data-share 4-file format per run
(config.json / bt_params.json / rounds.csv / summary.json); P2ETG uses
the ver6 summary schema, PrefLID the ver11 schema (without the
tuning-branch gt_* export fields).

Run:  PYTHONHASHSEED=0 python scripts/export_floor_scaling.py \
          --sizes 3 5 10
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
from scripts.run_ver_scaling import _dense_rows

FLOOR = 0.15
TARGET = 0.90
CONSTANT = 0.1
BUDGET = 20000
MAX_SAMPLES = 200_000
POST_STOP_TAIL = 200


def _market_config(n: int) -> Dict:
    import yaml

    from llm_matching.runner import DEFAULT_CONFIG, deep_merge

    with open(f"configs/bandit_scaling_{n}x{n}.yaml") as fh:
        config = deep_merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})
    config["preferences"] = {"mode": "symmetric"}  # mirror -> unique H*
    config["feedback"]["target_median_win_probability"] = TARGET
    config["feedback"]["probability_floor"] = FLOOR
    return config


def _run_p2etg(ctx, seed: int, out_root: Path) -> Dict:
    from p2etg import P2ETG

    n = ctx["N"]
    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"floor15::bt-p2etg::{seed}"),
        probability_floor=FLOOR,
    )
    learner = P2ETG(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=CONSTANT,
    )
    records: List = []

    class _Rec(list):
        def append(self, item):
            t, matching, disjoint = item
            records.append((t, matching))

    check_every = 2 * n * (n * (n - 1) // 2)  # one pass over all arms
    result = learner.run_until_stop(
        adaptive=True, check_every=check_every, max_samples=MAX_SAMPLES,
        verbose=False, rounds=_Rec(),
    )
    stopped = bool(result["stopped"])
    committed = result["matching"]
    h_star_pairs = _pairs_of(ctx)
    rows = _dense_rows(records, committed, stopped, h_star_pairs,
                       result["T_stop"] + (POST_STOP_TAIL if stopped else 0))

    ok_true, reason_true, _ = StabilityVerifier(ctx["prefs"]).is_stable(committed)
    prefs_hat = learner._build_preference_lists()
    ok_hat, reason_hat, _ = StabilityVerifier(prefs_hat).is_stable(committed)

    bt = simplex_thetas(ctx["theta_task"], ctx["theta_model"])
    run_dir = out_root / f"floor15_P2ETG_N{n}_K{n}_real_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "bt_params.json").write_text(json.dumps(bt, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "floor15_P2ETG", "baseline": "p2etg", "T0": 100,
        "N": n, "K": n, "alpha": None, "seed": seed,
        "adaptive": True, "check_every": check_every,
        "max_samples": MAX_SAMPLES, "constant": CONSTANT,
        "market_source": f"LLMRouterBench greedy distinct-winner "
                         f"scaling market {n}x{n} (full data, no split)",
        "preferences": "symmetric (mirror) — unique stable matching",
        "feedback": f"BT, eta-calibrated (target {TARGET}), "
                    f"probability_floor {FLOOR}",
    }, indent=2))
    summary = {
        "N": n, "K": n, "alpha": None, "seed": seed,
        "run_id": "floor15_P2ETG", "baseline": "p2etg", "T0": 100,
        "stopped": stopped, "T_stop": result["T_stop"],
        "n_epochs": len(records),
        "correct_at_stop": int(_pairs(committed) == h_star_pairs),
        "oracle_str": str(ctx["h_star"]),
        "committed_str": str(committed),
        "final_regret": rows[-1][4] if rows else None,
        "stable_under_truth": int(ok_true),
        "stable_under_hat": int(ok_hat),
        "reason_truth": reason_true, "reason_hat": reason_hat,
        "constant": CONSTANT,
        "min_bt_gap": min_bt_gap(bt),
        "n_stable_matchings": (count_stable_matchings(
            ctx["prefs"], ctx["men"], ctx["women"]) if n <= 5 else 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def _run_preflid(ctx, seed: int, out_root: Path,
                 preflid_constant: float = CONSTANT) -> Dict:
    from preflid import PrefLID

    n = ctx["N"]
    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"floor15::bt-preflid::{seed}"),
        probability_floor=FLOOR,
    )
    learner = PrefLID(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=preflid_constant, budget=BUDGET,
        center_policy="round_robin", min_samples_per_pair=10,
        min_sample_ratio=0.5,
    )
    records: List = []
    original_rrt = learner.rrt_round

    def rrt_round_with_record(center):
        n_samples = original_rrt(center)
        learner._refresh_estimates()
        records.append((learner.t, learner._current_gs_matching()))
        return n_samples

    learner.rrt_round = rrt_round_with_record
    max_iter = MAX_SAMPLES // (n * (n - 1) // 2)
    pl = learner.run_until_stop(max_iterations=max_iter, verbose=False)
    pl_stopped = bool(pl["stopped"])
    pl_committed = pl["matching"]
    pl_t_stop = int(pl["T_stop"])

    h_star_pairs = _pairs_of(ctx)
    t_end = MAX_SAMPLES if MAX_SAMPLES > pl_t_stop else pl_t_stop
    rows = _dense_rows(records, pl_committed, pl_stopped, h_star_pairs, t_end)

    run_dir = out_root / f"floor15_PrefLID_N{n}_K{n}_real_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "floor15_PrefLID", "N": n, "K": n, "alpha": None,
        "seed": seed, "budget": BUDGET, "constant": preflid_constant,
        "center_policy": "round_robin", "max_iterations": max_iter,
        "horizon": MAX_SAMPLES,
        "market_source": f"LLMRouterBench greedy distinct-winner "
                         f"scaling market {n}x{n} (full data, no split)",
        "preferences": "symmetric (mirror) — unique stable matching",
        "feedback": f"BT, eta-calibrated (target {TARGET}), "
                    f"probability_floor {FLOOR}",
    }, indent=2))
    summary = {
        "N": n, "K": n, "alpha": None, "seed": seed,
        "budget": BUDGET, "constant": preflid_constant,
        "horizon": MAX_SAMPLES,
        "preflid_stopped": pl_stopped,
        "preflid_T_stop": pl_t_stop,
        "preflid_iterations": pl.get("iterations"),
        "preflid_islands_final": pl.get("n_islands"),
        "preflid_correct": int(_pairs(pl_committed) == h_star_pairs),
        "preflid_stable_truth": int(
            StabilityVerifier(ctx["prefs"]).is_stable(pl_committed)[0]
        ),
        "final_regret": rows[-1][4] if rows else None,
        "market_source": f"LLMRouterBench real {n}x{n} scaling market "
                         f"(mirror, floor {FLOOR}, target {TARGET})",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def _pairs_of(ctx) -> frozenset:
    return _pairs(ctx["h_star"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[3, 5, 10])
    parser.add_argument("--which", choices=["p2etg", "preflid", "both"],
                        default="both")
    parser.add_argument("--preflid-constant", type=float, default=CONSTANT,
                        help="CI constant for the PrefLID series "
                             "(0.1 default; 0.3 needed for all-correct "
                             "certification at 5x5)")
    parser.add_argument("--out",
                        default="../gale_shapley_llm_routing/"
                                "gale_shapley_data_share")
    args = parser.parse_args()

    from llm_matching.upstream_rep import build_upstream_context

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for n in args.sizes:
        ctx = build_upstream_context(_market_config(n))
        if args.which in ("p2etg", "both"):
            for seed in range(10):
                s = _run_p2etg(ctx, seed, out_root)
                print(f"[{n}x{n} floor15 P2ETG s{seed}] "
                      f"stopped={s['stopped']} T_stop={s['T_stop']} "
                      f"correct={s['correct_at_stop']} "
                      f"regret={s['final_regret']}")
        if args.which in ("preflid", "both"):
            for seed in range(10):
                s = _run_preflid(ctx, seed, out_root,
                                 preflid_constant=args.preflid_constant)
                print(f"[{n}x{n} floor15 PrefLID s{seed}] "
                      f"stopped={s['preflid_stopped']} "
                      f"T_stop={s['preflid_T_stop']} "
                      f"correct={s['preflid_correct']} "
                      f"regret={s['final_regret']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
