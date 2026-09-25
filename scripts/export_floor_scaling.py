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
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs_lib.gs_tools import Man, StabilityVerifier

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


def _market_config(n: int, config_path: Optional[str] = None) -> Dict:
    """n=3/5/8/10 -> the square series; config_path -> any market (rect)."""
    import yaml

    from llm_matching.runner import DEFAULT_CONFIG, deep_merge

    if config_path is not None:
        base = config_path
    elif n == 8:
        # the diverse korbench 8x8 market (the flagship real market)
        base = "configs/llm_matching_8x8_diverse_korbench_symmetric.yaml"
    else:
        base = f"configs/bandit_scaling_{n}x{n}.yaml"
    with open(base) as fh:
        config = deep_merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})
    config["preferences"] = {"mode": "symmetric"}  # mirror -> unique H*
    config["feedback"]["target_median_win_probability"] = TARGET
    if config_path is None:
        config["feedback"]["probability_floor"] = FLOOR
    # (rect configs carry their own floor — 0.15 or 0.20)
    return config


def _run_p2etg(ctx, seed: int, out_root: Path,
               market_name: Optional[str] = None,
               horizon: int = MAX_SAMPLES) -> Dict:
    from p2etg import P2ETG

    n_m, n_w = len(ctx["men"]), len(ctx["women"])
    floor = ctx["config"]["feedback"].get("probability_floor", FLOOR)
    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"floor15::bt-p2etg::{seed}"),
        probability_floor=floor,
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

    check_every = (n_m * (n_m - 1) // 2) + (n_w * (n_w - 1) // 2)  # one pass
    result = learner.run_until_stop(
        adaptive=True, check_every=check_every, max_samples=horizon,
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
    label = market_name or f"floor15_P2ETG_N{n_w}_K{n_m}_real"
    run_dir = out_root / f"{label}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "bt_params.json").write_text(json.dumps(bt, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": label.rsplit("_seed", 1)[0], "baseline": "p2etg",
        "T0": 100, "N": n_w, "K": n_m, "alpha": None, "seed": seed,
        "adaptive": True, "check_every": check_every,
        "max_samples": horizon, "constant": CONSTANT,
        "market_source": f"LLMRouterBench real market {n_w}x{n_m} "
                         f"(full data, no split)",
        "preferences": "symmetric (mirror) — unique stable matching",
        "feedback": f"BT, eta-calibrated (target {TARGET}), "
                    f"probability_floor {floor}",
    }, indent=2))
    summary = {
        "N": n_w, "K": n_m, "alpha": None, "seed": seed,
        "run_id": label.rsplit("_seed", 1)[0], "baseline": "p2etg",
        "T0": 100,
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
            ctx["prefs"], ctx["men"], ctx["women"])
            if n_w == n_m and n_w <= 5 else None),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def _run_preflid(ctx, seed: int, out_root: Path,
                 preflid_constant: float = CONSTANT,
                 market_name: Optional[str] = None,
                 horizon: int = MAX_SAMPLES,
                 warmup_rounds: int = 0) -> Dict:
    from preflid import PrefLID

    n_m, n_w = len(ctx["men"]), len(ctx["women"])
    floor = ctx["config"]["feedback"].get("probability_floor", FLOOR)
    # square series: always use the series budget (rect configs carry
    # their own); small markets never bind it anyway
    is_rect = market_name is not None
    budget = (ctx["config"].get("preflid", {}).get("budget")
              if is_rect else BUDGET)
    provider = RouterBenchBTProvider(
        ctx["theta_task"], ctx["theta_model"],
        rng=random.Random(f"floor15::bt-preflid::{seed}"),
        probability_floor=floor,
    )
    learner = PrefLID(
        men=ctx["men"], women=ctx["women"], provider=provider,
        rng=random.Random(seed), constant=preflid_constant, budget=budget,
        center_policy="round_robin", min_samples_per_pair=10,
        min_sample_ratio=0.5, warmup_rounds=warmup_rounds,
    )
    records: List = []
    original_rrt = learner.rrt_round

    def rrt_round_with_record(center):
        n_samples = original_rrt(center)
        learner._refresh_estimates()
        records.append((learner.t, learner._current_gs_matching()))
        return n_samples

    learner.rrt_round = rrt_round_with_record
    if warmup_rounds > 0:
        # Uniform bootstrap mirroring P2ETG's explore loop exactly:
        # balanced-random sampling (least-sampled arm, uniform choice)
        # with refresh + record every G samples — the same sampling law
        # and the same refresh cadence as P2ETG's checks, so the
        # warm-up lock-in distribution matches P2ETG's. Standard
        # practice for UCB-style uniform initialization.
        from preflid import _canonical
        G = (n_m * (n_m - 1) // 2) + (n_w * (n_w - 1) // 2)
        arms = []
        for agent in learner.agent_states:
            opponents = (learner.women if isinstance(agent, Man)
                         else learner.men)
            opp = list(opponents)
            for i in range(len(opp)):
                for j in range(i + 1, len(opp)):
                    arms.append((agent, _canonical(opp[i], opp[j]),
                                  opp[i], opp[j]))
        warm_rng = random.Random(f"floor15::warmup::{seed}")
        for _ in range(warmup_rounds):
            for _ in range(G):
                min_count = min(
                    learner.agent_states[a].counts.total.get(k, 0)
                    for (a, k, _, _) in arms
                )
                candidates = [
                    (a, k, b1, b2) for (a, k, b1, b2) in arms
                    if learner.agent_states[a].counts.total.get(k, 0)
                    == min_count
                ]
                agent, key, b1, b2 = warm_rng.choice(candidates)
                x = learner._sample_comparison(agent, b1, b2)
                learner.observe(agent, b1, b2, x)
                learner.t += 1
            learner._refresh_estimates()
            records.append((learner.t, learner._current_gs_matching()))
    max_iter = horizon // max(n_m * (n_m - 1) // 2, n_w * (n_w - 1) // 2)
    pl = learner.run_until_stop(max_iterations=max_iter, verbose=False)
    pl_stopped = bool(pl["stopped"])
    pl_committed = pl["matching"]
    pl_t_stop = int(pl["T_stop"])

    h_star_pairs = _pairs_of(ctx)
    t_end = horizon if horizon > pl_t_stop else pl_t_stop
    # Unified evaluation grid: charge PrefLID's matching on the SAME
    # grid P2ETG is charged on (its check interval), instead of every
    # RRT round / warm-up agent step. This is an evaluation-protocol
    # choice, applied identically to both algorithms' outputs — it
    # removes the granularity artefact where a transient wrong
    # matching lasting <1 check interval is free for P2ETG but billed
    # to PrefLID at 3-sample granularity.
    grid = (n_m * (n_m - 1) // 2) + (n_w * (n_w - 1) // 2)
    if records:
        grid_records, last, idx = [], records[0], 0
        k = grid
        while k <= t_end:
            while idx < len(records) and records[idx][0] <= k:
                last = records[idx]
                idx += 1
            grid_records.append((k, last[1]))
            k += grid
        if grid_records:
            records = grid_records
    rows = _dense_rows(records, pl_committed, pl_stopped, h_star_pairs, t_end)

    label = market_name or f"floor15_PrefLID_N{n_w}_K{n_m}_real"
    run_dir = out_root / f"{label}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "bt_params.json").write_text(
        json.dumps(simplex_thetas(ctx["theta_task"], ctx["theta_model"]),
                   indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": label.rsplit("_seed", 1)[0], "N": n_w, "K": n_m,
        "alpha": None, "seed": seed, "budget": budget,
        "constant": preflid_constant,
        "center_policy": "round_robin", "max_iterations": max_iter,
        "horizon": horizon, "warmup_rounds": warmup_rounds,
        "market_source": f"LLMRouterBench real market {n_w}x{n_m} "
                         f"(full data, no split)",
        "preferences": "symmetric (mirror) — unique stable matching",
        "feedback": f"BT, eta-calibrated (target {TARGET}), "
                    f"probability_floor {floor}",
    }, indent=2))
    summary = {
        "N": n_w, "K": n_m, "alpha": None, "seed": seed,
        "budget": budget, "constant": preflid_constant,
        "horizon": horizon,
        "preflid_stopped": pl_stopped,
        "preflid_T_stop": pl_t_stop,
        "preflid_iterations": pl.get("iterations"),
        "preflid_islands_final": pl.get("n_islands"),
        "preflid_correct": int(_pairs(pl_committed) == h_star_pairs),
        "preflid_stable_truth": int(
            StabilityVerifier(ctx["prefs"]).is_stable(pl_committed)[0]
        ),
        "final_regret": rows[-1][4] if rows else None,
        "market_source": f"LLMRouterBench real {n_w}x{n_m} market "
                         f"(mirror, floor {floor}, target {TARGET})",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def _pairs_of(ctx) -> frozenset:
    return _pairs(ctx["h_star"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[3, 5, 8, 10])
    parser.add_argument("--rect", nargs="*", default=None,
                        help="rectangular-market config paths (e.g. "
                             "configs/rect_3x8.yaml); overrides --sizes")
    parser.add_argument("--horizon", type=int, default=MAX_SAMPLES,
                        help="horizon for the rect series (1_000_000 "
                             "for the 10x14 floor-0.20 market)")
    parser.add_argument("--which", choices=["p2etg", "preflid", "both"],
                        default="both")
    parser.add_argument("--preflid-constant", type=float, default=CONSTANT,
                        help="CI constant for the PrefLID series "
                             "(0.1 default; 0.3 needed for all-correct "
                             "certification at 5x5)")
    parser.add_argument("--preflid-warmup", type=int, default=0,
                        help="uniform warm-start rounds before the RRT "
                             "phase (one pass over all arms each)")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="seeds 0..N-1 per series (10 default; "
                             "30 recommended for stable medians)")
    parser.add_argument("--out",
                        default="../gale_shapley_llm_routing/"
                                "gale_shapley_data_share")
    args = parser.parse_args()

    from llm_matching.upstream_rep import build_upstream_context

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.rect:
        # rectangular series: one label per config path
        jobs = []
        for cfg_path in args.rect:
            config = _market_config(0, config_path=cfg_path)
            ctx = build_upstream_context(config)
            n_w, n_m = len(ctx["women"]), len(ctx["men"])
            floor = config["feedback"].get("probability_floor", FLOOR)
            tag = "floor20" if floor >= 0.2 else "floor15"
            jobs.append((ctx,
                         f"{tag}_{{algo}}_N{n_w}_K{n_m}_real",
                         args.horizon))
        for ctx, label_tpl, horizon in jobs:
            for algo in ("p2etg", "preflid"):
                if args.which not in (algo, "both"):
                    continue
                name = label_tpl.format(algo=algo.upper().replace("P2ETG", "P2ETG"))
                for seed in range(args.n_seeds):
                    if algo == "p2etg":
                        s = _run_p2etg(ctx, seed, out_root,
                                       market_name=name, horizon=horizon)
                        print(f"[{name} s{seed}] stopped={s['stopped']} "
                              f"T_stop={s['T_stop']} "
                              f"correct={s['correct_at_stop']}")
                    else:
                        s = _run_preflid(ctx, seed, out_root,
                                         preflid_constant=args.preflid_constant,
                                         market_name=name, horizon=horizon,
                                         warmup_rounds=args.preflid_warmup)
                        print(f"[{name} s{seed}] "
                              f"stopped={s['preflid_stopped']} "
                              f"T_stop={s['preflid_T_stop']} "
                              f"correct={s['preflid_correct']} "
                              f"regret={s['final_regret']}")
        return 0

    for n in args.sizes:
        ctx = build_upstream_context(_market_config(n))
        if args.which in ("p2etg", "both"):
            for seed in range(args.n_seeds):
                s = _run_p2etg(ctx, seed, out_root)
                print(f"[{n}x{n} floor15 P2ETG s{seed}] "
                      f"stopped={s['stopped']} T_stop={s['T_stop']} "
                      f"correct={s['correct_at_stop']} "
                      f"regret={s['final_regret']}")
        if args.which in ("preflid", "both"):
            for seed in range(args.n_seeds):
                s = _run_preflid(ctx, seed, out_root,
                                 preflid_constant=args.preflid_constant,
                                 warmup_rounds=args.preflid_warmup)
                print(f"[{n}x{n} floor15 PrefLID s{seed}] "
                      f"stopped={s['preflid_stopped']} "
                      f"T_stop={s['preflid_T_stop']} "
                      f"correct={s['preflid_correct']} "
                      f"regret={s['final_regret']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
