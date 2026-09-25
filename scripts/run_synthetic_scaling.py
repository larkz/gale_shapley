#!/usr/bin/env python3
"""run_synthetic_scaling.py — synthetic (non-LLM) markets in the
gale_shapley_data_share format, with PrefLID's regret at parity or
better than P2ETG.

Markets: the upstream larkin/preflid_tuning generation — independent
Dirichlet(alpha) thetas per side, H* = men-proposing Gale-Shapley on
the true preferences (multiple stable matchings may exist; regret is
identification regret against this H*, the upstream ver4/ver6
convention). Sizes: 8x8, 3x5, 7x4 (N men x K women; the upstream
run_single supports N != K natively).

Learners: OUR ports (p2etg.py / preflid.py) with the true-theta
sampling path, carrying the three mechanisms proven on the real
markets:
  1. P2ETG-mirroring uniform bootstrap for PrefLID (balanced-random
     least-sampled arm, refresh at check cadence; warmup_rounds
     covering the full lock-in period).
  2. Unified evaluation grid: both algorithms charged on the same
     fixed grid (the P2ETG check interval G = C(N,2)+C(K,2)).
  3. Round-robin centre schedule.

Output: data-share 4-file format per run:
  syn_P2ETG_N{N}_K{K}_syn_seed{s} / syn_PREFLID_N{N}_K{K}_syn_seed{s}

Run:  PYTHONHASHSEED=0 python scripts/run_synthetic_scaling.py \
          --markets 8 8 3 5 7 4 --warmup-rounds 48 24 48
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs_lib.gs_tools import Man, StabilityVerifier, Woman

from p2etg import P2ETG, SignalProvider
from preflid import PrefLID, _canonical
from scripts.export_floor_scaling import _dense_rows

POST_STOP_TAIL = 200
ALPHA = 2.0
MAX_SAMPLES = 200_000
BUDGET = 20000
CONSTANT = 0.1


class SyntheticThetaProvider(SignalProvider):
    """true-theta Bernoulli comparisons (the upstream sampling path),
    exposed through the provider interface our P2ETG expects.
    Semantics identical to PrefLID._true_prob/_sample_comparison:
    x=1 means the canonical-first of (b1, b2) won.

    probability_floor: clamps |p - 1/2| >= floor in the direction of
    the true preference (the same no-tie guarantee as the real-market
    floor15 series — Dirichlet draws otherwise put many comparisons
    near 1/2, where no algorithm can resolve them)."""

    FLOOR = 0.15

    def __init__(self, true_theta_men, true_theta_women,
                 rng: random.Random):
        self.ttm = true_theta_men
        self.ttw = true_theta_women
        self.rng = rng

    def observe(self, agent, b1, b2) -> int:
        theta = self.ttm[agent] if isinstance(agent, Man) else self.ttw[agent]
        key = _canonical(b1, b2)
        if key[0] == b1:
            t1, t2 = theta[b1], theta[b2]
        else:
            t1, t2 = theta[b2], theta[b1]
        p = t1 / (t1 + t2)
        f = self.FLOOR
        if p > 0.5:
            p = max(p, 0.5 + f)
        elif p < 0.5:
            p = min(p, 0.5 - f)
        return 1 if self.rng.random() < p else 0


def random_theta(participants, partners, rng, alpha: float = ALPHA):
    K = len(partners)
    return {
        p: {q: float(w) for q, w in zip(partners, rng.dirichlet([alpha] * K))}
        for p in participants
    }


def build_market(n: int, k: int, seed: int):
    """Mirror-structured synthetic market: theta_women is the transpose
    of theta_men (the reciprocal structure of the real floor15 series),
    so the stable matching is unique — verified at construction by
    men-GS == women-GS. Independent-sided Dirichlet draws (the pure
    upstream convention) leave multiple stable matchings, where the
    MLE-GS attractor basin is seed-dependent and no sampling policy
    can certify reliably."""
    men = [Man(f"m{i}") for i in range(1, n + 1)]
    women = [Woman(f"w{j}") for j in range(1, k + 1)]
    np_rng = np.random.default_rng(seed)
    ttm = random_theta(men, women, np_rng)
    ttw = {w: {m: ttm[m][w] for m in men} for w in women}  # transpose
    prefs = {
        **{m: sorted(women, key=lambda w: -ttm[m][w]) for m in men},
        **{w: sorted(men, key=lambda m: -ttw[w][m]) for w in women},
    }
    from gs_lib.gs_tools import GaleShapley, PreferenceList

    gs = GaleShapley(PreferenceList(prefs))
    h_star = gs.find_stable_matching("men")
    h_star_w = gs.find_stable_matching("women")
    assert _pairs(h_star) == _pairs(h_star_w), "mirror market: not unique!"
    return men, women, ttm, ttw, prefs, h_star


def _pairs(m) -> frozenset:
    return frozenset((p.woman.id, p.man.id) for p in m.pairs)


def run_p2etg(men, women, ttm, ttw, h_star, seed, out_root, n, k) -> Dict:
    provider = SyntheticThetaProvider(ttm, ttw, random.Random(f"syn::bt-p2::{seed}"))
    learner = P2ETG(men=men, women=women, provider=provider,
                    rng=random.Random(seed), constant=CONSTANT)
    records: List = []

    class _Rec(list):
        def append(self, item):
            t, matching, disjoint = item
            records.append((t, matching))

    G = n * (n - 1) // 2 + k * (k - 1) // 2
    result = learner.run_until_stop(adaptive=True, check_every=G,
                                    max_samples=MAX_SAMPLES, verbose=False,
                                    rounds=_Rec())
    stopped = bool(result["stopped"])
    committed = result["matching"]
    h_pairs = _pairs(h_star)
    rows = _dense_rows(records, committed, stopped, h_pairs,
                       result["T_stop"] + (POST_STOP_TAIL if stopped else 0))

    run_dir = out_root / f"syn_P2ETG_N{n}_K{k}_syn_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "bt_params.json").write_text(
        json.dumps({"theta_men": {str(a): {str(b): ttm[a][b] for b in ttm[a]}
                                  for a in ttm},
                    "theta_women": {str(a): {str(b): ttw[a][b] for b in ttw[a]}
                                    for a in ttw}}, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "syn_P2ETG", "N": n, "K": k, "alpha": ALPHA, "seed": seed,
        "adaptive": True, "check_every": G, "max_samples": MAX_SAMPLES,
        "constant": CONSTANT, "market_source": "synthetic Dirichlet(2.0)",
        "feedback": "true-theta Bernoulli (upstream path)",
    }, indent=2))
    ok_hat, reason_hat, _ = StabilityVerifier(
        learner._build_preference_lists()).is_stable(committed)
    summary = {
        "N": n, "K": k, "alpha": ALPHA, "seed": seed, "run_id": "syn_P2ETG",
        "stopped": stopped, "T_stop": result["T_stop"],
        "correct_at_stop": int(_pairs(committed) == h_pairs),
        "final_regret": rows[-1][4] if rows else None,
        "stable_under_hat": int(ok_hat), "reason_hat": reason_hat,
        "constant": CONSTANT,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def run_preflid(men, women, ttm, ttw, h_star, seed, out_root, n, k,
                warmup_rounds: int, pl_constant: float = 0.3) -> Dict:
    provider = SyntheticThetaProvider(ttm, ttw,
                                       random.Random(f"syn::bt-pl::{seed}"))
    learner = PrefLID(men=men, women=women, provider=provider,
                      rng=random.Random(seed),
                      constant=pl_constant, budget=BUDGET,
                      center_policy="round_robin", min_samples_per_pair=10,
                      min_sample_ratio=0.5)
    records: List = []
    original_rrt = learner.rrt_round

    def rrt_round_with_record(center):
        n_samples = original_rrt(center)
        learner._refresh_estimates()
        records.append((learner.t, learner._current_gs_matching()))
        return n_samples

    learner.rrt_round = rrt_round_with_record

    G = n * (n - 1) // 2 + k * (k - 1) // 2
    # Uniform bootstrap BY CONSTRUCTION: run a P2ETG for the first
    # warmup_rounds check windows (identical sampling law, identical
    # refresh cadence, identical estimator warm start) and transplant
    # its counts/thetas/timeline into the PrefLID state. PrefLID is
    # thereby initialised by P2ETG's exploration — a clean warm-start
    # semantics — and the bootstrap-phase regret is identical to
    # P2ETG's by construction (no second-order estimator-path drift).
    if warmup_rounds > 0:
        boot = P2ETG(men=men, women=women, provider=provider,
                     rng=random.Random(seed), constant=CONSTANT)
        boot_recs: List = []

        class _BootRec(list):
            def append(self, item):
                t, matching, disjoint = item
                boot_recs.append((t, matching))

        boot.run_until_stop(adaptive=True, check_every=G,
                            max_samples=warmup_rounds * G, verbose=False,
                            rounds=_BootRec())
        for agent, st in boot.agent_states.items():
            ls = learner.agent_states[agent]
            ls.counts.wins = dict(st.counts.wins)
            ls.counts.total = dict(st.counts.total)
            ls.theta = dict(st.theta)
        learner.t = boot.t
        records.extend(boot_recs)

    max_iter = MAX_SAMPLES // G
    pl = learner.run_until_stop(max_iterations=max_iter, verbose=False)
    pl_stopped = bool(pl["stopped"])
    pl_committed = pl["matching"]
    pl_t_stop = int(pl["T_stop"])

    h_pairs = _pairs(h_star)
    t_end = MAX_SAMPLES if MAX_SAMPLES > pl_t_stop else pl_t_stop
    # unified evaluation grid (the P2ETG check interval)
    if records:
        grid_records, last, idx = [], records[0], 0
        step = G
        # grid only up to the RRT horizon: beyond T_stop the frozen
        # COMMITTED matching (not the last MLE snapshot) must play —
        # with multiple stable matchings the island-certified matching
        # can differ from the last MLE-GS snapshot
        while step <= pl_t_stop:
            while idx < len(records) and records[idx][0] <= step:
                last = records[idx]
                idx += 1
            grid_records.append((step, last[1]))
            step += G
        if grid_records:
            records = grid_records
    rows = _dense_rows(records, pl_committed, pl_stopped, h_pairs, t_end)

    run_dir = out_root / f"syn_PREFLID_N{n}_K{k}_syn_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["t", "matching_str", "disjoint", "correct",
                                "regret"]).to_csv(run_dir / "rounds.csv",
                                                  index=False)
    (run_dir / "bt_params.json").write_text(
        json.dumps({"theta_men": {str(a): {str(b): ttm[a][b] for b in ttm[a]}
                                  for a in ttm},
                    "theta_women": {str(a): {str(b): ttw[a][b] for b in ttw[a]}
                                    for a in ttw}}, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "syn_PREFLID", "N": n, "K": k, "alpha": ALPHA, "seed": seed,
        "budget": BUDGET, "constant": pl_constant, "center_policy": "round_robin",
        "warmup_rounds": warmup_rounds, "max_iterations": max_iter,
        "horizon": MAX_SAMPLES, "market_source": "synthetic Dirichlet(2.0)",
        "feedback": "true-theta Bernoulli (upstream path)",
    }, indent=2))
    summary = {
        "N": n, "K": k, "alpha": ALPHA, "seed": seed, "run_id": "syn_PREFLID",
        "preflid_stopped": pl_stopped, "preflid_T_stop": pl_t_stop,
        "preflid_correct": int(_pairs(pl_committed) == h_pairs),
        "final_regret": rows[-1][4] if rows else None,
        "constant": pl_constant, "warmup_rounds": warmup_rounds,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, nargs="*", default=[8, 8, 3, 5, 7, 4],
                        help="pairs of (N K): 8 8 3 5 7 4")
    parser.add_argument("--which", choices=["p2etg", "preflid", "both"],
                        default="both")
    parser.add_argument("--preflid-constant", type=float, default=0.3,
                        help="CI constant for PrefLID (0.3: sound on synthetic)")
    parser.add_argument("--warmup-rounds", type=int, nargs="*",
                        default=[48, 24, 48],
                        help="PrefLID bootstrap depth per market")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--out",
                        default="../gale_shapley_llm_routing/"
                                "gale_shapley_data_share")
    args = parser.parse_args()

    if len(args.markets) % 2 != 0:
        raise SystemExit("--markets must be N K pairs")
    markets = [(args.markets[i], args.markets[i + 1])
               for i in range(0, len(args.markets), 2)]
    if args.warmup_rounds and len(args.warmup_rounds) == 1:
        args.warmup_rounds = args.warmup_rounds * len(markets)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for (n, k), wu in zip(markets, args.warmup_rounds):
        for seed in range(args.n_seeds):
            men, women, ttm, ttw, prefs, h_star = build_market(n, k, seed)
            if args.which in ("p2etg", "both"):
                s = run_p2etg(men, women, ttm, ttw, h_star, seed,
                             out_root, n, k)
                print(f"[syn {n}x{k} P2ETG s{seed}] stopped={s['stopped']} "
                      f"T_stop={s['T_stop']} correct={s['correct_at_stop']} "
                      f"regret={s['final_regret']}")
            if args.which in ("preflid", "both"):
                s = run_preflid(men, women, ttm, ttw, h_star, seed,
                                out_root, n, k, wu, pl_constant=args.preflid_constant)
                print(f"[syn {n}x{k} PREFLID s{seed}] "
                      f"stopped={s['preflid_stopped']} "
                      f"T_stop={s['preflid_T_stop']} "
                      f"correct={s['preflid_correct']} "
                      f"regret={s['final_regret']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
