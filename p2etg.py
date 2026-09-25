"""
p2etg.py
Pairwise Two-Sided Explore-then-Gale-Shapley (P2ETG) learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

import random

from gs_lib.gs_tools import (
    Man, Woman, Matching, PreferenceList, GaleShapley,
)
from gs_lib.bt import (
    PairCounts, _canonical,
    bt_mle_mm, bt_confidence_intervals, bt_ranking,
)


@dataclass
class AgentState:
    agent: Hashable
    partners: List[Hashable]
    counts: PairCounts = field(default_factory=PairCounts)
    theta: Dict[Hashable, float] = field(default_factory=dict)
    ci: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.theta:
            K = len(self.partners)
            self.theta = {p: 1.0 / max(K, 1) for p in self.partners}


class P2ETG:
    def __init__(
        self,
        men: List[Man],
        women: List[Woman],
        true_theta_men: Optional[Dict[Man, Dict[Woman, float]]] = None,
        true_theta_women: Optional[Dict[Woman, Dict[Man, float]]] = None,
        rng: Optional[random.Random] = None,
        constant: float = 0.1,
    ):
        self.men = list(men)
        self.women = list(women)
        self.true_theta_men = true_theta_men
        self.true_theta_women = true_theta_women
        self.rng = rng or random.Random(0)
        self.constant = constant

        self.agent_states: Dict[Hashable, AgentState] = {}
        for m in self.men:
            self.agent_states[m] = AgentState(agent=m, partners=self.women)
        for w in self.women:
            self.agent_states[w] = AgentState(agent=w, partners=self.men)

        self.t = 0
        self.epoch = 0
        self.stopped = False
        self.committed_matching: Optional[Matching] = None

        self._arms: List[Tuple[Hashable, Tuple, Hashable, Hashable]] = []
        for state in self.agent_states.values():
            agent = state.agent
            partners = state.partners
            for i in range(len(partners)):
                for j in range(i + 1, len(partners)):
                    key = _canonical(partners[i], partners[j])
                    self._arms.append((agent, key, partners[i], partners[j]))

        self._true_prob_cache: Dict[Tuple, float] = {}
        if (self.true_theta_men is not None
                and self.true_theta_women is not None):
            for (agent, key, b1, b2) in self._arms:
                self._true_prob_cache[(agent, key)] = self._true_prob(
                    agent, b1, b2
                )

    def _true_prob(self, agent: Hashable, b1: Hashable, b2: Hashable) -> float:
        if isinstance(agent, Man):
            theta = self.true_theta_men[agent]
        else:
            theta = self.true_theta_women[agent]
        key = _canonical(b1, b2)
        if key[0] == b1:
            t1, t2 = theta[b1], theta[b2]
        else:
            t1, t2 = theta[b2], theta[b1]
        return t1 / (t1 + t2)

    def _sample_comparison(self, agent: Hashable, b1: Hashable,
                           b2: Hashable) -> int:
        if not self._true_prob_cache:
            raise RuntimeError(
                "No ground-truth thetas provided; use observe() to inject "
                "external comparisons instead of running the sampler."
            )
        key = _canonical(b1, b2)
        p = self._true_prob_cache[(agent, key)]
        return 1 if self.rng.random() < p else 0

    def observe(self, agent: Hashable, b1: Hashable, b2: Hashable,
                x: int) -> None:
        self.agent_states[agent].counts.record(b1, b2, x)

    def _refresh_estimates(self) -> None:
        for state in self.agent_states.values():
            state.theta = bt_mle_mm(
                state.partners,
                state.counts.wins,
                state.counts.total,
                max_iter=50,
                tol=1e-6,
                theta_init=state.theta,
            )
            state.ci = bt_confidence_intervals(
                state.partners,
                state.theta,
                state.counts,
                self.t,
                self.constant,
            )

    def _build_preference_lists(self) -> PreferenceList:
        prefs: Dict[Hashable, List[Hashable]] = {}
        for m in self.men:
            prefs[m] = bt_ranking(self.agent_states[m].theta)
        for w in self.women:
            prefs[w] = bt_ranking(self.agent_states[w].theta)
        return PreferenceList(prefs)

    def current_matching(self) -> Matching:
        prefs = self._build_preference_lists()
        return GaleShapley(prefs).find_stable_matching(proposing_side="men")

    def _pairwise_disjoint(self) -> bool:
        for state in self.agent_states.values():
            n = len(state.partners)
            expected = n * (n - 1) // 2
            if len(state.ci) < expected:
                return False
            for (lo, hi) in state.ci.values():
                if not (lo > 0.5 or hi < 0.5):
                    return False
        return True

    def _run_epoch(self) -> int:
        R = 2 ** self.epoch
        new_samples = 0
        active: List[Tuple[Hashable, Tuple, Hashable, Hashable]] = []
        for arm in self._arms:
            agent, key, b1, b2 = arm
            state = self.agent_states[agent]
            if state.counts.total.get(key, 0) < R:
                active.append(arm)
        while active:
            idx = self.rng.randrange(len(active))
            agent, key, b1, b2 = active[idx]
            state = self.agent_states[agent]
            x = self._sample_comparison(agent, b1, b2)
            state.counts.record(b1, b2, x)
            self.t += 1
            new_samples += 1
            if state.counts.total.get(key, 0) >= R:
                active[idx] = active[-1]
                active.pop()
        return new_samples

    def _sample_batch(self, n: int) -> int:
        new_samples = 0
        for _ in range(n):
            min_count = min(
                self.agent_states[a].counts.total.get(k, 0)
                for (a, k, _, _) in self._arms
            )
            candidates = [
                (a, k, b1, b2) for (a, k, b1, b2) in self._arms
                if self.agent_states[a].counts.total.get(k, 0) == min_count
            ]
            agent, key, b1, b2 = self.rng.choice(candidates)
            x = self._sample_comparison(agent, b1, b2)
            self.observe(agent, b1, b2, x)
            self.t += 1
            new_samples += 1
        return new_samples

    def run_until_stop(
        self,
        max_epochs: int = 400,
        adaptive: bool = False,
        check_every: int = 500,
        max_samples: int = 2_000_000_000_000,
        verbose: bool = True,
        rounds: Optional[List[Tuple[int, Matching, bool]]] = None,
    ) -> Dict[str, object]:
        trace: List[Dict[str, object]] = []

        def _emit_line(epoch, R, new_samples, t, disjoint, matching):
            max_width = 0.0
            min_gap = float("inf")
            worst_overlap = None
            for state in self.agent_states.values():
                for (lo, hi) in state.ci.values():
                    w = hi - lo
                    if w > max_width:
                        max_width = w
                    p_hat = (lo + hi) / 2
                    gap = abs(p_hat - 0.5)
                    if gap < min_gap:
                        min_gap = gap
                    if lo <= 0.5 <= hi:
                        overlap = min(0.5 - lo, hi - 0.5)
                    else:
                        overlap = -min(lo - 0.5, 0.5 - hi)
                    if worst_overlap is None or overlap > worst_overlap:
                        worst_overlap = overlap
            if min_gap == float("inf"):
                min_gap = float("nan")
            if worst_overlap is None:
                worst_overlap = float("nan")
            if not getattr(self, "_trace_header_printed", False):
                hdr = (f"{'ep':>4} {'R':>8} {'new':>8} {'t':>10} "
                       f"{'disj':>6} {'maxWid':>8} {'minGap':>8} "
                       f"{'worstOv':>9}  matching")
                print(hdr)
                print("-" * len(hdr))
                self._trace_header_printed = True
            print(f"{epoch:>4} {R:>8} {new_samples:>8} {t:>10} "
                  f"{str(disjoint):>6} {max_width:>8.4f} {min_gap:>8.4f} "
                  f"{worst_overlap:>9.5f}  {matching}")
            return {
                "max_width": max_width,
                "min_gap": min_gap,
                "worst_overlap": worst_overlap,
            }

        if adaptive:
            while self.t < max_samples:
                # Never overshoot the budget: the final batch may be
                # smaller than check_every.
                batch = min(check_every, max_samples - self.t)
                self._sample_batch(batch)
                self._refresh_estimates()
                matching = self.current_matching()
                disjoint = self._pairwise_disjoint()

                if rounds is not None:
                    rounds.append((self.t, matching, disjoint))

                diag = _emit_line(
                    self.epoch, batch, batch,
                    self.t, disjoint, matching,
                ) if verbose else {}

                trace.append({
                    "epoch": self.epoch,
                    "R": check_every,
                    "new_samples": batch,
                    "t": self.t,
                    "matching": matching,
                    "pairwise_disjoint": disjoint,
                    **diag,
                })

                if disjoint:
                    self.stopped = True
                    self.committed_matching = matching
                    return {
                        "epochs": trace,
                        "T_stop": self.t,
                        "matching": matching,
                        "stopped": True,
                    }
                self.epoch += 1

            self.stopped = False
            self.committed_matching = trace[-1]["matching"] if trace else None
            return {
                "epochs": trace,
                "T_stop": self.t,
                "matching": self.committed_matching,
                "stopped": False,
            }

        for _ in range(max_epochs):
            if self.t >= max_samples:
                break

            new_samples = self._run_epoch()
            self._refresh_estimates()
            matching = self.current_matching()
            disjoint = self._pairwise_disjoint()

            if rounds is not None:
                rounds.append((self.t, matching, disjoint))

            diag = _emit_line(
                self.epoch, 2 ** self.epoch, new_samples,
                self.t, disjoint, matching,
            ) if verbose else {}

            trace.append({
                "epoch": self.epoch,
                "R": 2 ** self.epoch,
                "new_samples": new_samples,
                "t": self.t,
                "matching": matching,
                "pairwise_disjoint": disjoint,
                **diag,
            })

            if disjoint:
                self.stopped = True
                self.committed_matching = matching
                return {
                    "epochs": trace,
                    "T_stop": self.t,
                    "matching": matching,
                    "stopped": True,
                }

            self.epoch += 1

        self.stopped = False
        self.committed_matching = trace[-1]["matching"] if trace else None
        return {
            "epochs": trace,
            "T_stop": self.t,
            "matching": self.committed_matching,
            "stopped": False,
        }

    def run_with_trace(
        self,
        max_epochs: int = 400,
        adaptive: bool = False,
        check_every: int = 500,
        max_samples: int = 2_000_000_000_000,
        verbose: bool = False,
    ) -> Dict[str, object]:
        rounds: List[Tuple[int, Matching, bool]] = []
        result = self.run_until_stop(
            max_epochs=max_epochs,
            adaptive=adaptive,
            check_every=check_every,
            max_samples=max_samples,
            verbose=verbose,
            rounds=rounds,
        )
        result["rounds"] = rounds
        return result