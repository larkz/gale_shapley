"""
p2etg.py
Pairwise Two-Sided Explore-then-Gale-Shapley (P2ETG) learner.

Integrates with the stable matching library from earlier:
    Man, Woman, Matching, PreferenceList, GaleShapley

The learner runs in doubling epochs:
    R_l = 2^l
For each agent and each unordered pair of that agent's potential partners,
we accumulate R_l comparisons before refreshing the estimates.

Stopping condition:
    For every agent and every pair of their partners, the confidence intervals
    on theta are disjoint. (Pairwise disjointness, not per-parameter width.)
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


# ============================================================================
# Per-agent state
# ============================================================================

@dataclass
class AgentState:
    """State for a single agent learning preferences over the other side."""
    agent: Hashable
    partners: List[Hashable]
    counts: PairCounts = field(default_factory=PairCounts)
    theta: Dict[Hashable, float] = field(default_factory=dict)
    ci: Dict[Hashable, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.theta:
            K = len(self.partners)
            self.theta = {p: 1.0 / max(K, 1) for p in self.partners}


# ============================================================================
# Main learner
# ============================================================================

class P2ETG:
    """Pairwise Two-Sided Explore-then-Gale-Shapley learner.

    Args:
        men:   list of Man objects.
        women: list of Woman objects.
        true_theta_men:   dict Man -> {Woman: float}, optional ground truth.
        true_theta_women: dict Woman -> {Man: float}, optional ground truth.
            If provided, comparisons are sampled from the true BT model.
            Otherwise, comparisons must be injected via `observe`.
        rng: random.Random instance for reproducibility.
    """

    def __init__(
        self,
        men: List[Man],
        women: List[Woman],
        true_theta_men: Optional[Dict[Man, Dict[Woman, float]]] = None,
        true_theta_women: Optional[Dict[Woman, Dict[Man, float]]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.men = list(men)
        self.women = list(women)
        self.true_theta_men = true_theta_men
        self.true_theta_women = true_theta_women
        self.rng = rng or random.Random(0)

        # One AgentState per participant
        self.agent_states: Dict[Hashable, AgentState] = {}
        for m in self.men:
            self.agent_states[m] = AgentState(agent=m, partners=self.women)
        for w in self.women:
            self.agent_states[w] = AgentState(agent=w, partners=self.men)

        self.t = 0
        self.epoch = 0
        self.stopped = False
        self.committed_matching: Optional[Matching] = None

        # PATCH: precompute the arms list once (does not change across epochs).
        # Each arm is (agent, canonical_key, b1, b2).
        self._arms: List[Tuple[Hashable, Tuple, Hashable, Hashable]] = []
        for state in self.agent_states.values():
            agent = state.agent
            partners = state.partners
            for i in range(len(partners)):
                for j in range(i + 1, len(partners)):
                    key = _canonical(partners[i], partners[j])
                    self._arms.append((agent, key, partners[i], partners[j]))

        # PATCH: precompute true BT probabilities for every (agent, pair).
        # Eliminates per-sample dict lookups and canonicalisation.
        self._true_prob_cache: Dict[Tuple, float] = {}
        if (self.true_theta_men is not None
                and self.true_theta_women is not None):
            for (agent, key, b1, b2) in self._arms:
                self._true_prob_cache[(agent, key)] = self._true_prob(
                    agent, b1, b2
                )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _true_prob(self, agent: Hashable, b1: Hashable, b2: Hashable) -> float:
        """True BT probability that agent prefers canonical-first over second."""
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

    # def _sample_comparison(self, agent: Hashable, b1: Hashable,
    #                        b2: Hashable) -> int:
    #     """Sample a Bernoulli comparison outcome.

    #     Returns 1 if the canonical-first of (b1,b2) wins, else 0.
    #     Uses the precomputed probability cache.
    #     """
    #     if not self._true_prob_cache:
    #         raise RuntimeError(
    #             "No ground-truth thetas provided; use observe() to inject "
    #             "external comparisons instead of running the sampler."
    #         )
    #     key = _canonical(b1, b2)
    #     p = self._true_prob_cache[(agent, key)]
    #     draw = self.rng.random() < p
    #     return 1 if draw else 0

    def _sample_comparison(self, agent, b1, b2):
        key = _canonical(b1, b2)
        p = self._true_prob_cache[(agent, key)]
        draw = self.rng.random() < p
        return 1 if draw else 0

    def observe(self, agent: Hashable, b1: Hashable, b2: Hashable,
                x: int) -> None:
        """Inject an externally-supplied comparison outcome."""
        self.agent_states[agent].counts.record(b1, b2, x)

    # ------------------------------------------------------------------
    # Estimation and interval updates
    # ------------------------------------------------------------------

    def _refresh_estimates(self) -> None:
        # PATCH: warm-started, looser MLE.
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
                state.partners, state.theta, state.counts, self.t
            )

    # ------------------------------------------------------------------
    # Gale-Shapley
    # ------------------------------------------------------------------

    def _build_preference_lists(self) -> PreferenceList:
        prefs: Dict[Hashable, List[Hashable]] = {}
        for m in self.men:
            prefs[m] = bt_ranking(self.agent_states[m].theta)
        for w in self.women:
            prefs[w] = bt_ranking(self.agent_states[w].theta)
        return PreferenceList(prefs)

    def current_matching(self) -> Matching:
        """Run Gale-Shapley on the current empirical profile (A-side proposes)."""
        prefs = self._build_preference_lists()
        return GaleShapley(prefs).find_stable_matching(proposing_side="men")

    # ------------------------------------------------------------------
    # Stopping condition
    # ------------------------------------------------------------------

    # def _pairwise_disjoint(self) -> bool:
    #     """Stopping condition: every pair's CI on p̂ excludes 1/2."""
    #     for state in self.agent_states.values():
    #         for (lo, hi) in state.ci.values():
    #             if not (lo > 0.5 or hi < 0.5):
    #                 return False
    #     return True

    def _pairwise_disjoint(self) -> bool:
        for state in self.agent_states.values():
            n = len(state.partners)
            expected = n * (n - 1) // 2
            if len(state.ci) < expected:
                return False              # some pairs never compared
            for (lo, hi) in state.ci.values():
                if not (lo > 0.5 or hi < 0.5):
                    return False
        return True
    
    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(self) -> int:
        """Run one epoch of round-robin exploration.

        In epoch l, target R = 2^l comparisons per (agent, unordered pair).
        PATCH: uses an active-arm list; each sample is O(1) amortised.
        """
        R = 2 ** self.epoch
        new_samples = 0

        # Build the active list for this epoch. Each entry is
        # (agent, key, b1, b2); arms already at R are excluded.
        active: List[Tuple[Hashable, Tuple, Hashable, Hashable]] = []
        for arm in self._arms:
            agent, key, b1, b2 = arm
            state = self.agent_states[agent]
            if state.counts.total.get(key, 0) < R:
                active.append(arm)

        # Random-draw + swap-remove loop (uniform without replacement bias
        # only if we re-insert; since we never re-insert within an epoch,
        # this is uniform over the remaining active arms per draw).
        while active:
            idx = self.rng.randrange(len(active))
            agent, key, b1, b2 = active[idx]
            state = self.agent_states[agent]

            x = self._sample_comparison(agent, b1, b2)
            state.counts.record(b1, b2, x)
            self.t += 1
            new_samples += 1

            if state.counts.total.get(key, 0) >= R:
                # Swap-remove: O(1) removal
                active[idx] = active[-1]
                active.pop()

        return new_samples

    # ------------------------------------------------------------------
    # Adaptive sampling (PATCH): stop as soon as CIs separate
    # ------------------------------------------------------------------

    def _sample_batch(self, n: int) -> int:
        """Sample `n` comparisons round-robin across arms, weighted by
        how far each arm is from the current maximum count. Always picks
        the least-sampled arms first, in a random order."""
        new_samples = 0
        for _ in range(n):
            # Find the minimum count across arms
            min_count = min(
                self.agent_states[a].counts.total.get(k, 0)
                for (a, k, _, _) in self._arms
            )
            # Collect arms tied at min_count
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

    # ------------------------------------------------------------------
    # Top-level API
    # ------------------------------------------------------------------

    def run_until_stop(
        self,
        max_epochs: int = 400,
        adaptive: bool = False,
        check_every: int = 500,
        max_samples: int = 2_000_000_000_000,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """Run P2ETG until the stopping condition holds.

        Args:
            max_epochs:   cap on doubling epochs (used in non-adaptive mode).
            adaptive:     if True, sample in small batches and check the
                          stopping condition every `check_every` samples.
                          Otherwise, run full doubling epochs.
            check_every:  batch size for adaptive mode.
            max_samples:  hard cap on total samples (both modes).
            verbose:      if True, print a one-line live trace per epoch.

        Returns a trace dict.
        """
        trace: List[Dict[str, object]] = []

        # ------------------------------------------------------------------
        # Live-print helper
        # ------------------------------------------------------------------
        def _emit_line(epoch, R, new_samples, t, disjoint, matching):
            """Compute per-epoch diagnostics and print one line."""
            max_width = 0.0
            min_gap = float("inf")
            worst_overlap = None
            for state in self.agent_states.values():
                for (lo, hi) in state.ci.values():
                    w = hi - lo
                    if w > max_width:
                        max_width = w

                    # gap = distance of the interval centre from 1/2
                    p_hat = (lo + hi) / 2
                    gap = abs(p_hat - 0.5)
                    if gap < min_gap:
                        min_gap = gap

                    # overlap with 1/2:
                    #   positive if 1/2 is inside [lo, hi]
                    #   negative if the interval is entirely on one side of 1/2
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

            # Header (printed once)
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

        # ------------------------------------------------------------------
        # Adaptive mode: stop as soon as CIs separate
        # ------------------------------------------------------------------
        if adaptive:
            while self.t < max_samples:
                self._sample_batch(check_every)
                self._refresh_estimates()
                matching = self.current_matching()
                disjoint = self._pairwise_disjoint()

                diag = _emit_line(
                    self.epoch, check_every, check_every,
                    self.t, disjoint, matching,
                ) if verbose else {}

                trace.append({
                    "epoch": self.epoch,
                    "R": check_every,
                    "new_samples": check_every,
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

        # ------------------------------------------------------------------
        # Doubling mode
        # ------------------------------------------------------------------
        for _ in range(max_epochs):
            if self.t >= max_samples:
                break

            new_samples = self._run_epoch()
            self._refresh_estimates()
            matching = self.current_matching()
            disjoint = self._pairwise_disjoint()

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

        # After the print, before return:
        hook = getattr(self, "_emit_line_hook", None)
        if hook is not None:
            hook(t, matching, disjoint)

        return {
            "epochs": trace,
            "T_stop": self.t,
            "matching": self.committed_matching,
            "stopped": False,
        }

    def run_with_trace(
        self,
        max_epochs: int = 40,
        adaptive: bool = False,
        check_every: int = 500,
        max_samples: int = 2_000_000,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """Same as run_until_stop, but records the matching at every
        epoch/check so that per-round regret can be reconstructed.

        Returns a dict with an additional key 'rounds':
            rounds = [ (t, matching, disjoint), ... ]
        """
        rounds: List[Tuple[int, Matching, bool]] = []

        # We monkey-patch _emit_line so it also records the round.
        original_emit = getattr(self, "_emit_line_hook", None)

        def hook(t, matching, disjoint):
            rounds.append((t, matching, disjoint))

        self._emit_line_hook = hook
        try:
            result = self.run_until_stop(
                max_epochs=max_epochs,
                adaptive=adaptive,
                check_every=check_every,
                max_samples=max_samples,
                verbose=verbose,
            )
        finally:
            self._emit_line_hook = original_emit

        result["rounds"] = rounds
        return result