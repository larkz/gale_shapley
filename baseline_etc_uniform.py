"""
baseline_etc_uniform.py

Uniform Explore-then-Commit (ETC-Uniform) baseline for P2ETG comparison.

Fixed-budget counterpart to P2ETG. Same BT model, same MLE, same GS
integration, same regret definition. The ONLY difference: instead of an
adaptive stopping rule based on confidence intervals, it samples every
(agent, unordered pair) arm exactly T0 times, then commits to the
Gale-Shapley matching on the estimated preferences.

This is the natural baseline that isolates the contribution of P2ETG's
adaptive stopping rule from its other components.
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
# Per-agent state (identical to P2ETG)
# ============================================================================

@dataclass
class AgentState:
    agent: Hashable
    partners: List[Hashable]
    counts: PairCounts = field(default_factory=PairCounts)
    theta: Dict[Hashable, float] = field(default_factory=dict)
    ci: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.theta:
            K = len(self.partners)
            self.theta = {p: 1.0 / max(K, 1) for p in self.partners}


# ============================================================================
# ETC-Uniform
# ============================================================================

class ETCUniform:
    """Uniform Explore-then-Commit baseline.

    Exploration:
        Sample every arm (agent, unordered pair) exactly T0 times,
        in random order. Total samples = T0 * |arms|.

    Exploitation:
        Commit to Gale-Shapley on the estimated preference profile
        forever after. No stopping rule, no CI check.

    Args:
        men, women:                participants.
        true_theta_men/women:      ground-truth BT parameters (for sampling).
        rng:                       random.Random for reproducibility.
        T0:                        fixed samples per arm (hyperparameter).
        constant:                  CI width constant (for consistency with
                                   P2ETG; not used for stopping here).
    """

    def __init__(
        self,
        men: List[Man],
        women: List[Woman],
        true_theta_men: Optional[Dict[Man, Dict[Woman, float]]] = None,
        true_theta_women: Optional[Dict[Woman, Dict[Man, float]]] = None,
        rng: Optional[random.Random] = None,
        T0: int = 100,
        constant: float = 0.1,
    ):
        self.men = list(men)
        self.women = list(women)
        self.true_theta_men = true_theta_men
        self.true_theta_women = true_theta_women
        self.rng = rng or random.Random(0)
        self.T0 = T0
        self.constant = constant

        # One AgentState per participant, both sides learn.
        self.agent_states: Dict[Hashable, AgentState] = {}
        for m in self.men:
            self.agent_states[m] = AgentState(agent=m, partners=self.women)
        for w in self.women:
            self.agent_states[w] = AgentState(agent=w, partners=self.men)

        self.t = 0
        self.stopped = False
        self.committed_matching: Optional[Matching] = None

        # Precompute arms: (agent, canonical_key, b1, b2)
        self._arms: List[Tuple[Hashable, Tuple, Hashable, Hashable]] = []
        for state in self.agent_states.values():
            agent = state.agent
            partners = state.partners
            for i in range(len(partners)):
                for j in range(i + 1, len(partners)):
                    key = _canonical(partners[i], partners[j])
                    self._arms.append((agent, key, partners[i], partners[j]))

        # Precompute true BT probabilities for every (agent, pair).
        self._true_prob_cache: Dict[Tuple, float] = {}
        if (self.true_theta_men is not None
                and self.true_theta_women is not None):
            for (agent, key, b1, b2) in self._arms:
                self._true_prob_cache[(agent, key)] = self._true_prob(
                    agent, b1, b2
                )

    # ------------------------------------------------------------------
    # Sampling (identical to P2ETG)
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

    def _sample_comparison(
        self, agent: Hashable, b1: Hashable, b2: Hashable
    ) -> int:
        """Sample one Bernoulli comparison.

        Returns 1 if the canonical-first of (b1, b2) won, else 0.
        """
        if not self._true_prob_cache:
            raise RuntimeError(
                "No ground-truth thetas provided; use observe() to inject "
                "external comparisons instead of running the sampler."
            )
        key = _canonical(b1, b2)
        p = self._true_prob_cache[(agent, key)]
        return 1 if self.rng.random() < p else 0

    def observe(
        self, agent: Hashable, b1: Hashable, b2: Hashable, x: int
    ) -> None:
        """Inject an externally-supplied comparison outcome.

        x = 1 means the canonical-first of (b1, b2) won.
        x = 0 means the canonical-second won.
        """
        self.agent_states[agent].counts.record(b1, b2, x)

    # ------------------------------------------------------------------
    # Estimation (identical to P2ETG)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Gale-Shapley (identical to P2ETG)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Exploration: uniform, fixed-budget
    # ------------------------------------------------------------------

    def _run_exploration(self) -> int:
        """Sample every arm exactly T0 times, in random order.

        Returns the number of samples drawn.
        """
        new_samples = 0

        # Build a worklist: one entry per arm, tracking remaining draws.
        # Using a dict so we can remove exhausted arms in O(1).
        remaining: Dict[Tuple[Hashable, Tuple, Hashable, Hashable], int] = {
            arm: self.T0 for arm in self._arms
        }

        while remaining:
            # Uniformly pick an arm that still needs samples.
            arm = self.rng.choice(list(remaining.keys()))
            agent, key, b1, b2 = arm

            x = self._sample_comparison(agent, b1, b2)
            self.observe(agent, b1, b2, x)
            self.t += 1
            new_samples += 1

            remaining[arm] -= 1
            if remaining[arm] <= 0:
                del remaining[arm]

        return new_samples

    # ------------------------------------------------------------------
    # Public API — mirrors P2ETG
    # ------------------------------------------------------------------

    def run_until_stop(
        self,
        max_epochs: int = 1,
        adaptive: bool = False,
        check_every: int = 0,
        max_samples: int = 2_000_000_000_000,
        verbose: bool = True,
        rounds: Optional[List[Tuple[int, Matching, bool]]] = None,
    ) -> Dict[str, object]:
        """Run ETC-Uniform: fixed T0 samples per arm, then commit.

        The signature matches P2ETG so this is a drop-in replacement in
        experiment.py. The `adaptive`, `check_every`, and `max_epochs`
        arguments are ignored — ETC-Uniform has no stopping rule.
        """
        trace: List[Dict[str, object]] = []

        # Draw the entire fixed budget in one go.
        new_samples = self._run_exploration()

        # Estimate and commit.
        self._refresh_estimates()
        matching = self.current_matching()

        if rounds is not None:
            rounds.append((self.t, matching, True))

        trace.append({
            "epoch": 0,
            "R": self.T0,
            "new_samples": new_samples,
            "t": self.t,
            "matching": matching,
            "pairwise_disjoint": True,
        })

        self.stopped = True
        self.committed_matching = matching

        if verbose:
            print(f"ETC-Uniform: T0={self.T0}, arms={len(self._arms)}, "
                  f"total_samples={self.t}, committed={matching}")

        return {
            "epochs": trace,
            "T_stop": self.t,
            "matching": matching,
            "stopped": True,
        }

    def run_with_trace(
        self,
        max_epochs: int = 1,
        adaptive: bool = False,
        check_every: int = 0,
        max_samples: int = 2_000_000_000_000,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """Same as run_until_stop, but returns a per-round trace.

        For ETC-Uniform, `rounds` always has exactly one entry:
        (T_total, committed_matching, True). The regret curve is therefore
        flat before T_total (with the algorithm still exploring) and flat
        after (with the committed matching).
        """
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