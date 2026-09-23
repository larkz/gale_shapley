"""
preflid.py
Preference Resolution via Lattice Island Detection (PrefLID).

Built on the P2ETG learner for BT estimation. PrefLID adds a structural
layer on top: it partitions the two-sided ordering space into blocks
derived from pairwise confidence intervals, enumerates configurations
compatible with the current partition structure, builds lattices per
configuration, and certifies a matching when all lattices share a single
island.

Verbose diagnostics:
    Set `verbose=True` in `run_until_stop(...)` to enable per-iteration
    gate-failure prints, gate-breakdown summaries, and terminal arm-count
    dumps. The default is silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import random

from gs_lib.gs_tools import (
    Man, Woman, Matching, PreferenceList, GaleShapley,
)
from gs_lib.bt import (
    PairCounts, _canonical, bt_mle_mm, bt_confidence_intervals, bt_ranking,
)
from gs_lib.islands import (
    PartitionStructure, build_partitions_from_cis, enumerate_configurations,
    lattice_from_configuration, construct_islands, Island,
)
from p2etg import SignalProvider


# ============================================================================
# Per-agent state
# ============================================================================

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


# ============================================================================
# PrefLID
# ============================================================================

class PrefLID:
    """Preference Resolution via Lattice Island Detection."""

    def __init__(
        self,
        men: List[Man],
        women: List[Woman],
        true_theta_men: Optional[Dict[Man, Dict[Woman, float]]] = None,
        true_theta_women: Optional[Dict[Woman, Dict[Man, float]]] = None,
        rng: Optional[random.Random] = None,
        constant: float = 0.1,
        budget: int = 100,
        max_lattice_vertices: int = 5000,
        min_samples_per_pair: int = 10,
        min_sample_ratio: float = 0.5,
        provider: Optional[SignalProvider] = None,
        center_policy: str = "random",
        warmup_rounds: int = 0,
    ):
        self.men = list(men)
        self.women = list(women)
        self.true_theta_men = true_theta_men
        self.true_theta_women = true_theta_women
        self.provider = provider
        self.rng = rng or random.Random(0)
        self.constant = constant
        self.budget = budget
        self.max_lattice_vertices = max_lattice_vertices
        self.min_samples_per_pair = min_samples_per_pair
        self.min_sample_ratio = min_sample_ratio
        if center_policy not in ("random", "round_robin"):
            raise ValueError(
                f"center_policy must be 'random' or 'round_robin', "
                f"got {center_policy!r}"
            )
        self.center_policy = center_policy
        self.warmup_rounds = int(warmup_rounds)
        self._rr_queue: List = []

        self.agent_states: Dict[Hashable, AgentState] = {}
        for m in self.men:
            self.agent_states[m] = AgentState(agent=m, partners=self.women)
        for w in self.women:
            self.agent_states[w] = AgentState(agent=w, partners=self.men)

        self.t = 0
        self.stopped = False
        self.committed_matching: Optional[Matching] = None

        self._verbose = False

        self._true_prob_cache: Dict[Tuple, float] = {}
        if (self.true_theta_men is not None
                and self.true_theta_women is not None):
            for c in self.men:
                for i in range(len(self.women)):
                    for j in range(i + 1, len(self.women)):
                        b_i, b_j = self.women[i], self.women[j]
                        key = _canonical(b_i, b_j)
                        self._true_prob_cache[(c, key)] = self._true_prob(c, b_i, b_j)
            for c in self.women:
                for i in range(len(self.men)):
                    for j in range(i + 1, len(self.men)):
                        a_i, a_j = self.men[i], self.men[j]
                        key = _canonical(a_i, a_j)
                        self._true_prob_cache[(c, key)] = self._true_prob(c, a_i, a_j)

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------

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

    def _sample_comparison(self, agent: Hashable, b1: Hashable, b2: Hashable) -> int:
        """One comparison. Returns 1 if canonical-first won.

        If a SignalProvider is attached, the observation is delegated to
        it (x=1 means the canonical-first of (b1, b2) won). Otherwise the
        original true-theta Bernoulli draw is used.
        """
        if self.provider is not None:
            x = self.provider.observe(agent, b1, b2)
            if x not in (0, 1):
                raise ValueError(
                    f"SignalProvider returned {x!r} for ({agent}, {b1}, {b2})"
                )
            return int(x)
        key = _canonical(b1, b2)
        p = self._true_prob_cache[(agent, key)]
        return 1 if self.rng.random() < p else 0

    def observe(self, agent: Hashable, b1: Hashable, b2: Hashable, x: int) -> None:
        self.agent_states[agent].counts.record(b1, b2, x)

    # ------------------------------------------------------------------
    # Estimation
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

    def _pick_center(self, remaining: List) -> Hashable:
        """Choose the next RRT center.

        random (upstream default): iid uniform -> per-agent arm counts
        follow a multinomial fluctuation (the straggler agent can lag
        the mean by ~30% early on, delaying the matching lock).

        round_robin: every pass shuffles the eligible agents and cycles
        through them once, so each agent is centered EXACTLY once per
        pass -> per-agent arm counts grow in lockstep (like P2ETG's
        uniform round-robin), eliminating the straggler.
        """
        if self.center_policy == "random":
            return self.rng.choice(remaining)
        remaining_set = set(remaining)
        self._rr_queue = [a for a in self._rr_queue if a in remaining_set]
        if not self._rr_queue:
            self._rr_queue = list(remaining)
            self.rng.shuffle(self._rr_queue)
        return self._rr_queue.pop(0)

    # ------------------------------------------------------------------
    # RRT
    # ------------------------------------------------------------------

    def _warmup_round(self) -> int:
        """Uniform pass over ALL agents' pairwise arms (one sample each).

        A warm-start for the RRT phase: every agent's counts advance in
        lockstep (like uniform round-robin sampling), so the MLE
        matching stabilises early instead of waiting for each agent's
        turn as a centre. warmup_rounds=0 (default) preserves the
        upstream behaviour exactly.
        """
        n = 0
        for agent, state in self.agent_states.items():
            opponents = self.women if isinstance(agent, Man) else self.men
            for i in range(len(opponents)):
                for j in range(i + 1, len(opponents)):
                    b_i, b_j = opponents[i], opponents[j]
                    x = self._sample_comparison(agent, b_i, b_j)
                    self.observe(agent, b_i, b_j, x)
                    self.t += 1
                    n += 1
        return n

    def rrt_round(self, center: Hashable) -> int:
        if isinstance(center, Man):
            opponents = self.women
        else:
            opponents = self.men

        state = self.agent_states[center]

        pairs: List[Tuple[int, Hashable, Hashable]] = []
        for i in range(len(opponents)):
            for j in range(i + 1, len(opponents)):
                key = _canonical(opponents[i], opponents[j])
                n = state.counts.total.get(key, 0)
                pairs.append((n, opponents[i], opponents[j]))

        pairs.sort(key=lambda x: x[0])

        n = 0
        for _, b_i, b_j in pairs:
            x = self._sample_comparison(center, b_i, b_j)
            self.observe(center, b_i, b_j, x)
            self.t += 1
            n += 1
        return n

    # ------------------------------------------------------------------
    # Partition structure
    # ------------------------------------------------------------------

    def current_partition_structure(self) -> PartitionStructure:
        partitions: Dict[Hashable, List[List[Hashable]]] = {}
        for state in self.agent_states.values():
            partitions[state.agent] = build_partitions_from_cis(
                state.partners,
                state.theta,
                state.ci,
                counts=state.counts.total,
                min_samples_per_cut=max(10, self.min_samples_per_pair),
            )
        return PartitionStructure(partitions=partitions)

    # ------------------------------------------------------------------
    # Sampling sufficiency checks
    # ------------------------------------------------------------------

    def _all_pairs_sampled(self) -> bool:
        for state in self.agent_states.values():
            n_partners = len(state.partners)
            expected = n_partners * (n_partners - 1) // 2
            if len(state.counts.total) < expected:
                return False
        return True

    def _all_pairs_min_samples(self, min_samples: int) -> bool:
        for state in self.agent_states.values():
            n_partners = len(state.partners)
            expected = n_partners * (n_partners - 1) // 2
            if len(state.counts.total) < expected:
                if self._verbose:
                    print(f"  [gate fail] {state.agent}: "
                          f"{len(state.counts.total)}/{expected} arms sampled")
                return False
            for key, n in state.counts.total.items():
                if n < min_samples:
                    if self._verbose:
                        print(f"  [gate fail] {state.agent}: "
                              f"arm {key} has {n} < {min_samples} samples")
                    return False
        return True

    def _all_pairs_sufficiently_sampled(
        self,
        min_samples: int,
        min_ratio: float,
    ) -> bool:
        for state in self.agent_states.values():
            n_partners = len(state.partners)
            expected = n_partners * (n_partners - 1) // 2
            if len(state.counts.total) < expected:
                if self._verbose:
                    print(f"  [gate fail] {state.agent}: "
                          f"{len(state.counts.total)}/{expected} arms sampled")
                return False
            counts = list(state.counts.total.values())
            if not counts:
                return False
            max_count = max(counts)
            min_count = min(counts)
            if max_count < min_samples:
                if self._verbose:
                    print(f"  [gate fail] {state.agent}: "
                          f"max={max_count} < {min_samples}")
                return False
            if min_count < min_ratio * max_count:
                if self._verbose:
                    print(f"  [gate fail] {state.agent}: "
                          f"min={min_count} < {min_ratio}*{max_count}="
                          f"{min_ratio * max_count:.1f}")
                return False
        return True

    # ------------------------------------------------------------------
    # Lattice construction from a configuration
    # ------------------------------------------------------------------

    def build_lattices(
        self,
        configs: List[Dict[Hashable, List[Hashable]]],
    ) -> Optional[List[frozenset]]:
        lattices: List[frozenset] = []
        for config in configs:
            V = lattice_from_configuration(config, self.max_lattice_vertices)
            if V is None:
                return None
            lattices.append(V)
        return lattices

    # ------------------------------------------------------------------
    # Top-level loop
    # ------------------------------------------------------------------

    def run_until_stop(
        self,
        max_iterations: int = 200,
        rounds: Optional[List[Dict]] = None,
        verbose: bool = False,
    ) -> Dict[str, object]:
        if len(self.men) < 1 or len(self.women) < 2:
            raise ValueError(
                "PrefLID requires at least 1 A-side and 2 B-side agents."
            )

        self._verbose = verbose

        U: set = {self.men[0], self.women[0], self.women[1]}
        all_agents = self.men + self.women

        fair_share = self.budget / len(all_agents)

        last_islands: Optional[List[Island]] = None
        last_lattices: Optional[List[frozenset]] = None
        last_omega: Optional[List[Dict]] = None

        gate_counts = {
            "insufficient_data": 0,
            "over_budget": 0,
            "lattice_too_large": 0,
            "ok": 0,
        }

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            remaining = [a for a in all_agents if a not in U]
            if not remaining:
                remaining = all_agents
            center = self._pick_center(remaining)

            n_samples = self.rrt_round(center)

            U.add(center)

            self._refresh_estimates()

            if not self._all_pairs_sufficiently_sampled(
                min_samples=self.min_samples_per_pair,
                min_ratio=self.min_sample_ratio,
            ):
                gate_counts["insufficient_data"] += 1
                if rounds is not None:
                    rounds.append({
                        "t": self.t, "iteration": iteration,
                        "center": str(center), "n_samples": n_samples,
                        "omega_size": None, "n_islands": None,
                        "n_lattices_largest_island": None,
                        "support_max": None, "support_min": None,
                        "support_mean": None,
                        "H_star": None,
                        "H_star_str": None,
                        "status": "insufficient_data",
                    })
                if verbose and iteration % 50 == 0:
                    print(f"[iter {iteration}] t={self.t} center={center} "
                          f"— insufficient data, continuing. "
                          f"U size = {len(U)} / {len(all_agents)}")
                continue

            structure = self.current_partition_structure()
            omega_size = structure.size_of_omega()

            configs = enumerate_configurations(structure, budget=self.budget)

            if configs is None:
                gate_counts["over_budget"] += 1
                if rounds is not None:
                    rounds.append({
                        "t": self.t, "iteration": iteration,
                        "center": str(center), "n_samples": n_samples,
                        "omega_size": omega_size,
                        "n_islands": None,
                        "n_lattices_largest_island": None,
                        "support_max": None, "support_min": None,
                        "support_mean": None,
                        "H_star": None,
                        "H_star_str": None,
                        "status": "over_budget",
                    })
                if verbose:
                    print(f"[iter {iteration}] t={self.t} center={center} "
                          f"|Omega|={omega_size} > budget={self.budget}; "
                          f"refining.")
                continue

            lattices = self.build_lattices(configs)
            if lattices is None:
                gate_counts["lattice_too_large"] += 1
                if rounds is not None:
                    rounds.append({
                        "t": self.t, "iteration": iteration,
                        "center": str(center), "n_samples": n_samples,
                        "omega_size": omega_size,
                        "n_islands": None,
                        "n_lattices_largest_island": None,
                        "support_max": None, "support_min": None,
                        "support_mean": None,
                        "H_star": None,
                        "H_star_str": None,
                        "status": "lattice_too_large",
                    })
                if verbose:
                    print(f"[iter {iteration}] t={self.t} center={center} "
                          f"|Omega|={omega_size}; a lattice exceeded "
                          f"{self.max_lattice_vertices} vertices.")
                continue

            islands = construct_islands(lattices)
            gate_counts["ok"] += 1

            last_islands = islands
            last_lattices = lattices
            last_omega = configs

            largest = max(islands, key=lambda isl: len(isl.lattice_indices))
            if rounds is not None:
                rounds.append({
                    "t": self.t, "iteration": iteration,
                    "center": str(center), "n_samples": n_samples,
                    "omega_size": omega_size,
                    "n_islands": len(islands),
                    "n_lattices_largest_island": len(largest.lattice_indices),
                    "support_max": largest.support_max,
                    "support_min": largest.support_min,
                    "support_mean": largest.support_mean,
                    "H_star": largest.H_star,
                    "H_star_str": str(largest.H_star),
                    "support": largest.support,
                    "status": "ok",
                })
            if verbose:
                print(f"[iter {iteration}] t={self.t} center={center} "
                      f"|Omega|={omega_size} |islands|={len(islands)} "
                      f"H*_support={largest.support}")

            if len(islands) == 1:
                self.stopped = True
                self.committed_matching = islands[0].H_star
                if rounds is not None:
                    single = islands[0]
                    rounds.append({
                        "t": self.t, "iteration": iteration,
                        "center": str(center), "n_samples": n_samples,
                        "omega_size": omega_size,
                        "n_islands": 1,
                        "n_lattices_largest_island": len(single.lattice_indices),
                        "support_max": single.support_max,
                        "support_min": single.support_min,
                        "support_mean": single.support_mean,
                        "H_star": single.H_star,
                        "H_star_str": str(single.H_star),
                        "support": single.support,
                        "status": "committed",
                        "reason": "certified",
                    })

                if verbose:
                    print(f"\n--- Gate breakdown ---")
                    for k, v in gate_counts.items():
                        print(f"  {k:>20}: {v} iterations")
                    print(f"  Total: {iteration} iterations, t={self.t} samples")

                return {
                    "stopped": True,
                    "T_stop": self.t,
                    "matching": islands[0].H_star,
                    "n_islands": 1,
                    "iterations": iteration,
                    "islands": islands,
                    "lattices": lattices,
                    "omega": configs,
                    "reason": "certified",
                    "gate_counts": dict(gate_counts),
                }

            if len(lattices) >= fair_share and verbose:
                print(f"             entrant gate: {len(lattices)} "
                      f">= {fair_share:.1f}.")

        if last_islands is None:
            self._refresh_estimates()
            fallback = self._current_gs_matching()
            self.stopped = False
            self.committed_matching = fallback

            if verbose:
                print(f"\n--- Gate breakdown ---")
                for k, v in gate_counts.items():
                    print(f"  {k:>20}: {v} iterations")
                print(f"  Total: {iteration} iterations, t={self.t} samples")
                self._dump_arm_counts()

            return {
                "stopped": False,
                "T_stop": self.t,
                "matching": fallback,
                "n_islands": None,
                "iterations": iteration,
                "islands": None,
                "lattices": None,
                "omega": None,
                "reason": "insufficient_iterations",
                "gate_counts": dict(gate_counts),
            }

        largest = max(last_islands, key=lambda isl: len(isl.lattice_indices))
        self.stopped = False
        self.committed_matching = largest.H_star

        if verbose:
            print(f"\n--- Gate breakdown ---")
            for k, v in gate_counts.items():
                print(f"  {k:>20}: {v} iterations")
            print(f"  Total: {iteration} iterations, t={self.t} samples")
            self._dump_arm_counts()

        return {
            "stopped": False,
            "T_stop": self.t,
            "matching": largest.H_star,
            "n_islands": len(last_islands),
            "iterations": iteration,
            "islands": last_islands,
            "lattices": last_lattices,
            "omega": last_omega,
            "reason": "max_iterations",
            "gate_counts": dict(gate_counts),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_gs_matching(self) -> Matching:
        prefs_dict: Dict[Hashable, List[Hashable]] = {}
        for state in self.agent_states.values():
            prefs_dict[state.agent] = bt_ranking(state.theta)
        prefs = PreferenceList(prefs_dict)
        return GaleShapley(prefs).find_stable_matching("men")

    def _dump_arm_counts(self) -> None:
        print("\n=== Arm counts at termination ===")
        for state in self.agent_states.values():
            arms = list(state.counts.total.items())
            counts = sorted(n for _, n in arms)
            max_c = max(counts) if counts else 0
            min_c = min(counts) if counts else 0
            ratio = (min_c / max_c) if max_c > 0 else 0.0
            print(f"  {state.agent}: {len(arms)} arms, "
                  f"min={min_c}, max={max_c}, ratio={ratio:.2f}, "
                  f"counts={counts}")