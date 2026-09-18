"""matching_id.py

Exact Matching-ID: a certified, assignment-aware stopping rule.

Concept
-------
At time t, the learner's pairwise confidence intervals induce, for each
agent, a PARTIAL order over partners:

    i -> j  iff the current CI certifies  i preferred to j
             (CI of the pair lies strictly above or below 1/2).

Let Omega_t be the set of strict total preference profiles that are
linear extensions of these partial orders (one ranking per agent).
Let

    H_t = { GS_men(profile) : profile in Omega_t }.

Certified Matching-ID stopping condition:

    |H_t| = 1.

If every still-plausible preference profile yields the same
model-proposing GS matching, the algorithm commits to that matching
WITHOUT resolving every pairwise preference.

Correctness caveat: certification is sound only insofar as the CIs are
valid (contain the true win probability). This is the same assumption
P2ETG's full-preference stopping makes.

Search
------
- Per-agent linear extensions are enumerated with a backtracking
  topological-order search (`enumerate_linear_extensions`), NOT by
  permuting unresolved CI blocks (which could violate certified
  non-adjacent relations).
- Profiles are the Cartesian product of per-agent extensions,
  generated lazily; the search terminates as soon as TWO distinct GS
  matchings are found (NOT_IDENTIFIED) or the product is exhausted
  (IDENTIFIED if a single matching was seen).
- max_profiles / timeout_seconds caps are respected; an incomplete
  search NEVER counts as a positive certificate.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from gs_lib.gs_tools import (
    GaleShapley, Man, Matching, PreferenceList, Woman,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Result type
# ============================================================================

@dataclass
class MatchingIDResult:
    identified: bool
    matching: Optional[Matching]
    n_distinct_matchings_seen: int
    profiles_checked: int
    search_complete: bool
    hit_search_cap: bool
    reason: Optional[str] = None
    elapsed_seconds: float = 0.0


# ============================================================================
# Partial orders from confidence intervals
# ============================================================================

def certified_edges(state) -> List[Tuple[object, object]]:
    """Edges (i, j) meaning 'i is certified preferred to j' for one agent.

    state.ci maps canonical pair keys to (lo, hi) intervals of the win
    probability of the canonical-first partner.
    """
    edges: List[Tuple[object, object]] = []
    for (c1, c2), (lo, hi) in state.ci.items():
        if lo > 0.5:
            edges.append((c1, c2))
        elif hi < 0.5:
            edges.append((c2, c1))
    return edges


def has_cycle(partners: List, edges: List[Tuple[object, object]]) -> bool:
    """Kahn's algorithm cycle check on the certified preference DAG."""
    indeg = {p: 0 for p in partners}
    adj: Dict[object, List[object]] = {p: [] for p in partners}
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    queue = [p for p in partners if indeg[p] == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return seen != len(partners)


def enumerate_linear_extensions(
    partners: List,
    edges: List[Tuple[object, object]],
    cap: Optional[int] = None,
) -> Iterator[List]:
    """Yield all strict total orders consistent with the partial order.

    Backtracking topological enumeration. Candidate selection is
    deterministic (sorted by partner id) for reproducibility.
    """
    preds: Dict[object, Set] = {p: set() for p in partners}
    for a, b in edges:
        preds[b].add(a)

    n = len(partners)
    placed: List = []
    placed_set: Set = set()
    count = 0

    def _key(p) -> str:
        return str(getattr(p, "id", p))

    def _rec() -> Iterator[List]:
        nonlocal count
        if len(placed) == n:
            count += 1
            yield list(placed)
            return
        candidates = [
            p for p in partners
            if p not in placed_set and preds[p] <= placed_set
        ]
        for p in sorted(candidates, key=_key):
            placed.append(p)
            placed_set.add(p)
            yield from _rec()
            placed.pop()
            placed_set.discard(p)
            if cap is not None and count >= cap:
                return

    yield from _rec()


# ============================================================================
# Certification
# ============================================================================

def _matching_signature(matching: Matching) -> frozenset:
    return frozenset((p.man.id, p.woman.id) for p in matching.pairs)


def certify_matching(
    agent_states: Dict,
    men: List[Man],
    women: List[Woman],
    max_profiles: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    per_agent_extension_cap: int = 100_000,
) -> MatchingIDResult:
    """Decide whether all feasible profiles yield a single GS matching.

    agent_states: P2ETG-style mapping agent -> state with .ci and
    .partners (the CI dict may be a subset of pairs; unsampled pairs
    are simply unresolved).
    """
    t0 = time.monotonic()

    def _elapsed() -> float:
        return time.monotonic() - t0

    # ---- per-agent linear extensions ----
    extensions: Dict = {}
    for agent, state in agent_states.items():
        edges = certified_edges(state)
        partners = list(state.partners)
        if has_cycle(partners, edges):
            return MatchingIDResult(
                identified=False, matching=None,
                n_distinct_matchings_seen=0, profiles_checked=0,
                search_complete=False, hit_search_cap=False,
                reason="certified relations contain a cycle",
                elapsed_seconds=_elapsed(),
            )
        ext = list(
            enumerate_linear_extensions(partners, edges, cap=per_agent_extension_cap)
        )
        if not ext:
            return MatchingIDResult(
                identified=False, matching=None,
                n_distinct_matchings_seen=0, profiles_checked=0,
                search_complete=False, hit_search_cap=False,
                reason="no linear extension (empty)",
                elapsed_seconds=_elapsed(),
            )
        extensions[agent] = ext

    # Heuristic fast path for the NOT-identified verdict: test two
    # maximally diverse profiles first (first extension for every agent
    # vs last extension for every agent).
    diverse_pairs = []
    agents = list(extensions.keys())
    first_profile = [extensions[a][0] for a in agents]
    last_profile = [extensions[a][-1] for a in agents]
    if first_profile != last_profile:
        diverse_pairs.append(first_profile)
        diverse_pairs.append(last_profile)

    seen_signatures: Set = set()
    first_matching: Optional[Matching] = None
    profiles_checked = 0
    hit_cap = False
    cap_reason: Optional[str] = None

    def _check_profile(rankings: List) -> bool:
        """Run GS on one profile; returns True iff a second distinct
        matching was found (search can stop)."""
        nonlocal first_matching, profiles_checked
        prefs = {}
        for agent, ranking in zip(agents, rankings):
            prefs[agent] = list(ranking)
        matching = GaleShapley(PreferenceList(prefs)).find_stable_matching("men")
        profiles_checked += 1
        sig = _matching_signature(matching)
        if first_matching is None:
            first_matching = matching
            seen_signatures.add(sig)
            return False
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            return True
        return False

    # Heuristic probes
    for profile in diverse_pairs:
        if _check_profile(profile):
            return MatchingIDResult(
                identified=False, matching=None,
                n_distinct_matchings_seen=len(seen_signatures),
                profiles_checked=profiles_checked,
                search_complete=True, hit_search_cap=False,
                reason="two feasible profiles yield different matchings",
                elapsed_seconds=_elapsed(),
            )
        if timeout_seconds is not None and _elapsed() > timeout_seconds:
            return MatchingIDResult(
                identified=False, matching=first_matching,
                n_distinct_matchings_seen=len(seen_signatures),
                profiles_checked=profiles_checked,
                search_complete=False, hit_search_cap=True,
                reason="timeout during probe profiles",
                elapsed_seconds=_elapsed(),
            )

    # Full Cartesian product (lazy generator; no profile list retained)
    product = itertools.product(*(extensions[a] for a in agents))
    exhausted = True
    for rankings in product:
        if _check_profile(list(rankings)):
            return MatchingIDResult(
                identified=False, matching=None,
                n_distinct_matchings_seen=len(seen_signatures),
                profiles_checked=profiles_checked,
                search_complete=True, hit_search_cap=False,
                reason="two feasible profiles yield different matchings",
                elapsed_seconds=_elapsed(),
            )
        if max_profiles is not None and profiles_checked >= max_profiles:
            hit_cap = True
            cap_reason = "max_profiles reached"
            exhausted = False
            break
        if timeout_seconds is not None and _elapsed() > timeout_seconds:
            hit_cap = True
            cap_reason = "timeout"
            exhausted = False
            break
    else:
        exhausted = True

    if not exhausted:
        return MatchingIDResult(
            identified=False, matching=first_matching,
            n_distinct_matchings_seen=len(seen_signatures),
            profiles_checked=profiles_checked,
            search_complete=False, hit_search_cap=True,
            reason=cap_reason,
            elapsed_seconds=_elapsed(),
        )

    # Exhausted the product: certification iff exactly one matching.
    identified = len(seen_signatures) == 1
    return MatchingIDResult(
        identified=identified,
        matching=first_matching if identified else None,
        n_distinct_matchings_seen=len(seen_signatures),
        profiles_checked=profiles_checked,
        search_complete=True,
        hit_search_cap=False,
        reason=None if identified else "multiple matchings remain possible",
        elapsed_seconds=_elapsed(),
    )


# ============================================================================
# P2ETG wrapper: identical sampling, certified stopping
# ============================================================================

class MatchingIDP2ETG:
    """P2ETG with the stopping condition replaced by exact Matching-ID.

    Sampling, estimation, and CI construction are INHERITED UNCHANGED
    from P2ETG (composition around the P2ETG class); the only
    behavioural difference is the stopping certificate:

        stop iff certify_matching(...).identified

    A signature-keyed cache skips re-certification at checks where no
    certified relation changed.
    """

    def __init__(
        self,
        men: List[Man],
        women: List[Woman],
        provider,
        rng=None,
        constant: float = 0.1,
        max_profiles: int = 200_000,
        timeout_seconds: float = 5.0,
        certify_every: int = 1,
    ):
        from p2etg import P2ETG

        self._p2etg = P2ETG(
            men=men, women=women, provider=provider, rng=rng, constant=constant
        )
        self.max_profiles = max_profiles
        self.timeout_seconds = timeout_seconds
        self.certify_every = certify_every

        self.last_certification: Optional[MatchingIDResult] = None
        self._cert_cache_key = None
        self._cert_cache_result: Optional[bool] = None
        self._check_counter = 0
        self.certify_seconds_total = 0.0
        self.n_certifications = 0
        self.certify_profiles_total = 0

    # ---------------- P2ETG passthrough attributes ----------------

    def __getattr__(self, name):
        # Guard against recursion before _p2etg is assigned.
        if name.startswith("_") and name == "_p2etg":
            raise AttributeError(name)
        return getattr(self.__dict__.get("_p2etg", None), name)

    @property
    def agent_states(self):
        return self._p2etg.agent_states

    @property
    def men(self):
        return self._p2etg.men

    @property
    def women(self):
        return self._p2etg.women

    @property
    def t(self):
        return self._p2etg.t

    @property
    def stopped(self):
        return self._p2etg.stopped

    def committed_matching(self):
        return self._p2etg.committed_matching()

    def estimated_preferences(self, agent):
        return self._p2etg.estimated_preferences(agent)

    def current_matching(self):
        return self._p2etg.current_matching()

    # ---------------- certification ----------------

    def _cert_signature(self):
        """Signature of the certified-relation set (edge set per agent)."""
        parts = []
        for agent, state in self._p2etg.agent_states.items():
            edges = certified_edges(state)
            sig = tuple(
                sorted(
                    (a.id, b.id) for a, b in edges
                )
            )
            parts.append((str(agent.id), sig))
        return tuple(sorted(parts))

    def _pairwise_disjoint(self) -> bool:  # replaces P2ETG's stopping check
        self._check_counter += 1
        if self.certify_every > 1 and self._check_counter % self.certify_every != 0:
            # Not a certification check: never stop here.
            return False

        key = self._cert_signature()
        if key == self._cert_cache_key and self._cert_cache_result is not None:
            return self._cert_cache_result

        t0 = time.monotonic()
        result = certify_matching(
            self._p2etg.agent_states,
            self._p2etg.men,
            self._p2etg.women,
            max_profiles=self.max_profiles,
            timeout_seconds=self.timeout_seconds,
        )
        self.certify_seconds_total += time.monotonic() - t0
        self.n_certifications += 1
        self.certify_profiles_total += result.profiles_checked
        self.last_certification = result
        self._cert_cache_key = key
        self._cert_cache_result = result.identified
        return result.identified

    # ---------------- top-level run ----------------

    def run_until_stop(
        self,
        max_epochs: int = 400,
        adaptive: bool = False,
        check_every: int = 500,
        max_samples: int = 2_000_000_000_000,
        verbose: bool = False,
        rounds=None,
    ):
        """Run P2ETG's loop with the certification-based stopping rule.

        Returns the P2ETG result dict, with:
          * 'matching' replaced by the CERTIFIED matching when certified;
          * 'certification' = MatchingIDResult of the last check.
        """
        p = self._p2etg
        # Temporarily patch the stopping predicate.
        original = p._pairwise_disjoint
        p._pairwise_disjoint = self._pairwise_disjoint
        try:
            result = p.run_until_stop(
                max_epochs=max_epochs,
                adaptive=adaptive,
                check_every=check_every,
                max_samples=max_samples,
                verbose=verbose,
                rounds=rounds,
            )
        finally:
            p._pairwise_disjoint = original

        if result.get("stopped") and self.last_certification is not None:
            certified = self.last_certification.matching
            if certified is not None:
                result["matching"] = certified
                self.certified_matching = certified
        result["certification"] = self.last_certification
        result["certify_seconds_total"] = self.certify_seconds_total
        result["n_certifications"] = self.n_certifications
        return result
