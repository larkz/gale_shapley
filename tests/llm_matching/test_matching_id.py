"""Exact Matching-ID correctness tests (section 28 of the Phase-2 spec).

Cases:
  1. Fully resolved preferences -> exactly the GS matching.
  2. Completely unresolved 2x2 with multiple possible GS matchings
     -> identified = False.
  3. Partially resolved market where several profiles remain but all
     produce the same GS matching -> identified = True (KEY case).
  4. Two feasible profiles yielding different GS matchings
     -> identified = False.
  5. Search timeout / cap -> never incorrectly certifies.
"""

from __future__ import annotations

import random

import pytest
from gs_lib.bt import _canonical
from gs_lib.gs_tools import GaleShapley, Man, Matching, PreferenceList, Woman
from p2etg import AgentState, P2ETG

from llm_matching.matching_id import (
    MatchingIDP2ETG,
    certify_matching,
    enumerate_linear_extensions,
)


class CoinProvider:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def observe(self, agent, b1, b2) -> int:
        return 1 if self.rng.random() < 0.5 else 0

    def warmup(self, agent, partners) -> None:
        return None


def make_state(agent, partners, ci_pairs: dict) -> AgentState:
    """AgentState with handcrafted CIs.

    ci_pairs: {(p1, p2): 'p1' | 'p2'} — which partner is certified
    preferred. Entries are stored under the canonical key with the
    interval on the correct side of 1/2.
    """
    state = AgentState(agent=agent, partners=list(partners))
    for (p1, p2), winner in ci_pairs.items():
        key = _canonical(p1, p2)
        if winner is p1:
            interval = (0.9, 1.0) if key[0] is p1 else (0.0, 0.1)
        else:
            interval = (0.9, 1.0) if key[0] is p2 else (0.0, 0.1)
        state.ci[key] = interval
        # ensure counts exist so the state is realistic
        state.counts.total[key] = 100
        state.counts.wins[key] = 95 if interval[0] > 0.5 else 5
    return state


def gs(prefs: PreferenceList) -> Matching:
    return GaleShapley(prefs).find_stable_matching("men")


def test_linear_extensions_basic():
    a, b, c = "a", "b", "c"
    ext = list(enumerate_linear_extensions([a, b, c], [(a, c)]))
    # a before c; b anywhere
    assert [a, b, c] in ext
    assert [b, a, c] in ext
    assert [a, c, b] in ext
    assert len(ext) == 3
    for order in ext:
        assert order.index(a) < order.index(c)


def test_linear_extensions_chain_unique():
    ext = list(enumerate_linear_extensions(["x", "y", "z"], [("x", "y"), ("y", "z")]))
    assert ext == [["x", "y", "z"]]


def test_linear_extensions_cap():
    ext = list(enumerate_linear_extensions(list("abcd"), [], cap=5))
    assert len(ext) == 5


# ---------------------------------------------------------------------------
# Case 1: fully resolved
# ---------------------------------------------------------------------------

def test_fully_resolved_returns_gs_matching():
    m1, m2 = Man("m1"), Man("m2")
    w1, w2 = Woman("w1"), Woman("w2")
    men, women = [m1, m2], [w1, w2]

    prefs = PreferenceList(
        {
            m1: [w1, w2], m2: [w2, w1],
            w1: [m2, m1], w2: [m1, m2],
        }
    )
    expected = gs(prefs)

    states = {
        m1: make_state(m1, women, {(w1, w2): w1}),
        m2: make_state(m2, women, {(w1, w2): w2}),
        w1: make_state(w1, men, {(m1, m2): m2}),
        w2: make_state(w2, men, {(m1, m2): m1}),
    }
    result = certify_matching(states, men, women)
    assert result.identified is True
    assert result.search_complete is True
    assert result.profiles_checked == 1
    assert frozenset((p.man.id, p.woman.id) for p in result.matching.pairs) == \
        frozenset((p.man.id, p.woman.id) for p in expected.pairs)


# ---------------------------------------------------------------------------
# Case 2: completely unresolved 2x2, multiple GS matchings
# ---------------------------------------------------------------------------

def test_completely_unresolved_not_identified():
    m1, m2 = Man("m1"), Man("m2")
    w1, w2 = Woman("w1"), Woman("w2")
    men, women = [m1, m2], [w1, w2]

    states = {
        m1: make_state(m1, women, {}),
        m2: make_state(m2, women, {}),
        w1: make_state(w1, men, {}),
        w2: make_state(w2, men, {}),
    }
    result = certify_matching(states, men, women)
    assert result.identified is False
    assert result.search_complete is True
    assert result.n_distinct_matchings_seen >= 2


# ---------------------------------------------------------------------------
# Case 3: partially resolved, all profiles agree  (KEY CASE)
# ---------------------------------------------------------------------------

def test_partially_resolved_single_gs_identified():
    """Several feasible profiles remain, but all give the same
    model-proposing GS matching: certification must fire."""
    m1, m2 = Man("m1"), Man("m2")
    w1, w2 = Woman("w1"), Woman("w2")
    men, women = [m1, m2], [w1, w2]

    # Certified: m1: w1>w2; m2: w1>w2; w1: m1>m2; w2: UNRESOLVED.
    # Whatever w2's ranking, men-proposing GS yields (m1-w1, m2-w2):
    #   m1 proposes w1 (accepted); m2 proposes w1 (rejected, w1 prefers
    #   m1), then m2 proposes w2 (accepted regardless of w2's ranking).
    states = {
        m1: make_state(m1, women, {(w1, w2): w1}),
        m2: make_state(m2, women, {(w1, w2): w1}),
        w1: make_state(w1, men, {(m1, m2): m1}),
        w2: make_state(w2, men, {}),
    }
    result = certify_matching(states, men, women)
    assert result.identified is True
    assert result.search_complete is True
    # 2 heuristic probe profiles + the 2-profile Cartesian product.
    assert result.profiles_checked == 4
    got = {(p.man.id, p.woman.id) for p in result.matching.pairs}
    assert got == {("m1", "w1"), ("m2", "w2")}


# ---------------------------------------------------------------------------
# Case 4: two feasible profiles, different GS matchings
# ---------------------------------------------------------------------------

def test_two_profiles_different_matchings_not_identified():
    m1, m2 = Man("m1"), Man("m2")
    w1, w2 = Woman("w1"), Woman("w2")
    men, women = [m1, m2], [w1, w2]

    # Only m1's ranking resolved (w1 > w2); everyone else free.
    # w1's ranking decides which man ends up with w1 -> different
    # matchings are possible.
    states = {
        m1: make_state(m1, women, {(w1, w2): w1}),
        m2: make_state(m2, women, {}),
        w1: make_state(w1, men, {}),
        w2: make_state(w2, men, {}),
    }
    result = certify_matching(states, men, women)
    assert result.identified is False
    assert result.search_complete is True
    assert result.reason == "two feasible profiles yield different matchings"


def test_inconsistent_certified_relations_cycle():
    """Cyclic certified edges cannot yield any linear extension."""
    m1, m2, m3 = Man("m1"), Man("m2"), Man("m3")
    w1, w2, w3 = Woman("w1"), Woman("w2"), Woman("w3")
    men, women = [m1, m2, m3], [w1, w2, w3]

    # w1 certified: m1>m2, m2>m3, m3>m1  (a cycle)
    states = {
        m1: make_state(m1, women, {}),
        m2: make_state(m2, women, {}),
        m3: make_state(m3, women, {}),
        w1: make_state(w1, men, {(m1, m2): m1, (m2, m3): m2, (m3, m1): m3}),
        w2: make_state(w2, men, {}),
        w3: make_state(w3, men, {}),
    }
    result = certify_matching(states, men, women)
    assert result.identified is False
    assert result.search_complete is False
    assert "cycle" in (result.reason or "")


# ---------------------------------------------------------------------------
# Case 5: timeout / cap never certifies
# ---------------------------------------------------------------------------

def test_max_profiles_cap_never_certifies():
    m1, m2 = Man("m1"), Man("m2")
    w1, w2 = Woman("w1"), Woman("w2")
    men, women = [m1, m2], [w1, w2]
    states = {
        m1: make_state(m1, women, {}),
        m2: make_state(m2, women, {}),
        w1: make_state(w1, men, {}),
        w2: make_state(w2, men, {}),
    }
    # Cap of 1 profile: search incomplete even though the first two
    # profiles likely already differ — force cap by giving a market
    # where the first profile pair agrees? With cap=1 the loop checks
    # 1 profile then stops: never a positive certificate.
    result = certify_matching(states, men, women, max_profiles=1)
    assert result.identified is False
    assert result.search_complete is False
    assert result.hit_search_cap is True


def test_timeout_never_certifies():
    m1, m2 = Man("m1"), Man("m2")
    w1, w2 = Woman("w1"), Woman("w2")
    men, women = [m1, m2], [w1, w2]
    states = {
        m1: make_state(m1, women, {(w1, w2): w1}),
        m2: make_state(m2, women, {(w1, w2): w1}),
        w1: make_state(w1, men, {(m1, m2): m1}),
        w2: make_state(w2, men, {}),
    }
    # Timeout of 0 seconds: probe phase aborts immediately.
    result = certify_matching(states, men, women, timeout_seconds=0.0)
    assert result.identified is False
    assert result.search_complete is False


# ---------------------------------------------------------------------------
# Wrapper: certified stopping fires before full resolution
# ---------------------------------------------------------------------------

def test_matching_id_p2etg_stops_early_on_unambiguous_market():
    """A market where the matching is fixed by a few strong preferences:
    Matching-ID must stop before every CI resolves."""
    m1, m2 = Man("m1"), Man("m2")
    w1, w2 = Woman("w1"), Woman("w2")
    men, women = [m1, m2], [w1, w2]

    class StrongProvider:
        """Deterministic preferences: m1: w1>w2, m2: w1>w2,
        w1: m1>m2. w2's preference (m2 vs m1) returns STRICTLY
        ALTERNATING outcomes, so its p_hat stays exactly 0.5 forever
        (CI never excludes 1/2) — yet it cannot change the GS outcome:
        m1 -> w1 (accepted), m2 -> w1 (rejected), m2 -> w2 (accepted
        regardless of w2's ranking)."""

        def __init__(self) -> None:
            self.rng = random.Random(0)
            self._flip_counts = {}

        def observe(self, agent, b1, b2):
            a = agent.id

            def beats(x, y):
                if a in ("m1", "m2"):
                    return x == "w1" and y == "w2"
                if a == "w1":
                    return x == "m1" and y == "m2"
                return None  # w2: alternating, never resolves

            c1, c2 = _canonical(b1, b2)
            r1 = beats(c1.id, c2.id)
            r2 = beats(c2.id, c1.id)
            if r1:
                return 1
            if r2:
                return 0
            key = (a, frozenset((b1.id, b2.id)))
            n = self._flip_counts.get(key, 0)
            self._flip_counts[key] = n + 1
            return n % 2  # alternating 0,1,0,1 -> p_hat == 0.5 exactly

        def warmup(self, agent, partners):
            return None

        def warmup(self, agent, partners):
            return None

    wrapper = MatchingIDP2ETG(
        men, women, StrongProvider(), rng=random.Random(0),
        max_profiles=10_000, timeout_seconds=10.0,
    )
    result = wrapper.run_until_stop(
        adaptive=True, check_every=4, max_samples=2000, verbose=False
    )

    assert result["stopped"] is True, "Matching-ID should certify early"
    cert = result["certification"]
    assert cert is not None and cert.identified is True

    # Not every arm must be resolved at certification.
    n_resolved = sum(
        1
        for state in wrapper.agent_states.values()
        for lo, hi in state.ci.values()
        if lo > 0.5 or hi < 0.5
    )
    total_arms = sum(
        len(s.partners) * (len(s.partners) - 1) // 2
        for s in wrapper.agent_states.values()
    )
    assert n_resolved < total_arms, (
        "certification should occur before full preference resolution"
    )

    got = {(p.man.id, p.woman.id) for p in result["matching"].pairs}
    assert got == {("m1", "w1"), ("m2", "w2")}

    # P2ETG with the same provider and budget: full-resolution stopping
    # cannot fire as early (coin-flip arms never resolve).
    plain = P2ETG(men, women, StrongProvider(), rng=random.Random(0))
    plain_result = plain.run_until_stop(
        adaptive=True, check_every=4, max_samples=2000, verbose=False
    )
    assert plain_result["stopped"] is False
    assert result["T_stop"] <= plain_result["T_stop"]
