"""Tests for the BT probability floor (no-tie reward guarantee)."""

from __future__ import annotations

import random

from gs_lib.bt import _canonical
from llm_matching.providers import RouterBenchBTProvider


class _Id:
    """Stable identity: hash/equality by id string (NOT by address),
    so the canonical order cannot flip between observe() calls."""

    def __init__(self, s):
        self.id = s

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, _Id) and self.id == other.id


def _tables():
    # agent d compares models a/b/c: strong gap a>b, NEAR-TIE b~c
    theta_task = {"d": {"a": 1.0, "b": 0.6, "c": 0.5999999}}
    # model a compares tasks t1/t2 with an exact-ish tie (t2 hair ahead)
    theta_model = {
        "a": {"t1": 1.0, "t2": 1.0000001},
        "b": {"t1": 1.0, "t2": 0.5},
        "c": {"t1": 1.0, "t2": 0.5},
    }
    return theta_task, theta_model


def _sample_mean(provider, agent, x, y, n=4000):
    ones = sum(provider.observe(agent, x, y) for _ in range(n))
    return ones / n


def _expected_floored(theta, x, y, floor):
    """Expected P(x=1) with the floor, honouring the canonical order."""
    c1, c2 = _canonical(x, y)
    t1, t2 = theta[c1.id], theta[c2.id]
    p = t1 / (t1 + t2)
    if p > 0.5:
        return max(p, 0.5 + floor)
    if p < 0.5:
        return min(p, 0.5 - floor)
    return (0.5 + floor) if t1 >= t2 else (0.5 - floor)


def test_probability_floor_clamps_near_ties():
    theta_task, theta_model = _tables()
    d, a, b, c = _Id("d"), _Id("a"), _Id("b"), _Id("c")
    provider = RouterBenchBTProvider(
        theta_task, theta_model, rng=random.Random(0), probability_floor=0.1
    )
    # near-tie b vs c: floored to 0.5 ± 0.1 in the canonical-first
    # direction (whichever it is in this process)
    expected = _expected_floored(theta_task["d"], b, c, 0.1)
    assert min(abs(expected - 0.4), abs(expected - 0.6)) < 1e-9, expected
    p_hat = _sample_mean(provider, d, b, c)
    assert abs(p_hat - expected) < 0.03, (p_hat, expected)

    # strong signal a vs b: p = 1.0/1.6 = 0.625 (canonical (a,b)) — the
    # floor never touches |p-0.5| >= 0.1 signals
    expected = _expected_floored(theta_task["d"], a, b, 0.1)
    assert min(abs(expected - 0.625), abs(expected - 0.375)) < 1e-9, expected
    p_hat = _sample_mean(provider, d, a, b)
    assert abs(p_hat - expected) < 0.03, (p_hat, expected)

    # reversed call direction (c vs a) gives the complement
    expected = _expected_floored(theta_task["d"], c, a, 0.1)
    p_hat = _sample_mean(provider, d, c, a)
    assert abs(p_hat - expected) < 0.03, (p_hat, expected)


def test_probability_floor_exact_tie_uses_theta_direction():
    theta_task, theta_model = _tables()
    m, t1, t2 = _Id("a"), _Id("t1"), _Id("t2")
    provider = RouterBenchBTProvider(
        theta_task, theta_model, rng=random.Random(0), probability_floor=0.05
    )
    # t1 vs t2 for model a: essentially tied thetas; floored to 0.5±0.05
    # in the canonical-first direction
    expected = _expected_floored(theta_model["a"], t1, t2, 0.05)
    assert min(abs(expected - 0.45), abs(expected - 0.55)) < 1e-9, expected
    p_hat = _sample_mean(provider, m, t1, t2)
    assert abs(p_hat - expected) < 0.03, (p_hat, expected)


def test_no_floor_preserves_original_behaviour():
    theta_task, theta_model = _tables()
    d, b, c = _Id("d"), _Id("b"), _Id("c")
    provider = RouterBenchBTProvider(
        theta_task, theta_model, rng=random.Random(0)
    )
    p_hat = _sample_mean(provider, d, b, c)
    assert abs(p_hat - 0.5) < 0.03, p_hat  # near-tie stays a coin flip
