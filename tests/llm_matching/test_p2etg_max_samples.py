"""Regression tests: P2ETG adaptive mode must respect max_samples exactly."""

from __future__ import annotations

import random

from gs_lib.gs_tools import Man, Woman
from p2etg import P2ETG


class CoinProvider:
    """Every comparison is a fair coin; CIs never resolve."""

    def observe(self, agent, b1, b2) -> int:
        return 1 if self.rng.random() < 0.5 else 0

    def warmup(self, agent, partners) -> None:
        return None

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng


def _market(n: int = 4):
    men = [Man(f"m{i}") for i in range(n)]
    women = [Woman(f"w{i}") for i in range(n)]
    return men, women


def test_adaptive_never_overshoots_max_samples():
    men, women = _market()
    provider = CoinProvider(random.Random(0))
    learner = P2ETG(men, women, provider, rng=random.Random(0))
    result = learner.run_until_stop(
        adaptive=True, check_every=7, max_samples=50, verbose=False
    )
    assert learner.t == 50
    assert result["T_stop"] == 50
    assert result["stopped"] is False


def test_adaptive_partial_final_batch():
    """max_samples not a multiple of check_every -> exact stop."""
    men, women = _market()
    provider = CoinProvider(random.Random(1))
    learner = P2ETG(men, women, provider, rng=random.Random(1))
    learner.run_until_stop(
        adaptive=True, check_every=13, max_samples=40, verbose=False
    )
    assert learner.t == 40


def test_adaptive_multiple_of_check_every_unchanged():
    men, women = _market()
    provider = CoinProvider(random.Random(2))
    learner = P2ETG(men, women, provider, rng=random.Random(2))
    learner.run_until_stop(
        adaptive=True, check_every=10, max_samples=40, verbose=False
    )
    assert learner.t == 40
    # 4 full batches, no partial batch
