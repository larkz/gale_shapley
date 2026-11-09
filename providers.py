"""Built-in signal providers for P2ETG / PrefLID.

BradleyTerryProvider draws a Bernoulli comparison from fixed ground-truth
Bradley-Terry weights:  P(b1 > b2 | agent) = theta[b1] / (theta[b1] + theta[b2]).
It follows the provider convention of p2etg.SignalProvider: the return
value is 1 iff the canonical-first element of (b1, b2) wins.
"""

from __future__ import annotations

import random
from typing import Dict, Hashable, Optional

from gs_lib.bt import _canonical
from gs_lib.gs_tools import Man
from p2etg import SignalProvider


class BradleyTerryProvider(SignalProvider):
    def __init__(self, theta_men: Dict, theta_women: Dict,
                 rng: Optional[random.Random] = None):
        self.theta_men = theta_men
        self.theta_women = theta_women
        self.rng = rng or random.Random(0)

    def observe(self, agent: Hashable, b1: Hashable, b2: Hashable) -> int:
        theta = (self.theta_men[agent] if isinstance(agent, Man)
                 else self.theta_women[agent])
        first, second = _canonical(b1, b2)
        p = theta[first] / (theta[first] + theta[second])
        return 1 if self.rng.random() < p else 0
