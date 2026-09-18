"""BT calibration tests.

The calibrated eta must give a median-gap comparison a win probability
approximately equal to the configured target.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from llm_matching.providers import RouterBenchBTProvider, build_bt_thetas, calibrate_eta
from llm_matching.utilities import median_positive_gap

import random


def _gap_utility(gap: float) -> pd.DataFrame:
    """Two agents x three partners with controlled gaps.

    Agent a1: partners at 0, gap, 2*gap.
    Agent a2: partners at 0.5, 0.5+gap, 0.5+2*gap.
    Positive gaps: per agent {gap, gap, 2*gap} -> median = gap.
    """
    return pd.DataFrame(
        [
            {"p1": 0.0, "p2": gap, "p3": 2 * gap},
            {"p1": 0.5, "p2": 0.5 + gap, "p3": 0.5 + 2 * gap},
        ],
        index=["a1", "a2"],
    )


@pytest.mark.parametrize("target", [0.60, 0.70, 0.80])
def test_median_gap_induces_target_win_probability(target):
    gap = 0.07
    util = _gap_utility(gap)
    med = median_positive_gap(util, tie_epsilon=1e-9)
    assert med == pytest.approx(gap)

    eta = calibrate_eta(med, target)

    theta = {
        "a1": {p: math.exp(eta * v) for p, v in util.loc["a1"].items()},
        "a2": {p: math.exp(eta * v) for p, v in util.loc["a2"].items()},
    }
    # p2 vs p1 is a median-gap comparison.
    p = theta["a1"]["p2"] / (theta["a1"]["p2"] + theta["a1"]["p1"])
    assert p == pytest.approx(target, abs=1e-9)
    # Symmetric direction.
    p_rev = theta["a1"]["p1"] / (theta["a1"]["p1"] + theta["a1"]["p2"])
    assert p_rev == pytest.approx(1 - target, abs=1e-9)


def test_empirical_win_rate_at_median_gap():
    """Monte Carlo: the BT provider's empirical win rate for a
    median-gap pair approximates the target probability."""
    gap = 0.05
    util = _gap_utility(gap)
    eta = calibrate_eta(gap, 0.70)

    theta_table = {
        "a1": {p: math.exp(eta * float(v)) for p, v in util.loc["a1"].items()}
    }
    provider = RouterBenchBTProvider(
        theta_table, {}, rng=random.Random(123)
    )

    from gs_lib.gs_tools import Man

    agent = Man("a1")  # agent id lookup hits theta_table (model side)
    b1, b2 = Man("p1"), Man("p2")
    n = 40000
    wins = 0
    for _ in range(n):
        x = provider.observe(agent, b1, b2)
        from gs_lib.bt import _canonical

        c1, _ = _canonical(b1, b2)
        p2_won = (x == 1) == (c1.id == "p2")
        wins += p2_won
    assert wins / n == pytest.approx(0.70, abs=0.015)


def test_eta_monotone_in_target():
    gap = 0.1
    etas = [calibrate_eta(gap, t) for t in (0.6, 0.7, 0.8)]
    assert etas[0] < etas[1] < etas[2]


def test_eta_inverse_in_gap():
    """Same target, larger gap -> smaller eta."""
    small = calibrate_eta(0.05, 0.7)
    large = calibrate_eta(0.20, 0.7)
    assert large < small


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        calibrate_eta(0.1, 0.5)
    with pytest.raises(ValueError):
        calibrate_eta(0.1, 1.0)
    with pytest.raises(ValueError):
        calibrate_eta(0.0, 0.7)


def test_build_bt_thetas_exponential():
    task_util = pd.DataFrame(
        [[0.0, 0.1]], index=["t1"], columns=["mA", "mB"]
    )
    model_util = pd.DataFrame(
        [[0.1, 0.0]], index=["mA"], columns=["t1", "t2"]
    )
    theta_task, theta_model = build_bt_thetas(task_util, model_util, 2.0, 3.0)
    assert theta_task["t1"]["mB"] == pytest.approx(math.exp(0.2))
    assert theta_model["mA"]["t1"] == pytest.approx(math.exp(0.3))
