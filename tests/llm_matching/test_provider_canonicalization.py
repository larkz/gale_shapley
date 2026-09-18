"""Provider canonicalisation tests.

Critical property: observe() returns 1 iff the CANONICAL-FIRST element
of (b1, b2) wins, regardless of the order in which the caller passes
the partners. P2ETG stores wins relative to _canonical order, so the
provider must mirror that convention exactly.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest
from gs_lib.bt import _canonical
from gs_lib.gs_tools import Man, Woman

from llm_matching.providers import (
    RouterBenchBTProvider,
    RouterBenchReplayProvider,
    build_bt_thetas,
    calibrate_eta,
)

from conftest import make_records

TASKS = ["taskX", "taskY"]
MODELS = ["modelA", "modelB", "modelC"]


def _bt_provider(rng: random.Random) -> RouterBenchBTProvider:
    theta_task = {
        "taskX": {"modelA": 100.0, "modelB": 1.0, "modelC": 1.0},
        "taskY": {"modelA": 1.0, "modelB": 100.0, "modelC": 1.0},
    }
    theta_model = {
        "modelA": {"taskX": 100.0, "taskY": 1.0},
        "modelB": {"taskX": 1.0, "taskY": 100.0},
        "modelC": {"taskX": 1.0, "taskY": 1.0},
    }
    return RouterBenchBTProvider(theta_task, theta_model, rng)


def test_bt_provider_canonical_orientation():
    """Extreme thetas make the winner deterministic. observe() must
    return 1 iff the canonical-first element is that winner, no matter
    the argument order."""
    provider = _bt_provider(random.Random(0))
    task = Woman("taskX")  # modelA >> modelB == modelC

    for b1, b2 in [(Man("modelA"), Man("modelB")),
                   (Man("modelB"), Man("modelA")),
                   (Man("modelA"), Man("modelC")),
                   (Man("modelC"), Man("modelA"))]:
        x = provider.observe(task, b1, b2)
        c1, _ = _canonical(b1, b2)
        # modelA always wins; x == 1 iff canonical-first IS modelA.
        assert x == (1 if c1.id == "modelA" else 0), (b1, b2, x)


def test_bt_provider_symmetry_under_argument_swap():
    """observe(a, b1, b2) and observe(a, b2, b1) must describe the same
    winner: the first returns 'canonical-first won', the second returns
    'canonical-first won' with the same canonical pair."""
    provider = _bt_provider(random.Random(42))
    task = Woman("taskX")
    b1, b2 = Man("modelB"), Man("modelC")  # tied thetas -> p = 0.5

    n = 4000
    wins_first_order = sum(provider.observe(task, b1, b2) for _ in range(n))
    wins_second_order = sum(provider.observe(task, b2, b1) for _ in range(n))
    # With p=0.5 both orders win ~half the time, regardless of which
    # element is canonical-first.
    assert 0.35 * n < wins_first_order < 0.65 * n
    assert 0.35 * n < wins_second_order < 0.65 * n


def test_bt_provider_probability_matches_theta_ratio():
    import math

    eta = 1.7
    theta_task = {
        "taskX": {
            "modelA": math.exp(eta * 0.30),
            "modelB": math.exp(eta * 0.10),
        }
    }
    theta_model = {
        "modelA": {"taskX": 1.0, "taskY": 1.0},
        "modelB": {"taskX": 1.0, "taskY": 1.0},
    }
    provider = RouterBenchBTProvider(theta_task, theta_model, random.Random(7))

    p_true = math.exp(eta * 0.30) / (math.exp(eta * 0.30) + math.exp(eta * 0.10))
    task = Woman("taskX")
    b1, b2 = Man("modelA"), Man("modelB")

    n = 20000
    wins = 0
    for _ in range(n):
        x = provider.observe(task, b1, b2)
        c1, _ = _canonical(b1, b2)
        winner_is_a = (x == 1) == (c1 == b1)
        wins += winner_is_a
    assert wins / n == pytest.approx(p_true, abs=0.02)


def test_p2etg_stores_provider_wins_correctly():
    """End-to-end orientation check: P2ETG fed a deterministic provider
    must rank the always-winning model first."""
    from p2etg import P2ETG

    class AlwaysFirst:
        """modelA always preferred by every task; taskX always preferred
        by every model."""

        def observe(self, agent, b1, b2):
            a_id = agent.id if hasattr(agent, "id") else str(agent)
            b1_id = b1.id if hasattr(b1, "id") else str(b1)
            b2_id = b2.id if hasattr(b2, "id") else str(b2)

            def beats(x, y):
                if a_id in ("taskX", "taskY"):
                    return x == "modelA" and y != "modelA"
                # model agents prefer taskX over taskY
                return x == "taskX" and y == "taskY"

            c1, c2 = _canonical(b1, b2)
            if beats(c1.id, c2.id):
                return 1
            if beats(c2.id, c1.id):
                return 0
            # not comparable in this provider's rule -> coin via canonical id
            return 1 if c1.id < c2.id else 0

        def warmup(self, agent, partners):
            return None

    men = [Man(m) for m in MODELS]
    women = [Woman(t) for t in TASKS]
    learner = P2ETG(men, women, AlwaysFirst(), rng=random.Random(0))

    # Force many samples of every arm via the doubling epochs.
    learner.run_until_stop(max_epochs=8, adaptive=False, verbose=False)

    for task in women:
        ranking = learner.estimated_preferences(task)
        assert ranking[0].id == "modelA"
    for model in men:
        ranking = learner.estimated_preferences(model)
        assert ranking[0].id == "taskX"


def test_replay_provider_task_side_orientation():
    scores = {
        "taskX": {
            "modelA": {i: 1.0 for i in range(30)},
            "modelB": {i: 0.0 for i in range(30)},
        }
    }
    records = make_records(scores, ["modelA", "modelB"])
    splits = pd.DataFrame(
        [(d, i, "train") for d in ["taskX"] for i in range(30)],
        columns=["dataset", "record_index", "split"],
    )
    provider = RouterBenchReplayProvider(
        records, splits, ["taskX"], ["modelA", "modelB"],
        rng=random.Random(3),
    )
    task = Woman("taskX")
    b1, b2 = Man("modelA"), Man("modelB")

    # modelA scores 1.0 on every train instance -> always wins.
    for _ in range(200):
        x = provider.observe(task, b1, b2)
        c1, _ = _canonical(b1, b2)
        winner_a = (x == 1) == (c1 == b1)
        assert winner_a


def test_replay_provider_coin_on_exact_tie():
    scores = {
        "taskX": {
            "modelA": {i: 0.5 for i in range(50)},
            "modelB": {i: 0.5 for i in range(50)},
        }
    }
    records = make_records(scores, ["modelA", "modelB"])
    splits = pd.DataFrame(
        [("taskX", i, "train") for i in range(50)],
        columns=["dataset", "record_index", "split"],
    )
    provider = RouterBenchReplayProvider(
        records, splits, ["taskX"], ["modelA", "modelB"], rng=random.Random(11)
    )
    task = Woman("taskX")
    b1, b2 = Man("modelA"), Man("modelB")
    n = 2000
    wins_a = 0
    for _ in range(n):
        x = provider.observe(task, b1, b2)
        c1, _ = _canonical(b1, b2)
        wins_a += (x == 1) == (c1 == b1)
    assert 0.40 * n < wins_a < 0.60 * n


def test_build_bt_thetas_and_calibration():
    import math

    task_util = pd.DataFrame(
        [[0.0, 0.05]], index=["taskX"], columns=["modelA", "modelB"]
    )
    model_util = pd.DataFrame(
        [[0.05, 0.0]], index=["modelA"], columns=["taskX", "taskY"]
    )
    eta = calibrate_eta(0.05, 0.70)
    assert eta == pytest.approx(math.log(0.7 / 0.3) / 0.05)

    theta_task, theta_model = build_bt_thetas(task_util, model_util, eta, eta)
    assert theta_task["taskX"]["modelA"] == pytest.approx(math.exp(eta * 0.0))
    assert theta_task["taskX"]["modelB"] == pytest.approx(math.exp(eta * 0.05))
    assert theta_model["modelA"]["taskX"] == pytest.approx(math.exp(eta * 0.05))
