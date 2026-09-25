"""Tests for the ported PrefLID tuning (upstream larkin/preflid_tuning).

Covers:
  * the new per-agent sampling-balance gate _all_pairs_sufficiently_sampled
    (min_samples AND min_count >= min_ratio * max_count);
  * SignalProvider delegation still works with the updated constructor;
  * backward compatibility with true-theta sampling (no provider).
"""

from __future__ import annotations

import random

from gs_lib.bt import _canonical
from gs_lib.gs_tools import Man, Woman
from preflid import PrefLID


class AlwaysCanonicalFirstProvider:
    """Deterministic: canonical-first partner always wins (p_hat == 1)."""

    def observe(self, agent, b1, b2) -> int:
        return 1

    def warmup(self, agent, partners) -> None:
        return None


def _market3():
    men = [Man(f"m{i}") for i in range(3)]
    women = [Woman(f"w{i}") for i in range(3)]
    return men, women


def _fill_arms(learner, n: int) -> None:
    for state in learner.agent_states.values():
        partners = state.partners
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                key = _canonical(partners[i], partners[j])
                state.counts.total[key] = n
                state.counts.wins[key] = n // 2


def test_sufficiently_sampled_gate_requires_coverage():
    men, women = _market3()
    learner = PrefLID(
        men, women, rng=random.Random(0),
        min_samples_per_pair=5, min_sample_ratio=0.5,
    )
    # nothing sampled -> insufficient
    assert not learner._all_pairs_sufficiently_sampled(5, 0.5)


def test_sufficiently_sampled_gate_accepts_balanced_counts():
    men, women = _market3()
    learner = PrefLID(
        men, women, rng=random.Random(0),
        min_samples_per_pair=5, min_sample_ratio=0.5,
    )
    _fill_arms(learner, 10)
    assert learner._all_pairs_sufficiently_sampled(5, 0.5)


def test_sufficiently_sampled_gate_rejects_imbalance():
    men, women = _market3()
    learner = PrefLID(
        men, women, rng=random.Random(0),
        min_samples_per_pair=5, min_sample_ratio=0.5,
    )
    _fill_arms(learner, 10)
    state = next(iter(learner.agent_states.values()))
    key = next(iter(state.counts.total))
    state.counts.total[key] = 4  # 4 < 0.5 * 10
    assert not learner._all_pairs_sufficiently_sampled(5, 0.5)

    # restore, then drop everything below min_samples
    state.counts.total[key] = 10
    assert learner._all_pairs_sufficiently_sampled(5, 0.5)
    _fill_arms(learner, 4)  # max = 4 < 5
    assert not learner._all_pairs_sufficiently_sampled(5, 0.5)


def test_preflid_with_provider_certifies_unambiguous_market():
    men, women = _market3()
    provider = AlwaysCanonicalFirstProvider()
    learner = PrefLID(
        men, women, provider=provider, rng=random.Random(0),
        min_samples_per_pair=1, min_sample_ratio=0.5,
        budget=100, max_lattice_vertices=5000,
    )
    rounds = []
    result = learner.run_until_stop(max_iterations=200, rounds=rounds, verbose=False)

    assert learner.t > 0
    # deterministic p=1 feedback: every sampled arm's p_hat is exactly 1
    for state in learner.agent_states.values():
        for key, n in state.counts.total.items():
            if n:
                assert state.counts.wins.get(key, 0) == n

    # with fully deterministic feedback the structure certifies
    assert result["stopped"] is True
    assert result.get("reason") == "certified"
    assert len(result["matching"].pairs) == 3

    # rounds records carry the new tuning fields
    ok_rows = [r for r in rounds if r.get("status") == "ok"]
    assert ok_rows
    for r in ok_rows:
        assert "n_lattices_largest_island" in r
        assert "support_max" in r
        assert "support_mean" in r
        assert r["H_star"] is not None


def test_preflid_true_theta_backward_compatible():
    men, women = _market3()
    theta_men = {m: {w: 1.0 for w in women} for m in men}
    theta_women = {w: {m: 1.0 for w in [m] and True} for m in [] for w in women}
    # simpler: all thetas equal -> p = 0.5 everywhere
    theta_women = {w: {m: 1.0 for m in men} for w in women}
    learner = PrefLID(
        men, women,
        true_theta_men=theta_men,
        true_theta_women=theta_women,
        rng=random.Random(1),
        min_samples_per_pair=1, min_sample_ratio=0.5,
    )
    result = learner.run_until_stop(max_iterations=5, verbose=False)
    # only asserts the legacy path runs and samples
    assert learner.t > 0
    assert result["T_stop"] == learner.t


def test_round_robin_center_policy_balances_counts():
    men, women = _market3()
    learner = PrefLID(
        men, women, provider=AlwaysCanonicalFirstProvider(), rng=random.Random(0),
        min_samples_per_pair=1, min_sample_ratio=0.5,
        center_policy="round_robin", budget=100,
    )
    learner.run_until_stop(max_iterations=100, verbose=False)
    counts = [
        min(state.counts.total.values())
        for state in learner.agent_states.values()
    ]
    # every agent centered within 2 of every other (lockstep per pass;
    # the <=2 tolerance covers the entrant-set warmup transient where
    # the queue is rebuilt with a subset of agents for the first few
    # iterations)
    assert max(counts) - min(counts) <= 2, counts
    assert max(counts) >= 6  # ~100 iterations / 6 agents


def test_round_robin_invalid_policy_rejected():
    men, women = _market3()
    try:
        PrefLID(men, women, provider=AlwaysCanonicalFirstProvider(),
                center_policy="bogus")
        raise SystemExit("should have raised")
    except ValueError:
        pass
