"""providers.py

Offline pairwise-feedback SignalProviders for P2ETG / PrefLID.

Mode A: RouterBenchBTProvider
    Real (tie-strictified) utilities + Bradley-Terry feedback. Matches
    P2ETG's statistical modelling assumptions:

        theta_{d,m} = exp(eta_T * U_d(m))       (task side)
        theta_{m,d} = exp(eta_M * V_m(d))       (model side)

        P(b1 beats b2 | agent) = theta[b1] / (theta[b1] + theta[b2])

Mode B: RouterBenchReplayProvider
    Empirical benchmark replay from stored per-instance outcomes. No
    parametric assumption is made: feedback is sampled from actual
    instance-level scores.

Both providers honour the repository convention: observe() returns 1 iff
the CANONICAL-FIRST element of (b1, b2) wins, where canonical order is
gs_lib.bt._canonical(b1, b2). The provider never assumes b1 is
canonical-first.

No LLM inference is performed anywhere in this module.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, Hashable, List, Optional

import pandas as pd

from gs_lib.bt import _canonical
from p2etg import SignalProvider

logger = logging.getLogger(__name__)


# ============================================================================
# BT calibration
# ============================================================================

def calibrate_eta(median_gap: float, target_win_probability: float) -> float:
    """eta such that a median-gap comparison has the target win prob."""
    if not 0.5 < target_win_probability < 1.0:
        raise ValueError(
            f"target_median_win_probability must be in (0.5, 1), got "
            f"{target_win_probability}"
        )
    if median_gap <= 0:
        raise ValueError(f"median_gap must be positive, got {median_gap}")
    return math.log(target_win_probability / (1.0 - target_win_probability)) / median_gap


# ============================================================================
# Mode A: BT feedback from real utilities
# ============================================================================

class RouterBenchBTProvider(SignalProvider):
    """Bradley-Terry feedback with thetas exp(eta * strict utility).

    theta_task[task][model]  = exp(eta_T * U_d^strict(m))
    theta_model[model][task] = exp(eta_M * V_m^strict(d))

    probability_floor > 0 GUARANTEES NO TIES in the reward channel:
    every pairwise win probability is clamped away from 1/2 by at
    least `probability_floor`, in the direction of the (jittered)
    strict preference. Exact utility ties — which would otherwise
    give p = 0.500000x and be structurally unresolvable — become
    p = 0.5 + floor in the tie-broken direction; weak-but-real
    signals below the floor are boosted to the floor. Preferences,
    the oracle H*, and welfare are UNCHANGED: only the observation
    channel is separated. Resolution guarantee: every arm resolves
    at n ~ c*ln(t)/floor^2 samples (c = the CI constant).
    """

    def __init__(
        self,
        theta_task: Dict[Hashable, Dict[Hashable, float]],
        theta_model: Dict[Hashable, Dict[Hashable, float]],
        rng: random.Random,
        probability_floor: float = 0.0,
    ) -> None:
        self.theta_task = theta_task
        self.theta_model = theta_model
        self.rng = rng
        self.probability_floor = float(probability_floor)

    def observe(self, agent: Hashable, b1: Hashable, b2: Hashable) -> int:
        # Normalise to stable id strings (Man/Woman objects do not
        # compare equal to plain strings in dict lookups).
        agent_id = agent.id if hasattr(agent, "id") else str(agent)

        # Determine which side the agent is on and its theta table.
        # Tasks (Women) compare models; models (Men) compare tasks.
        if agent_id in self.theta_task:
            table = self.theta_task[agent_id]
        elif agent_id in self.theta_model:
            table = self.theta_model[agent_id]
        else:
            raise KeyError(f"Unknown agent {agent!r} for BT provider")

        c1, c2 = _canonical(b1, b2)
        c1_id = c1.id if hasattr(c1, "id") else str(c1)
        c2_id = c2.id if hasattr(c2, "id") else str(c2)
        if c1_id not in table or c2_id not in table:
            raise KeyError(
                f"Unknown partners {c1_id!r}/{c2_id!r} for agent {agent_id!r}"
            )

        t1, t2 = table[c1_id], table[c2_id]
        p_c1_wins = t1 / (t1 + t2)
        if self.probability_floor > 0.0:
            eps = self.probability_floor
            if p_c1_wins > 0.5:
                p_c1_wins = max(p_c1_wins, 0.5 + eps)
            elif p_c1_wins < 0.5:
                p_c1_wins = min(p_c1_wins, 0.5 - eps)
            else:
                # theta collision (should not happen after jitter):
                # deterministic direction toward the larger theta
                p_c1_wins = 0.5 + eps if t1 >= t2 else 0.5 - eps
        return 1 if self.rng.random() < p_c1_wins else 0


def build_bt_thetas(
    task_util_strict: pd.DataFrame,
    model_util_strict: pd.DataFrame,
    eta_task: float,
    eta_model: float,
) -> "tuple[dict, dict]":
    """BT theta tables keyed by the same id strings as the utilities."""
    theta_task: Dict[str, Dict[str, float]] = {}
    for dataset in task_util_strict.index:
        theta_task[dataset] = {
            model: math.exp(eta_task * float(task_util_strict.loc[dataset, model]))
            for model in task_util_strict.columns
        }
    theta_model: Dict[str, Dict[str, float]] = {}
    for model in model_util_strict.index:
        theta_model[model] = {
            dataset: math.exp(eta_model * float(model_util_strict.loc[model, dataset]))
            for dataset in model_util_strict.columns
        }
    return theta_task, theta_model


# ============================================================================
# Mode B: empirical benchmark replay
# ============================================================================

class RouterBenchReplayProvider(SignalProvider):
    """Empirical pairwise feedback from stored per-instance outcomes.

    Task side (agent = task d, models m1 vs m2):
        sample a TRAIN instance x of d; the model with the higher stored
        score on x wins; exact ties are broken by a seeded fair coin.

    Model side (agent = model m, tasks d1 vs d2):
        sample TRAIN instances x1 ~ D_{d1}, x2 ~ D_{d2}; compute
        per-instance comparative advantages
            a_i = s_{m,d_i,x_i} - mean_{m' != m} s_{m',d_i,x_i}
        and decide by a configurable mode:
          * probabilistic (default):
                P(d1 wins) = clip(0.5 + gamma*(a1 - a2), eps, 1 - eps)
          * direct: d1 wins iff a1 > a2 (ties broken by coin).
    """

    def __init__(
        self,
        records: pd.DataFrame,
        splits: pd.DataFrame,
        datasets: List[str],
        models: List[str],
        rng: random.Random,
        replay_gamma: float = 0.25,
        replay_probability_epsilon: float = 0.01,
        model_replay_mode: str = "probabilistic",
    ) -> None:
        if model_replay_mode not in ("probabilistic", "direct"):
            raise ValueError(
                f"model_replay_mode must be 'probabilistic' or 'direct', "
                f"got {model_replay_mode!r}"
            )
        self.rng = rng
        self.replay_gamma = float(replay_gamma)
        self.replay_probability_epsilon = float(replay_probability_epsilon)
        self.model_replay_mode = model_replay_mode
        self.models = list(models)

        merged = records.merge(splits, on=["dataset", "record_index"], how="inner")
        train = merged[merged["split"] == "train"]

        # score[dataset][record_index][model] and train indices per dataset
        self.scores: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.train_indices: Dict[str, List[int]] = {}
        for dataset in datasets:
            sub = train[train["dataset"] == dataset]
            self.scores[dataset] = {}
            for row in sub.itertuples(index=False):
                self.scores[dataset].setdefault(row.record_index, {})[
                    row.model
                ] = float(row.score)
            self.train_indices[dataset] = sorted(self.scores[dataset].keys())

        # Per-instance comparative advantage:
        #   adv[dataset][record_index][model] =
        #       s - mean_{m' != m} s_{m'}
        self.adv: Dict[str, Dict[int, Dict[str, float]]] = {
            d: {} for d in datasets
        }
        for dataset in datasets:
            for idx, row_scores in self.scores[dataset].items():
                total = sum(row_scores[m] for m in self.models)
                n_models = len(self.models)
                self.adv[dataset][idx] = {
                    m: row_scores[m] - (total - row_scores[m]) / (n_models - 1)
                    for m in self.models
                }

        self._dataset_set = set(datasets)

    # ------------------------------------------------------------------

    def _task_side_outcome(self, dataset: str, m1: str, m2: str) -> str:
        """Winner model for a task-side comparison (raw, not canonicalised)."""
        indices = self.train_indices.get(dataset)
        if not indices:
            raise KeyError(f"No train instances for dataset {dataset!r}")
        x = self.rng.choice(indices)
        row = self.scores[dataset][x]
        s1, s2 = row[m1], row[m2]
        if s1 > s2:
            return m1
        if s2 > s1:
            return m2
        return m1 if self.rng.random() < 0.5 else m2

    def _model_side_outcome(self, model: str, d1: str, d2: str) -> str:
        """Winner task for a model-side comparison (raw, not canonicalised)."""
        x1 = self.rng.choice(self.train_indices[d1])
        x2 = self.rng.choice(self.train_indices[d2])
        a1 = self.adv[d1][x1][model]
        a2 = self.adv[d2][x2][model]

        if self.model_replay_mode == "direct":
            if a1 > a2:
                return d1
            if a2 > a1:
                return d2
            return d1 if self.rng.random() < 0.5 else d2

        p = 0.5 + self.replay_gamma * (a1 - a2)
        eps = self.replay_probability_epsilon
        p = min(max(p, eps), 1.0 - eps)
        return d1 if self.rng.random() < p else d2

    def observe(self, agent: Hashable, b1: Hashable, b2: Hashable) -> int:
        agent_id = agent.id if hasattr(agent, "id") else str(agent)
        b1_id = b1.id if hasattr(b1, "id") else str(b1)
        b2_id = b2.id if hasattr(b2, "id") else str(b2)

        if agent_id in self._dataset_set:
            winner = self._task_side_outcome(agent_id, b1_id, b2_id)
        else:
            winner = self._model_side_outcome(agent_id, b1_id, b2_id)

        c1, c2 = _canonical(b1, b2)
        return 1 if winner == (b1_id if c1 == b1 else b2_id) else 0
