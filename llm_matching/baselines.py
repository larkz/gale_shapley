"""baselines.py

Reference assignments for the LLM-task market.

A. Oracle GS          -> llm_matching.oracle (H*)
B. Random bijection   -> uniform random one-to-one assignment
C. Hungarian welfare  -> scipy linear_sum_assignment maximising
                         sum_d U_d(H(d)) (task-side utility only;
                         ignores model-side preferences/stability)
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from gs_lib.gs_tools import Man, Matching, Woman


def random_matching(men: List[Man], women: List[Woman], rng: random.Random) -> Matching:
    """Uniformly random model-task bijection."""
    shuffled = list(women)
    rng.shuffle(shuffled)
    matching_dict = {m: w for m, w in zip(men, shuffled)}
    return Matching.from_dict(matching_dict, set(men), set(women))


def hungarian_matching(task_util: pd.DataFrame) -> Matching:
    """Maximum-welfare matching on the given task-side utility matrix.

    task_util: index=dataset (tasks), columns=model.
    Maximises sum_d U_d(H(d)) via scipy.optimize.linear_sum_assignment.
    """
    cost = -task_util.to_numpy(dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost)
    models = list(task_util.columns)
    datasets = list(task_util.index)

    matching_dict: Dict[Man, Woman] = {}
    for d_idx, m_idx in zip(row_ind, col_ind):
        matching_dict[Man(models[m_idx])] = Woman(datasets[d_idx])
    all_men = {Man(m) for m in models}
    all_women = {Woman(d) for d in datasets}
    return Matching.from_dict(matching_dict, all_men, all_women)


def random_welfare_stats(
    task_util: pd.DataFrame,
    n_seeds: int = 200,
    seed: int = 3407,
) -> Tuple[float, float]:
    """(mean, std) of sum_d U_d(H_random(d)) over random bijections."""
    datasets = list(task_util.index)
    models = list(task_util.columns)
    util = task_util.to_numpy(dtype=float)
    n = len(datasets)

    welfares = []
    for i in range(n_seeds):
        rng = random.Random(f"random-baseline::{seed}::{i}")
        order = list(range(n))
        rng.shuffle(order)
        # order[i] = model index assigned to dataset i
        welfares.append(float(sum(util[i, order[i]] for i in range(n))))
    return float(np.mean(welfares)), float(np.std(welfares))


def datasetwise_best_ignoring_capacity(task_util: pd.DataFrame) -> Dict[str, str]:
    """For each task, its single best model ignoring capacity constraints.

    A 'trivial assignment' reference for the sanity checks: if every
    task's best model is the same, the market has weak specialization.
    """
    return {
        dataset: str(task_util.loc[dataset].idxmax())
        for dataset in task_util.index
    }
