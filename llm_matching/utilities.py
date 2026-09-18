"""utilities.py

Latent utilities for both sides of the market.

Task side (Women):
    U_d(m) = mean score of model m on dataset d's TRAIN instances.

Model side (Men), comparative advantage:
    V_m(d) = U_d(m) - mean_{m' != m} U_d(m')

Model preference is deliberately NOT raw accuracy: models prefer tasks
on which they beat the other available models by the widest margin.

Tie handling: exact or numerically indistinguishable ties are broken by
a deterministic SHA256-based jitter that depends only on stable entity
names and the split seed (never Python's hash()).
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Utility matrices
# ============================================================================

def task_utility(
    records: pd.DataFrame,
    splits: pd.DataFrame,
    split: str,
    datasets: List[str],
    models: List[str],
) -> pd.DataFrame:
    """U_d(m): mean score over instances of dataset d in the given split.

    Returns a DataFrame indexed by dataset with one column per model.
    """
    merged = records.merge(splits, on=["dataset", "record_index"], how="inner")
    merged = merged[merged["split"] == split]
    util = (
        merged.groupby(["dataset", "model"])["score"].mean().unstack("model")
    )
    util = util.reindex(index=datasets, columns=models)
    if util.isna().any().any():
        missing = util.isna().sum().sum()
        raise ValueError(
            f"task_utility has {missing} missing entries for split={split}; "
            "aligned records do not cover every (dataset, model, split)"
        )
    util.index.name = "dataset"
    return util


def model_utility(task_util: pd.DataFrame) -> pd.DataFrame:
    """V_m(d) = U_d(m) - mean_{m' != m} U_d(m').

    Input:  task_util  (index=dataset, columns=model)
    Output: model_util (index=model,   columns=dataset)
    """
    row_sum = task_util.sum(axis=1)
    # row_sum - task_util with ROW alignment (rsub(axis=0)); a plain
    # `row_sum - task_util` would align on columns and produce NaNs.
    other_mean = task_util.rsub(row_sum, axis="index") / (
        task_util.shape[1] - 1
    )
    v = (task_util - other_mean).T
    v.index.name = "model"
    v.columns.name = "dataset"
    return v


# ============================================================================
# Deterministic tie breaking
# ============================================================================

def deterministic_jitter(agent_id: str, partner_id: str, seed: int) -> float:
    """Reproducible value in [-1, 1) from SHA256 of stable entity names."""
    digest = hashlib.sha256(
        f"{agent_id}|{partner_id}|{seed}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / (2 ** 63) - 1.0


def strictify_utilities(
    utility: pd.DataFrame,
    tie_epsilon: float,
    split_seed: int,
    agent_kind: str,
) -> Tuple[pd.DataFrame, int, List[dict]]:
    """Break numerically indistinguishable utility ties deterministically.

    For every agent (row), partners (columns) whose utilities are within
    tie_epsilon of each other (transitive grouping over sorted values)
    form a tie group. Every member of a multi-member group receives an
    additive jitter of at most tie_epsilon/4, so:

      * ties receive a strict, reproducible order;
      * groups stay strictly separated from non-tied neighbours
        (min inter-group gap after jitter is tie_epsilon / 2);
      * non-tied utilities are untouched.

    Returns (strict_utility, n_changed_entries, tie_report).
    """
    strict = utility.copy()
    n_changed = 0
    tie_report: List[dict] = []
    scale = tie_epsilon / 4.0

    for agent in utility.index:
        row = utility.loc[agent]
        partners = list(row.index)
        order = sorted(partners, key=lambda p: row[p])
        groups: List[List[str]] = []
        for p in order:
            if groups and abs(row[p] - row[groups[-1][-1]]) <= tie_epsilon:
                groups[-1].append(p)
            else:
                groups.append([p])

        for group in groups:
            if len(group) < 2:
                continue
            tie_report.append(
                {
                    "agent_kind": agent_kind,
                    "agent": str(agent),
                    "partners": [str(p) for p in group],
                    "values": [float(row[p]) for p in group],
                }
            )
            for p in group:
                strict.loc[agent, p] = float(row[p]) + scale * deterministic_jitter(
                    str(agent), str(p), split_seed
                )
                n_changed += 1

    if tie_report:
        logger.warning(
            "Detected %d utility tie group(s) for %s side; applied "
            "deterministic jitter to %d entries.",
            len(tie_report), agent_kind, n_changed,
        )
        for entry in tie_report:
            logger.warning("Tie group: %s", entry)
    else:
        logger.info("No utility ties detected for %s side.", agent_kind)

    return strict, n_changed, tie_report


# ============================================================================
# Gap statistics (for BT calibration)
# ============================================================================

def positive_gaps(utility: pd.DataFrame, tie_epsilon: float) -> List[float]:
    """All within-agent positive utility gaps larger than tie_epsilon.

    utility rows are agents, columns are partners (raw, pre-jitter).
    """
    gaps: List[float] = []
    partners = list(utility.columns)
    for agent in utility.index:
        row = utility.loc[agent]
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                g = abs(float(row[partners[i]]) - float(row[partners[j]]))
                if g > tie_epsilon:
                    gaps.append(g)
    return gaps


def median_positive_gap(utility: pd.DataFrame, tie_epsilon: float) -> float:
    gaps = positive_gaps(utility, tie_epsilon)
    if not gaps:
        raise ValueError(
            "No positive utility gaps found; BT calibration is undefined "
            "(all pairs tied within tie_epsilon)."
        )
    gaps.sort()
    n = len(gaps)
    if n % 2 == 1:
        return float(gaps[n // 2])
    return (gaps[n // 2 - 1] + gaps[n // 2]) / 2.0
