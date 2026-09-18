"""splits.py

Deterministic train/validation/test split of aligned record indices.

The split is made once per dataset over the aligned record IDs (never
per model), using a string-seeded random.Random. String seeding is
stable across processes and platforms, unlike Python's hash().
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import pandas as pd

TRAIN = "train"
VAL = "val"
TEST = "test"


def split_record_indices(
    indices: List[int], dataset: str, split_seed: int,
    train_frac: float, val_frac: float,
) -> Dict[int, str]:
    """Deterministically assign record indices to splits for one dataset."""
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError(
            f"Invalid split fractions: train={train_frac}, val={val_frac}"
        )
    shuffled = sorted(indices)
    rng = random.Random(f"{dataset}::{split_seed}")
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    if n_train + n_val >= n and n > 0:
        n_val = max(0, n - n_train - 1)

    assignment: Dict[int, str] = {}
    for i, idx in enumerate(shuffled):
        if i < n_train:
            assignment[idx] = TRAIN
        elif i < n_train + n_val:
            assignment[idx] = VAL
        else:
            assignment[idx] = TEST
    return assignment


def make_splits(
    records: pd.DataFrame,
    split_seed: int,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> pd.DataFrame:
    """Build the split table: dataset | record_index | split."""
    rows: List[Tuple[str, int, str]] = []
    for dataset in sorted(records["dataset"].unique()):
        indices = sorted(
            records.loc[records["dataset"] == dataset, "record_index"]
            .unique()
            .tolist()
        )
        assignment = split_record_indices(
            indices, dataset, split_seed, train_frac, val_frac
        )
        for idx, split in assignment.items():
            rows.append((dataset, idx, split))
    return pd.DataFrame(rows, columns=["dataset", "record_index", "split"])


def split_sizes(splits: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset split counts, long format."""
    return (
        splits.groupby(["dataset", "split"]).size().reset_index(name="n")
    )
