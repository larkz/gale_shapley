"""oracle.py

Hidden oracle preference profile and model-proposing Gale-Shapley.

The oracle builds STRICT rankings from the (tie-strictified) TRAIN
utilities:

  * each task ranks models by descending U_d(m);
  * each model ranks tasks by descending V_m(d);

and computes the model-proposing stable matching H*_train.

The online learner NEVER sees these rankings; they exist only for
evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from gs_lib.gs_tools import (
    GaleShapley, Man, Matching, PreferenceList, StabilityVerifier, Woman,
)

logger = logging.getLogger(__name__)


def build_market(
    task_util_strict: pd.DataFrame,
    model_util_strict: pd.DataFrame,
) -> Tuple[List[Man], List[Woman], PreferenceList]:
    """Create the matching market from strict utilities.

    Men = models, Women = tasks.
    """
    models: List[str] = list(model_util_strict.index)
    datasets: List[str] = list(task_util_strict.index)

    men = [Man(m) for m in models]
    women = [Woman(d) for d in datasets]

    preferences: Dict[object, List[object]] = {}
    for man, model in zip(men, models):
        ranked = sorted(
            datasets, key=lambda d: (-float(model_util_strict.loc[model, d]), d)
        )
        preferences[man] = [Woman(d) for d in ranked]
    for woman, dataset in zip(women, datasets):
        ranked = sorted(
            models, key=lambda m: (-float(task_util_strict.loc[dataset, m]), m)
        )
        preferences[woman] = [Man(m) for m in ranked]

    return men, women, PreferenceList(preferences)


def oracle_matching(prefs: PreferenceList) -> Matching:
    """Model-proposing Gale-Shapley under the given preferences."""
    return GaleShapley(prefs).find_stable_matching(proposing_side="men")


def verify_stable(prefs: PreferenceList, matching: Matching) -> bool:
    ok, _, _ = StabilityVerifier(prefs).is_stable(matching)
    return ok


def matching_to_dict(matching: Matching) -> Dict[str, str]:
    """Readable {task -> model} view (women keyed)."""
    out: Dict[str, str] = {}
    for pair in matching.pairs:
        out[pair.woman.id] = pair.man.id
    return dict(sorted(out.items()))


def preference_lists_to_dict(prefs: PreferenceList) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for person, plist in prefs.preferences.items():
        out[str(person.id)] = [str(p.id) for p in plist]
    return dict(sorted(out.items()))


def matching_equal(m1: Matching, m2: Matching) -> bool:
    """Exact equality of the matched pair sets."""
    return m1.pairs == m2.pairs


def save_oracle_outputs(
    out_dir: Path,
    prefs_train: PreferenceList,
    h_train: Matching,
    h_val: Matching,
    h_test: Matching,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "oracle_preferences_train.json", "w") as fh:
        json.dump(preference_lists_to_dict(prefs_train), fh, indent=2)

    payload = {
        "train": matching_to_dict(h_train),
        "val": matching_to_dict(h_val),
        "test": matching_to_dict(h_test),
        "train_equals_val": matching_equal(h_train, h_val),
        "train_equals_test": matching_equal(h_train, h_test),
    }
    with open(out_dir / "oracle_matching_train.json", "w") as fh:
        json.dump(matching_to_dict(h_train), fh, indent=2)
    with open(out_dir / "oracle_matching_val.json", "w") as fh:
        json.dump(matching_to_dict(h_val), fh, indent=2)
    with open(out_dir / "oracle_matching_test.json", "w") as fh:
        json.dump(matching_to_dict(h_test), fh, indent=2)
    with open(out_dir / "oracle_generalization.json", "w") as fh:
        json.dump(
            {
                "train_equals_val": payload["train_equals_val"],
                "train_equals_test": payload["train_equals_test"],
            },
            fh,
            indent=2,
        )
