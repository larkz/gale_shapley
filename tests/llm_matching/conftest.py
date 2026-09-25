"""Shared fixtures for llm_matching tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

GALE_SHAPLEY_ROOT = Path(__file__).resolve().parents[2]
if str(GALE_SHAPLEY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALE_SHAPLEY_ROOT))

ROUTERBENCH_ROOT = GALE_SHAPLEY_ROOT.parent / "LLMRouterBench"
BENCH_DIR = ROUTERBENCH_ROOT / "results" / "bench"

requires_routerbench_data = pytest.mark.skipif(
    not BENCH_DIR.is_dir(),
    reason="LLMRouterBench results not downloaded",
)


def write_fake_bench(
    root: Path,
    scores: Dict[str, Dict[str, Dict[int, float]]],
    split: str = "test",
    timestamp: str = "20250101_000000",
    demo: bool = False,
) -> None:
    """Create a fake results/bench tree.

    scores[dataset][model] = {record_index: score}
    """
    for dataset, models in scores.items():
        for model, records in models.items():
            model_dir = root / "results" / "bench" / dataset / split / model
            model_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset_name": dataset,
                "split": split,
                "model_name": model,
                "demo": demo,
                "records": [
                    {
                        "index": idx,
                        "score": score,
                        "cost": 0.001,
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                    }
                    for idx, score in sorted(records.items())
                ],
            }
            name = f"{dataset}-{split}-{model}-{timestamp}.json"
            with open(model_dir / name, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)


def make_records(
    scores: Dict[str, Dict[str, Dict[int, float]]],
    models_order: List[str],
) -> pd.DataFrame:
    """Long-format records DataFrame from a scores dict."""
    rows = []
    for dataset, models in scores.items():
        for model, records in models.items():
            for idx, score in records.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "record_index": idx,
                        "model": model,
                        "score": float(score),
                        "cost": 0.001,
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                    }
                )
    return pd.DataFrame(rows)
