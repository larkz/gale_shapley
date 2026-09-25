"""routerbench_loader.py

Load and align LLMRouterBench per-instance benchmark results.

The loader discovers the actual on-disk dataset directories / splits
(capitalisation may differ from internal names), reads the latest result
JSON per (dataset, split, model) following the LLMRouterBench convention
(dedupe by filename timestamp, skip demo files), and produces a single
long-format aligned table:

    dataset | record_index | model | score | cost | prompt_tokens | completion_tokens

Alignment: for every dataset only the intersection of instance IDs
available for ALL selected models is retained, so scores are never
compared across different examples.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})\.json$")

RECORD_COLUMNS = [
    "dataset",
    "record_index",
    "model",
    "score",
    "cost",
    "prompt_tokens",
    "completion_tokens",
]


class RouterBenchDataError(RuntimeError):
    """Raised when required LLMRouterBench results are missing or unusable."""


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved on-disk location for one internal dataset name."""

    internal_name: str
    dataset_id: str
    split: str


@dataclass
class AlignmentStats:
    dataset: str
    dataset_id: str
    split: str
    raw_min_records: int
    raw_max_records: int
    aligned_records: int
    missing_fraction: float

    def as_row(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "raw_min_records": self.raw_min_records,
            "raw_max_records": self.raw_max_records,
            "aligned_records": self.aligned_records,
            "missing_fraction": self.missing_fraction,
        }


@dataclass
class AlignedTable:
    """Long-format aligned score table plus resolution metadata."""

    records: pd.DataFrame
    dataset_specs: Dict[str, DatasetSpec]
    stats: List[AlignmentStats]
    models: List[str]
    datasets: List[str]

    def scores_for(self, dataset: str) -> pd.DataFrame:
        """Wide matrix [record_index x model] of scores for one dataset."""
        sub = self.records[self.records["dataset"] == dataset]
        return sub.pivot(
            index="record_index", columns="model", values="score"
        ).sort_index()

    def train_records(self, splits: pd.DataFrame) -> pd.DataFrame:
        """Records restricted to the train split."""
        merged = self.records.merge(
            splits, on=["dataset", "record_index"], how="inner"
        )
        return merged[merged["split"] == "train"]


def _is_demo(data: dict) -> bool:
    return bool(data.get("demo", False))


def _latest_json(model_dir: Path) -> Optional[Path]:
    """Latest non-demo result JSON for one (dataset, split, model) dir."""
    files = sorted(model_dir.glob("*.json"))
    if not files:
        return None

    def timestamp(p: Path) -> int:
        m = _TIMESTAMP_RE.search(p.name)
        if m:
            return int(m.group(1) + m.group(2))
        return int(p.stat().st_mtime)

    candidates: List[Path] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                head = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Unreadable result file %s: %s", f, exc)
            continue
        if _is_demo(head):
            continue
        candidates.append(f)

    if not candidates:
        return None
    return max(candidates, key=timestamp)


def _read_records(path: Path, dataset_id: str, split: str, model: str) -> Dict[int, dict]:
    """Read one result file into {record_index: fields}."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("dataset_name", dataset_id) != dataset_id:
        logger.debug(
            "dataset_name mismatch in %s: %s != %s",
            path, data.get("dataset_name"), dataset_id,
        )
    if data.get("split", split) != split:
        logger.debug(
            "split mismatch in %s: %s != %s", path, data.get("split"), split
        )

    out: Dict[int, dict] = {}
    for rec in data.get("records", []):
        idx = rec.get("index", rec.get("record_index"))
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        score = rec.get("score")
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        out[idx] = {
            "score": score,
            "cost": rec.get("cost"),
            "prompt_tokens": rec.get("prompt_tokens"),
            "completion_tokens": rec.get("completion_tokens"),
        }
    return out


class RouterBenchLoader:
    """Load an aligned score tensor for selected datasets and models."""

    def __init__(
        self,
        root: Path,
        datasets: List[str],
        models: List[str],
        min_aligned_records: int = 100,
    ) -> None:
        self.root = Path(root)
        self.datasets = list(datasets)
        self.models = list(models)
        self.min_aligned_records = min_aligned_records
        self.bench_root = self.root / "results" / "bench"

    # ------------------------------------------------------------------
    # Dataset / split resolution
    # ------------------------------------------------------------------

    def _check_bench_root(self) -> None:
        if not self.bench_root.is_dir():
            raise RouterBenchDataError(
                f"LLMRouterBench results not found under {self.bench_root}.\n"
                "Download the official pre-collected 'bench-release.tar.gz' "
                "from the LLMRouterBench dataset release "
                "(https://huggingface.co/datasets/NPULH/LLMRouterBench) and "
                "extract it into <LLMRouterBench>/results/ so that "
                "<LLMRouterBench>/results/bench/<dataset>/ exists.\n"
                "You can also use scripts/fetch_routerbench_results.py."
            )

    # ------------------------------------------------------------------
    # Dataset / split resolution
    # ------------------------------------------------------------------

    def _available_dataset_dirs(self) -> Dict[str, Path]:
        return {
            d.name.lower(): d for d in sorted(self.bench_root.iterdir()) if d.is_dir()
        }

    def _resolve_dataset_dir(self, internal: str, available: Dict[str, Path]) -> Path:
        key = internal.strip().lower()
        if key in available:
            return available[key]
        raise RouterBenchDataError(
            f"Dataset '{internal}' not found under {self.bench_root}. "
            f"Available datasets: {sorted(available.keys())}"
        )

    def _resolve_split(self, dataset_dir: Path, internal: str) -> str:
        """Pick the split in which every selected model is present.

        If several splits qualify, prefer the one whose first selected
        model has the most records (largest evaluation).
        """
        splits = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir())
        if not splits:
            raise RouterBenchDataError(
                f"No split directories under {dataset_dir}"
            )

        missing: Dict[str, List[str]] = {}
        sizes: Dict[str, int] = {}
        for split in splits:
            split_dir = dataset_dir / split
            miss = [m for m in self.models if not (split_dir / m).is_dir()]
            if miss:
                missing[split] = miss
                continue
            latest = _latest_json(split_dir / self.models[0])
            n = 0
            if latest is not None:
                with open(latest, "r", encoding="utf-8") as fh:
                    n = len(json.load(fh).get("records", []))
            sizes[split] = n

        if not sizes:
            detail = "; ".join(
                f"{s}: missing {miss}" for s, miss in sorted(missing.items())
            )
            raise RouterBenchDataError(
                f"Dataset '{internal}' ({dataset_dir}): no split contains all "
                f"selected models. Details: {detail}"
            )

        return max(sizes, key=lambda s: (sizes[s], s))

    def resolve_specs(self) -> Dict[str, DatasetSpec]:
        """Resolve every internal dataset name to (dataset_id, split)."""
        self._check_bench_root()
        available = self._available_dataset_dirs()
        specs: Dict[str, DatasetSpec] = {}
        for internal in self.datasets:
            ds_dir = self._resolve_dataset_dir(internal, available)
            split = self._resolve_split(ds_dir, internal)
            specs[internal] = DatasetSpec(
                internal_name=internal, dataset_id=ds_dir.name, split=split
            )
            logger.info(
                "Resolved dataset %s -> %s (split=%s)", internal, ds_dir.name, split
            )
        return specs

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> AlignedTable:
        specs = self.resolve_specs()

        per_model: Dict[str, Dict[str, Dict[int, dict]]] = {}
        missing_entries: List[str] = []

        for internal, spec in specs.items():
            per_model.setdefault(internal, {})
            for model in self.models:
                model_dir = (
                    self.bench_root / spec.dataset_id / spec.split / model
                )
                latest = _latest_json(model_dir)
                if latest is None:
                    missing_entries.append(
                        f"{internal} (={spec.dataset_id}/{spec.split}): "
                        f"model '{model}' has no result file"
                    )
                    continue
                per_model[internal][model] = _read_records(
                    latest, spec.dataset_id, spec.split, model
                )

        if missing_entries:
            raise RouterBenchDataError(
                "Missing model/dataset results in LLMRouterBench data:\n  "
                + "\n  ".join(missing_entries)
            )

        rows: List[dict] = []
        stats: List[AlignmentStats] = []
        below_threshold: List[str] = []

        for internal in self.datasets:
            spec = specs[internal]
            index_sets = {
                m: set(per_model[internal][m].keys()) for m in self.models
            }
            common = set.intersection(*index_sets.values())
            raw_counts = [len(index_sets[m]) for m in self.models]
            raw_min = min(raw_counts)
            raw_max = max(raw_counts)
            aligned = len(common)
            missing_fraction = (
                1.0 - aligned / raw_min if raw_min > 0 else 1.0
            )

            stats.append(
                AlignmentStats(
                    dataset=internal,
                    dataset_id=spec.dataset_id,
                    split=spec.split,
                    raw_min_records=raw_min,
                    raw_max_records=raw_max,
                    aligned_records=aligned,
                    missing_fraction=missing_fraction,
                )
            )
            if aligned < self.min_aligned_records:
                below_threshold.append(
                    f"{internal}: {aligned} aligned records < "
                    f"min_aligned_records={self.min_aligned_records}"
                )

            for idx in sorted(common):
                for model in self.models:
                    fields = per_model[internal][model][idx]
                    rows.append(
                        {
                            "dataset": internal,
                            "record_index": idx,
                            "model": model,
                            "score": fields["score"],
                            "cost": fields["cost"],
                            "prompt_tokens": fields["prompt_tokens"],
                            "completion_tokens": fields["completion_tokens"],
                        }
                    )

        if below_threshold:
            raise RouterBenchDataError(
                "Aligned instance count below threshold:\n  "
                + "\n  ".join(below_threshold)
            )

        records = pd.DataFrame(rows, columns=RECORD_COLUMNS)
        return AlignedTable(
            records=records,
            dataset_specs=specs,
            stats=stats,
            models=list(self.models),
            datasets=list(self.datasets),
        )


def format_alignment_stats(stats: List[AlignmentStats]) -> str:
    header = (
        f"{'Dataset':<18} {'on-disk id':<18} {'split':<12} "
        f"{'raw_min':>8} {'raw_max':>8} {'aligned':>8} {'missing_frac':>13}"
    )
    lines = [header, "-" * len(header)]
    for s in stats:
        lines.append(
            f"{s.dataset:<18} {s.dataset_id:<18} {s.split:<12} "
            f"{s.raw_min_records:>8d} {s.raw_max_records:>8d} "
            f"{s.aligned_records:>8d} {s.missing_fraction:>13.4f}"
        )
    return "\n".join(lines)
