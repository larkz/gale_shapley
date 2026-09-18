#!/usr/bin/env python3
"""fetch_routerbench_results.py

Optionally download the official pre-collected LLMRouterBench benchmark
results (bench-release.tar.gz) from the HuggingFace dataset release
NPULH/LLMRouterBench and extract them into <LLMRouterBench>/results/.

The experiment itself never needs LLM inference; this script only
fetches static JSON benchmark outputs.

Usage (from the gale_shapley repository root):
    python scripts/fetch_routerbench_results.py --routerbench-root ../LLMRouterBench
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tarfile
from pathlib import Path

LOG = logging.getLogger("fetch_routerbench_results")

HF_URL = (
    "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/"
    "bench-release.tar.gz"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routerbench-root",
        type=Path,
        default=Path("../LLMRouterBench"),
        help="Path to the LLMRouterBench checkout",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded tar.gz after extraction",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = args.routerbench_root.resolve()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = results_dir / "bench"
    if bench_dir.is_dir() and any(bench_dir.iterdir()):
        LOG.info(
            "Benchmark results already present at %s; nothing to do.",
            bench_dir,
        )
        return 0

    archive = results_dir / "bench-release.tar.gz"
    LOG.info("Downloading %s ...", HF_URL)
    rc = subprocess.call(["curl", "-L", "-o", str(archive), HF_URL])
    if rc != 0 or not archive.exists():
        LOG.error("Download failed (curl rc=%d).", rc)
        return 1

    LOG.info("Extracting %s ...", archive)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(results_dir)

    extracted = results_dir / "bench-release"
    if extracted.is_dir() and not bench_dir.is_dir():
        extracted.rename(bench_dir)
    elif extracted.is_dir():
        LOG.warning(
            "Both %s and %s exist after extraction; keeping both.", extracted, bench_dir
        )

    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    if bench_dir.is_dir():
        LOG.info("Done. Benchmark results at %s", bench_dir)
        return 0
    LOG.error("Extraction finished but %s not found.", bench_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
