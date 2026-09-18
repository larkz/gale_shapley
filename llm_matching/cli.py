"""cli.py

Command-line entry point for the LLM-task matching experiments.

Usage:
    python -m llm_matching.cli --config configs/llm_matching_smoke.yaml
    python -m llm_matching.cli --config configs/llm_matching_8x8.yaml \
        --algorithm p2etg --feedback bt --seed 0

CLI arguments override YAML values.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from llm_matching.runner import DEFAULT_CONFIG, deep_merge, run_experiment


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m llm_matching.cli",
        description="Offline LLM-task stable matching experiment on "
        "LLMRouterBench data (no LLM inference).",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to YAML config"
    )
    parser.add_argument(
        "--algorithm",
        choices=("p2etg", "preflid"),
        default=None,
        help="Learner (overrides config)",
    )
    parser.add_argument(
        "--feedback",
        choices=("bt", "replay"),
        default=None,
        help="Feedback mode (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        action="append",
        help="Run only this seed (repeatable; overrides config seeds)",
    )
    parser.add_argument(
        "--routerbench-root",
        type=Path,
        default=None,
        help="Path to the LLMRouterBench checkout (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip plot generation"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip matching report"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging"
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    return deep_merge(DEFAULT_CONFIG, user_cfg)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)

    algorithm = args.algorithm or "p2etg"
    feedback = args.feedback

    if args.routerbench_root is not None:
        config["routerbench_root"] = str(args.routerbench_root)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    seeds: Optional[List[int]] = None
    if args.seed is not None:
        seeds = [int(s) for s in args.seed]

    # Persist the fully resolved configuration.
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    per_seed = run_experiment(
        config,
        algorithm=algorithm,
        feedback=feedback,
        seeds_override=seeds,
        make_plots=not args.no_plots,
        make_report=not args.no_report,
    )
    n_stopped = int(pd.to_numeric(per_seed["stopped"]).sum()) if len(per_seed) else 0
    print(
        f"Done: {len(per_seed)} seed(s), {n_stopped} stopped. "
        f"Outputs in {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
