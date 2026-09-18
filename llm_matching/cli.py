"""cli.py

Command-line entry point for the LLM-task matching experiments.

Main experiment (legacy form, unchanged):
    python -m llm_matching.cli --config configs/llm_matching_smoke.yaml
    python -m llm_matching.cli --config configs/llm_matching_8x8.yaml \
        --algorithm p2etg --feedback bt --seed 0

Phase-2 subcommands:
    python -m llm_matching.cli bootstrap     --config C [--num-bootstrap N]
    python -m llm_matching.cli sensitivity   --config C
    python -m llm_matching.cli diagnostics   --config C [--seed 0 --seed 1 ...]
    python -m llm_matching.cli matching-id   --config C [--n-seeds 20]
    python -m llm_matching.cli market-analysis --config C

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

SUBCOMMANDS = (
    "bootstrap", "sensitivity", "diagnostics", "matching-id",
    "market-analysis", "regret",
)


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
        choices=("p2etg", "preflid", "matching_id"),
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
    # Subcommand-specific options (ignored by the legacy main run).
    parser.add_argument(
        "--num-bootstrap", type=int, default=500, help="bootstrap replicates"
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=20260918, help="bootstrap seed"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=20, help="seeds for matching-id study"
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    return deep_merge(DEFAULT_CONFIG, user_cfg)


def _apply_common_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.routerbench_root is not None:
        config["routerbench_root"] = str(args.routerbench_root)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    return config


def _require_config(rest: List[str], subcmd: str) -> str:
    """Extract the --config value from a subcommand argv (or exit)."""
    it = iter(rest)
    for token in it:
        if token == "--config":
            try:
                return next(it)
            except StopIteration:
                break
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    print(f"error: subcommand '{subcmd}' requires --config PATH", file=sys.stderr)
    sys.exit(2)


def _run_subcommand(subcmd: str, argv: List[str]) -> int:
    rest = list(argv[1:])
    config_path = _require_config(rest, subcmd)
    args = parse_args(["--config", config_path] + rest)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _apply_common_overrides(load_config(Path(config_path)), args)

    if subcmd == "bootstrap":
        from llm_matching.bootstrap import run_bootstrap, save_bootstrap_outputs
        from llm_matching.runner import build_context

        ctx = build_context(config)
        result = run_bootstrap(
            ctx,
            num_bootstrap=args.num_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        save_bootstrap_outputs(ctx.out_dir, result)
        print(
            f"Bootstrap done: P(H_b == H_train) = "
            f"{result.p_equal_train_oracle:.3f}, "
            f"{result.n_distinct_matchings} distinct matchings. "
            f"Outputs in {ctx.out_dir / 'bootstrap'}"
        )
    elif subcmd == "sensitivity":
        from llm_matching.bootstrap import run_bootstrap
        from llm_matching.runner import build_context
        from llm_matching.sensitivity import (
            compute_arm_sensitivity,
            save_sensitivity_outputs,
        )

        ctx = build_context(config)
        boot = run_bootstrap(
            ctx,
            num_bootstrap=args.num_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        arms = compute_arm_sensitivity(ctx, bootstrap_result=boot)
        save_sensitivity_outputs(ctx.out_dir, arms)
        n_critical = int(
            arms["adjacent_reversal_changes_matching"].fillna(False).sum()
        )
        print(
            f"Sensitivity done: {len(arms)} arms, {n_critical} adjacent "
            f"matching-critical. Outputs in {ctx.out_dir / 'sensitivity'}"
        )
    elif subcmd == "diagnostics":
        from llm_matching.diagnostics import run_diagnostics_study

        seeds = [int(s) for s in args.seed] if args.seed else [0, 1, 2, 3, 4]
        combined = run_diagnostics_study(
            config,
            seeds=seeds,
            feedbacks=["bt", "replay"],
            num_bootstrap=args.num_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(
            f"Diagnostics done: {len(combined)} runs. Outputs in "
            f"{Path(config['output_dir']) / 'diagnostics_5seed'}"
        )
    elif subcmd == "matching-id":
        from llm_matching.diagnostics import run_matching_id_study

        seeds = (
            [int(s) for s in args.seed]
            if args.seed
            else list(range(args.n_seeds))
        )
        combined = run_matching_id_study(
            config, seeds=seeds, feedbacks=["bt", "replay"],
            algorithms=["p2etg", "matching_id"],
        )
        print(
            f"Matching-ID study done: {len(combined)} runs. Outputs in "
            f"{Path(config['output_dir']) / 'matching_id'}"
        )
    elif subcmd == "market-analysis":
        from llm_matching.market_analysis import run_market_analysis

        matrix = run_market_analysis(config)
        print(
            f"Market analysis done: {matrix.shape[0]} datasets x "
            f"{matrix.shape[1]} models. Outputs in "
            f"{Path(config['output_dir']) / 'market_analysis'}"
        )
    elif subcmd == "regret":
        from llm_matching.regret import write_regret_plots

        summary = write_regret_plots(Path(config["output_dir"]))
        print(
            f"Regret plots done. Outputs in "
            f"{Path(config['output_dir']) / 'regret'}"
        )
        cols = [
            "group", "n_seeds", "n_stopped", "final_regret_mean",
            "best_regret_mean", "committed_regret_mean",
        ]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in SUBCOMMANDS:
        return _run_subcommand(argv[0], argv)

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _apply_common_overrides(load_config(args.config), args)

    algorithm = args.algorithm or "p2etg"
    feedback = args.feedback

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
