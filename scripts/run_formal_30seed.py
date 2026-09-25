#!/usr/bin/env python3
"""run_formal_30seed.py

Formal 30-seed 8x8 experiment launcher (P2ETG, BT or replay feedback).

PYTHONHASHSEED=0 is REQUIRED: gs_lib.bt._canonical uses Python's
hash(), which is otherwise randomised per process and makes runs
irreproducible (see the Phase-2 report).

Usage (from the gale_shapley repository root):
    PYTHONHASHSEED=0 python scripts/run_formal_30seed.py --feedback bt
    PYTHONHASHSEED=0 python scripts/run_formal_30seed.py --feedback replay \
        --output-dir outputs/llm_matching_8x8_replay

Arm annotations (sensitivity/arm_sensitivity.csv from the Phase-2
diagnostics study) are loaded when present so per-seed summaries carry
unresolved critical/non-critical counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_matching.runner import DEFAULT_CONFIG, deep_merge, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/llm_matching_8x8.yaml",
        help="Path to the 8x8 config",
    )
    parser.add_argument(
        "--feedback", choices=("bt", "replay"), required=True
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: config output_dir; use a "
             "separate dir for replay)",
    )
    args = parser.parse_args()

    if args.feedback == "replay" and args.output_dir is None:
        print(
            "error: replay runs must use --output-dir so the BT run is "
            "not overwritten",
            file=sys.stderr,
        )
        return 2

    with open(args.config, "r", encoding="utf-8") as fh:
        config = deep_merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})
    if args.output_dir:
        config["output_dir"] = args.output_dir

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    annotations = None
    ann_path = Path("outputs/llm_matching_8x8/sensitivity/arm_sensitivity.csv")
    if ann_path.exists():
        annotations = pd.read_csv(ann_path)
        print(f"Loaded arm annotations: {len(annotations)} arms")
    else:
        print(
            "WARNING: no arm annotations at "
            f"{ann_path}; summaries will lack critical-arm counts"
        )

    per_seed = run_experiment(
        config,
        algorithm="p2etg",
        feedback=args.feedback,
        arm_annotations=annotations,
    )
    stopped = int(pd.to_numeric(per_seed["stopped"]).sum()) if len(per_seed) else 0
    print(
        f"Formal {args.feedback} run complete: {len(per_seed)} seeds, "
        f"{stopped} stopped. Outputs in {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
