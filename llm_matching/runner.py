"""runner.py

Experiment orchestration for the offline LLM-task stable matching study.

Pipeline:
  1. Load + align LLMRouterBench results (RouterBenchLoader)
  2. Deterministic train/val/test split
  3. Latent utilities U (task side) and V (model side, comparative
     advantage), strictified against ties
  4. Hidden oracle preferences + model-proposing GS matchings
     (train / val / test)
  5. Baselines: random bijections, Hungarian max-welfare
  6. BT calibration (eta_task, eta_model) from real utility gaps
  7. Per-seed learner runs (P2ETG primary; PrefLID optional) against an
     offline SignalProvider (BT or empirical replay)
  8. Traces, per-seed summary, aggregate summary, plots, report

The learner NEVER sees the oracle preference lists or the full utility
matrices; it only queries the provider for pairwise feedback.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gs_lib.gs_tools import Man, Matching, PreferenceList, Woman

from llm_matching.baselines import (
    hungarian_matching,
    random_welfare_stats,
)
from llm_matching.metrics import (
    count_blocking_pairs,
    exact_oracle_match,
    is_stable,
    matching_str,
    matching_to_dict,
    mean_test_score,
    normalized_welfare,
    resolved_fraction,
    test_welfare,
)
from llm_matching.oracle import (
    build_market,
    matching_equal,
    oracle_matching,
    preference_lists_to_dict,
)
from llm_matching.providers import (
    RouterBenchBTProvider,
    RouterBenchReplayProvider,
    build_bt_thetas,
    calibrate_eta,
)
from llm_matching.routerbench_loader import (
    AlignedTable,
    RouterBenchLoader,
    format_alignment_stats,
)
from llm_matching.splits import make_splits
from llm_matching.utilities import (
    median_positive_gap,
    model_utility,
    strictify_utilities,
    task_utility,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict = {
    "routerbench_root": "../LLMRouterBench",
    "min_aligned_records": 100,
    "datasets": [],
    "models": [],
    "split": {"train": 0.60, "val": 0.20, "test": 0.20, "seed": 3407},
    "feedback": {
        "mode": "bt",
        "target_median_win_probability": 0.70,
        "replay_gamma": 0.25,
        "replay_probability_epsilon": 0.01,
        "model_replay_mode": "probabilistic",
    },
    "ties": {"policy": "deterministic_jitter", "epsilon": 1.0e-6},
    "p2etg": {
        "adaptive": True,
        "check_every": 448,
        "max_samples": 200000,
        "constant": 0.1,
    },
    "preflid": {
        "budget": 100,
        "max_iterations": 200,
        "constant": 0.1,
        "max_lattice_vertices": 5000,
        "min_samples_per_pair": 10,
    },
    "experiment": {"seeds": 30, "random_baseline_seeds": 200},
    "output_dir": "outputs/llm_matching",
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# ============================================================================
# Context
# ============================================================================

@dataclass
class ExperimentContext:
    config: Dict
    out_dir: Path
    aligned: AlignedTable
    splits: pd.DataFrame
    task_util: Dict[str, pd.DataFrame]
    model_util: Dict[str, pd.DataFrame]
    task_util_strict: Dict[str, pd.DataFrame]
    model_util_strict: Dict[str, pd.DataFrame]
    tie_report: Dict[str, dict]
    men: List[Man]
    women: List[Woman]
    prefs: Dict[str, PreferenceList]
    oracle: Dict[str, Matching]
    hungarian: Matching
    random_welfare_test: Tuple[float, float]
    calibration: Dict[str, float]
    theta_task: Dict[str, Dict[str, float]] = field(default_factory=dict)
    theta_model: Dict[str, Dict[str, float]] = field(default_factory=dict)


def build_context(config: Dict) -> ExperimentContext:
    """Run all deterministic pre-processing and save context outputs."""
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(config["datasets"])
    models = list(config["models"])
    if not datasets or not models:
        raise ValueError("Config must define non-empty 'datasets' and 'models'")

    # ---- 1. Load + align -------------------------------------------------
    loader = RouterBenchLoader(
        root=Path(config["routerbench_root"]),
        datasets=datasets,
        models=models,
        min_aligned_records=int(config["min_aligned_records"]),
    )
    aligned = loader.load()
    print(format_alignment_stats(aligned.stats))
    alignment_df = pd.DataFrame([s.as_row() for s in aligned.stats])
    alignment_df.to_csv(out_dir / "data_alignment.csv", index=False)

    # ---- 2. Splits -------------------------------------------------------
    split_cfg = config["split"]
    splits = make_splits(
        aligned.records,
        split_seed=int(split_cfg["seed"]),
        train_frac=float(split_cfg["train"]),
        val_frac=float(split_cfg["val"]),
    )
    splits.to_csv(out_dir / "splits.csv", index=False)
    sizes = (
        splits.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    )
    logger.info("Split sizes:\n%s", sizes.to_string())

    # ---- 3. Utilities ----------------------------------------------------
    task_util: Dict[str, pd.DataFrame] = {}
    model_util: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        tu = task_utility(aligned.records, splits, split, datasets, models)
        task_util[split] = tu
        model_util[split] = model_utility(tu)
        tu.to_csv(out_dir / f"task_utility_{split}.csv")
        model_util[split].to_csv(out_dir / f"model_utility_{split}.csv")

    # ---- 4. Tie strictification ------------------------------------------
    tie_cfg = config["ties"]
    tie_epsilon = float(tie_cfg["epsilon"])
    split_seed = int(split_cfg["seed"])
    task_util_strict: Dict[str, pd.DataFrame] = {}
    model_util_strict: Dict[str, pd.DataFrame] = {}
    tie_report: Dict[str, dict] = {}

    for split in ("train", "val", "test"):
        tu_s, task_changed, task_groups = strictify_utilities(
            task_util[split], tie_epsilon, split_seed, agent_kind="task"
        )
        mu_s, model_changed, model_groups = strictify_utilities(
            model_util[split], tie_epsilon, split_seed, agent_kind="model"
        )
        task_util_strict[split] = tu_s
        model_util_strict[split] = mu_s
        tie_report[split] = {
            "task_side_changed_entries": task_changed,
            "task_side_groups": task_groups,
            "model_side_changed_entries": model_changed,
            "model_side_groups": model_groups,
        }
    with open(out_dir / "tie_report.json", "w") as fh:
        json.dump(tie_report, fh, indent=2, default=str)

    # ---- 5. Oracle markets -------------------------------------------------
    men, women, prefs_train = build_market(
        task_util_strict["train"], model_util_strict["train"]
    )
    oracle: Dict[str, Matching] = {}
    prefs: Dict[str, PreferenceList] = {"train": prefs_train}
    for split in ("train", "val", "test"):
        if split == "train":
            p = prefs_train
        else:
            _, _, p = build_market(
                task_util_strict[split], model_util_strict[split]
            )
        prefs[split] = p
        oracle[split] = oracle_matching(p)

    with open(out_dir / "oracle_preferences_train.json", "w") as fh:
        json.dump(preference_lists_to_dict(prefs_train), fh, indent=2)
    for split in ("train", "val", "test"):
        with open(out_dir / f"oracle_matching_{split}.json", "w") as fh:
            json.dump(matching_to_dict(oracle[split]), fh, indent=2)
    with open(out_dir / "oracle_generalization.json", "w") as fh:
        json.dump(
            {
                "train_equals_val": matching_equal(
                    oracle["train"], oracle["val"]
                ),
                "train_equals_test": matching_equal(
                    oracle["train"], oracle["test"]
                ),
            },
            fh,
            indent=2,
        )

    # ---- 6. Baselines ------------------------------------------------------
    hungarian = hungarian_matching(task_util["train"])
    with open(out_dir / "hungarian_matching.json", "w") as fh:
        json.dump(matching_to_dict(hungarian), fh, indent=2)

    n_rand = int(config["experiment"].get("random_baseline_seeds", 200))
    rand_mean, rand_std = random_welfare_stats(
        task_util["test"], n_seeds=n_rand, seed=split_seed
    )
    with open(out_dir / "random_baseline.json", "w") as fh:
        json.dump(
            {
                "n_seeds": n_rand,
                "test_welfare_mean": rand_mean,
                "test_welfare_std": rand_std,
            },
            fh,
            indent=2,
        )

    # ---- 7. BT calibration --------------------------------------------------
    feedback_cfg = config["feedback"]
    target_p = float(feedback_cfg["target_median_win_probability"])
    median_task_gap = median_positive_gap(task_util["train"], tie_epsilon)
    median_model_gap = median_positive_gap(model_util["train"], tie_epsilon)
    eta_task = calibrate_eta(median_task_gap, target_p)
    eta_model = calibrate_eta(median_model_gap, target_p)
    calibration = {
        "target_median_win_probability": target_p,
        "median_task_gap": median_task_gap,
        "median_model_gap": median_model_gap,
        "eta_task": eta_task,
        "eta_model": eta_model,
    }
    with open(out_dir / "bt_calibration.json", "w") as fh:
        json.dump(calibration, fh, indent=2)
    print(
        f"BT calibration: eta_task={eta_task:.4f} (median gap "
        f"{median_task_gap:.6f}), eta_model={eta_model:.4f} (median gap "
        f"{median_model_gap:.6f}), target win prob={target_p}"
    )

    theta_task, theta_model = build_bt_thetas(
        task_util_strict["train"], model_util_strict["train"],
        eta_task, eta_model,
    )

    return ExperimentContext(
        config=config,
        out_dir=out_dir,
        aligned=aligned,
        splits=splits,
        task_util=task_util,
        model_util=model_util,
        task_util_strict=task_util_strict,
        model_util_strict=model_util_strict,
        tie_report=tie_report,
        men=men,
        women=women,
        prefs=prefs,
        oracle=oracle,
        hungarian=hungarian,
        random_welfare_test=(rand_mean, rand_std),
        calibration=calibration,
        theta_task=theta_task,
        theta_model=theta_model,
    )


# ============================================================================
# Trace recording (extends run_with_trace's rounds mechanism)
# ============================================================================

class TraceRecorder(list):
    """Records one full metric row per algorithm stopping check.

    P2ETG's run_until_stop(rounds=...) appends (t, matching, disjoint)
    after every check; this list subclass expands each tuple into the
    per-check trace row required by the experiment spec.
    """

    def __init__(
        self,
        ctx: ExperimentContext,
        seed: int,
        algorithm: str,
        feedback_mode: str,
        learner,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.seed = seed
        self.algorithm = algorithm
        self.feedback_mode = feedback_mode
        self.learner = learner

    def append(self, item) -> None:  # type: ignore[override]
        t, matching, disjoint = item
        ctx = self.ctx
        row = {
            "seed": self.seed,
            "feedback_mode": self.feedback_mode,
            "algorithm": self.algorithm,
            "t": t,
            "matching": matching_str(matching),
            "exact_oracle_match": exact_oracle_match(matching, ctx.oracle["train"]),
            "train_stable": is_stable(ctx.prefs["train"], matching),
            "train_blocking_pairs": count_blocking_pairs(ctx.prefs["train"], matching),
            "test_welfare": test_welfare(matching, ctx.task_util["test"]),
            "test_mean_score": mean_test_score(matching, ctx.task_util["test"]),
            "pairwise_resolved_fraction": resolved_fraction(
                self.learner.agent_states
            ),
            "stopped": bool(disjoint),
        }
        super().append(row)


# ============================================================================
# Learner runs
# ============================================================================

def build_provider(ctx: ExperimentContext, feedback: str, seed: int):
    feedback_cfg = ctx.config["feedback"]
    if feedback == "bt":
        return RouterBenchBTProvider(
            theta_task=ctx.theta_task,
            theta_model=ctx.theta_model,
            rng=random.Random(f"bt-provider::{seed}"),
        )
    if feedback == "replay":
        return RouterBenchReplayProvider(
            records=ctx.aligned.records,
            splits=ctx.splits,
            datasets=list(ctx.config["datasets"]),
            models=list(ctx.config["models"]),
            rng=random.Random(f"replay-provider::{seed}"),
            replay_gamma=float(feedback_cfg["replay_gamma"]),
            replay_probability_epsilon=float(
                feedback_cfg["replay_probability_epsilon"]
            ),
            model_replay_mode=str(feedback_cfg.get("model_replay_mode", "probabilistic")),
        )
    raise ValueError(f"Unknown feedback mode {feedback!r}")


def run_p2etg_seed(
    ctx: ExperimentContext, seed: int, feedback: str
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    from p2etg import P2ETG

    cfg = ctx.config["p2etg"]
    provider = build_provider(ctx, feedback, seed)
    learner = P2ETG(
        men=ctx.men,
        women=ctx.women,
        provider=provider,
        rng=random.Random(f"p2etg::{seed}"),
        constant=float(cfg.get("constant", 0.1)),
    )
    recorder = TraceRecorder(
        ctx, seed=seed, algorithm="p2etg", feedback_mode=feedback, learner=learner
    )
    result = learner.run_until_stop(
        max_epochs=int(cfg.get("max_epochs", 400)),
        adaptive=bool(cfg.get("adaptive", True)),
        check_every=int(cfg.get("check_every", 448)),
        max_samples=int(cfg.get("max_samples", 200000)),
        verbose=False,
        rounds=recorder,
    )

    final_matching: Matching = result["matching"]
    stopped = bool(result["stopped"])
    t_stop = int(result["T_stop"])

    trace_df = pd.DataFrame(recorder)
    if not trace_df.empty and not stopped:
        trace_df.loc[trace_df.index[-1], "stopped"] = False

    rand_mean, _ = ctx.random_welfare_test
    hung_test = test_welfare(ctx.hungarian, ctx.task_util["test"])
    welfare = test_welfare(final_matching, ctx.task_util["test"])

    summary: Dict[str, object] = {
        "seed": seed,
        "algorithm": "p2etg",
        "feedback_mode": feedback,
        "stopped": stopped,
        "T_stop": t_stop,
        "exact_oracle_match": exact_oracle_match(final_matching, ctx.oracle["train"]),
        "stable_train": is_stable(ctx.prefs["train"], final_matching),
        "blocking_pairs_train": count_blocking_pairs(ctx.prefs["train"], final_matching),
        "resolved_fraction_at_stop": resolved_fraction(learner.agent_states),
        "test_mean_score": mean_test_score(final_matching, ctx.task_util["test"]),
        "test_welfare": welfare,
        "normalized_test_welfare": normalized_welfare(welfare, rand_mean, hung_test),
        "n_checks": len(trace_df),
        "final_matching": matching_str(final_matching),
    }
    return trace_df, summary


def run_preflid_seed(
    ctx: ExperimentContext, seed: int, feedback: str
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    from preflid import PrefLID

    cfg = ctx.config["preflid"]
    provider = build_provider(ctx, feedback, seed)
    learner = PrefLID(
        men=ctx.men,
        women=ctx.women,
        provider=provider,
        rng=random.Random(f"preflid::{seed}"),
        constant=float(cfg.get("constant", 0.1)),
        budget=int(cfg.get("budget", 100)),
        max_lattice_vertices=int(cfg.get("max_lattice_vertices", 5000)),
        min_samples_per_pair=int(cfg.get("min_samples_per_pair", 10)),
    )
    rounds: List[Dict] = []
    result = learner.run_until_stop(
        max_iterations=int(cfg.get("max_iterations", 200)),
        rounds=rounds,
        verbose=False,
    )

    final_matching: Matching = result["matching"]
    stopped = bool(result["stopped"])
    t_stop = int(result["T_stop"])

    trace_df = pd.DataFrame(rounds)
    for col in ("support", "H_star_str"):
        if col in trace_df.columns:
            trace_df[col] = trace_df[col].astype(str)

    rand_mean, _ = ctx.random_welfare_test
    hung_test = test_welfare(ctx.hungarian, ctx.task_util["test"])
    welfare = test_welfare(final_matching, ctx.task_util["test"])

    summary: Dict[str, object] = {
        "seed": seed,
        "algorithm": "preflid",
        "feedback_mode": feedback,
        "stopped": stopped,
        "T_stop": t_stop,
        "exact_oracle_match": exact_oracle_match(final_matching, ctx.oracle["train"]),
        "stable_train": is_stable(ctx.prefs["train"], final_matching),
        "blocking_pairs_train": count_blocking_pairs(ctx.prefs["train"], final_matching),
        "resolved_fraction_at_stop": resolved_fraction(learner.agent_states),
        "test_mean_score": mean_test_score(final_matching, ctx.task_util["test"]),
        "test_welfare": welfare,
        "normalized_test_welfare": normalized_welfare(welfare, rand_mean, hung_test),
        "n_checks": len(trace_df),
        "final_matching": matching_str(final_matching),
    }
    return trace_df, summary


# ============================================================================
# Aggregation
# ============================================================================

SUMMARY_NUMERIC_COLUMNS = [
    "T_stop",
    "exact_oracle_match",
    "stable_train",
    "blocking_pairs_train",
    "resolved_fraction_at_stop",
    "test_mean_score",
    "test_welfare",
    "normalized_test_welfare",
]


def aggregate_summaries(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in SUMMARY_NUMERIC_COLUMNS:
        if col not in per_seed.columns:
            continue
        # Booleans (exact_oracle_match, stable_train, ...) must become
        # floats before mean/std quantiles; numpy forbids bool subtract.
        series = pd.to_numeric(per_seed[col], errors="coerce").astype("float64")
        rows.append(
            {
                "metric": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Top-level
# ============================================================================

def run_experiment(
    config: Dict,
    algorithm: str = "p2etg",
    feedback: Optional[str] = None,
    seeds_override: Optional[List[int]] = None,
    make_plots: bool = True,
    make_report: bool = True,
) -> pd.DataFrame:
    """Run the full experiment; returns the per-seed summary frame."""
    if feedback is None:
        feedback = str(config["feedback"].get("mode", "bt"))
    if algorithm not in ("p2etg", "preflid"):
        raise ValueError(f"Unknown algorithm {algorithm!r}")

    ctx = build_context(config)

    seeds: List[int]
    if seeds_override is not None:
        seeds = [int(s) for s in seeds_override]
    else:
        seeds = list(range(int(config["experiment"]["seeds"])))

    traces_dir = ctx.out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    # Guard against silently overwriting traces of a different run mode.
    probe = traces_dir / f"seed_{seeds[0]:03d}.csv"
    if probe.exists():
        try:
            old = pd.read_csv(probe)
            old_combo = (
                str(old["algorithm"].iloc[0]), str(old["feedback_mode"].iloc[0])
            )
            if old_combo != (algorithm, feedback):
                logger.warning(
                    "Overwriting traces from a different run (%s %s) with "
                    "(%s %s). Use a separate output_dir per run mode.",
                    *old_combo, algorithm, feedback,
                )
        except Exception:  # noqa: BLE001 - best-effort guard
            pass

    run_fn = run_p2etg_seed if algorithm == "p2etg" else run_preflid_seed

    summaries: List[Dict[str, object]] = []
    all_traces: List[pd.DataFrame] = []
    for seed in seeds:
        logger.info("Running %s (%s feedback), seed=%d ...", algorithm, feedback, seed)
        trace_df, summary = run_fn(ctx, seed, feedback)
        trace_df.to_csv(traces_dir / f"seed_{seed:03d}.csv", index=False)
        all_traces.append(trace_df)
        summaries.append(summary)
        logger.info(
            "seed=%d stopped=%s T_stop=%s exact=%s welfare=%.4f",
            seed, summary["stopped"], summary["T_stop"],
            summary["exact_oracle_match"], float(summary["test_welfare"]),
        )

    per_seed = pd.DataFrame(summaries)
    per_seed.to_csv(ctx.out_dir / "per_seed_summary.csv", index=False)
    aggregate = aggregate_summaries(per_seed)
    n_stopped = int(pd.to_numeric(per_seed["stopped"]).sum()) if len(per_seed) else 0
    with open(ctx.out_dir / "aggregate_summary.csv", "w") as fh:
        aggregate.to_csv(fh, index=False)
    with open(ctx.out_dir / "run_meta.json", "w") as fh:
        json.dump(
            {
                "algorithm": algorithm,
                "feedback_mode": feedback,
                "seeds": seeds,
                "n_seeds": len(seeds),
                "n_stopped": n_stopped,
            },
            fh,
            indent=2,
        )

    print(
        f"\n=== {algorithm} / {feedback} over {len(seeds)} seed(s): "
        f"{n_stopped} stopped ==="
    )
    if len(per_seed):
        cols = [
            "seed", "stopped", "T_stop", "exact_oracle_match",
            "resolved_fraction_at_stop", "test_welfare",
        ]
        print(per_seed[cols].to_string(index=False))

    if make_plots:
        from llm_matching.plots import make_all_plots

        make_all_plots(ctx, all_traces, per_seed)

    if make_report:
        from llm_matching.plots import write_matching_report

        write_matching_report(ctx, per_seed, algorithm=algorithm, feedback=feedback)

    return per_seed
