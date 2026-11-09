"""Draw the three panels of Figure 3 from run directories.

Uses the plotting routine of plotting.py (`plot_mean_regret_with_std`) with
the settings of the paper figure:
  * each run's cumulative regret is extended to the horizon with its final
    value (a stopped learner keeps playing its committed matching), so the
    mean at round t is taken over ALL seeds;
  * line = mean over seeds, shaded band = mean +/- 0.1 standard deviation;
  * x-axis "Iteration (t)" shown up to t = 50,000, y-axis "Cumulative Regret".

Inputs (default layout, see README):
  results/llm_routing/{p2etg,preflid}/<run>_seed<s>/  LLM-routing panel
      (included in this package)
  results/synthetic/<run>_seed<s>/                    synthetic panels
      (run directories written by experiment.py / experiment_preflid.py;
      the run names below are those of the submitted figure)
Runs shipped as rounds.csv.gz are decompressed on first use. A panel whose
runs are not present is skipped with a message.

Usage:  python make_figure3.py [--results results] [--out results/figures]
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import plotting as P

MAX_ITERATIONS = 50_000
STD_MULTIPLIER = 0.1

PANELS = {
    "mean_cum_regret_equal_nk": dict(
        title="Mean Cumulative Regret - Equal Set Agent Sizes",
        family="synthetic",
        series=[  # (subdir, config name, label)
            ("", "ver6_P2ETG_N5_K5_a2.0",        "N=5, K=5 P2ETG"),
            ("", "ver2_PrefLID_N5_K5_b20000",    "N=5, K=5 PrefLID"),
            ("", "ver6_P2ETG_N8_K8_a2.0",        "N=8, K=8 P2ETG"),
            ("", "ver25_PrefLID_N8_K8_b20000",   "N=8, K=8 PrefLID"),
        ]),
    "mean_cum_regret_unequal_nk": dict(
        title="Mean Cumulative Regret - Unequal Set Agent Sizes",
        family="synthetic",
        series=[
            ("", "ver1_P2ETG_N3_K5_a2.0",        "N=3, K=5 P2ETG"),
            ("", "ver1_PrefLID_N3_K5_b200000",   "N=3, K=5 PrefLID"),
            ("", "ver1_P2ETG_N5_K8_a2.0",        "N=5, K=8 P2ETG"),
            ("", "ver2_PrefLID_N5_K8_b200000",   "N=5, K=8 PrefLID"),
        ]),
    "mean_cum_regret_llm": dict(
        title="Mean Cumulative Regret - LLM Routing",
        family="llm_routing",
        series=[
            ("p2etg",   "floor15_P2ETG_N5_K5_real",   "N=5, K=5 P2ETG"),
            ("preflid", "floor15_PrefLID_N5_K5_real", "N=5, K=5 PrefLID"),
            ("p2etg",   "floor15_P2ETG_N8_K8_real",   "N=8, K=8 P2ETG"),
            ("preflid", "floor15_PrefLID_N8_K8_real", "N=8, K=8 PrefLID"),
        ]),
}
# first market olive, second blue; P2ETG dark, PrefLID light (as in the paper)
TONES = ["olive", "olive", "blue", "blue"]


def _ensure_dense(folder: Path) -> None:
    dense, packed = folder / "rounds.csv", folder / "rounds.csv.gz"
    if not dense.exists() and packed.exists():
        with gzip.open(packed, "rb") as src, open(dense, "wb") as dst:
            shutil.copyfileobj(src, dst)


def load_series(root: Path, subdir: str, config: str):
    data = {}
    for folder in sorted((root / subdir).glob(f"{config}_seed*")):
        seed = int(folder.name.rsplit("_seed", 1)[1])
        _ensure_dense(folder)
        data[seed] = P.load_regret_data(folder, max_iterations=MAX_ITERATIONS)
    if not data:
        raise FileNotFoundError(f"no runs {root / subdir / config}_seed*")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--panels", nargs="*", default=list(PANELS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.panels:
        spec = PANELS[name]
        root = Path(args.results) / spec["family"]
        all_data, labels, groups = {}, {}, {}
        try:
            for (subdir, config, label), tone in zip(spec["series"], TONES):
                all_data[config] = load_series(root, subdir, config)
                labels[config] = label
                groups[label] = tone
        except FileNotFoundError as err:
            print(f"{name}: skipped ({err})")
            continue
        fig, ax = P.plot_mean_regret_with_std(
            all_data, max_iterations=MAX_ITERATIONS,
            save_path=None, show_individual=False,
            std_multiplier=STD_MULTIPLIER, labels=labels,
            title=spec["title"], figsize=(7, 6), color_groups=groups)
        ax.title.set_fontweight("normal")  # paper figures use a regular title
        fig.savefig(out / f"{name}.pdf", format="pdf")
        fig.savefig(out / f"{name}.png", dpi=150)
        print(f"{name}: " + ", ".join(
            f"{labels[c]} n={len(d)}" for c, d in all_data.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
