# regret_plotting.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_palette("husl")


# ===========================================================================
# Helpers
# ===========================================================================

def _first_present(summary, *keys, default=None):
    """Return the first key present (and non-None) in summary, else default."""
    for k in keys:
        if k in summary and summary[k] is not None:
            return summary[k]
    return default


def _shade(hex_color, factor):
    """Blend a hex color toward white (factor<1) or keep as-is (factor=1)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r * factor + 255 * (1 - factor))
    g = int(g * factor + 255 * (1 - factor))
    b = int(b * factor + 255 * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def resolve_config_colors(config_names, color_groups=None, labels=None):
    """
    Return {config_name: color}.

    Group lookup is done on the DISPLAY LABEL (labels.get(cfg, cfg)),
    so COLOR_GROUPS keys should match the values in LABELS.

    - configs in color_groups: get a shaded variant within their group.
      Group base colors come from a fixed lookup; members are shaded
      light -> dark in the order they appear in config_names.
    - configs not in color_groups: fall back to the husl palette.
    """
    GROUP_BASES = {
        "blue":   "#1f77b4",
        "green":  "#2ca02c",
        "red":    "#d62728",
        "orange": "#ff7f0e",
        "purple": "#9467bd",
        "brown":  "#8c564b",
        "pink":   "#e377c2",
        "gray":   "#7f7f7f",
        "olive":  "#bcbd22",
        "cyan":   "#17becf",
    }

    color_groups = color_groups or {}
    labels = labels or {}
    label_of = lambda cfg: labels.get(cfg, cfg)

    group_members = {}      # group_name -> [config_name, ...]
    cfg_to_group = {}       # config_name -> group_name
    for cfg in config_names:
        display = label_of(cfg)
        g = color_groups.get(display)
        if g is not None:
            cfg_to_group[cfg] = g
            group_members.setdefault(g, []).append(cfg)

    ungrouped = [c for c in config_names if c not in cfg_to_group]
    fallback = sns.color_palette("husl", max(len(ungrouped), 1))

    colors = {}
    shade_positions = {}

    for cfg in config_names:
        g = cfg_to_group.get(cfg)
        if g is None:
            idx = ungrouped.index(cfg)
            colors[cfg] = fallback[idx]
            continue

        base = GROUP_BASES.get(g, "#333333")
        members = group_members[g]
        i = shade_positions.get(g, 0)
        shade_positions[g] = i + 1

        n = max(len(members), 1)
        factor = 0.55 + 0.45 * (i / max(n - 1, 1)) if n > 1 else 1.0
        colors[cfg] = _shade(base, factor)

    return colors


def _coerce_flag(x):
    """Coerce a 0/1 / true-false / None value into an int 0/1, or None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        if x == 0:
            return 0
        if x == 1:
            return 1
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1"):
            return 1
        if s in ("false", "0"):
            return 0
    return None


def _coerce_int(x):
    """Best-effort coercion to int, or None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def compute_min_bt_gap(summary):
    """Minimum gap between any two theta values in the BT preference matrices."""
    values = []

    for key in ("gt_true_theta_men", "gt_true_theta_women"):
        block = summary.get(key)
        if not isinstance(block, dict):
            continue
        for inner in block.values():
            if not isinstance(inner, dict):
                continue
            for v in inner.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))

    if len(values) < 2:
        return None

    values.sort()
    diffs = np.diff(values)
    return float(diffs.min())


def extract_min_bt_gap(summary):
    """Get min_bt_gap from a summary, preferring the explicit key."""
    explicit = summary.get("min_bt_gap")
    if explicit is not None and not isinstance(explicit, bool):
        if isinstance(explicit, (int, float)):
            return float(explicit)
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass

    return compute_min_bt_gap(summary)


def extract_in_mstar(summary):
    """Extract the 'correctly stopped / found stable matching' flag."""
    raw = _first_present(
        summary,
        "stable_under_truth",
        "preflid_in_mstar_true",
        "stable_under_hat",
        "preflid_stable_truth",
        "preflid_correct",
        "correct_at_stop",
    )
    if raw is not None:
        return _coerce_flag(raw)

    preflid_block = summary.get("preflid")
    if isinstance(preflid_block, dict):
        raw = _first_present(
            preflid_block,
            "in_mstar_true",
            "stable_truth",
            "correct",
        )
        if raw is not None:
            return _coerce_flag(raw)

    return None


def extract_n_stable_matchings(summary):
    """Unified stable-matchings count."""
    raw = _first_present(
        summary,
        "n_stable_matchings",
        "gt_stable_matchings_count",
    )
    return _coerce_int(raw)


# ===========================================================================
# Folder discovery & data loading
# ===========================================================================

def find_config_folders(base_dir):
    """Find all configuration folders in a FLAT directory."""
    base_dir = Path(base_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {base_dir}")

    configs = {}
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        name = folder.name
        if "_seed" not in name:
            continue

        parts = name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue

        config_name, seed_str = parts
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        configs.setdefault(config_name, []).append((seed, folder))

    for cfg in configs:
        configs[cfg].sort(key=lambda x: x[0])

    return configs


def load_regret_data(folder, max_iterations=200000):
    """Load one replication's data. Handles BOTH summary schemas."""
    rounds_path = folder / "rounds.csv"
    summary_path = folder / "summary.json"

    df = pd.read_csv(rounds_path)

    if "regret" not in df.columns:
        raise ValueError(f"'regret' column missing in {rounds_path}")
    if "correct" not in df.columns:
        raise ValueError(f"'correct' column missing in {rounds_path}")

    regret = df["regret"].to_numpy(dtype=float)

    expected = np.cumsum(1 - df["correct"].to_numpy(dtype=int))
    if len(regret) == len(expected) and not np.array_equal(regret, expected):
        print(f"  WARNING: 'regret' != cumsum(1 - 'correct') in {folder.name}")

    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    stopped = _first_present(summary, "stopped", "preflid_stopped",
                             default=False)
    T_stop = _first_present(summary, "T_stop", "preflid_T_stop",
                            "p2etg_T_stop", default=len(regret))
    correct_at_stop = _first_present(
        summary, "correct_at_stop", "preflid_correct", "p2etg_correct",
        default=None,
    )

    final_regret = regret[-1] if len(regret) > 0 else 0.0
    if len(regret) < max_iterations:
        padding = np.full(max_iterations - len(regret), final_regret)
        regret = np.concatenate([regret, padding])
    elif len(regret) > max_iterations:
        regret = regret[:max_iterations]

    return {
        "regret": regret,
        "stopped": stopped,
        "T_stop": T_stop,
        "correct_at_stop": correct_at_stop,
        "N": summary.get("N"),
        "K": summary.get("K"),
        "alpha": summary.get("alpha"),
    }


def load_all_configs(base_dir, max_iterations=200000):
    """Load regret data for every config/seed under base_dir."""
    config_folders = find_config_folders(base_dir)

    print(f"Found {len(config_folders)} configuration(s):")
    for cfg, sf in config_folders.items():
        print(f"  {cfg}: {len(sf)} seed(s)")

    all_data = {}
    for config_name, seeds_folders in config_folders.items():
        all_data[config_name] = {}
        for seed, folder in seeds_folders:
            try:
                all_data[config_name][seed] = load_regret_data(
                    folder, max_iterations
                )
            except Exception as e:
                print(f"  ERROR loading {folder.name}: "
                      f"{type(e).__name__}: {e}")

    return all_data


# ===========================================================================
# Plot 1: individual regret curves
# ===========================================================================

def plot_individual_regrets(all_data, max_iterations=200000,
                            save_path=None, figsize=(12, 7),
                            labels=None, color_groups=None, title=None):
    """One thin line per seed, colored by configuration."""
    labels = labels or {}
    label_of = lambda cfg: labels.get(cfg, cfg)

    fig, ax = plt.subplots(figsize=figsize)

    config_names = sorted(all_data.keys())
    config_colors = resolve_config_colors(config_names, color_groups, labels)

    for config_name in config_names:
        seeds_data = all_data[config_name]
        first_seed = min(seeds_data.keys())
        for seed, data in sorted(seeds_data.items()):
            regret = data["regret"]
            x = np.arange(1, len(regret) + 1)
            ax.plot(
                x, regret,
                color=config_colors[config_name],
                alpha=0.4,
                linewidth=0.8,
                label=label_of(config_name) if seed == first_seed else None,
            )

    ax.set_xlabel("Iteration (t)", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title(
        title if title else "Individual Cumulative Regret per Seed",
        fontsize=14,
    )
    ax.set_xlim(0, max_iterations)
    ax.legend(title="Configuration", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax


# ===========================================================================
# Plot 2: mean regret with shaded band
# ===========================================================================

def plot_mean_regret_with_std(all_data, max_iterations=200000,
                              save_path=None, figsize=(12, 7),
                              show_individual=True,
                              std_multiplier=1.0,
                              use_sem=False,
                              labels=None, title=None,
                              color_groups=None):
    """Mean cumulative regret per config with a mean ± k*spread shaded band."""
    labels = labels or {}
    label_of = lambda cfg: labels.get(cfg, cfg)

    fig, ax = plt.subplots(figsize=figsize)

    config_names = sorted(all_data.keys())
    config_colors = resolve_config_colors(config_names, color_groups, labels)

    for config_name in config_names:
        color = config_colors[config_name]
        seeds_data = all_data[config_name]

        regret_arrays = []
        for seed, data in sorted(seeds_data.items()):
            regret = data["regret"]
            if len(regret) < max_iterations:
                regret = np.pad(regret, (0, max_iterations - len(regret)),
                                mode="edge")
            regret_arrays.append(regret[:max_iterations])

        if not regret_arrays:
            continue

        regret_matrix = np.vstack(regret_arrays)
        n = regret_matrix.shape[0]
        mean_regret = regret_matrix.mean(axis=0)
        std_regret = (regret_matrix.std(axis=0, ddof=1)
                      if n > 1 else np.zeros_like(mean_regret))

        spread = std_regret / np.sqrt(n) if (use_sem and n > 1) else std_regret
        x = np.arange(1, max_iterations + 1)

        if std_multiplier > 0 and n > 1:
            ax.fill_between(
                x,
                mean_regret - std_multiplier * spread,
                mean_regret + std_multiplier * spread,
                color=color,
                alpha=0.2,
                linewidth=0,
            )

        ax.plot(x, mean_regret, color=color, linewidth=2,
                label=labels.get(config_name,
                                 f"{label_of(config_name)} (n={n})"))

        if show_individual:
            for regret in regret_arrays:
                ax.plot(x, regret, color=color, alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Iteration (t)", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    band_desc = f"±{std_multiplier:g}·{'SEM' if use_sem else 'std'}"
    default_title = f"Mean Cumulative Regret ({band_desc}, across seeds)"
    ax.set_title(title if title else default_title, fontsize=14)
    ax.set_xlim(0, max_iterations)
    ax.legend(title="Configuration", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax


# ===========================================================================
# Plot 3: summary bars (final regret, T_stop, correctness rate)
# ===========================================================================

def plot_summary_stats(all_data, save_path=None, figsize=(14, 5),
                       labels=None, titles=None, title_suffix=""):
    """Three bar charts: mean final regret, mean T_stop, correctness rate."""
    labels = labels or {}
    label_of = lambda cfg: labels.get(cfg, cfg)

    config_names = sorted(all_data.keys())
    display_names = [label_of(c) for c in config_names]

    final_regrets, t_stops, correct_rates = [], [], []
    for config_name in config_names:
        seeds_data = all_data[config_name]
        fr = [d["regret"][-1] for d in seeds_data.values()]
        ts = [d["T_stop"] for d in seeds_data.values()
              if d.get("T_stop") is not None]
        cr = [d["correct_at_stop"] for d in seeds_data.values()
              if d.get("correct_at_stop") is not None]

        final_regrets.append(np.mean(fr) if fr else np.nan)
        t_stops.append(np.mean(ts) if ts else np.nan)
        correct_rates.append(np.mean(cr) if cr else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].bar(display_names, final_regrets, color="steelblue")
    axes[0].set_title(f"Mean Final Regret{title_suffix}")
    axes[0].set_ylabel("Regret")

    axes[1].bar(display_names, t_stops, color="coral")
    axes[1].set_title(f"Mean T_stop{title_suffix}")
    axes[1].set_ylabel("Iterations")

    axes[2].bar(display_names, correct_rates, color="seagreen")
    axes[2].set_title(f"Correctness Rate at Stop{title_suffix}")
    axes[2].set_ylabel("Fraction Correct")
    axes[2].set_ylim(0, 1.05)

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig, axes


# ===========================================================================
# Plot 4: summary.json fields — mean ± std per configuration
# ===========================================================================

def plot_summary_fields(avg_df, fields, labels=None, figsize=None,
                        save_path=None, errorbars="std",
                        float_fmt="{:.3g}"):
    """Bar chart of averaged summary.json fields with error bars."""
    labels = labels or {}
    label_of = lambda cfg: labels.get(cfg, cfg)

    n_fields = len(fields)
    if figsize is None:
        figsize = (4.0 * n_fields, 4.5)

    config_names = avg_df["config"].tolist()
    display_names = [label_of(c) for c in config_names]
    x = np.arange(len(config_names))

    fig, axes = plt.subplots(1, n_fields, figsize=figsize, squeeze=False)
    axes = axes[0]

    palette = sns.color_palette("husl", len(config_names))

    for ax, field in zip(axes, fields):
        mean_col = f"{field}_mean"
        std_col = f"{field}_std"

        if mean_col not in avg_df.columns:
            ax.set_title(f"{field}\n(not present)")
            ax.set_xticks([])
            continue

        means = avg_df[mean_col].to_numpy(dtype=float)

        if errorbars and std_col in avg_df.columns:
            stds = avg_df[std_col].to_numpy(dtype=float)
            if errorbars == "sem":
                n = (avg_df["n_seeds"].to_numpy(dtype=float)
                     if "n_seeds" in avg_df.columns else 1)
                errs = np.where(n > 0, stds / np.sqrt(np.maximum(n, 1)), 0.0)
                err_label = "SEM"
            else:
                errs = np.nan_to_num(stds, nan=0.0)
                err_label = "std"
        else:
            errs = None
            err_label = None

        ax.bar(
            x, means,
            color=palette,
            yerr=errs,
            capsize=4,
            error_kw=dict(ecolor="black", lw=1),
        )

        for xi, m in zip(x, means):
            if np.isnan(m):
                continue
            ax.text(xi, m, float_fmt.format(m),
                    ha="center", va="bottom", fontsize=8)

        title = field if err_label is None else f"{field} (±{err_label})"
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(y=0.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig, axes


# ===========================================================================
# Summary.json averaging
# ===========================================================================

_NON_NUMERIC_FIELDS = {
    "run_id", "baseline", "oracle_str", "committed_str",
    "reason_truth", "reason_hat", "preflid_reason", "seed",
    "gt_true_theta_men", "gt_true_theta_women",
    "gt_prefs_men", "gt_prefs_women",
    "gt_h_star_men_pairs", "gt_h_star_women_pairs",
    "gt_stable_matchings_list",
    "gt_h_star_men_str", "gt_h_star_women_str",
    "gt_stable_matchings_error",
    "preflid", "p2etg",
}

_DROP_SOURCES = {
    "stable_under_truth",
    "stable_under_hat",
}


def _is_scalar_numeric(x):
    """int/float, not bool, not NaN. Bools are coerced to int upstream."""
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return not (isinstance(x, float) and np.isnan(x))
    return False


def collect_summaries(base_dir, verbose=False):
    """Load every *_seed*/summary.json under base_dir into a tidy DataFrame."""
    base_dir = Path(base_dir).expanduser().resolve()
    rows = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir() or "_seed" not in folder.name:
            continue

        cfg, seed_str = folder.name.rsplit("_seed", 1)
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        summary_path = folder / "summary.json"
        if not summary_path.exists():
            print(f"  [skip] no summary.json in {folder.name}")
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        scalar_summary = {k: v for k, v in summary.items()
                          if not isinstance(v, (dict, list))}

        gap_value = extract_min_bt_gap(summary)
        raw_gap = summary.get("min_bt_gap")
        if gap_value is None:
            gap_source = "missing"
        elif (raw_gap is not None and not isinstance(raw_gap, bool)
              and (isinstance(raw_gap, (int, float))
                   or (isinstance(raw_gap, str)
                       and raw_gap.strip() not in ("", "nan")))):
            gap_source = "explicit"
        else:
            gap_source = "computed"

        rounds_path = folder / "rounds.csv"
        final_regret = None
        if rounds_path.exists():
            try:
                rr = pd.read_csv(rounds_path, usecols=["regret"])
                if len(rr):
                    final_regret = float(rr["regret"].iloc[-1])
            except Exception as e:
                print(f"  [warn] could not read rounds.csv in "
                      f"{folder.name}: {e}")

        raw_flag = _first_present(summary, "stable_under_truth",
                                  "preflid_in_mstar_true")
        in_mstar = extract_in_mstar(summary)
        n_sm = extract_n_stable_matchings(summary)

        for k in _DROP_SOURCES:
            scalar_summary.pop(k, None)

        scalar_summary["min_bt_gap"] = gap_value
        scalar_summary["final_regret"] = final_regret
        scalar_summary["in_mstar"] = in_mstar if in_mstar is not None else 0
        scalar_summary["n_stable_matchings"] = (
            n_sm if n_sm is not None else 0
        )

        if verbose:
            print(f"  {folder.name:<40} "
                  f"min_bt_gap={gap_value!r:<22}[{gap_source}] "
                  f"in_mstar={scalar_summary['in_mstar']!r:<5}"
                  f"(raw={raw_flag!r}, type={type(raw_flag).__name__})")

        rows.append({"config": cfg, "seed": seed, **scalar_summary})

    if not rows:
        raise RuntimeError(f"No summary.json found under {base_dir}")

    return pd.DataFrame(rows)


def average_summaries(df, by="config", include_std=True, include_count=True,
                      min_count=1):
    """Average every numeric field, grouped by `by`."""
    numeric_cols = []
    for col in df.columns:
        if col in _NON_NUMERIC_FIELDS or col == by or col == "seed":
            continue
        vals = df[col].dropna()
        if len(vals) < min_count:
            continue
        if all(_is_scalar_numeric(v) for v in vals):
            numeric_cols.append(col)

    grouped = df.groupby(by, dropna=False)
    out = pd.DataFrame(index=grouped.size().index)

    if include_count:
        out["n_seeds"] = grouped.size()

    for col in numeric_cols:
        out[f"{col}_mean"] = grouped[col].mean()
        if include_std:
            out[f"{col}_std"] = grouped[col].std(ddof=1)

    return out.reset_index()


def print_summary_table(avg_df, fields=None, float_fmt="{:.3f}"):
    """Pretty-print a subset of averaged fields."""
    if fields is None:
        fields = sorted({c.rsplit("_mean", 1)[0] for c in avg_df.columns
                         if c.endswith("_mean")})

    header = f"{'config':<20} {'n':>4}  " + "  ".join(f"{f:>14}" for f in fields)
    print(header)
    print("-" * len(header))

    for _, row in avg_df.iterrows():
        cells = []
        for f in fields:
            m = row.get(f"{f}_mean", np.nan)
            s = row.get(f"{f}_std", np.nan)
            if pd.isna(m):
                cells.append(f"{'-':>14}")
            elif pd.isna(s):
                cells.append(f"{float_fmt.format(m):>14}")
            else:
                cells.append(f"{float_fmt.format(m)}±{float_fmt.format(s):>6}")
        print(f"{str(row['config']):<20} {int(row['n_seeds']):>4}  "
              + "  ".join(cells))


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # -------- CONFIG --------
    BASE_DIR = "path/to/your/results"
    MAX_ITERATIONS = 200000
    OUTPUT_DIR = "plots"
    VERBOSE = True

    LABELS = {
        "ver1_P2ETG_N3_K3_a2.0":     "N=3, K=3 P2ETG",
        "ver1_PrefLID_N3_K3_b20000": "N=3, K=3 PrefLID",
        "ver3_P2ETG_N5_K5_a2.0":     "N=5, K=5 P2ETG",
    }

    # COLOR_GROUPS keys are DISPLAY LABELS (the values in LABELS).
    # Configs sharing a group get the same base hue, shaded light -> dark.
    COLOR_GROUPS = {
        "N=3, K=3 P2ETG":     "green",
        "N=3, K=3 PrefLID":   "blue",
        "N=5, K=5 P2ETG":     "red",
    }

    MEAN_REGRET_TITLE = "Mean Cumulative Regret"
    INDIVIDUAL_TITLE  = "Individual Cumulative Regret per Seed"

    SUMMARY_FIELDS = [
        "final_regret",
        "min_bt_gap",
        "n_stable_matchings",
        "in_mstar",
        "preflid_is_hstar_men",
        "preflid_T_stop",
        "preflid_iterations",
        "T_stop_ratio",
        "T_stop", "n_epochs",
    ]
    # ------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Regret plots ----
    all_data = load_all_configs(BASE_DIR, max_iterations=MAX_ITERATIONS)

    # --- sanity check: do COLOR_GROUPS keys match any display label? ---
    display_labels = {LABELS.get(c, c) for c in all_data}
    print("\nDisplay labels found:", sorted(display_labels))
    print("COLOR_GROUPS keys   :", sorted(COLOR_GROUPS.keys()))
    missing = [k for k in COLOR_GROUPS if k not in display_labels]
    if missing:
        print("!! COLOR_GROUPS keys not matching any display label:", missing)

    plot_individual_regrets(
        all_data,
        max_iterations=MAX_ITERATIONS,
        save_path=os.path.join(OUTPUT_DIR, "individual_regrets.png"),
        labels=LABELS,
        color_groups=COLOR_GROUPS,
        title=INDIVIDUAL_TITLE,
    )

    plot_mean_regret_with_std(
        all_data,
        max_iterations=MAX_ITERATIONS,
        save_path=os.path.join(OUTPUT_DIR, "mean_regret_std.png"),
        show_individual=False,
        std_multiplier=0.1,
        labels=LABELS,
        title=MEAN_REGRET_TITLE,
        figsize=(7, 6),
        color_groups=COLOR_GROUPS,
    )

    plot_summary_stats(
        all_data,
        save_path=os.path.join(OUTPUT_DIR, "summary_stats.png"),
        labels=LABELS,
    )

    # ---- Summary averaging ----
    df = collect_summaries(BASE_DIR, verbose=VERBOSE)
    avg = average_summaries(df, by="config")
    avg.to_csv(os.path.join(OUTPUT_DIR, "summary_averages.csv"), index=False)

    print("\nAveraged summary fields:")
    print_summary_table(avg, fields=SUMMARY_FIELDS)

    # ---- Bar charts of the averaged fields ----
    plot_summary_fields(
        avg,
        fields=SUMMARY_FIELDS,
        labels=LABELS,
        errorbars="std",
        save_path=os.path.join(OUTPUT_DIR, "summary_fields.png"),
    )