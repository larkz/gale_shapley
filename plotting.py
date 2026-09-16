# regret_plotting.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def find_config_folders(base_dir):
    """
    Find all configuration folders matching pattern <config>_seed<N>.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {base_dir}")
    
    print(f"Scanning: {base_dir}")
    
    # Simpler, more robust: split on last "_seed"
    configs = {}
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        name = folder.name
        # Must contain "_seed"
        if "_seed" not in name:
            print(f"  Skipping (no '_seed'): {name}")
            continue
        
        # rsplit on "_seed" — everything before is config, after is seed number
        parts = name.rsplit("_seed", 1)
        if len(parts) != 2:
            print(f"  Skipping (bad format): {name}")
            continue
        
        config_name, seed_str = parts
        try:
            seed = int(seed_str)
        except ValueError:
            print(f"  Skipping (seed not int): {name}")
            continue
        
        configs.setdefault(config_name, []).append((seed, folder))
        print(f"  Found: config={config_name!r}, seed={seed}")
    
    # Sort seeds within each config
    for config_name in configs:
        configs[config_name].sort(key=lambda x: x[0])
    
    return configs


def load_regret_data(folder, max_iterations=200000):
    """
    Load regret data from a single configuration folder.

    The rounds.csv has columns:
        t, matching_str, disjoint, correct, regret
    - correct: 1 if matching is correct, 0 otherwise
    - regret:  already CUMULATIVE regret (do not cumsum)

    After the algorithm stops (summary.json: stopped=True, T_stop=X),
    regret stays flat at its final value for the remaining iterations.
    """
    rounds_path = folder / "rounds.csv"
    summary_path = folder / "summary.json"

    df = pd.read_csv(rounds_path)

    # Sanity check: 'regret' is cumulative, 'correct' is the 0/1 per-step flag
    if "regret" not in df.columns:
        raise ValueError(f"'regret' column missing in {rounds_path}")
    if "correct" not in df.columns:
        raise ValueError(f"'correct' column missing in {rounds_path}")

    # Cumulative regret is ALREADY in the column -- just take it as-is.
    regret = df["regret"].to_numpy(dtype=float)

    # Optional consistency check: regret should equal cumsum(1 - correct)
    expected = np.cumsum(1 - df["correct"].to_numpy(dtype=int))
    if len(regret) == len(expected) and not np.array_equal(regret, expected):
        print(f"  WARNING: 'regret' != cumsum(1 - 'correct') in {folder.name}")

    # Load summary
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)

    stopped = summary.get("stopped", False)
    T_stop = summary.get("T_stop", len(regret))
    correct_at_stop = summary.get("correct_at_stop", None)

    # ---- Pad with the FINAL regret value after stopping ----
    # This is what makes the curve "linear then flat" (step function).
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
    config_folders = find_config_folders(base_dir)
    
    print(f"\nFound {len(config_folders)} configuration(s):")
    for cfg, sf in config_folders.items():
        print(f"  {cfg}: {len(sf)} seeds")
    
    all_data = {}
    for config_name, seeds_folders in config_folders.items():
        all_data[config_name] = {}
        for seed, folder in seeds_folders:
            try:
                data = load_regret_data(folder, max_iterations)
                all_data[config_name][seed] = data
            except Exception as e:
                print(f"  ERROR loading {folder}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    
    return all_data


def plot_individual_regrets(all_data, max_iterations=200000, 
                            save_path=None, figsize=(12, 7)):
    """
    Plot individual cumulative regret curves for each seed and configuration.
    
    Parameters
    ----------
    all_data : dict
        Nested dict from load_all_configs.
    max_iterations : int
        Maximum number of iterations for x-axis.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    config_names = sorted(all_data.keys())
    palette = sns.color_palette("husl", len(config_names))
    config_colors = dict(zip(config_names, palette))
    
    for config_name in config_names:
        seeds_data = all_data[config_name]
        for seed, data in sorted(seeds_data.items()):
            regret = data["regret"]
            x = np.arange(1, len(regret) + 1)
            ax.plot(
                x, regret,
                color=config_colors[config_name],
                alpha=0.4,
                linewidth=0.8,
                label=config_name if seed == min(seeds_data.keys()) else None,
            )
    
    ax.set_xlabel("Iteration (t)", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title("Individual Cumulative Regret per Seed", fontsize=14)
    ax.set_xlim(0, max_iterations)
    ax.legend(title="Configuration", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax


def plot_mean_regret_with_std(all_data, max_iterations=200000,
                              save_path=None, figsize=(12, 7),
                              show_individual=True,
                              std_multiplier=1.0,
                              use_sem=False,
                              labels=None):
    """
    Plot mean cumulative regret across seeds with ±k*spread shaded area.

    Parameters
    ----------
    all_data : dict
        Nested dict from load_all_configs.
    max_iterations : int
        Maximum number of iterations for x-axis.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    figsize : tuple
        Figure size.
    show_individual : bool
        Whether to overlay individual seed curves (faintly).
    std_multiplier : float
        Multiplier k for the shaded band: mean ± k * spread.
    use_sem : bool
        If True, use standard error of the mean (std / sqrt(n)) instead of std.
    labels : dict, optional
        Map from config_name (folder prefix) -> display label in the legend.
        Missing entries fall back to the raw config_name.
        Example: {"N3_K3_a0.5": "N=3, K=3, α=0.5"}
    """
    labels = labels or {}
    label_of = lambda cfg: labels.get(cfg, cfg)

    fig, ax = plt.subplots(figsize=figsize)

    config_names = sorted(all_data.keys())
    palette = sns.color_palette("husl", len(config_names))

    for config_name, color in zip(config_names, palette):
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

        regret_matrix = np.vstack(regret_arrays)      # (n_seeds, T)
        n = regret_matrix.shape[0]
        mean_regret = regret_matrix.mean(axis=0)
        std_regret = regret_matrix.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean_regret)

        if use_sem and n > 1:
            spread = std_regret / np.sqrt(n)
        else:
            spread = std_regret

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
                label=f"{label_of(config_name)} (n={n})")

        if show_individual:
            for regret in regret_arrays:
                ax.plot(x, regret, color=color, alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Iteration (t)", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    band_desc = f"±{std_multiplier:g}·{'SEM' if use_sem else 'std'}"
    ax.set_title(f"Mean Cumulative Regret ({band_desc}, across seeds)", fontsize=14)
    ax.set_xlim(0, max_iterations)
    ax.legend(title="Configuration", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax




def plot_summary_stats(all_data, save_path=None, figsize=(12, 5)):
    """
    Plot bar charts of final regret, T_stop, and correctness rate.
    
    Parameters
    ----------
    all_data : dict
        Nested dict from load_all_configs.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    figsize : tuple
        Figure size.
    """
    config_names = sorted(all_data.keys())
    
    final_regrets = []
    t_stops = []
    correct_rates = []
    
    for config_name in config_names:
        seeds_data = all_data[config_name]
        fr = [d["regret"][-1] for d in seeds_data.values()]
        ts = [d["T_stop"] for d in seeds_data.values()]
        cr = [d["correct_at_stop"] for d in seeds_data.values()
              if d["correct_at_stop"] is not None]
        
        final_regrets.append(np.mean(fr))
        t_stops.append(np.mean(ts))
        correct_rates.append(np.mean(cr) if cr else 0)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].bar(config_names, final_regrets, color="steelblue")
    axes[0].set_title("Mean Final Regret")
    axes[0].set_ylabel("Regret")
    
    axes[1].bar(config_names, t_stops, color="coral")
    axes[1].set_title("Mean T_stop")
    axes[1].set_ylabel("Iterations")
    
    axes[2].bar(config_names, correct_rates, color="seagreen")
    axes[2].set_title("Correctness Rate at Stop")
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- CONFIGURE THIS ----
    BASE_DIR = "runs"   # <-- change me
    MAX_ITERATIONS = 200000
    OUTPUT_DIR = "plots"                # where to save figures
    # ------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all data
    print(f"Scanning {BASE_DIR} ...")
    all_data = load_all_configs(BASE_DIR, max_iterations=MAX_ITERATIONS)
    
    if not all_data:
        raise SystemExit("No configuration folders found. Check BASE_DIR.")
    
    print(f"Found {len(all_data)} configurations:")
    for cfg, seeds in all_data.items():
        print(f"  {cfg}: {len(seeds)} seeds")
    
    # 1) Individual regret curves
    plot_individual_regrets(
        all_data,
        max_iterations=MAX_ITERATIONS,
        save_path=os.path.join(OUTPUT_DIR, "individual_regrets.png"),
    )
    
    # 2) Mean ± std regret curves
    plot_mean_regret_with_std(
        all_data,
        max_iterations=MAX_ITERATIONS,
        save_path=os.path.join(OUTPUT_DIR, "mean_regret_std.png"),
    )
    
    # 3) Summary stats
    plot_summary_stats(
        all_data,
        save_path=os.path.join(OUTPUT_DIR, "summary_stats.png"),
    )