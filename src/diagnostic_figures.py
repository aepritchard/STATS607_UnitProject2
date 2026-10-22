'''
Generate diagnostic figures
    Examine running bias vs number of repeitions to see if it stabilizes
        Can check if 1000 is enough repetitions
Initial version of code was generated with assistance from Claude, then revised and edited
'''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import get_valid_reps
from figure_settings import (N_SAMPLES, CONDITIONS, SEPARATE_SCALE_EXPERIMENT, ESTIMATOR_STYLES, DPI,
                             select_condition, remove_top_right_spines, set_shared_y_limits)

def running_bias(estimates, true_delta):
    '''
    Calculates running mean of estimator bias by number of repeitions
    '''
    errors = estimates - true_delta
    cumulative_errors = np.cumsum(errors)
    n_reps = np.arange(1, len(errors) + 1)
    return n_reps, cumulative_errors / n_reps

def plot_running_bias(runs, out_dir, n_sample = 5000):
    fig, axes = plt.subplots(2,3, figsize=(10,8))
    axes = axes.flatten()

    axes_and_conditions = []
    shared_low_values = []
    shared_high_values = []

    for ax, condition in zip(axes, CONDITIONS):
        subset = select_condition(runs, condition["experiment"], condition["pi_treat"])
        subset = subset[subset["n_sample"] == n_sample]
        valid, counts = get_valid_reps(subset)
        valid = valid.sort_values("sim_index")
        true_delta = subset["true_delta"].iloc[0]

        for estimator, style in ESTIMATOR_STYLES.items():
            n_repetitions, bias_so_far = running_bias(valid[estimator].to_numpy(), true_delta)

            ax.plot(n_repetitions, bias_so_far,
                    color=style["color"], marker=style["marker"], markevery=100,
                    linestyle=style["linestyle"], markerfacecolor=style["markerfacecolor"],
                    label=style["label"], linewidth=1.2, markersize=5)

            if condition["experiment"] != SEPARATE_SCALE_EXPERIMENT:
                shared_low_values.append(np.nanmin(bias_so_far))
                shared_high_values.append(np.nanmax(bias_so_far))

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        title = condition["label"]
        ax.set_title(title, fontsize=11)

        remove_top_right_spines(ax)
        axes_and_conditions.append((ax, condition))

    set_shared_y_limits(axes_and_conditions, shared_low_values, shared_high_values)

    fig.supxlabel("Number of repetitions")
    fig.supylabel(r"Running bias (estimate $-\ \delta$)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(Path(out_dir) / "running_bias_n5000.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def run_diagnostic_figures(runs_path="results/raw/runs.csv", out_dir="results/figures"):
    '''
    Reads raw results, creates diagnostic figures
    '''
    # Create output folder
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw results
    runs = pd.read_csv(runs_path)
    runs["failure_reason"] = runs["failure_reason"].fillna("")

    plot_running_bias(runs, out_dir)