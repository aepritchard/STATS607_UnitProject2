'''
Creates publication quality figures
Examines bias versus sample size for all experiments
'''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_settings import (N_SAMPLES, CONDITIONS, SEPARATE_SCALE_EXPERIMENT, ESTIMATOR_STYLES, DPI,
                             select_condition, remove_top_right_spines, set_shared_y_limits)

def plot_bias_vs_n(performance, out_dir):
    '''
    Plots bias of beta2_hat, delta_tilde, delta_hat vs n
    One panel per condition w/ logit model
    Adds error bars that are +/- 1 Monte Carlo SE
    '''
    fig, axes = plt.subplots(2,3, figsize=(10,8))
    axes = axes.flatten()

    axes_and_conditions = []
    shared_low_values = []
    shared_high_values = []

    for ax, condition in zip(axes, CONDITIONS):
        subset = select_condition(performance, condition['experiment'], condition['pi_treat'])
        
        for estimator, style in ESTIMATOR_STYLES.items():
            rows = subset[subset["estimator"] == estimator].sort_values("n_sample")
            n_sample = rows["n_sample"].to_numpy()
            bias = rows["bias"].to_numpy()
            half_width = 1 * rows["mcse_bias"].to_numpy()

            ax.errorbar(n_sample, bias, yerr=half_width,
                        color=style["color"], marker=style["marker"], label=style["label"],
                        linestyle=style["linestyle"], markerfacecolor=style["markerfacecolor"],
                        capsize=3, linewidth=1.5, markersize=5)

            if condition["experiment"] != SEPARATE_SCALE_EXPERIMENT:
                shared_low_values.append(np.nanmin(bias - half_width))
                shared_high_values.append(np.nanmax(bias + half_width))

        ax.axhline(0, color='gray', linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks(N_SAMPLES)
        ax.set_xticklabels([str(n) for n in N_SAMPLES])
        ax.minorticks_off()
        title = condition['label']
        ax.set_title(title, fontsize=11)
        remove_top_right_spines(ax)
        axes_and_conditions.append((ax, condition))

    set_shared_y_limits(axes_and_conditions, shared_low_values, shared_high_values)

    fig.supxlabel(r'Sample size $n$')
    fig.supylabel(r"Bias (estimate $-\ \delta$), log-odds scale")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(Path(out_dir) / "bias_vs_n.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def run_publication_figures(performance_path="results/tables/performance.csv",
                            runs_path="results/raw/runs.csv",
                            out_dir="results/figures"):
    '''
    Reads the performance table, creates publication-quality figure
    '''
    # Create output folder
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load performance table and raw results
    performance = pd.read_csv(performance_path)
    runs = pd.read_csv(runs_path)
    runs["failure_reason"] = runs["failure_reason"].fillna("")

    plot_bias_vs_n(performance, out_dir)
    plot_probit_vs_logit(runs, out_dir)
