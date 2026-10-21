'''
Creates publication quality figures
Examines bias versus sample size for all experiments
'''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_settings import (N_SAMPLES, CONDITIONS, SEPARATE_SCALE_EXPERIMENT, ESTIMATOR_STYLES, DPI,
                             N_SAMPLE_COLORS, COEFFICIENT_STYLES,
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

def plot_probit_vs_logit(paired, out_dir):
    '''
    Experiment 1: probit vs logit coefficients
    One panel per coefficient (beta1, beta2, beta3): probit vs logit estimate,
            points colored by n, reference line probit = (5/8) logit
    '''
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, (coefficient, coefficient_style) in zip(axes, COEFFICIENT_STYLES.items()):
        all_logit_values = []

        for n_sample in N_SAMPLES:
            at_n = paired[paired["n_sample"] == n_sample]
            logit_values = at_n[f"{coefficient}_logit"].to_numpy()
            probit_values = at_n[f"{coefficient}_probit"].to_numpy()

            ax.scatter(logit_values, probit_values, s=6, alpha=0.5,
                       color=N_SAMPLE_COLORS[n_sample], label=f"n = {n_sample}")
            all_logit_values.append(logit_values)

        all_logit_values = np.concatenate(all_logit_values)
        x_line = np.linspace(np.min(all_logit_values), np.max(all_logit_values), 100)
        ax.plot(x_line, (5 / 8) * x_line, color="black", linestyle="--", linewidth=1,
                label=r"probit $= (5/8) \times$ logit")

        ax.set_title(coefficient_style["label"], fontsize=11)
        ax.set_xlabel("Logit estimate")
        ax.set_ylabel("Probit estimate")
        remove_top_right_spines(ax)

    # One legend: n colors and reference line
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, markerscale=3)

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    fig.savefig(Path(out_dir) / "probit_vs_logit.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def run_publication_figures(performance_path="results/tables/performance.csv",
                            paired_path="results/tables/probit_logit_paired.csv",
                            out_dir="results/figures"):
    '''
    Reads the summary tables made by make analyze, creates publication-quality figures
    '''
    # Create output folder
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tables
    performance = pd.read_csv(performance_path)
    paired = pd.read_csv(paired_path)

    plot_bias_vs_n(performance, out_dir)
    plot_probit_vs_logit(paired, out_dir)
