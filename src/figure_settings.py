'''
Define settings to be shared across figures to reduce repeat code
'''

import numpy as np

N_SAMPLES = [100,500,1000,5000]

# Order of experiments for panels
CONDITIONS = [
    {"label": "E1: baseline (Table 1)", "experiment": "e1_table1", "pi_treat": 0.75},
    {"label": "E2: 50% treated", "experiment": "e2_pct_treated", "pi_treat": 0.50},
    {"label": "E2: 25% treated", "experiment": "e2_pct_treated", "pi_treat": 0.25},
    {"label": "E3: cut offs 0.95, 1.5", "experiment": "e3_cutoffs", "pi_treat": 0.75},
    {"label": "E4: Z = U + V", "experiment": "e4_covariate", "pi_treat": 0.75},
    {"label": "E5: (U, V) ~ Beta(2, 5)", "experiment": "e5_joint_dist", "pi_treat": 0.75},
]

# Colors for sample sizes, light (n = 100) to dark (n = 5000), used when points are colored by n
N_SAMPLE_COLORS = {100: "#cbc9e2", 500: "#9e9ac8", 1000: "#756bb1", 5000: "#54278f"}

# Colors and markers for the model coefficients (not used for any estimator)
COEFFICIENT_STYLES = {
    "beta1_hat": {"label": r"$\hat{\beta}_1$ (intercept)", "color": "#E69F00", "marker": "o"},
    "beta2_hat": {"label": r"$\hat{\beta}_2$ (treatment)", "color": "#CC79A7", "marker": "s"},
    "beta3_hat": {"label": r"$\hat{\beta}_3$ (covariate)", "color": "#56B4E9", "marker": "^"},
}

# For all experiments except E4, have very small bias
# Want to share axes for easy comparison, but allow E4 to have its own axis
#   That way all plots are still visible
SEPARATE_SCALE_EXPERIMENT = "e4_covariate"

# Define colors and markers for each estimator
ESTIMATOR_STYLES = {
    "beta2_hat": {"label": r"$\hat{\beta}_2$ (logit coefficient)", "color": "#D55E00", "marker": "o",
                  "linestyle": "-", "markerfacecolor": "#D55E00"},
    "plug_in": {"label": r"Plug-in $\tilde{\delta}$", "color": "#0072B2", "marker": "s",
                "linestyle": "-", "markerfacecolor": "#0072B2"},
    "itt": {"label": r"ITT $\hat{\delta}$", "color": "#009E73", "marker": "^",
            "linestyle": "--", "markerfacecolor": "none"},
}

DPI = 300

def select_condition(frame, experiment, pi_treat, model="logit"):
    '''
    Returns the rows of a results table that match the condition and model
    '''
    # Check that given options are allowed
    is_experiment = frame["experiment"] == experiment
    is_pi_treat = np.isclose(frame["pi_treat"], pi_treat)
    is_model = frame["model"] == model

    # Return matching data
    return frame[is_experiment & is_pi_treat & is_model]

def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def set_shared_y_limits(axes_and_conditions, low_values, high_values):
    '''
    Gives all panels a shared y axis limit (except for the specified experiment w/ separate scale)
    '''                    
    low = np.nanmin(low_values)
    high = np.nanmax(high_values)
    padding = 0.05 * (high - low)

    for ax, condition in axes_and_conditions:
        if condition["experiment"] != SEPARATE_SCALE_EXPERIMENT:
            ax.set_ylim(low - padding, high + padding)