from pathlib import Path
import numpy as np
import pandas as pd

ESTIMATORS = ["beta1_hat", "beta2_hat", "beta3_hat", "plug_in", "itt"]
DELTA_ESTIMATORS = ["beta2_hat", "plug_in", "itt"]
GROUP_COLUMNS = ["scenario_id", "experiment", "model", "n_sample", "pi_treat"]

def get_valid_reps(group):
    '''
    For a given group, gets repetitions that are valid, counts failures
    Rep is valid if it raised no errors/warnings
    '''
    is_error = group["beta2_hat"].isna()
    is_warning = (~is_error) & (group["failure_reason"] != "")
    is_valid = (~is_error) & (~is_warning)

    counts = {
        "n_reps": len(group),
        "n_valid": int(is_valid.sum()),
        "n_error": int(is_error.sum()),
        "n_warning": int(is_warning.sum()),
    }
    return group[is_valid], counts

def estimator_performance(estimates, true_delta, targets_delta):
    '''
    Performance of an estimator across valid repetitions
    Returns mean and empirical SD
    If estimator targets Delta (targets_delta == True), 
        returns bias and RMSE, with Monte Carlo SE
    '''
    n_valid = len(estimates)
    mean = np.mean(estimates)
    sd = np.std(estimates, ddof=1)

    result = {
        "mean": mean,
        "mcse_mean": sd / np.sqrt(n_valid),
        "sd": sd,
        "mcse_sd": sd / np.sqrt(2 * (n_valid - 1)),
        "bias": np.nan,
        "mcse_bias": np.nan,
        "rmse": np.nan,
        "mcse_rmse": np.nan,
    }

    if targets_delta == True:
        errors = estimates - true_delta
        squared_errors = errors ** 2
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        mcse_mse = np.std(squared_errors, ddof=1) / np.sqrt(n_valid)

        result["bias"] = mean - true_delta
        result["mcse_bias"] = sd / np.sqrt(n_valid)
        result["rmse"] = rmse
        result["mcse_rmse"] = mcse_mse / (2 * rmse) 

    return result

def performance_table(runs):
    '''
    Creates table summarizing performance
    One row per scenario/model/estimator combination
    '''
    rows = []
    for keys, group in runs.groupby(GROUP_COLUMNS):
        valid, counts = get_valid_reps(group)
        true_delta = group["true_delta"].iloc[0]   # constant within a scenario-model

        for estimator in ESTIMATORS:
            row = dict(zip(GROUP_COLUMNS, keys))
            row["estimator"] = estimator
            row["true_delta"] = true_delta
            row.update(counts)

            if counts["n_valid"] >= 2:
                performance = estimator_performance(
                    valid[estimator].to_numpy(), true_delta,
                    targets_delta=(estimator in DELTA_ESTIMATORS))
                row.update(performance)

            rows.append(row)

    return pd.DataFrame(rows)

def table1(performance, experiment="e1_table1", model="logit"):
    '''
    Reproduces Freedman's Table 1
    '''
    subset = performance[(performance["experiment"] == experiment)
                         & (performance["model"] == model)]

    rows = []
    for n_sample, group in subset.groupby("n_sample"):
        row = {"n_sample": n_sample}
        for estimator in ["beta1_hat", "beta2_hat", "beta3_hat", "plug_in"]:
            estimator_row = group[group["estimator"] == estimator].iloc[0]
            row[f"{estimator}_mean"] = estimator_row["mean"]
            row[f"{estimator}_sd"] = estimator_row["sd"]
        row["truth"] = group["true_delta"].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)



COEFFICIENTS = ["beta1_hat", "beta2_hat", "beta3_hat"]

def paired_probit_logit(runs, experiment="e1_table1"):
    '''
    Logit and probit coefficients fit to the same randomizations, matched by sim_index
    Keeps repetitions valid for both models
    One row per (n_sample, sim_index) with columns <coefficient>_logit and <coefficient>_probit
    '''
    experiment_runs = runs[runs["experiment"] == experiment]

    frames = []
    for n_sample, at_n in experiment_runs.groupby("n_sample"):
        valid_logit, counts_logit = get_valid_reps(at_n[at_n["model"] == "logit"])
        valid_probit, counts_probit = get_valid_reps(at_n[at_n["model"] == "probit"])

        columns = ["sim_index"] + COEFFICIENTS
        paired = pd.merge(valid_logit[columns], valid_probit[columns],
                          on="sim_index", suffixes=("_logit", "_probit"))
        paired.insert(0, "n_sample", n_sample)
        frames.append(paired)

    return pd.concat(frames, ignore_index=True)

def ratio_of_means(probit_estimates, logit_estimates):
    '''
    Ratio of mean probit estimate to mean logit estimate, with Monte Carlo SE by the delta method
    Var(a/b) ~ Var(a)/b^2 + a^2 Var(b)/b^4 - 2 a Cov(a, b)/b^3,
            where a, b are the two means and the covariance uses the pairing
    '''
    n_valid = len(probit_estimates)
    mean_probit = np.mean(probit_estimates)
    mean_logit = np.mean(logit_estimates)
    ratio = mean_probit / mean_logit

    var_mean_probit = np.var(probit_estimates, ddof=1) / n_valid
    var_mean_logit = np.var(logit_estimates, ddof=1) / n_valid
    cov_means = np.cov(probit_estimates, logit_estimates, ddof=1)[0, 1] / n_valid

    var_ratio = (var_mean_probit / mean_logit ** 2
                 + mean_probit ** 2 * var_mean_logit / mean_logit ** 4
                 - 2 * mean_probit * cov_means / mean_logit ** 3)
    mcse_ratio = np.sqrt(var_ratio)

    return ratio, mcse_ratio

def probit_logit_ratio_table(paired):
    '''
    One row per (n_sample, coefficient): ratio of mean probit to mean logit estimate and its MCSE
    '''
    rows = []
    for n_sample, at_n in paired.groupby("n_sample"):
        for coefficient in COEFFICIENTS:
            ratio, mcse_ratio = ratio_of_means(at_n[f"{coefficient}_probit"].to_numpy(),
                                               at_n[f"{coefficient}_logit"].to_numpy())
            rows.append({
                "n_sample": n_sample,
                "coefficient": coefficient,
                "n_paired": len(at_n),
                "ratio": ratio,
                "mcse_ratio": mcse_ratio,
            })

    return pd.DataFrame(rows)

def run_all_metrics(runs_path="results/raw/runs.csv", out_dir="results/tables"):
    '''
    Reads raw simulation results, computes performance measures, saves summary tables
    '''
    # Create output folder
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw results
    runs = pd.read_csv(runs_path)

    # Empty failure reasons are read back from the CSV as NaN, so set them back to ""
    runs["failure_reason"] = runs["failure_reason"].fillna("")

    # Performance for estimator/scenario
    performance = performance_table(runs)
    performance_path = out_dir / "performance.csv"
    performance.to_csv(performance_path, index=False)

    # Reproduce Table 1
    table1_reproduction = table1(performance)
    table1_path = out_dir / "table1_reproduction.csv"
    table1_reproduction.to_csv(table1_path, index=False)

    # Probit vs logit coefficients (experiment 1)
    paired = paired_probit_logit(runs)
    paired_path = out_dir / "probit_logit_paired.csv"
    paired.to_csv(paired_path, index=False)

    ratio_table = probit_logit_ratio_table(paired)
    ratio_path = out_dir / "probit_logit_ratio.csv"
    ratio_table.to_csv(ratio_path, index=False)