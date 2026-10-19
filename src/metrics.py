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