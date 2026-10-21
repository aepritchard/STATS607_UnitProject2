'''
Reproducibility test
    Verify that rerunning w/ the same seed gives identical results
    Verify that rerunning w/ a different seed gives different results
'''

import numpy as np
import pandas as pd
from simulation import SimulationStudy

# Test Experiment 1 w/ small number of samples and small number of simulations
TEST_SETTINGS = {
    "experiment": "test",
    "covariate_distribution": "uniform",
    "models": ["logit", "probit"],
    "n_sample": 100,
    "pi_treat": 0.75,
    "cut_control": 0.5,
    "cut_treat": 0.75,
    "observed_covariate": "V",
}

N_SIM = 50
SEED = 607

def run_test_scenario(scenario_id, out_dir):
    '''
    Runs one scenario w/ test settings and saves to out_dir
    '''
    scenarios = {scenario_id: TEST_SETTINGS}
    study = SimulationStudy(scenarios, n_sim=N_SIM, seed=SEED, out_dir=out_dir)
    results = study.run_scenario(scenario_id, TEST_SETTINGS)
    return results

def test_same_seed_gives_identical_results(tmp_path):
    '''
    Test that independent runs of the same scenario using the same seed match
    Each run writes to its own folder to prevent runs from overriding each other
    '''
    results_1 = run_test_scenario("test_scenario", tmp_path / "run_1a")
    results_2 = run_test_scenario("test_scenario", tmp_path / "run_2a")

    # Drop timing column since timing can vary b/w runs but doesn't affect results
    results_1 = results_1.drop(columns="seconds")
    results_2 = results_2.drop(columns="seconds")

    pd.testing.assert_frame_equal(results_1, results_2)

def test_diff_scenarios_gives_diff_results(tmp_path):
    '''
    Test that independent runs of different scenarios get different random streams
    Each run writes to its own folder to prevent runs from overriding each other
    '''
    results_1 = run_test_scenario("test_scenario_1b", tmp_path / "run_1b")
    results_2 = run_test_scenario("test_scenario_2b", tmp_path / "run_2b")

    # Compare beta2 estimates
    beta2_first = results_1["beta2_hat"].to_numpy()
    beta2_second = results_2["beta2_hat"].to_numpy()

    assert not np.allclose(beta2_first, beta2_second)