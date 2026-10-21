'''
Data Generating Processs Tests  
    Verify that the correct number of samples are assigned trt for each randomization
    Verify that the population remains the same (frozen) across repetitions
'''
import numpy as np
from dgps import DGP_Beta, DGP_Uniform

def test_trt_randomization():
    '''
    Every repetition assigns n_sample * pi_treat subjects to treatment
    '''
    n_sample = 100
    all_pi_treat = [0.25, 0.5, 0.75]

    rng = np.random.default_rng(607)
    for pi_treat in all_pi_treat:
        dgp_unif = DGP_Uniform(experiment="test",
                                n_sample = n_sample,
                                rng = rng,
                                n_sim = 20,
                                pi_treat=pi_treat)
        unif_repetitions = dgp_unif.generate_repetitions()

        dgp_beta = DGP_Beta(experiment="test",
                                n_sample = n_sample,
                                rng = rng,
                                n_sim = 50,
                                pi_treat=pi_treat)
        beta_repetitions = dgp_beta.generate_repetitions()

        for unif_rep in unif_repetitions:
            assert len(unif_rep['trt']) == n_sample
            assert unif_rep['trt'].sum() == n_sample * pi_treat

        for beta_rep in beta_repetitions:
                    assert len(beta_rep['trt']) == n_sample
                    assert beta_rep['trt'].sum() == n_sample * pi_treat

def test_population_same_across_reps():
    '''
    Across repetitions, make sure only treatment assignment changes
    Outcome under control or under treatment should remain the same 
        whenever a sample recieves control or treatment
    '''
    rng = np.random.default_rng(607)
    dgp_unif = DGP_Uniform(experiment="test",
                                    n_sample = 100,
                                    rng = rng,
                                    n_sim = 50,
                                    pi_treat=0.75)
    unif_repetitions = dgp_unif.generate_repetitions()
      
    for subject in range(100):
        # Make sure that outcome under trt and outcome under ctrl is consistent for each sample
        outcomes_under_trt = []
        outcomes_under_ctrl = []

        for rep in unif_repetitions:
            if rep['trt'][subject] == 1:
                outcomes_under_trt.append(rep['Y_observed'][subject])
            else:
                outcomes_under_ctrl.append(rep['Y_observed'][subject]) 

        assert len(set(outcomes_under_trt)) <= 1
        assert len(set(outcomes_under_ctrl)) <= 1

        # Make sure that treatment assignment DOES change across repetitions
        first_trt_assignment = unif_repetitions[0]['trt']
        assignment_changes = False
        for rep in unif_repetitions[1:]:
                if not np.array_equal(rep['trt'], first_trt_assignment):
                    assignment_changes = True
        assert assignment_changes == True