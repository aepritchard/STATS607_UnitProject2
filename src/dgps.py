import numpy as np

class DGP:
    '''
    Generate covariates and define outcomes based on covariates with length n_sample
    Randomize treatment assignment n_sim times with probability pi_treat
    '''
    def __init__(self, n_sample, rng, n_sim=1000, pi_treat=0.75,
                 cut_control=0.5, cut_treat=0.75, observed_covariate="V"):
        self.n_sim = n_sim
        self.n_sample = n_sample
        self.rng = rng
        self.pi_treat = pi_treat
        self.cut_control = cut_control
        self.cut_treat = cut_treat
        self.covariate = "V"
        self.n_treat = int(round(pi_treat * n_sample))

    def generate_covariates(self):
        '''
        Covariates U and V with length n_sample are drawn from Uniform distribution
        Returns (2 x n_sample) matrix of values
        '''
        return self.rng.uniform(low=0.0, high=1.0, size=(2,self.n_sample))

    def generate_outcomes(self):
        '''
        Generates covariates, then uses covariates to define outcome under treatment
                and outcome under control
        '''
        covariates = self.generate_covariates()
        U = covariates[0, :]
        V = covariates[1, :]

        # Define outcomes based on covariates
        Y_control = np.zeros(self.n_sample, dtype=int)
        Y_control[U > self.cut_control] = 1

        Y_treatment = np.zeros(self.n_sample, dtype=int)
        Y_treatment[U + V > self.cut_treat] = 1

        if self.covariate == "V":
            Z = V
        elif self.covariate == "U+V":
            Z = U + V
        else:
            raise ValueError(rf"covariate must be 'V' or 'U+V', recieved {self.covariate}")

        self.Z = Z
        self.Y_control = Y_control
        self.Y_treatment = Y_treatment

    def generate_treatment(self, rng):
        '''
        Randomly assign treatment for one simulation repitition
        '''
        trt_assignment = np.zeros(self.n_sample, dtype=int)
        treated_index = self.rng.choice(self.n_sample, size=self.n_treat, replace=False)
        trt_assignment[treated_index] = 1
        return trt_assignment

    def generate_experiments(self):
        '''
        Generate experiements
        '''
        self.generate_outcomes()

        all_experiments = []
        for sim_index in range(self.n_sim):
            trt_assignment = self.generate_treatment()
            Y_observed = trt_assignment * self.Y_treatment + (1-trt_assignment) * self.Y_control

            experiment = {
                "sim_index": sim_index,
                "trt": trt_assignment,
                "Y_observed": Y_observed
            }
            all_experiments.append(experiment)

        return all_experiments

        