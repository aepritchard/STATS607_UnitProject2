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
        control_outcome = np.zeros(self.n_sample, dtype=int)
        control_outcome[U > self.cut_control] = 1

        treat_outcome = np.zeros(self.n_sample, dtype=int)
        treat_outcome[U + V > self.cut_treat] = 1

        if self.covariate == "V":
            Z= V
        elif self.covariate == "U+V":
            Z = U + V
        else:
            raise ValueError(rf"covariate must be 'V' or 'U+V', recieved {self.covariate}")

        