import numpy as np
from abc import ABC, abstractmethod


class DGP(ABC):
    '''
    Generate covariates and define outcomes based on covariates with length n_sample
    Randomize treatment assignment n_sim times with probability pi_treat
    '''
    def __init__(self, experiment, n_sample, rng, n_sim=1000, pi_treat=0.75,
                 cut_control=0.5, cut_treat=0.75, observed_covariate="V"):
        self.experiment = experiment
        self.n_sim = n_sim
        self.n_sample = n_sample
        self.rng = rng
        self.pi_treat = pi_treat
        self.cut_control = cut_control
        self.cut_treat = cut_treat
        self.covariate = observed_covariate
        self.n_treat = int(round(pi_treat * n_sample))

    def true_delta(self, link):
        '''
        Finite-population differential parameter on the scale of the link:
                log odds for the logit (eq 2), Phi^{-1} for the probit (Section 6)
        '''
        alpha_treat = np.mean(self.Y_treatment)
        alpha_control = np.mean(self.Y_control)
        return link(alpha_treat) - link(alpha_control)

    @abstractmethod
    def generate_covariates(self):
        '''
        Covariates U and V with length n_sample are drawn from specific distribution
        Returns (2 x n_sample) matrix of values
        '''
        pass

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

    def generate_treatment(self):
        '''
        Randomly assign treatment for one simulation repitition
        '''
        trt_assignment = np.zeros(self.n_sample, dtype=int)
        treated_index = self.rng.choice(self.n_sample, size=self.n_treat, replace=False)
        trt_assignment[treated_index] = 1
        return trt_assignment

    def generate_repetitions(self):
        '''
        Generate experiements
        '''
        self.generate_outcomes()

        all_repetitions = []
        for sim_index in range(self.n_sim):
            trt_assignment = self.generate_treatment()
            Y_observed = trt_assignment * self.Y_treatment + (1-trt_assignment) * self.Y_control

            repetition = {
                "experiment": self.experiment,
                "sim_index": sim_index,
                "trt": trt_assignment,
                "Y_observed": Y_observed
            }
            all_repetitions.append(repetition)

        return all_repetitions

class DGP_Uniform(DGP):
    def __init__(self, experiment, n_sample, rng, n_sim=1000, pi_treat=0.75, 
                 cut_control=0.5, cut_treat=0.75, observed_covariate="V"):
        super().__init__(experiment, n_sample, rng, n_sim, pi_treat, 
                         cut_control, cut_treat, observed_covariate)

    def generate_covariates(self):
        return self.rng.uniform(low=0.0, high=1.0, size=(2,self.n_sample))

class DGP_Beta(DGP):
    def __init__(self, experiment, n_sample, rng, n_sim=1000, pi_treat=0.75, 
                     cut_control=0.5, cut_treat=0.75, observed_covariate="V",
                     alpha = 2.0, beta=5.0):
        super().__init__(experiment, n_sample, rng, n_sim, pi_treat, 
                             cut_control, cut_treat, observed_covariate)
        
        if isinstance(alpha, (int, float)):
            if alpha > 0:
                self.alpha = alpha
            else:
                raise ValueError(rf"alpha must be > 0, recieved {alpha}")
        else:
            raise TypeError(rf"alpha must be a number, recieved {alpha}")

        if isinstance(beta, (int, float)):
            if beta > 0:
                self.beta = beta
            else:
                raise ValueError(rf"beta must be > 0, recieved {beta}")
        else:
            raise TypeError(rf"beta must be a number, recieved {beta}")
    
    def generate_covariates(self):
        return self.rng.beta(a=self.alpha, b=self.beta, size=(2,self.n_sample))
