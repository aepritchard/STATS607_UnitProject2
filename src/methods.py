import numpy as np
import statsmodels.api as sm

from abc import ABC, abstractmethod

class Methods(ABC):
    '''
    Estimators for one randomized experiment
    Fits the logit model, then computes the plug-in and ITT estimators
            of the differential log odds
    '''

    def __init__(self, trt, Y_observed, covariate_observed):
        self.trt = trt
        self.Y_observed = Y_observed
        self.covariate_observed = covariate_observed

    @staticmethod
    def logit(p):
        '''
        Returns logit of probability p
        '''
        if p <= 0 or p >= 1:
            raise ValueError(rf"p must be in (0,1), recieved {p}")
        return np.log(p / (1 - p))

    @staticmethod
    def p(beta, trt, covariate_observed):
        '''
        Fits equation (4) of paper
        Conditional probability that Y_i=1 given beta, X_i, Z_i
        '''
        linear_predictor = beta[0] + beta[1] * trt + beta[2] * covariate_observed
        return np.exp(linear_predictor) / (1 + np.exp(linear_predictor))

    @abstractmethod
    def fit_model(self):
        '''
        Fits model of Y on (1, trt, covariate_observed) with maximum likelihood
        Returns beta hat estimates
        '''
        pass
        

    def plug_in_estimator(self, beta_hat):
        '''
        Returns plug-in estimator for the differential log odds
        Gets average predicted probabilities (eq 10a) to get estimated differential log odds (eq 10b)
        '''
        # Get average predicted probabilities (eq 10a)
        alpha_tilde_treated = np.mean(self.p(beta_hat, 1, self.covariate_observed))
        alpha_tilde_control = np.mean(self.p(beta_hat, 0, self.covariate_observed))

        # Get estimated differential log odds (eq 10b)
        delta_tilde = self.logit(alpha_tilde_treated) - self.logit(alpha_tilde_control)
        return delta_tilde

    def itt_estimate(self):
        '''
        Returns ITT estimator of the differential log odds
        Gets ITT estimators (eq 11a) to get ITT estimated differential log odds (eq 11b)
        '''
        alpha_treat_hat = np.mean(self.Y_observed[self.trt == 1])
        alpha_control_hat = np.mean(self.Y_observed[self.trt == 0])

        delta_hat = self.logit(alpha_treat_hat) - self.logit(alpha_control_hat)
        return delta_hat

    def run_methods(self):
        '''
        Runs all methods on the stored experiment
        Returns a dict with the logit coefficients, plug-in estimate, and ITT estimate
        '''
        beta_hat = self.fit_logit()
        delta_tilde = self.plug_in_estimator(beta_hat)
        delta_hat = self.itt_estimate()

        results = {
            "beta1_hat": beta_hat[0],
            "beta2_hat": beta_hat[1],
            "beta3_hat": beta_hat[2],
            "plug_in": delta_tilde,
            "itt": delta_hat,
        }
        return results

class Methods_logit(Methods):
    def __init__(self, trt, Y_observed, covariate_observed):
        super().__init__(trt, Y_observed, covariate_observed)

    def fit_model(self):
        intercept = np.ones(len(self.Y_observed))
        design_matrix = np.column_stack((intercept, self.trt, self.covariate_observed))

        model = sm.Logit(self.Y_observed, design_matrix)
        fit = model.fit(disp=0)

        beta_hat = fit.params
        return beta_hat

class Methods_probit(Methods):
    def __init__(self, trt, Y_observed, covariate_observed):
        super().__init__(trt, Y_observed, covariate_observed)

    def fit_model(self):
        intercept = np.ones(len(self.Y_observed))
        design_matrix = np.column_stack((intercept, self.trt, self.covariate_observed))

        model = sm.Probit(self.Y_observed, design_matrix)
        fit = model.fit(disp=0)

        beta_hat = fit.params
        return beta_hat