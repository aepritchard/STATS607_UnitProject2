'''
Test Method Correctness
    For a small dataset, calculate ITT by hand and compare to function
'''

import numpy as np
from dgps import DGP_Uniform
from methods import Methods_logit 

def test_itt_estimate_by_hand():
    '''
    For 10 subjects, 5 treated (w/ 4 success) and 5 control (w/ 2 success)
        Then alpha_T=4/5 and alpha_C=2/5
    ITT = logit(4/5) - logit(2/5) = log 6
    The covariate is not used by the ITT estimator
    '''
    trt = np.array([1,1,1,1,1,0,0,0,0,0])
    Y_observed = np.array([1,1,1,1,0,1,1,0,0,0])
    covariate_observed = np.zeros(10)

    methods = Methods_logit(trt, Y_observed, covariate_observed)
    delta_hat = methods.itt_estimate()

    assert np.isclose(delta_hat, np.log(6))

def test_plug_in_estimate_by_hand():
    '''
    beta = (0, log 3, 0), then p(1)=3/4 and p(0)=1/2
    Plug-in estimator = logit(3/4) - logit(1/2) = log 3
    '''
    trt = np.array([1,1,0])
    Y_observed = np.array([1,0,1])
    covariate_observed = np.array([0.1, 0.5, 0.9])
    beta = np.array([0.0, np.log(3), 0.0])

    methods = Methods_logit(trt, Y_observed, covariate_observed)
    delta_tilde = methods.plug_in_estimator(beta)

    assert np.isclose(delta_tilde, np.log(3))