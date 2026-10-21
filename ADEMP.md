**Aims**:
Evaluate the performance of the logit model under misspecification as analyzed in Freedman (2008). 
In particular, we want to examine three estimators (MLE $\hat{\beta}_2$, plug-in estimator $\tilde{\delta}$, and ITT estimator $\hat{\delta}$) of the differential log odds ($\Delta$). 
Then, we want to examine:

* If $\hat{\beta}_2$ correctly estimates $\Delta$ (experiment 1).
* The plug-in estimator $\tilde{\delta}$ and the ITT estimator $\hat{\delta}$ are unbiased for $\Delta$ (experiment 1).
* The plug-in and ITT estimators have different variances (experiment 1).
* Whether changing the logit model to a probit model affects bias/variance of the estimators (experiment 1).
* Whether changing the proportion of samples treated affects bias/variance of the estimators (experiment 2).
* Whether changing the cut-off values affects bias/variance of the estimators (experiment 3).
* Whether changing the observed covariate affects bias/variance of the estimators (experiment 4).
* Whether changing the joint distribution of (U, V) affects bias/variance of the estimators (experiment 5).


**Data-generating mechanisms**:
Generate $n$ samples each of $U$ and $V$ that are drawn independently from a pre-specified distribution. 
Then, define the observed covariate $Z$. 
For each observation $i$ ($i=1,\dots,n$), define the outcome under control as $Y_i^C=1$ if $U_i>\gamma_C$ and as $Y_i^C=0$ otherwise. 
Define the outcome under treatment as $Y_i^T=1$ if $U_i+V_i> \gamma_T$ and as $Y_i^T=0$ otherwise. 
Freeze these variable definitions to use under $1000$ simulations. 
For each simulation, using complete randomization, assign $(\pi_C)*100\%$ of the samples to receive control (C) and assign $(1-\pi_C)*100\%$ of the samples to recieve treatment (T). 
Repeat this process for each level of $n$, where $n=100,500,1000,5000$.

| Experiment | Model(s) | Distribution of (U,V) | Observed Covariate | $\gamma_C$ | $\gamma_T$ | $\pi_C$ | $n$ |
|---|---|---|---|---|---|---|---|
| 1 | Logit, Probit | Uniform(0,1) | Z = V | $0.5$ | $0.75$ | $0.25$ | 100, 500, 1000, 5000 |
| 2 | Logit | Uniform(0,1) | Z = V | $0.5$ | $0.75$ | $0.25, 0.5, 0.75$ | 100, 500, 1000, 5000 |
| 3 | Logit | Uniform(0,1) | Z = V | $0.95$ | $1.5$ | $0.25$ | 100, 500, 1000, 5000 |
| 4 | Logit | Uniform(0,1) | Z = V + U | $0.5$ | $0.75$ | $0.25$ | 100, 500, 1000, 5000 |
| 5 | Logit | Beta(2, 5) | Z = V | $0.5$ | $0.75$ | $0.25$ | 100, 500, 1000, 5000 |

The exact values and specifications are set in `scenarios.py`. 
Each scenario uses its own random stream: the master seed (607) and a SHA-256 hash of the scenario ID are combined in a NumPy `SeedSequence`. The same stream generates the population and then the 1000 treatment assignments.

**Estimands**:
The finite-population differential log odds of success
$$\delta = \log\frac{\alpha^T}{1-\alpha^T} - \log\frac{\alpha^C}{1-\alpha^C},$$
where $\alpha^T$ and $\alpha^C$ are the population means of $Y^T$ and $Y^C$.

**Methods**:
For each treatment assignment randomization for each level of $n$, use maximum likelihood to fit logit/probit models of $Y$ on an intercept, treatment assignment, and observed covariate.
Compute the MLE coefficient estimates $\hat{\beta}$. 
The MLE estimates are used to compute average predicted probabilities ($\tilde{\alpha}^T, \tilde{\alpha}^C$), which are used to compute the plug-in differential log-odds estimator $\tilde{\Delta}$. 
The ITT (intention-to-treat) estimators ($\hat{\alpha}^T, \hat{\alpha}^C$) are used to compute the differential log-odds estimator $\hat{\delta}$. 
In experiment 1, both the probit and logit models are fit to the same randomizations to allow for direct comparison.

**Performance measures**:
For each scenario, model, and estimator, the following are calculated using all valid repetitions (repetitions without errors or warnings):

- Mean and empirical SD (with Monte Carlo SE)
- Bias relative to $\delta$ and RMSE (with Monte Carlo SE)
- Number of failed repetitions.