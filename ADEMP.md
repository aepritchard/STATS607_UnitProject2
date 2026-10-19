**Aims**:
Evaluate the performance of the logit model under misspecification (when the data generating
mechism is not logit).

**Data-generating mechanisms**:
Generate $n$ samples each of $U$ and $V$ that are drawn independently from a pre-specified distribution. Then, define the observed coviarate Z. For each observation $i$ ($i=1,\dots,n$), define the outcome under control as $Y_i^C=1$ if $U_i>\gamma_C$ and as $Y_i^C=0$ otherwise. Define the outcome under treatment as $Y_i^T=1$ if $U_i+V_i> \gamma_T$ and as $Y_i^T=0$ otherwise. Freeze these variable definitions to use under $1000$ simulations. For each simulations, randomly assign $(pi_C)*100%$ of the samples to recieve control (C) and assign $(1-\pi_C)*100%$ of the samples to recieve treatment (T). Repeat this process for each level of $n$, where $n=100,500,1000,5000$.

| Experiment | Distribution of (U,V) | Observed Covariate | $\gamma_C$ | $\gamma_T$ | $\pi_C$
| 1 (Basline) | Uniform(0,1) | Z = V | $1/2$ | $3/4$ | $1/4$ |
| 2 | Uniform(0,1) | Z = V | $1/2$ | $3/4$ | $1/4, 1/2, 3/4$ |
| 3 | Uniform(0,1) | Z = V | $0.95$ | $1.5$ | $1/4$ |
| 4 | Uniform(0,1) | Z = V + U | $1/2$ | $3/4$ | $1/4$ |
| 5 | Beta(2, 5) | Z = V | $1/2$ | $3/4$ | $1/4$ |

The exact values and specifications are set in `scenarios.py`. 

**Estimands**:
MLE estimates $\hat{\beta}$ based on probit and logit models. 
MLE and ITT estimates of $\delta$.

**Methods**:
For each treatment assignment randomization for each level of $n$, calculate the true differntial log odds of success $\delta$. Fit logit and probit models to the data and compute the MLE $\hat{\beta}$. The MLE estimates are then used to compute average predicted probabilities ($\tilde{\alpha}^T, \tilde{\alpha}^C$), which are used to compute the plug-in differential log-odds estimator $\tilde{\delta}$. Additionally, the ITT (intention-to-treat) estimators ($\hat{\alpha}^T, \hat{\alpha}^C$) are used to compute the differential log-odds estimator $\hat{\delta}$.

**Performance measures**: