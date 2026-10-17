*Aims*:
Evaluate the performance of the logit model under misspecification (when the data generating
mechism is not logit).

*Data-generating mechanisms*:
Generate $n$ samples each of $U$ and $V$ that are drawn independently from the Uniform distribution. Then, define Z=V. For each observation $i$ ($i=1,\dots,n$), define the outcome under control as $Y_i^C=1$ if $U_i>1/2$ and as $Y_i^C=0$ otherwise. Define the outcome under treatment as $Y_i^T=1$ if $U_i+V_i> 3/4$ and as $Y_i^T=0$ otherwise. Freeze these variable definitions to use under $1000$ simulations. For each simulations, randomly assign $25\%$ of the samples to recieve control (C) and assign $75\%$ of the samples to recieve treatment (T). Repeat this process for each level of $n$, where $n=100,500,1000,5000$.

*Methods*:
For each treatment assignment randomization for each level of $n$, fit a logit model to the data and compute the MLE $\hat{\beta}$ and the plug-in estimator $\tilde{\delta}$

*Estimands*:


*Performance measures*: