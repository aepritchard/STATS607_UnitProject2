'''
Defines scenarios to be used in the simulation
'''

#### Experiment 1: Reproduce Table 1
# 25% f subjects assigned to control, 75% assigned to trt
# Control cut off = 1/2, treatment cut off = 3/4
# Observed covariate Z = V

#### Experiment 2: Change % treated
# Use same set-up as experiment 1, BUT change the % assigned to trt/ctrl
# Specific values were not specified in the paper

#### Experiment 3: Change cut offs
# Use same set-up as experiment 1, BUT change control and treatment cut offs
# Control cut off = 0.95, treatment cut off = 1.5

#### Experiment 4: Change observed covariate
# Use same set-up as experiment 1, BUT change observed covariate
# Observed covariate Z = V + U

#### Experiment 5: Change joint distribution
# Use same set-up as experiment 1, BUT change joint distribution of (U_i, V_i)
# Specific distribution used were not specified in the paper
# Try: Beta (same support)

#### Experiment 6: Fit probit model
# Use same set-up as experiment 1, BUT fit a probit model instead of a lofit model
