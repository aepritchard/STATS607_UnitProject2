# STATS 607 Unit Project 2

## Project Description:
This project reproduces the analysis of the paper "Randomization Does Not Justify Logistic Regression" (Freedman 2008). The simulation examines the use of the logit model under a non-logit data generating mechanism to examine the bias and variance of estimators of the differential log odds of success. Experiments are used to compare the results under different sample sizes, number of treated, data generating distributions, outcome definitions, and under a probit model.

## Setup Instructions
Python 3.13+ must be installed on the user's system. All required python packages are stored in `requirements.txt` and are installed using the command `make setup`. This setup process also runs when the `make all` command is used.

## How to Run the Analysis
In the main project directory, the following commands are defined in the Makefile:
* `make all`: installs required python packages, runs simulation and analysis, generates all figures
* `make setup`: installs required python packages
* `make simulate`: runs simulations and saves raw results
* `make analyze`: processes raw results and generates summary statistics
* `make clean`: removes all genrated files
* `make test`: runs all tests

The entire setup, simulation, analysis, and figure generating process takes xxxx minutes to run.

## Summary of Key Findings