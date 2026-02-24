# KRGLM
Numerical experiments for our paper: Pseudo-Labeling for Unsupervised Domain Adaptation with Kernel GLMs"

## Solvers
rkhs_glm_scaled.py provides the basic solver for ridge-regularized kernel GLMs, using only numpy and scipy. It is appropriate for relatively small sample sizes ($n \le 5000$). 
rkhs_glm_scaled_KeOps.py leverages the PyKeOps library to compute matvec operations on the fly on the GPU. 
For more details, report to Section 6.1 of the paper.

This can be used as a general solver for kernel ridge regression, kernel logistic regression and kernel Poisson regression. Standard kernels are available e.g. linear, polynomial, RBF, Sobolev (first-order). 

## Synthetic experiments of Section 6.2
pseudo_label_experiment_general.py (resp. pseudo_label_experiment_general_KeOps.py for the KeOps version) implements our synthetic experiment for logistic regression with the first-order Sobolev kernel, detailed in section 6.2. 
It can be run with run_experiments_logistic.ipynb. 

As the full experiment can take some time to run, we provided the final results of the experiments in results_logistic_torchcpu_1_5_cos_0_4_shift.zip (covariate shift strength $B=(0.4)^n$) and results_logistic_torchcpu_1_5_cos_0_45_shift.zip (covariate shift strength $B=(0.45)^n$). 

The associated plots are logistic_errors_04.pdf and logistic_errors_045.pdf.

## Real experiment on the Raisin dataset from Section 6.3
The notebook final_exp_raisin.ipynb contains everything to reproduce the results presented in Section 6.3. It is self-contained.
