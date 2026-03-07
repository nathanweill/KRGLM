# KRGLM: Pseudo-Labeling for Unsupervised Domain Adaptation with Kernel GLMs

This repository contains the numerical experiments and generic solvers for our paper, *"Pseudo-Labeling for Unsupervised Domain Adaptation with Kernel GLMs."* Our goal is to minimize prediction error in the target domain by leveraging labeled source data and unlabeled target data, despite differences in covariate distributions. We partition the labeled source data into two batches: one for training a family of candidate models, and the other for building an imputation model. This imputation model generates pseudo-labels for the target data, enabling robust model selection.

## 🧮 Algorithmic Details
We implemented a generic solver for kernel GLMs in Python, using the Fisher scoring method. For full mathematical details and notes on our scalable GPU implementation with KeOps, please see our [Algorithmic Details document](ALGORITHM.md).

## 🛠️ Solvers
This repository provides a general solver for kernel ridge regression, kernel logistic regression, and kernel Poisson regression. Standard kernels are available (e.g., linear, polynomial, RBF, first-order Sobolev).

* `rkhs_glm_scaled.py`: Provides the basic solver for ridge-regularized kernel GLMs. For relatively small sample sizes ($n \le 5000$), a simple version using only Numpy and Scipy is enough. 
* `rkhs_glm_scaled_KeOps.py`: For larger problems, we implement the IRLS inner linear solves using kernel matvec oracles computed on-the-fly on the GPU, using the KeOps library. 

## 📊 Experiments

### Synthetic Data (Section 6.1)
We test our approach using logistic regression with the first-order Sobolev kernel. 
* **Run the experiment:** Use `run_experiments_logistic.ipynb`. This notebook calls `pseudo_label_experiment_general.py` (or `pseudo_label_experiment_general_KeOps.py` for the KeOps version).
* **Results:** Because the full experiment is computationally intensive, we have provided the final results in:
    * `results_logistic_torchcpu_1_5_cos_0_4_shift.zip` (covariate shift strength $B=n^{0.4}$) 
    * `results_logistic_torchcpu_1_5_cos_0_45_shift.zip` (covariate shift strength $B=n^{0.45}$) 
* **Plotting:** The results can be plotted using `plot_curves_synthetic.ipynb`, which outputs `logistic_errors_04.pdf` and `logistic_errors_045.pdf`.

### Real-World Data (Section 6.2)
We evaluate our method on the Raisin dataset. 
* **Run the experiment:** The notebook `final_exp_raisin.ipynb` contains everything needed to reproduce the results presented in the paper.

## 📝 Citation
If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{weill2026pseudolabeling,
  title={Pseudo-Labeling for Unsupervised Domain Adaptation with Kernel GLMs},
  author={Weill, Nathan and Wang, Kaizheng},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}



# KRGLM
Numerical experiments for our paper: Pseudo-Labeling for Unsupervised Domain Adaptation with Kernel GLMs"

## Solvers
rkhs_glm_scaled.py provides the basic solver for ridge-regularized kernel GLMs, using only numpy and scipy. It is appropriate for relatively small sample sizes ($n \le 5000$). 
rkhs_glm_scaled_KeOps.py leverages the PyKeOps library to compute matvec operations on the fly on the GPU. 
For more details, report to Section 6.1 and Appendix G.1 of the paper.

This can be used as a general solver for kernel ridge regression, kernel logistic regression and kernel Poisson regression. Standard kernels are available e.g. linear, polynomial, RBF, Sobolev (first-order). 

## Synthetic experiments of Section 6.1
pseudo_label_experiment_general.py (resp. pseudo_label_experiment_general_KeOps.py for the KeOps version) implements our synthetic experiment for logistic regression with the first-order Sobolev kernel, detailed in section 6.1. 
It can be run with run_experiments_logistic.ipynb. 

As the full experiment can take some time to run, we provided the final results of the experiments in results_logistic_torchcpu_1_5_cos_0_4_shift.zip (covariate shift strength $B=n^(0.4)$) and results_logistic_torchcpu_1_5_cos_0_45_shift.zip (covariate shift strength $B=n^(0.45)$). 

The results can be plotted using plot_curves_synthetic.ipynb. We included the final output graphs in logistic_errors_04.pdf and logistic_errors_045.pdf respectively.

## Real experiment on the Raisin dataset from Section 6.2
The notebook final_exp_raisin.ipynb contains everything to reproduce the results presented in Section 6.2.
