import numpy as np
from rkhs_glm_scaled_KeOps import RKHSGLM_KeOps
from rkhs_glm_scaled import sigmoid, softplus, KERNELS
import torch

FAMILIES = {"gaussian", "logistic", "poisson", "neg-bin"}

def make_data(family, eta, rng, **params):
    n = eta.shape[0]
    if family == "gaussian":
        sigma = params.get("sigma", 1.0)
        return eta.astype(np.float32) , (eta + rng.normal(0, sigma, size=n)).astype(np.float32) 
    
    elif family == 'logistic':
        p = sigmoid(np.clip(eta, -10, 10))
        return p.astype(np.float32) , rng.binomial(1, p).astype(np.float32)
    
    elif family == 'poisson':
        rate_scale = params.get("rate_scale", 1.0)
        mu = np.exp(np.clip(rate_scale * eta, -8, 8))
        return mu.astype(np.float32) , rng.poisson(mu).astype(np.float32)
    
    elif family == 'neg-bin':
        alpha = params.get("alpha", None)
        rate_scale = params.get("rate_scale", 1.0)
        if alpha is None or alpha <= 0:
            raise ValueError("neg-bin requires alpha > 0 (overdispersion).")
        mu = np.exp(np.clip(rate_scale * eta, -8, 8))
        shape = 1.0 / alpha
        scale = alpha * mu
        lam = rng.gamma(shape, scale)
        return mu.astype(np.float32), rng.poisson(lam).astype(np.float32)
    raise ValueError(f"unknown family {family}")

def mean(family, eta, **params):
    if family == "gaussian":
        return eta.astype(np.float32)
    
    elif family == 'logistic':
        return sigmoid(np.clip(eta, -10, 10)).astype(np.float32)
    
    elif family == 'poisson':
        rate_scale = params.get("rate_scale", 1.0)
        mu = np.exp(np.clip(rate_scale * eta, -8, 8)).astype(np.float32)
        return mu
    
    elif family == 'neg-bin':
        alpha = params.get("alpha", None)
        rate_scale = params.get("rate_scale", 1.0)
        if alpha is None or alpha <= 0:
            raise ValueError("neg-bin requires alpha > 0 (overdispersion).")
        mu = np.exp(np.clip(rate_scale * eta, -8, 8))
        return mu.astype(np.float32)
    raise ValueError(f"unknown family {family}")
    

def a_logpartition(theta, family, **params):
    """
    Return log-partition a(theta).
    """
    theta = np.asarray(theta, np.float32)
    if family == "gaussian":
        return 0.5 * theta**2
    if family == "logistic":
        return softplus(theta)
    if family == "poisson":
        return np.exp(theta)
    if family == "neg-bin":
        alpha = params.get("alpha", None)
        # A(theta) = -(1/alpha) * log(1 - alpha * exp(theta))
        # Guard domain: alpha*exp(theta) < 1
        t = np.minimum(theta, np.log((1.0 - 1e-12)/alpha))
        return -(1.0/alpha) * np.log(1.0 - alpha * np.exp(t))
    raise ValueError(f"unknown family {family}")



class KGLM_covariate_shift:
    def __init__(self, n, n_0, B, fcn, family, kernel, seed, device=None, **kwargs): # preparations, sample generation
        # fcn is now a callable function e.g. lambda function passed directly as an argument
        # kwargs for distribution parameters: gaussian std, poisson rate scale, negbin rate scale and overdispersion
        assert B >= 1
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel '{kernel}'")
        if family not in FAMILIES:
            raise ValueError(f"Unknown family '{family}'")
        
        self.B = B
        rng=np.random.default_rng(seed)
        self.family = family
        self.kernel = kernel
        self.fcn = fcn
        self.alog = lambda theta: a_logpartition(theta, family, **kwargs)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # source data
        self.n = n        
        tmp = int(n * self.B / (self.B + 1))
        self.X = np.concatenate((rng.random(tmp) / 2, 1/2 + rng.random(n - tmp) / 2))
        rng.shuffle(self.X)
        eta = self.fcn(self.X).astype(np.float32) 
        self.mean, self.y = make_data(family, eta, rng, **kwargs)

        # target data
        self.n_0 = n_0
        tmp = int(n_0 * self.B / (self.B + 1))
        self.X_0 = np.concatenate((rng.random(n_0 - tmp) / 2, 1/2 + rng.random(tmp) / 2))
        eta_0 = self.fcn(self.X_0).astype(np.float32) 
        self.mean_0, _ = make_data(family, eta_0, rng, **kwargs)
        self.y_0 = self.mean_0 # noiseless response
        
    
    
    def fit(self, rho = 0.5, beta = 2): # KRR under covariate shift
        assert beta > 1
        assert 0 < rho and rho < 1
        
        # data splitting
        self.n_1 = int( (1 - rho) * self.n )
        self.n_2 = self.n - self.n_1
        self.X_1, self.y_1, self.mean_1 = self.X[0:self.n_1], self.y[0:self.n_1], self.mean[0:self.n_1]
        self.X_2, self.y_2, self.mean_2 = self.X[self.n_1:self.n], self.y[self.n_1:self.n], self.mean[self.n_1:self.n]  
        
        self.X_1 = self.X_1.astype(np.float32, copy=False)
        self.X_2 = self.X_2.astype(np.float32, copy=False)
        self.X_0 = self.X_0.astype(np.float32, copy=False)
        
        # penalty parameters: one for imputation, a geometric sequence for training
        lbd_tilde = 0.1 / self.n # for the imputation model
        lbd_min, lbd_max = 0.1 / self.n, 1 # min and max for training
        m = np.log(lbd_max / lbd_min) / np.log(beta)
        m = max( int(np.ceil(m)) , 2 ) + 1
        self.Lambda = lbd_min * ( beta ** np.array( range(m) ) ) # for training

        # pseudo-labeling (call the KRR solver in sklearn)
        mdl = RKHSGLM_KeOps(family=self.family, kernel=self.kernel, lam=lbd_tilde, fit_intercept=False)# clip_f=15, cg_tol=1e-5, precond="jacobi")
        mdl.fit(self.X_2, self.y_2)
        self.y_tilde = mdl.predict(self.X_0)
        self.mean_tilde = mdl.predict_mean(self.X_0)
        self.model_tilde = mdl

        # training (call the KRR solver in sklearn)
        self.Y_lbd = np.zeros((m, self.n_0))
        self.M_lbd = np.zeros((m, self.n_0))
        self.F_lbd = np.zeros((m, self.n_0))
        self.err_est_naive = np.zeros(m)  
        self.err_est_pseudo = np.zeros(m)
        self.err_est_real = np.zeros(m)    
        self.model = []
        for (j, lbd) in enumerate(self.Lambda):
            # KGLM
            mdl = RKHSGLM_KeOps(family=self.family, kernel=self.kernel, lam=lbd, fit_intercept=False)
            mdl.fit(self.X_1, self.y_1)
            y_lbd = mdl.predict(self.X_0)
            self.Y_lbd[j] = y_lbd
            f_lbd = mdl.decision_function(self.X_0)
            self.F_lbd[j] = f_lbd
            mean_lbd = mdl.predict_mean(self.X_0)
            self.M_lbd[j] = mean_lbd
            self.model.append(mdl)
            
            self.err_est_naive[j] = np.mean( self.alog(mdl.decision_function(self.X_2)) - self.y_2 * mdl.decision_function(self.X_2) )
            self.err_est_pseudo[j] = np.mean( self.alog(f_lbd) - self.mean_tilde * f_lbd )
            self.err_est_real[j] = np.mean( self.alog(f_lbd) - self.mean_0 * f_lbd )


        
        # selection
        self.j_naive = np.argmin(self.err_est_naive)
        self.lbd_naive = self.Lambda[self.j_naive]
        self.y_naive = self.Y_lbd[self.j_naive]
        self.mean_naive = self.M_lbd[self.j_naive]
        self.model_naive = self.model[self.j_naive]
       
        self.j_pseudo = np.argmin(self.err_est_pseudo)
        self.lbd_pseudo = self.Lambda[self.j_pseudo]
        self.y_pseudo = self.Y_lbd[self.j_pseudo]
        self.mean_pseudo = self.M_lbd[self.j_pseudo]
        self.model_pseudo = self.model[self.j_pseudo]
        
        self.j_real = np.argmin(self.err_est_real)
        self.lbd_real = self.Lambda[self.j_real]
        self.y_real = self.Y_lbd[self.j_real]
        self.mean_real = self.M_lbd[self.j_real]
        self.model_real = self.model[self.j_real]


    ############################################
    # evaluation of candidates
    def predict_candidates(self, X_new, list_idx_candidates, **kwargs): # make predictions using candidates
        eta_new = self.fcn(X_new).astype(np.float32) 
        self.mean_new_true = mean(self.family, eta_new, **kwargs)
        self.mean_new_candidates = []
        self.f_new_candidates = []
        for j in list_idx_candidates:
            model = self.model[j]
            self.mean_new_candidates.append(model.predict_mean(X_new))
            self.f_new_candidates.append(model.decision_function(X_new))

    def evaluate_candidates(self, distribution, list_idx_candidates, N_test, seed, **kwargs): # evaluate excess risk on the source or the target distribution
        rng=np.random.default_rng(seed)
        tmp = int(N_test * self.B / (self.B + 1))
        if distribution == 'target':
            self.X_test_0 = np.concatenate((rng.random(N_test - tmp) / 2, 1/2 + rng.random(tmp) / 2))
        elif distribution == 'source':
            self.X_test_0 = np.concatenate((rng.random(tmp) / 2, 1/2 + rng.random(N_test - tmp) / 2))

        self.predict_candidates(self.X_test_0, list_idx_candidates, **kwargs)
        self.err_candidates = []
        self.err_candidates_ste = []
        sqrt_N = np.sqrt(N_test)
        for i in range(len(list_idx_candidates)):
            tmp = self.alog(self.f_new_candidates[i]) - self.mean_new_true * self.f_new_candidates[i] #?? to respect the theory

            self.err_candidates.append( np.mean(tmp) )
            self.err_candidates_ste.append( np.std(tmp) / sqrt_N )


    ############################################
    # evaluation of selected models
    def predict_final(self, X_new, **kwargs): # make predictions using selected models
        eta_new = self.fcn(X_new).astype(np.float32) 
        self.mean_new_true = mean(self.family, eta_new, **kwargs)
        self.f_new_true = eta_new

        self.f_new_tilde = self.model_tilde.decision_function(X_new)
        self.mean_new_tilde = self.model_tilde.predict_mean(X_new)
        self.y_new_tilde = self.model_tilde.predict(X_new)
        
        self.f_new_naive = self.model_naive.decision_function(X_new)
        self.f_new_pseudo = self.model_pseudo.decision_function(X_new)
        self.f_new_real = self.model_real.decision_function(X_new)

        self.mean_new_naive = self.model_naive.predict_mean(X_new)
        self.mean_new_pseudo = self.model_pseudo.predict_mean(X_new)
        self.mean_new_real = self.model_real.predict_mean(X_new)

        self.y_new_naive = self.model_naive.predict(X_new)
        self.y_new_pseudo = self.model_pseudo.predict(X_new)
        self.y_new_real = self.model_real.predict(X_new)


    def evaluate_final(self, N_test, seed, **kwargs): # evaluate excess risk on the target distribution using selected models and newly generated samples
        rng=np.random.default_rng(seed)
        tmp = int(N_test * self.B / (self.B + 1))
        self.X_test_0 = np.concatenate((rng.random(N_test - tmp) / 2, 1/2 + rng.random(tmp) / 2))
        self.predict_final(self.X_test_0, **kwargs)

        sqrt_N = np.sqrt(N_test)
        
        tmp = self.alog(self.f_new_true) - self.mean_new_true * self.f_new_true #?? to respect the theory
        self.err_true = np.mean(tmp)
        self.err_true_ste = np.std(tmp) / sqrt_N
        
        tmp = self.alog(self.f_new_naive) - self.mean_new_true * self.f_new_naive #?? to respect the theory
        self.err_naive = np.mean(tmp) - self.err_true
        self.err_naive_ste = np.std(tmp) / sqrt_N

        tmp = self.alog(self.f_new_pseudo) - self.mean_new_true * self.f_new_pseudo #?? to respect the theory
        self.err_pseudo = np.mean(tmp) - self.err_true
        self.err_pseudo_ste = np.std(tmp) / sqrt_N

        tmp = self.alog(self.f_new_real) - self.mean_new_true * self.f_new_real  #?? to respect the theory
        self.err_real = np.mean(tmp) - self.err_true
        self.err_real_ste = np.std(tmp) / sqrt_N



# run experiments in Section 5.2

def run_experiment(n_list, seed_list, family, fcn, kernel, **kwargs):
    # idx: 1 to 100
    # n_list: list of sample sizes, e.g., [2000, 4000, 8000, 16000, 32000]
    # seed_list: list of random seeds

    #fcn = 'lin' # true function
    #sigma = 1 # standard deviation of noise
    beta = 2 # ratio parameter in the grid of lambdas
    N_test = 10000

    res = np.zeros((len(n_list), len(seed_list), 3))
    for (j, n) in enumerate(n_list):
        B = n ** (0.45)
        #B = int(round(n ** (1/3)))
        n_0 = n
        for (k, seed) in enumerate(seed_list):
            test = KGLM_covariate_shift(n, n_0, B, fcn, family, kernel, seed, **kwargs)
            test.fit(beta = beta)
            test.evaluate_final(N_test = N_test, seed = seed)
            res[j, k] = [test.err_naive, test.err_pseudo, test.err_real]

    return res
