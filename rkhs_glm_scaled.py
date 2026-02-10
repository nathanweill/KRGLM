
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, cholesky
from scipy.sparse.linalg import LinearOperator, cg
from dataclasses import dataclass

# ===================== Hyperparameter tuning advice =====================
# For Poisson and NB2, select intercept regularization as 1e-6 of λ times average diag(K)

# ===================== Kernels =====================

"""def kernel_linear(X, Y, **kw):
    X = np.asarray(X)
    if X.ndim == 1:  # allow a single sample vector
        X = X[None, :]
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y[None, :]
    return X @ Y.T"""


def kernel_linear(X, Y=None):
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)

    # both 1-D -> outer product (Gram matrix)
    if X.ndim == 1 and Y.ndim == 1:
        if X.shape[0] != Y.shape[0]:
            # If you truly want outer regardless of length, remove this check.
            pass
        return np.outer(X, Y)

    # 1-D vs 2-D -> outer with each row/col appropriately
    if X.ndim == 1 and Y.ndim == 2:
        if X.shape[0] != Y.shape[1]:
            raise ValueError(f"Feature dims differ: {X.shape} vs {Y.shape}")
        # result (1, m): treat X as a single sample
        return (X[None, :] @ Y.T)
    if X.ndim == 2 and Y.ndim == 1:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(f"Feature dims differ: {X.shape} vs {Y.shape}")
        # result (n, 1): treat Y as a single sample
        return (X @ Y[:, None])

    # both 2-D
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Feature dims differ: {X.shape[1]} vs {Y.shape[1]}")
    return X @ Y.T

def kernel_linear_intercept(X, Y=None):
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)

    # both 1-D -> outer product (Gram matrix)
    if X.ndim == 1 and Y.ndim == 1:
        n_1, n_2 = len(X), len(Y)
        if X.shape[0] != Y.shape[0]:
            # If you truly want outer regardless of length, remove this check.
            pass
        return np.outer(X, Y) + np.ones((n_1, n_2))

    # 1-D vs 2-D -> outer with each row/col appropriately
    if X.ndim == 1 and Y.ndim == 2:
        if X.shape[0] != Y.shape[1]:
            raise ValueError(f"Feature dims differ: {X.shape} vs {Y.shape}")
        # result (1, m): treat X as a single sample
        return (X[None, :] @ Y.T) + np.ones((1, Y.shape[0]))
    if X.ndim == 2 and Y.ndim == 1:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(f"Feature dims differ: {X.shape} vs {Y.shape}")
        # result (n, 1): treat Y as a single sample
        return (X @ Y[:, None]) + np.ones((X.shape[0], 1))

    # both 2-D
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Feature dims differ: {X.shape[1]} vs {Y.shape[1]}")
    return X @ Y.T + np.ones((X.shape[0], Y.shape[0]))

def kernel_rbf(X, Y, lengthscale=1.0, **kw):
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2 * (X @ Y.T)
    ls2 = (lengthscale ** 2)
    return np.exp(-0.5 * D2 / ls2)

def kernel_rbf_intercept(X, Y=None, lengthscale=1.0, **kw):
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
    
    # Handle 1D case (though RBF is typically used with 2D)
    if X.ndim == 1 and Y.ndim == 1:
        n_1, n_2 = len(X), len(Y)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
    else:
        n_1, n_2 = X.shape[0], Y.shape[0]
    
    # Compute RBF kernel
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2 * (X @ Y.T)
    ls2 = (lengthscale ** 2)
    K = np.exp(-0.5 * D2 / ls2)
    
    # Add intercept term (constant matrix of ones)
    return K + np.ones((n_1, n_2))

def kernel_poly(X, Y, degree=3, c0=1.0, **kw):
    if X.ndim == 1 and Y.ndim == 1:
        if X.shape[0] != Y.shape[0]:
            # If you truly want outer regardless of length, remove this check.
            pass
        return (np.outer(X, Y) + c0) ** degree
    return (X @ Y.T + c0) ** degree

def kernel_poly_intercept(X, Y, degree=3, c0=1.0, **kw):
    if X.ndim == 1 and Y.ndim == 1:
        n_1, n_2 = len(X), len(Y)
        if X.shape[0] != Y.shape[0]:
            # If you truly want outer regardless of length, remove this check.
            pass
        return (np.outer(X, Y) + c0) ** degree + np.ones((n_1, n_2))
    return (X @ Y.T + c0) ** degree

def kernel_laplacian(X, Y, lengthscale=1.0, **kw):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for j in range(Y.shape[0]):
        K[:, j] = np.sum(np.abs(X - Y[j]), axis=1)
    return np.exp(-K / lengthscale)

def kernel_sobolev(X, Y, order=1, **kw):
    n_1, n_2 = len(X), len(Y)
    K = np.minimum(np.ones((n_1, 1)) @ Y.reshape(1, -1), X.reshape(-1, 1) @ np.ones((1, n_2)))
    return K #/ n_2

def kernel_sobolev_intercept(X, Y, order=1, **kw):
    n_1, n_2 = len(X), len(Y)
    K = np.minimum(np.ones((n_1, 1)) @ Y.reshape(1, -1), X.reshape(-1, 1) @ np.ones((1, n_2))) + np.ones((n_1, n_2))
    return K #/ n_2

KERNELS = {
    "rbf": kernel_rbf,
    "linear": kernel_linear,
    "linear_intercept": kernel_linear_intercept,
    "poly": kernel_poly,
    "poly_intercept": kernel_poly_intercept,
    "laplacian": kernel_laplacian,
    "sobolev": kernel_sobolev,
    "sobolev_intercept": kernel_sobolev_intercept,
    "rbf_intercept": kernel_rbf_intercept,
}

# ===================== Utils =====================

def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def _sobolev1_quantile_landmarks(x, r):
    qs = (np.arange(r)+0.5)/r
    return np.quantile(x.reshape(-1), qs)


@dataclass
class FitResult:
    alpha: np.ndarray
    train_obj: list
    iters: int
    solver: str

# ===================== RKHS-GLM =====================

class RKHSGLM_Scaled:
    def __init__(
        self,
        family="logistic",
        kernel="rbf",
        kernel_params=None,
        lam=1e-1,
        nb_alpha=0.5,
        solver="irls",
        max_iter=1000,
        tol=1e-6,
        backtrack=10,
        ls_c=1e-4,          # Armijo constant
        step0=1.0,
        verbose=False,
        clip_f=30.0,
        jitter=1e-8,
        fit_intercept=False, 
        lam_intercept=0.0,
        cg_tol=1e-6, 
        cg_maxiter=None,
        precond="jacobi",   # "jacobi" or "nystrom"
        nystrom_rank=0,     # >0 to enable Nyström PC
        random_state=None,
        normalize_kernel=False,
        ew_eta_max=0.5,
        ew_beta=0.5
    ):
        self.family = family
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.lam = float(lam)
        self.nb_alpha = float(nb_alpha)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.backtrack = int(backtrack)
        self.step0 = float(step0)
        self.verbose = verbose
        self.clip_f = float(clip_f)
        self.jitter = float(jitter)
        self.fit_intercept = bool(fit_intercept)
        self.lam_intercept = float(lam_intercept)
        self.b_ = 0.0
        self.normalize_kernel = bool(normalize_kernel)
        self._K_scale = 1.0  # store scale to rescale predictions/regs if needed
        self.ew_eta_max = float(ew_eta_max) # cap on forcing term
        self.ew_beta = float(ew_beta) # exponent in Eisenstat-Walker rule
        self.ew_prev = None # stores previous resiudal norm

        self.X_train_ = None
        self.K_ = None
        self.alpha_ = None
        self.train_history_ = None

        # ⟶ new (CG / PC config)
        self.cg_tol      = float(cg_tol)
        self.cg_maxiter  = None if cg_maxiter is None else int(cg_maxiter)
        self.precond     = str(precond)
        self.nystrom_rank = int(nystrom_rank)
        self.random_state = None if random_state is None else int(random_state)
        self.ls_c = float(ls_c)           # Armijo constant (tiny)

        # caches (filled in fit)
        self._K_fac = None         # cached factor for (K + jitter I)
        self._use_scipy_fac = False
        self._pc_L = None          # Nyström preconditioner factor L (n x r)
        self._pc_P = None          # small r x r inverse: (I + (1/lam_n) L^T L)^(-1)
        self._pc_lam_n = None      # lam_n used for _pc_P

    # ---------- kernels ----------
    def _gram(self, X, Y):
        if self.kernel not in KERNELS:
            raise ValueError(f"Unknown kernel '{self.kernel}'")
        return KERNELS[self.kernel](X, Y, **self.kernel_params)

    # ---------- mean & derivatives ----------
    def _pseudo_and_weights(self, f, y):
        fam = self.family
        f_c = np.clip(f, -self.clip_f, self.clip_f)

        if fam == "gaussian":  # identity link
            mu = f_c
            dmu = np.ones_like(f_c)
            V   = np.ones_like(f_c)   # σ^2 absorbed into loss scaling
        elif fam == "logistic":  # binomial, logit link
            mu = sigmoid(f_c)
            dmu = mu * (1.0 - mu)
            V   = mu * (1.0 - mu)
        elif fam == "poisson":   # log link
            mu = np.exp(f_c)
            dmu = mu
            V   = mu
        elif fam == "neg-bin":   # NB2, log link, dispersion theta = 1/a
            a = self.nb_alpha                        # a = 1/theta
            mu = np.exp(f_c)
            dmu = mu
            V   = mu + a * mu**2                     # NB2 variance
        else:
            raise ValueError(f"Unknown family '{fam}'")

        # Fisher scoring (universal):
        eps = 1e-16
        W = (dmu**2) / np.maximum(V, eps)            # diag weights
        z = f_c + (y - mu) / np.maximum(dmu, eps)    # pseudo-response
        return z, W, mu
    
    # ---------- SPD solver cached to retrieve alpha from f ----------
    def _prep_K_factor(self, K):
        A = 0.5*(K + K.T) + self.jitter * np.eye(K.shape[0])
        try:
            self._K_fac = cho_factor(A, lower=True, check_finite=False)
            self._use_scipy_fac = True
        except Exception:
            self._K_L = np.linalg.cholesky(A)
            self._use_scipy_fac = False

    def _solve_with_K(self, f):
        if self._use_scipy_fac:
            return cho_solve(self._K_fac, f, check_finite=False)
        # NumPy fallback
        y = np.linalg.solve(self._K_L, f)
        return np.linalg.solve(self._K_L.T, y)
    
    # ---------- Preconditioner ----------
    def _prep_nystrom_pc(self, K, lam_n):
        # rank-r Nyström using uniform landmark selection (fast & simple)
        n = K.shape[0]
        r = self.nystrom_rank
        if r <= 0 or r > n:
            self._pc_L = self._pc_P = self._pc_lam_n = None
            return
        rng = np.random.default_rng(self.random_state)
        J = rng.choice(n, size=r, replace=False)

        C = K[:, J]                          # n x r
        W = K[np.ix_(J, J)]                  # r x r
        # Stabilize & invert sqrt(W)
        W = 0.5*(W + W.T) + self.jitter*np.eye(r)
        evals, evecs = np.linalg.eigh(W)
        evals = np.clip(evals, 1e-12, None)
        Winvhalf = (evecs * (evals**-0.5)) @ evecs.T    # W^{-1/2}
        L = C @ Winvhalf                                 # n x r, so that K ≈ L L^T
        LtL = L.T @ L                                    # r x r
        P = np.linalg.inv(np.eye(r) + (1.0/lam_n) * LtL) # (I + (1/lam) L^T L)^{-1}

        self._pc_L = L
        self._pc_P = P
        self._pc_lam_n = float(lam_n)

    def _make_preconditioner(self, K, s, lam_n):
        """
        Return a LinearOperator that approximates (K_s + lam_n I)^{-1}
        where K_s = S K S, S = diag(s).
        """
        n = K.shape[0]

        if self.precond == "nystrom" and (self._pc_L is not None) and (abs(self._pc_lam_n - lam_n) <= 1e-12):
            L = self._pc_L            # n x r
            P = self._pc_P            # r x r inverse
            s_inv = 1.0 / s
            L = L.astype(np.float32, copy=False)

            # Apply M^{-1} r = S^{-1} * ( (lam I + L L^T)^{-1} (S^{-1} r) ) with Woodbury
            # (lam I + L L^T)^{-1} u = (1/lam) * (u - L * (I + (1/lam) L^T L)^{-1} * L^T u)
            def mapply(r):
                u = s_inv * r
                tmp = L.T @ u                  # r
                y  = P @ tmp                   # r      (small system already inverted)
                w  = (u - L @ y) / lam_n       # n
                return s_inv * w
            return LinearOperator((n, n), matvec=mapply, dtype=K.dtype)

        # default Jacobi: diag(K_s) + lam_n ≈ s^2 * diag(K) + lam_n
        dK = np.diag(K).copy()
        Mdiag = (s**2) * dK + lam_n
        Mdiag[Mdiag == 0] = 1.0
        return LinearOperator((n, n), matvec=lambda r: r / Mdiag, dtype=K.dtype)


    def _solve_spd(self, A, b):
        # Symmetrize + jitter to ensure numerical SPD
        A = 0.5 * (A + A.T)
        A = A + self.jitter * np.eye(A.shape[0])
        #c, lower = cho_factor(A, lower=True, check_finite=False, overwrite_a=False)
        #x = cho_solve((c, lower), b, check_finite=False, overwrite_b=False)
        L = cholesky(A, lower=True, check_finite=False)
        y = solve_triangular(L, b, lower=True, check_finite=False)
        x = solve_triangular(L.T, y, lower=False, check_finite=False)
        #L = np.linalg.cholesky(A)
        #y = np.linalg.solve(L, b)
        #return np.linalg.solve(L.T, y)
        return x

    # ---------- Loss and objective ----------
    def _nll(self, f, y):
        fam = self.family
        if fam == "gaussian":
            r = y - f
            return 0.5 * np.sum(r * r)
        if fam == "logistic":
            f_c = np.clip(f, -self.clip_f, self.clip_f)
            return np.sum(softplus(f_c) - y * f_c)
        if fam == "poisson":
            f_c = np.clip(f, -self.clip_f, self.clip_f)
            mu = np.exp(f_c)
            return np.sum(mu - y * f_c)
        if fam == "neg-bin":
            a = self.nb_alpha
            f_c = np.clip(f, -self.clip_f, self.clip_f)
            mu = np.exp(f_c)
            return np.sum((y + 1.0/a) * np.log1p(a * mu) - y * f_c)
        raise ValueError(f"Unknown family '{fam}'")

    def _objective(self, alpha, K, y, b=0.0):
        f = K @ alpha + (b if self.fit_intercept else 0.0)
        n = len(y)
        reg = 0.5 * self.lam * float(alpha.T @ K @ alpha)
        if self.fit_intercept and self.lam_intercept > 0.0:
            reg += 0.5 * self.lam_intercept * (b * b)
        return self._nll(f, y) / n + reg

    # ---------- public API ----------
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        self.X_train_ = X
        self.K_ = self._gram(X, X)
        #self.K_ = self.K_.astype(np.float32, copy=False) # use float32 to save memory

        # Normalize K so mean diag ~ 1 (improves λ robustness)
        if self.normalize_kernel:
            md = float(np.mean(np.diag(self.K_)))
            if md > 0:
                self._K_scale = md
                self.K_ = self.K_ / md
            else:
                self._K_scale = 1.0
        else:
            self._K_scale = 1.0
        # Don't do it for pseudo-labeling because this standardizes the regularization scale

        if self.solver == "irls":
            return self._fit_irls(self.K_, y)
        else:
            raise ValueError("solver must be 'irls'")

    def decision_function(self, Xnew):
        if self.alpha_ is None:
            raise RuntimeError("Call fit() first.")
        Kx = self._gram(np.asarray(Xnew, float), self.X_train_)
        return Kx @ self.alpha_ + (self.b_ if self.fit_intercept else 0.0)

    def predict_mean(self, Xnew):
        f = self.decision_function(Xnew)
        fam = self.family
        if fam == "gaussian":
            return f
        if fam == "logistic":
            return sigmoid(np.clip(f, -self.clip_f, self.clip_f))
        if fam in ("poisson", "neg-bin"):
            return np.exp(np.clip(f, -self.clip_f, self.clip_f))
        raise ValueError("Unknown family")

    def predict(self, Xnew, threshold=0.5):
        if self.family == "logistic":
            return (self.predict_mean(Xnew) >= threshold).astype(int)
            #rng = np.random.default_rng()
            #p_pred = self.predict_mean(Xnew)
            #y_pred = rng.binomial(1, p_pred).astype(int)
            #return y_pred
        return self.predict_mean(Xnew)

    # --- retrieve coefficients ---
    def get_dual_coef(self):
        """Return dual coefficients α (one per training sample)."""
        if self.alpha_ is None:
            raise RuntimeError("Call fit() first.")
        return self.alpha_.copy()

    def get_intercept(self):
        """Return intercept b."""
        if getattr(self, "b_", None) is None:
            raise RuntimeError("Call fit() first.")
        return float(self.b_) if self.fit_intercept else 0.0

    def get_primal_coef(self):
        """
        Return primal weights w (only meaningful for the linear kernel, or if you
        trained with an explicit feature map like RFF/Nyström and stored it).
        For linear kernel: w = X_train^T α.
        """
        if self.alpha_ is None or self.X_train_ is None:
            raise RuntimeError("Call fit() first.")
        if self.kernel != "linear":
            raise ValueError("Primal coefficients are only defined for the linear kernel (or explicit features).")
        return self.X_train_.T @ self.alpha_

    # ---------- score ----------
    def _deviance(self, y, mu):
        y = np.asarray(y, float).reshape(-1)
        mu = np.asarray(mu, float).reshape(-1)
        fam = self.family

        # small eps to avoid log(0)
        eps = 1e-12
        if fam == "gaussian":
            # up to sigma^2 scaling; constants drop in comparisons
            return float(np.sum((y - mu)**2))

        if fam == "poisson":
            # 2 * sum( y log(y/mu) - (y - mu) ), with 0*log(0) := 0
            y_safe = np.maximum(y, 0.0)
            term = np.where(y_safe > 0, y_safe * (np.log(np.maximum(y_safe, eps)) - np.log(np.maximum(mu, eps))), 0.0)
            return float(2.0 * np.sum(term - (y_safe - mu)))

        if fam == "neg-bin":
            # NB2 with dispersion theta = 1/a
            a = self.nb_alpha
            theta = 1.0 / a
            # 2 * sum[ y*log(y/mu) - (y+theta)*log((y+theta)/(mu+theta)) ]
            y_safe = np.maximum(y, 0.0)
            mu_th = mu + theta
            y_th  = y_safe + theta
            t1 = np.where(y_safe > 0, y_safe * (np.log(np.maximum(y_safe, eps)) - np.log(np.maximum(mu, eps))), 0.0)
            t2 = y_th * (np.log(np.maximum(y_th, eps)) - np.log(np.maximum(mu_th, eps)))
            return float(2.0 * np.sum(t1 - t2))

        if fam == "logistic":
            # binomial deviance (up to constants): 2 * sum[ y log(y/p) + (1-y) log((1-y)/(1-p)) ]
            p = np.clip(mu, eps, 1 - eps)
            y_safe = np.clip(y, 0.0, 1.0)
            t1 = np.where(y_safe > 0, y_safe * (np.log(np.maximum(y_safe, eps)) - np.log(p)), 0.0)
            t0 = np.where(y_safe < 1, (1 - y_safe) * (np.log(np.maximum(1 - y_safe, eps)) - np.log(1 - p)), 0.0)
            return float(2.0 * np.sum(t1 + t0))

        raise ValueError("Unknown family")
    
    def _mcfadden_pseudo_r2(self, X, y):
        # model loglik ~ -NLL_model; null loglik ~ -NLL_null (intercept-only)
        y = np.asarray(y, float).reshape(-1)
        f_model = self.decision_function(X)
        nll_model = self._nll(f_model, y)

        # null model: intercept-only (b̂ = argmin NLL)
        fam = self.family
        if fam == "gaussian":
            mu0 = np.full_like(y, y.mean())
            f0 = mu0  # identity link
        elif fam == "logistic":
            p0 = np.clip(y.mean(), 1e-12, 1 - 1e-12)
            f0 = np.full_like(y, np.log(p0/(1 - p0)))
        elif fam in ("poisson", "neg-bin"):
            m = np.clip(y.mean(), 1e-12, None)
            f0 = np.full_like(y, np.log(m))
        else:
            raise ValueError("Unknown family")
        nll_null = self._nll(f0, y)

        # McFadden: 1 - (LL_model / LL_null) = 1 - (NLL_model / NLL_null) (constants cancel in differences/ratios)
        # guard if null is pathological
        if nll_null <= 0:
            return np.nan
        return 1.0 - (nll_model / nll_null)
    
    def score(self, X, y, metric=None, threshold=0.5):
        """
        sklearn-like score:
        - default: R^2 for regression families; accuracy for logistic (classification)
        - metric options: 'r2', 'rmse', 'mae', 'deviance', 'mcfadden', 'logloss', 'accuracy'
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        fam = self.family

        # Defaults per family
        if metric is None:
            metric = "accuracy" if fam == "logistic" else "r2"

        # Common predictions
        mu = self.predict_mean(X)

        if metric == "r2":
            # sklearn's RegressorMixin.score
            y_bar = y.mean()
            ss_res = float(np.sum((y - mu)**2))
            ss_tot = float(np.sum((y - y_bar)**2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if metric == "rmse":
            return float(np.sqrt(np.mean((y - mu)**2)))

        if metric == "mae":
            return float(np.mean(np.abs(y - mu)))

        if metric == "deviance":
            return self._deviance(y, mu)

        if metric == "mcfadden":
            return self._mcfadden_pseudo_r2(X, y)

        if metric == "logloss":
            # average negative log-likelihood (without constants)
            f = self.decision_function(X)
            return self._nll(f, y) / len(y)

        if metric == "accuracy":
            if fam != "logistic":
                raise ValueError("accuracy is only for logistic (classification).")
            yhat = (mu >= threshold).astype(float)
            return float(np.mean(yhat == y))

        raise ValueError(f"Unknown metric '{metric}'")

    # ---------- solvers ----------
    def _irls_fspace_cg_step(self, K, y, alpha, b, lam_n, u0=None):
        n = len(y)

        # Fisher pieces
        f  = K @ alpha + (b if self.fit_intercept else 0.0)
        z, w, _ = self._pseudo_and_weights(f, y)
        w = np.clip(w, 1e-12, 1e6)
        s = np.sqrt(w)
        z_eff = z - (b if self.fit_intercept else 0.0)

        # A v = (K_s + lam_n I) v = s * (K @ (s * v)) + lam_n * v
        def A_mv(v): return s * (K @ (s * v)) + lam_n * v
        A = LinearOperator(K.shape, matvec=A_mv, dtype=K.dtype)

        # RHS: K_s (s*(z-b)) = s * K @ (w * (z - b))
        rhs = s * (K @ (w * z_eff))

        # EW forcing term to set inner tol adaptively: approximate F(x)=gradient residual via the RH norm; cheap and effective RIP
        """rhs_norm = float(np.linalg.norm(rhs))
        if getattr(self, "ew_prev", None) is None:
            eta = 0.5 # loose at first
        else:
            # Type 1 EW: eta_k = min(eta_max, (||F_{k}|| / ||F_{k-1}||)^beta)
            ratio = rhs_norm / (max(self.ew_prev, 1e-16))
            eta = min(self.ew_eta_max, ratio**self.ew_beta)
        # Target CG tolerance = max(base_tol, eta * ||rhs||) in relative form
        cg_rel_tol = max(self.cg_tol, eta) if rhs_norm > 0 else self.cg_tol"""

        # Preconditioner
        M = self._make_preconditioner(K, s, lam_n)

        # Warm start: u0 = S * f_old if provided
        if u0 is None:
            u0 = s * (K @ alpha)  # S f_old

        maxit = self.cg_maxiter if self.cg_maxiter is not None else min(500, n)
        u, info = cg(A, rhs, tol=self.cg_tol, maxiter=maxit, M=M, x0=u0)
        #u, info = cg(A, rhs, tol=cg_rel_tol, maxiter=maxit, M=M, x0=u0)

        #self.ew_prev = rhs_norm
        if info != 0 and self.verbose:
            print(f"[IRLS f-CG] info={info}")

        f_new = u / s
        alpha_new = self._solve_with_K(f_new)

        # Scalar Schur for intercept
        if self.fit_intercept:
            num = float((w * (z - f_new)).sum())
            den = float(w.sum() + n * self.lam_intercept + self.jitter)
            b_new = num / den
        else:
            b_new = 0.0

        return alpha_new, b_new, u  # also return u for warm-start next iter



    def _fit_irls(self, K, y):
        n = len(y)
        alpha = np.zeros(n)
        b = 0.0
        """if self.fit_intercept and self.family in ("poisson", "neg-bin"):
            # initialize intercept to log(mean(y)) for faster convergence
            m = float(np.mean(np.clip(y, 0, None)))
            b = np.log(max(m, 1e-12))"""
        hist = []
        lam_n = n * self.lam

        self._prep_K_factor(K)

        if self.precond == "nystrom" and self.nystrom_rank > 0:
            self._prep_nystrom_pc(K, lam_n)

        obj = self._objective(alpha, K, y, b=b)
        hist.append(obj)
        u_ws = None

        for it in range(1, self.max_iter + 1):
            # full step (candidate)
            alpha_try, b_try, u_try = self._irls_fspace_cg_step(K, y, alpha, b, lam_n, u0=u_ws)
            obj_try = self._objective(alpha_try, K, y, b=b_try)

            # Armijo-like check on objective decrease
            accept = (obj_try <= obj)
            tau = 1.0
            if not accept:
                f_old  = K @ alpha
                f_full = K @ alpha_try
                for _ in range(self.backtrack):
                    tau *= 0.5
                    f_damped = f_old + tau * (f_full - f_old)
                    b_damped = b + tau * (b_try - b)
                    alpha_damped = self._solve_with_K(f_damped)
                    obj_damped = self._objective(alpha_damped, K, y, b=b_damped)
                    if obj_damped <= obj:
                        alpha_try, b_try, obj_try = alpha_damped, b_damped, obj_damped
                        break

            # ---- compute deltas BEFORE assignment (bug fix) ----
            delta_param = np.linalg.norm(alpha_try - alpha) + abs(b_try - b)

            # accept
            alpha, b, obj = alpha_try, b_try, obj_try
            hist.append(obj)

            if self.verbose:
                print(f"[IRLS {it:3d}] obj={obj:.6f}  Δ={delta_param:.2e}")

            # update warm start: u_ws = S f (with current weights)
            f_now = K @ alpha + (b if self.fit_intercept else 0.0)
            _, w_now, _ = self._pseudo_and_weights(f_now, y)
            w_now = np.clip(w_now, 1e-12, 1e6)
            s_now = np.sqrt(w_now)
            u_ws = s_now * (K @ alpha)

            # stopping: either tiny param change or tiny objective change
            if delta_param < self.tol:
                break
            if len(hist) >= 2 and abs(hist[-1] - hist[-2]) < self.tol * max(1.0, hist[-2]):
                break

        self.alpha_ = alpha
        self.b_ = b
        self.train_history_ = hist
        return FitResult(alpha=alpha, train_obj=hist, iters=it, solver="irls")




    

# ===================== Synthetic Data =====================

def sample_latent_f(X, kernel="rbf", kernel_params=None, seed=0, jitter=1e-6):
    rng = np.random.default_rng(seed)
    K = KERNELS[kernel](X, X, **(kernel_params or {}))
    w, U = np.linalg.eigh(K + jitter * np.eye(len(X)))
    w = np.maximum(w, 0.0)
    z = rng.normal(size=len(X))
    return U @ (np.sqrt(w) * z)

def make_gaussian_data(n=400, d=4, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    f = sample_latent_f(X, "rbf", {"lengthscale": 1.2}, seed=seed)
    y = f + rng.normal(0, noise, size=n)
    return X, y, f

def make_logistic_data(n=500, d=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    f = sample_latent_f(X, "rbf", {"lengthscale": 1.0}, seed=seed)
    p = sigmoid(np.clip(f, -10, 10))
    y = rng.binomial(1, p).astype(float)
    return X, y, f

def make_poisson_data(n=500, d=4, rate_scale=0.8, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    f = sample_latent_f(X, "rbf", {"lengthscale": 1.0}, seed=seed)
    mu = np.exp(np.clip(rate_scale * f, -8, 8))
    y = rng.poisson(mu).astype(float)
    return X, y, f

def make_negbin_data(n=600, d=4, alpha=0.5, rate_scale=0.8, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    f = sample_latent_f(X, "rbf", {"lengthscale": 1.0}, seed=seed)
    mu = np.exp(np.clip(rate_scale * f, -8, 8))
    shape = 1.0 / alpha
    scale = alpha * mu
    lam = rng.gamma(shape, scale)
    y = rng.poisson(lam).astype(float)
    return X, y, f, alpha
