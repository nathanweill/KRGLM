
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, cholesky
from scipy.sparse.linalg import LinearOperator, cg
from dataclasses import dataclass

import torch
from pykeops.torch import LazyTensor


# --------- numerically-stable logistic pieces in torch ----------
def torch_sigmoid_stable(z):
    # stable sigmoid in float32
    return torch.where(z >= 0, 1.0 / (1.0 + torch.exp(-z)), torch.exp(z) / (1.0 + torch.exp(z)))
    
def _ensure_2d_torch(X_t):
    # X_t: torch tensor on correct device
    if X_t.dim() == 1:
        return X_t[:, None].contiguous()
    return X_t.contiguous()
    
def pick_pred_block(n, safety=0.20):
    """
    Choose B so that the matvec uses ~safety fraction of free VRAM.
    Very conservative; OK for A100/L4/T4.
    """
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        bytes_per = 4  # float32
        # Heuristic: ~ (n + B) working elements touched; add a small overhead factor
        overhead = 8 * 1024 * 1024  # 8 MB cushion
        B = int((safety * free - overhead) // (bytes_per * (n + 1)))
    else:
        B = 8192
    return max(1024, min(B, 32768))  # clamp to a reasonable range


def pseudo_and_weights_torch(f, y, family, clip_f=30.0, nb_alpha=0.5):
    f = torch.clamp(f, -clip_f, clip_f)
    if family == "gaussian":
        mu  = f
        dmu = torch.ones_like(f)
        V   = torch.ones_like(f)
    elif family == "logistic":
        mu  = torch_sigmoid_stable(f)
        dmu = mu * (1 - mu)
        V   = dmu
    elif family == "poisson":
        mu  = torch.exp(f)
        dmu = mu
        V   = mu
    elif family == "neg-bin":
        a   = nb_alpha
        mu  = torch.exp(f)
        dmu = mu
        V   = mu + a * mu * mu
    else:
        raise ValueError("unknown family")
    eps = 1e-16
    W = (dmu * dmu) / torch.clamp(V, min=eps)
    z = f + (y - mu) / torch.clamp(dmu, min=eps)
    return z, W, mu

# --------- KeOps exact kernel operator on GPU ----------
# If you pass a flat array, just reshape with .reshape(-1,1) before calling fit.
class KeOpsKernel:
    """
    Exact RBF/Laplacian/Linear kernels with KeOps LazyTensors.
    X: (n, d) torch.float32 on CUDA
    """
    def __init__(self, X, kernel="rbf", kernel_params=None, pred_block=8192):
        #assert X.is_cuda 
        assert X.dtype == torch.float32
        self.X = _ensure_2d_torch(X)
        self.X = X.contiguous()
        self.kernel = kernel
        self.params = dict(kernel_params or {})
        self.n, self.d = self.X.shape
        self.pred_block = int(pred_block)

        Xi = LazyTensor(self.X[:, None, :])   # (n,1,d)
        Xj = LazyTensor(self.X[None, :, :])   # (1,n,d)
        self._Xi = Xi
        self._Xj = Xj

        if kernel == "rbf":
            ell = float(self.params.get("ell", 1.0))
            self._scale = torch.tensor(ell * ell, device=X.device, dtype=X.dtype)
            self._Ksym = ( -((Xi - Xj) ** 2).sum(-1) / self._scale )  # exponent *before* exp
            self._diag = 1.0
            self._kind = "exp"
        elif kernel == "laplacian":
            ell = float(self.params.get("ell", 1.0))
            self._scale = torch.tensor(ell, device=X.device, dtype=X.dtype)
            self._Ksym = ( -((Xi - Xj).abs().sum(-1)) / self._scale )
            self._diag = 1.0
            self._kind = "exp"
        elif kernel in ("sobolev", "sobolev_intercept"):
            # Require d==1
            if self.d != 1:
                raise ValueError(f"{kernel} expects 1D input; got d={self.d}")
            # min(a,b) = 0.5*(a + b - |a - b|)
            # Xi, Xj shapes: (n,1,1) and (1,n,1); reduce last dim to scalar:
            S = 0.5 * (Xi + Xj - (Xi - Xj).abs()).sum(-1)   # (n,n)
            if kernel == "sobolev_intercept":
                S = S + 1.0
                self._diag_offset = 1.0
            else:
                self._diag_offset = 0.0
            self._Sob = S
            self._kind = "sob1d"
        elif kernel == "linear":
            self._G = (Xi * Xj).sum(-1)       # dot product
            self._diag = None                 # will compute as (x·x)
            self._kind = "dot"
        elif kernel == "poly":
            # params: gamma, c0, degree (default: gamma=1.0, c0=1.0, degree=2)
            self.gamma = float(self.params.get("gamma", 1.0))
            self.c0    = float(self.params.get("c0", 1.0))
            self.degree = int(self.params.get("degree", 2))
            self._G = (Xi * Xj).sum(-1)  # <x_i, x_j>
            self._kind = "poly"
        else:
            raise ValueError("unknown kernel")

    @torch.inference_mode()
    def mv(self, v):
        """Compute K @ v (exact) on GPU."""
        v = v.contiguous()
        if self._kind == "exp":               # rbf/laplacian
            return (self._Ksym.exp() @ v).contiguous()
        elif self._kind == "dot":             # linear
            return (self._G @ v).contiguous()
        elif self._kind == "poly":            # --- NEW ---
            Ksym = (self.gamma * self._G + self.c0)
            # integer power; for non-integer degrees use torch.pow
            Ksym = Ksym ** self.degree
            return (Ksym @ v).contiguous()
        else:                                 # sobolev1d / sobolev1d+1
            return (self._Sob @ v).contiguous()

    @torch.inference_mode()
    def diag(self):
        if self._kind == "exp":
            return torch.full((self.n,), 1.0, device=self.X.device, dtype=self.X.dtype)
        elif self._kind == "dot":
            return (self.X * self.X).sum(dim=1)               # ||x||^2
        elif self._kind == "poly":                             # --- NEW ---
            norm2 = (self.X * self.X).sum(dim=1)              # ||x||^2
            return (self.gamma * norm2 + self.c0) ** self.degree
        else:
            d = self.X[:, 0]
            if getattr(self, "_diag_offset", 0.0) != 0.0:
                d = d + self._diag_offset
            return d.contiguous()

    @torch.inference_mode()
    def kmv_newX(self, Xnew, alpha):
        Xnew = _ensure_2d_torch(Xnew)
        Xnew = Xnew.contiguous()
        B = self.pred_block
        out = torch.empty(Xnew.shape[0], device=self.X.device, dtype=self.X.dtype)
        for i0 in range(0, Xnew.shape[0], B):
            i1 = min(i0+B, Xnew.shape[0])
            Xn = Xnew[i0:i1]
            Xi = LazyTensor(Xn[:, None, :])
            Xj = self._Xj
            if self._kind == "exp":
                val = ( -((Xi - Xj) ** 2).sum(-1) / self._scale ).exp() if self.kernel == "rbf" \
                      else ( -((Xi - Xj).abs().sum(-1)) / self._scale ).exp()
                Kblk = (val @ alpha)
            elif self._kind == "dot":
                Kblk = ( (Xi * Xj).sum(-1) @ alpha )
            elif self._kind == "poly":                            # --- NEW ---
                Gnew = (Xi * Xj).sum(-1)
                Kblk = ((self.gamma * Gnew + self.c0) ** self.degree) @ alpha
            else:
                S = 0.5 * (Xi + Xj - (Xi - Xj).abs()).sum(-1)
                if self.kernel == "sobolev_intercept":
                    S = S + 1.0
                Kblk = (S @ alpha)
            out[i0:i1] = Kblk
        return out.contiguous()
        
        
class Sobolev1DExactOp:
    """ Exact O(n) matvec for k(x,z)=min(x,z) (optionally +1). Works on GPU. """
    def __init__(self, X, plus_one=False):
        # X: (n,1) float32 on device
        assert X.dim() == 2 and X.shape[1] == 1
        self.device = X.device
        self.plus_one = bool(plus_one)
        x = X[:,0]
        self.x_sorted, self.idx_sort = torch.sort(x)                  # ascending
        self.inv_idx = torch.empty_like(self.idx_sort)
        self.inv_idx[self.idx_sort] = torch.arange(x.numel(), device=x.device)
        self.n = x.numel()

    @torch.inference_mode()
    def mv(self, v):
        # v: (n,)
        v_sorted = v[self.idx_sort]
        # prefix sums
        S1 = torch.cumsum(v_sorted, dim=0)                            # ∑ v_j
        Sx = torch.cumsum(self.x_sorted * v_sorted, dim=0)            # ∑ x_j v_j
        total_S1 = S1[-1]
        # Kv (sorted order)
        Kv_sorted = Sx + self.x_sorted * (total_S1 - S1)
        if self.plus_one:
            Kv_sorted = Kv_sorted + total_S1
        # unsort
        return Kv_sorted[self.inv_idx].contiguous()

    @torch.inference_mode()
    def kmv_newX(self, Xnew, alpha):
        # K(Xnew,X) @ alpha
        # For each x_new: sum_{x_j <= x_new} x_j α_j + x_new * sum_{x_j > x_new} α_j [+ ∑α]
        xnew = Xnew[:,0]
        v_sorted = alpha[self.idx_sort]
        S1 = torch.cumsum(v_sorted, dim=0)
        Sx = torch.cumsum(self.x_sorted * v_sorted, dim=0)
        total_S1 = S1[-1]
        # locate positions by binary search
        # torch.searchsorted expects sorted sequence
        pos = torch.searchsorted(self.x_sorted, xnew, right=True)     # index of last <= xnew + 1
        # gather prefix sums at pos-1 (clip)
        posm1 = torch.clamp(pos-1, min=0)
        prefix_S1 = torch.where(pos>0, S1[posm1], torch.zeros_like(posm1, dtype=S1.dtype))
        prefix_Sx = torch.where(pos>0, Sx[posm1], torch.zeros_like(posm1, dtype=Sx.dtype))
        out = prefix_Sx + xnew * (total_S1 - prefix_S1)
        if self.plus_one:
            out = out + total_S1
        return out.contiguous()



# --------- GPU Conjugate Gradient with optional Jacobi precond ----------
"""@torch.inference_mode()
def cg_torch(A_mv, b, M_inv=None, x0=None, rtol=1e-3, maxiter=200):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_mv(x)
    z = M_inv(r) if M_inv is not None else r
    p = z.clone()
    rz_old = (r * z).sum()
    b_norm = b.norm()
    tol = rtol * (b_norm + 1e-12)
    for _ in range(maxiter):
        Ap = A_mv(p)
        denom = (p * Ap).sum().clamp_min(1e-30)
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if r.norm() <= tol:
            break
        z = M_inv(r) if M_inv is not None else r
        rz_new = (r * z).sum()
        beta = rz_new / rz_old.clamp_min(1e-30)
        p = z + beta * p
        rz_old = rz_new
    return x"""
    
@torch.inference_mode()
def cg_torch(A_mv, b, M_inv=None, x0=None, rtol=1e-6, maxiter=500):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_mv(x)
    z = M_inv(r) if M_inv is not None else r
    p = z.clone()
    rz_old = (r * z).sum()
    b_norm = b.norm()
    tol = rtol * (b_norm + 1e-30)

    it = 0
    while it < maxiter:
        Ap = A_mv(p)
        denom = (p * Ap).sum().clamp_min(1e-30)
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        it += 1
        r_norm = r.norm()
        if r_norm <= tol:
            break
        z = M_inv(r) if M_inv is not None else r
        rz_new = (r * z).sum()
        beta = rz_new / rz_old.clamp_min(1e-30)
        p = z + beta * p
        rz_old = rz_new

    relres = float(r.norm() / (b_norm + 1e-30))
    return x, relres, it


def refine_once(A_mv, x, b, M_inv=None, rtol=3e-7, maxiter=200):
    # compute residual in float64, even if vectors are float32
    x64 = x.double()
    b64 = b.double()
    def A_mv64(v64):
        return A_mv(v64.float()).double()  # reuse float32 mv, cast result to float64
    r64 = b64 - A_mv64(x64)
    # Solve A * delta = r   (in float32 CG for speed)
    r = r64.float()
    delta, _, _ = cg_torch(A_mv, r, M_inv=M_inv, x0=None, rtol=rtol, maxiter=maxiter)
    return (x + delta).contiguous()

    
    
class RKHSGLM_KeOps:
    def __init__(self,
                 family="logistic",
                 kernel="rbf",
                 kernel_params=None,
                 lam=1e-1,
                 nb_alpha=0.5,
                 max_iter=100,
                 tol=1e-6,
                 clip_f=30.0,
                 lam_intercept=0.0,
                 fit_intercept=False,
                 normalize_kernel=False,
                 cg_tol=1e-6,
                 cg_maxiter=450,
                 verbose=False,
                 device=None,
                 ew_adapt=True,
                 use_armijo=True):
        self.family = family
        self.kernel = kernel
        self.kernel_params = dict(kernel_params or {})
        self.lam = float(lam)
        self.nb_alpha = float(nb_alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.clip_f = float(clip_f)
        self.fit_intercept = bool(fit_intercept)
        self.lam_intercept = float(lam_intercept)
        self.normalize_kernel = bool(normalize_kernel)
        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)
        self.verbose = verbose
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._ew_prev_rhs_norm = None   # for adaptive CG tolerance (Eisenstat–Walker)

        # ---- NEW TUNING KNOBS ----
        self.ew_adapt = ew_adapt             # set False to use fixed CG tol
        self.ew_eta0 = 2e-4              # initial target (loose) when ew_adapt=True
        self.ew_eta_max = 2e-4           # cap for adaptivity
        self.ew_beta = 0.5               # exponent in EW rule
        
        self.cg_tol_f_floor = 1e-5       # floors for adaptive mode (prevent too-loose solves)
        self.cg_tol_alpha_floor = 5e-6
        
        self.use_armijo = use_armijo           # set False to disable backtracking
        self.armijo_max_halves = 8
        self.armijo_eps = 1e-12          # tiny slack to ignore fp jitter
        
        # If ew_adapt=False, these are the fixed tolerances used:
        # (fall back to existing self.cg_tol for both unless you want two)
        self.cg_tol_fixed_f = self.cg_tol
        self.cg_tol_fixed_alpha = self.cg_tol
        
        # Final polish tol (tight last Kα solve)
        self.final_polish_tol = 1e-6


        # learned params
        self.alpha_ = None
        self.b_ = 0.0
        self.X_train_ = None
        self._K = None
        self._K_scale = 1.0

    @torch.inference_mode()
    def fit(self, X, y):
        # move to GPU, float32
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).view(-1)
        X = _ensure_2d_torch(X)

        self.X_train_ = X
        n = X.shape[0]
        lam_n = self.lam * n
        
        use_sobolev_exact = (self.kernel in ("sobolev", "sobolev_intercept")) and (X.shape[1] == 1)
        if use_sobolev_exact:
            self._sob = Sobolev1DExactOp(X, plus_one=(self.kernel=="sobolev_intercept"))
            self._K_scale = 1.0  # keep semantics identical
            def Kmv(v):  # exact O(n) matvec
                return self._sob.mv(v)
            # --- NEW: cache row sums for Gershgorin preconditioner ---
            ones = torch.ones(n, device=self.device, dtype=torch.float32)
            self._sob_row_sum = self._sob.mv(ones)  # (n,)
        else:
            B = pick_pred_block(n)
            K = KeOpsKernel(X, kernel=self.kernel, kernel_params=self.kernel_params, pred_block=B)
            self._K = K
            if self.normalize_kernel:
                dK = K.diag()
                md = float(dK.mean().item())
                self._K_scale = md if md > 0 else 1.0
            else:
                self._K_scale = 1.0
            scale = 1.0 / self._K_scale
            def Kmv(v):
                return scale * K.mv(v)
                
        #print("Using Sobolev exact:", hasattr(self, "_sob") and (self._sob is not None))



        """# Build KeOps kernel
        B = pick_pred_block(n)
        K = KeOpsKernel(X, kernel=self.kernel, kernel_params=self.kernel_params, pred_block=B)
        self._K = K

        # Optional normalization: scale K so mean diag ≈ 1
        if self.normalize_kernel:
            dK = K.diag()
            md = float(dK.mean().item())
            self._K_scale = md if md > 0 else 1.0
        else:
            self._K_scale = 1.0
        scale = 1.0 / self._K_scale

        # helpers
        def Kmv(v):  # K @ v (scaled if normalized)
            return scale * K.mv(v)"""

        # init
        alpha = torch.zeros(n, device=self.device, dtype=torch.float32)
        b = torch.tensor(0.0, device=self.device)
        hist = []
        
        # helpers
        def _nll_torch_local(f, y):
            f_c = torch.clamp(f, -self.clip_f, self.clip_f)
            if self.family == "gaussian":
                return 0.5 * torch.mean((y - f_c)**2)
            if self.family == "logistic":
                return torch.mean(torch.nn.functional.softplus(f_c) - y * f_c)
            if self.family == "poisson":
                return torch.mean(torch.exp(f_c) - y * f_c)
            # neg-bin
            a = self.nb_alpha
            mu = torch.exp(f_c)
            return torch.mean((y + 1.0/a) * torch.log1p(a * mu) - y * f_c)
        
        def _objective_local(Kmv, alpha, b, y, n):
            f = Kmv(alpha) + (b if self.fit_intercept else 0.0)
            nll = _nll_torch_local(f, y)
            reg = 0.5 * self.lam * (alpha * Kmv(alpha)).sum()
            if self.fit_intercept and self.lam_intercept > 0:
                reg = reg + 0.5 * self.lam_intercept * (b*b) * n
            return nll + reg


        for it in range(1, self.max_iter + 1):
            # f = K alpha + b
            f = Kmv(alpha) + (b if self.fit_intercept else 0.0)

            # Fisher pieces
            z, W, _ = pseudo_and_weights_torch(f, y, self.family, clip_f=self.clip_f, nb_alpha=self.nb_alpha)
            W = torch.clamp(W, 1e-12, 1e6)
            s = torch.sqrt(W)

            # Solve (S K S + lam_n I) u = S K (W z_eff), where z_eff = z - b
            z_eff = z - (b if self.fit_intercept else 0.0)

            def A_mv(v):
                return s * Kmv(s * v) + lam_n * v

            rhs = s * Kmv(W * z_eff)
            
            # ---- Choose CG tolerances: adaptive (Eisenstat–Walker style) or fixed ----
            if self.ew_adapt:
                rhs_norm = float(torch.linalg.norm(rhs).item())
                if it == 1 or self._ew_prev_rhs_norm is None:
                    eta = self.ew_eta0
                else:
                    ratio = rhs_norm / max(self._ew_prev_rhs_norm, 1e-12)
                    eta = min(self.ew_eta_max, ratio ** self.ew_beta)
                f_cg_rtol  = max(self.cg_tol_f_floor, eta)
                ka_cg_rtol = max(self.cg_tol_alpha_floor, 0.3 * f_cg_rtol)
                self._ew_prev_rhs_norm = rhs_norm
            
                # Optional: tighten slightly on the very last IRLS pass
                if it == self.max_iter - 1:
                    f_cg_rtol  = min(f_cg_rtol, 1e-5)
                    ka_cg_rtol = min(ka_cg_rtol, 5e-6)
            else:
                # STRICT, FIXED TOLERANCES (e.g., 1e-6)
                f_cg_rtol  = self.cg_tol_fixed_f
                ka_cg_rtol = self.cg_tol_fixed_alpha
            # -------------------------------------------
            
            # --- Preconditioner for A = S K S + lam_n I ---
            if hasattr(self, "_sob") and (self._sob is not None):
                # Gershgorin: use exact row-sum R = K @ 1  (stronger than diag(K))
                R = torch.clamp(self._sob_row_sum, min=1e-6)         # small clamp for stability
                Mdiag = (s * s) * R + lam_n
                M_inv = lambda r: r / torch.clamp(Mdiag, min=1e-12)
            else:
                # fallback: Jacobi on diag(K)
                dK = K.diag() * scale
                dK = torch.clamp(dK, min=1e-6)                   # keep your min clamp
                Mdiag = (s * s) * dK + lam_n
                M_inv = lambda r: r / torch.clamp(Mdiag, min=1e-12)
            

            # warm start in "u"-space
            u0 = s * Kmv(alpha)
            #u = cg_torch(A_mv, rhs, M_inv=M_inv, x0=u0, rtol=self.cg_tol, maxiter=self.cg_maxiter)
            #u = cg_torch(A_mv, rhs, M_inv=M_inv, x0=u0, rtol=f_cg_rtol, maxiter=self.cg_maxiter)
            u, rel_f, it_f = cg_torch(A_mv, rhs, M_inv=M_inv, x0=u0, rtol=f_cg_rtol, maxiter=self.cg_maxiter)
            if rel_f > 1.5 * f_cg_rtol:
                # one retry tighter
                u, rel_f, it_f = cg_torch(A_mv, rhs, M_inv=M_inv, x0=u, rtol=max(f_cg_rtol*0.3, 1e-7), maxiter=self.cg_maxiter+100)
            # iterative refinement (one pass)
            u = refine_once(A_mv, u, rhs, M_inv=M_inv, rtol=3e-7, maxiter=150)



            # map back to f and alpha
            f_new = u / torch.clamp(s, min=1e-12)
            # Solve K alpha = f_new - b
            rhs2 = f_new - (b if self.fit_intercept else 0.0)
            if hasattr(self, "_sob") and (self._sob is not None):
                R = torch.clamp(self._sob_row_sum, min=1e-6)
                M2_inv = lambda r: r / torch.clamp(R + 1e-8, min=1e-12)
            else:
                M2_inv = lambda r: r / torch.clamp(dK + 1e-8, min=1e-12)
            
            
            #alpha_new = cg_torch(Kmv, rhs2, M_inv=M2_inv, x0=alpha, rtol=max(self.cg_tol, 3e-4), maxiter=self.cg_maxiter)
            #alpha_new = cg_torch(Kmv, rhs2, M_inv=M2_inv, x0=alpha, rtol=ka_cg_rtol, maxiter=self.cg_maxiter)
            alpha_new, rel_a, it_a = cg_torch(Kmv, rhs2, M_inv=M2_inv, x0=alpha, rtol=ka_cg_rtol, maxiter=self.cg_maxiter)
            if rel_a > 1.5 * ka_cg_rtol:
                alpha_new, rel_a, it_a = cg_torch(Kmv, rhs2, M_inv=M2_inv, x0=alpha_new, rtol=max(ka_cg_rtol*0.3, 1e-7), maxiter=self.cg_maxiter+100)
            # iterative refinement (one pass; do two if you still see drift)
            alpha_new = refine_once(Kmv, alpha_new, rhs2, M_inv=M2_inv, rtol=3e-7, maxiter=150)

            # Intercept (scalar Schur complement)
            if self.fit_intercept:
                # recompute with alpha_new
                f_tmp = Kmv(alpha_new) + b
                _, Wtmp, _ = pseudo_and_weights_torch(f_tmp, y, self.family, clip_f=self.clip_f, nb_alpha=self.nb_alpha)
                Wtmp = torch.clamp(Wtmp, 1e-12, 1e6)
                num = (Wtmp * (z - (Kmv(alpha_new) + b))).sum()
                den = Wtmp.sum() + self.lam_intercept * n + 1e-12
                b_new = num / den
            else:
                b_new = torch.tensor(0.0, device=self.device)

            # Objective (for progress + stopping)
            f_eval = Kmv(alpha_new) + (b_new if self.fit_intercept else 0.0)
            if self.family == "gaussian":
                nll = 0.5 * torch.sum((y - f_eval) ** 2) / n
            elif self.family == "logistic":
                # softplus(f) - y f
                nll = torch.mean(torch.nn.functional.softplus(torch.clamp(f_eval, -self.clip_f, self.clip_f)) - y * torch.clamp(f_eval, -self.clip_f, self.clip_f))
            elif self.family == "poisson":
                f_c = torch.clamp(f_eval, -self.clip_f, self.clip_f)
                nll = torch.mean(torch.exp(f_c) - y * f_c)
            else:  # neg-bin
                a = self.nb_alpha
                f_c = torch.clamp(f_eval, -self.clip_f, self.clip_f)
                mu = torch.exp(f_c)
                nll = torch.mean((y + 1.0/a) * torch.log1p(a * mu) - y * f_c)

            reg = 0.5 * self.lam * (alpha_new * Kmv(alpha_new)).sum()
            if self.fit_intercept and self.lam_intercept > 0:
                reg = reg + 0.5 * self.lam_intercept * (b_new * b_new) * n
            obj = (nll + reg).item()
            hist.append(obj)
            
            # ---- Armijo (optional) ----
            if self.use_armijo:
                obj_old = _objective_local(Kmv, alpha, b, y, n).item()
                obj_try = obj
                if obj_try > obj_old + self.armijo_eps:  # ignore tiny jitter
                    tau = 1.0
                    for _ in range(self.armijo_max_halves):
                        tau *= 0.5
                        a_d = alpha + tau * (alpha_new - alpha)
                        b_d = b + tau * (b_new - b)
                        obj_d = _objective_local(Kmv, a_d, b_d, y, n).item()
                        if obj_d <= obj_old + self.armijo_eps:
                            alpha_new, b_new = a_d, b_d
                            obj_try = obj_d
                            break
                    obj = obj_try
            # ---------------------------

            # progress / stopping
            delta = (alpha_new - alpha).norm().item() + float(torch.abs(b_new - b))
            alpha, b = alpha_new, b_new
            if self.verbose:
                print(f"[IRLS {it:03d}] obj={obj:.6f}  Δ={delta:.3e}")
            if delta < self.tol:
                break
            if len(hist) >= 2 and abs(hist[-1] - hist[-2]) < self.tol * max(1.0, abs(hist[-2])):
                break
        
        # ---- Final polish: tighten Kα solve once for accuracy parity ----
        if self.fit_intercept:
            f_final = Kmv(alpha) + b
        else:
            f_final = Kmv(alpha)
        rhs2 = f_final - (b if self.fit_intercept else 0.0)
        if hasattr(self, "_sob") and (self._sob is not None):
            R = torch.clamp(self._sob_row_sum, min=1e-6)
            M2_inv = lambda r: r / torch.clamp(R + 1e-8, min=1e-12)
        else:
            dK = self._K.diag() * (1.0 / self._K_scale)
            dK = torch.clamp(dK, min=1e-6)
            M2_inv = lambda r: r / torch.clamp(dK + 1e-8, min=1e-12)
        #alpha = cg_torch(Kmv, rhs2, M_inv=M2_inv, x0=alpha, rtol=1e-6, maxiter=max(200, self.cg_maxiter))
        #alpha = cg_torch(Kmv, rhs2, M_inv=M2_inv, x0=alpha, rtol=self.final_polish_tol, maxiter=max(200, self.cg_maxiter))
        alpha, _, _ = cg_torch(Kmv, rhs2, M_inv=M2_inv, x0=alpha, rtol=self.final_polish_tol, maxiter=max(200, self.cg_maxiter))
        alpha = refine_once(Kmv, alpha, rhs2, M_inv=M2_inv, rtol=3e-7, maxiter=150)
        
        self.alpha_ = alpha.detach().cpu().numpy()
        self.b_ = float(b.item())
        self.train_history_ = hist
        return self


    @torch.inference_mode()
    def decision_function(self, Xnew):
        Xnew = torch.as_tensor(Xnew, dtype=torch.float32, device=self.device)
        Xnew = _ensure_2d_torch(Xnew)
        alpha = torch.as_tensor(self.alpha_, dtype=torch.float32, device=self.device)
        
        if hasattr(self, "_sob") and self._sob is not None:
            Kx_alpha = self._sob.kmv_newX(Xnew, alpha)
        else:
            Kx_alpha = self._K.kmv_newX(Xnew, alpha) * (1.0 / self._K_scale)
        return (Kx_alpha + (self.b_ if self.fit_intercept else 0.0)).detach().cpu().numpy()

        """K = self._K
        out = K.kmv_newX(Xnew, alpha) + (self.b_ if self.fit_intercept else 0.0)
        return out.detach().cpu().numpy()"""

    @torch.inference_mode()
    def predict_mean(self, Xnew):
        f = torch.as_tensor(self.decision_function(Xnew), dtype=torch.float32)
        if self.family == "gaussian":
            return f.cpu().numpy()
        if self.family == "logistic":
            return torch_sigmoid_stable(torch.clamp(f, -self.clip_f, self.clip_f)).cpu().numpy()
        if self.family in ("poisson", "neg-bin"):
            return torch.exp(torch.clamp(f, -self.clip_f, self.clip_f)).cpu().numpy()

    @torch.inference_mode()
    def predict(self, Xnew, threshold=0.5):
        if self.family == "logistic":
            p = self.predict_mean(Xnew)
            return (p >= threshold).astype(int)
        return self.predict_mean(Xnew)

