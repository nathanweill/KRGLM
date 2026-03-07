### Algorithmic Details and Implementation

We implemented a generic solver for kernel GLMs in Python, using the Fisher scoring method. The code for the solver, and the two experiments, are available at the GitHub link provided. 

#### KRGLM via Fisher Scoring in Function Space

Recall that we need to fit a ridge-regularized kernel GLM. By the representer theorem, $f(\cdot)=\sum_{j=1}^n \alpha_j K(x_j,\cdot)$, hence the optimization can be carried out in *function space* over the vector of training evaluations $\mathbf{f} =(f(\mathbf{x}_1),\dots,f(\mathbf{x}_n))^\top = \mathbf{K}\boldsymbol{\alpha}$, where $\mathbf{K}\in\mathbb{R}^{n\times n}$ is the kernel matrix. Denote in what follows $\eta_i := f(\mathbf{x}_i), \forall i$, and let $\mu_i=\mathbb{E}[y_i\mid \eta_i]$ and $V_i=\text{Var}(y_i\mid \eta_i)$ under the chosen GLM family.

We use *Fisher scoring* (McCullagh & Nelder, 1989), i.e., Newton's method with the Hessian replaced by the expected Hessian (Fisher information), which yields the standard *iteratively reweighted least squares* (IRLS) updates. At the current iterate $\eta=f$, define the *weights* $W_i$ and *pseudo-response* $z_i$:

$$W_i \;=\; \frac{\left(\frac{d\mu_i}{d\eta_i}\right)^2}{V_i}, \qquad z_i \;=\; \eta_i \;+\;\frac{y_i-\mu_i}{\frac{d\mu_i}{d\eta_i}}.$$

Then the next iterate $f_{\text{new}}$ is obtained by solving the weighted kernel ridge regression problem:

$$\min_{\tilde f\in\mathcal H} \frac{1}{2n}\sum_{i=1}^n W_i \bigl(z_i - \tilde f(x_i)\bigr)^2 +\frac{\lambda}{2}\||\tilde f\||_{\mathcal H}^2.$$

(Implementation details: we clip $\eta$ before exponentials/logits and clip $W_i$ away from $0$ to avoid overflow/underflow and division by very small numbers.)

Expressing the RKHS norm through coefficients gives $\||\tilde f\||_{\mathcal H}^2=\alpha^\top K\alpha$ with $\tilde f = K\alpha$ on the training set. Eliminating $\alpha$ yields the normal equation in $f$-space:

$$W(z-f) = n\lambda K^{-1}f \quad\Longleftrightarrow\quad (KW + n\lambda I)f = K W z,$$

where $W=\text{diag}(W_1,\dots,W_n)$. To obtain a symmetric positive definite (SPD) linear system, define $S=\text{diag}(\sqrt{W_1},\dots,\sqrt{W_n})$ and the change of variables $u := S f$. Since $W=S^2$, multiplying by $S$ gives:

$$\underbrace{(S K S + n\lambda I)}_{\text{SPD}} u = S K W z, \qquad f = S^{-1}u.$$

Using $SKS$ is numerically preferable because it preserves symmetry/SPD (needed for stable CG) and distributes the weights as $S$ on both sides of $K$, avoiding extreme scalings from $W$. In practice we also add a small diagonal "jitter" to $K$ and/or to the linear system to ensure strict SPD. 

#### Conjugate Gradient (CG) and Preconditioning

CG (Hestenes & Stiefel, 1952) solves SPD linear systems $Au=r$ using only matrix-vector products with $A$ (no matrix factorization). After $t$ iterations, $u_t$ lies in the Krylov subspace $\mathcal{K}_t(A,r)=\mathrm{span}\{r,Ar,\dots,A^{t-1}r\}$ and minimizes the quadratic energy $\frac12 u^\top A u - r^\top u$ over that subspace. To accelerate convergence, we use preconditioned CG with a preconditioner $M\approx A^{-1}$, improving the effective conditioning of the system. 

In our GitHub repository, we propose two implementations of this IRLS-CG algorithm. For relatively small sample sizes ($n\le5000$), a simple version (`rkhs_glm_scaled.py`) using only NumPy and SciPy is enough. For larger problems, forming and storing the full dense kernel matrix becomes a $\mathcal{O}(n^2)$ memory bottleneck. We thus implement the IRLS inner linear solves using *kernel matvec oracles* computed on-the-fly on the GPU, using the KeOps library (`rkhs_glm_scaled_KeOps.py`). 

#### Scalable KeOps/PyTorch-GPU Implementation

Concretely, the KeOps backend represents pairwise kernel interactions symbolically via `pykeops.torch.LazyTensor` and evaluates contractions such as $$(\mathbf{K} v)_i=\sum_{j=1}^n K(x_i,x_j)\,v_j$$ without ever materializing $\mathbf{K}$ as an $n\times n$ tensor. All arrays are moved to `torch.float32` on the chosen device (typically CUDA), and the GLM-specific pieces (pseudo-response/weights and stable sigmoid/softplus) are computed directly in PyTorch. The IRLS system is solved by conjugate gradients using only applications of the SPD operator $A(v)=s\odot \mathbf{K}(s\odot v)+n\lambda v$, implemented as a callable `A_mv` that composes elementwise scaling with the KeOps matvec `Kmv` (again avoiding any dense matrix formation). For prediction, we similarly compute $\mathbf{K}(\mathbf{X}_{\mathrm{new}},\mathbf{X})\boldsymbol{\alpha}$ in GPU-friendly blocks whose size is chosen from available VRAM (via `torch.cuda.mem_get_info`) to prevent memory overflows. Overall, this keeps the *same update equations* as the dense implementation but changes memory usage from $\mathcal{O}(n^2)$ to roughly $\mathcal{O}(nd)$ ($d$ being the input dimension) plus a small constant number of vectors, while each CG iteration costs one (or a few) GPU kernel-matvec reductions rather than dense linear algebra.
