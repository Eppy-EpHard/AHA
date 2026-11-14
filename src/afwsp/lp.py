"""The linear program for v in a rank-1 down_proj edit ΔW = u vᵀ. Adapted from weight-space-lp (MIT).

The edit changes the layer's write into the residual stream by u · (v·h_t) at token t, so every constraint is
linear in v:

    min  Σ_j c_j |v_j| + C Σ slack
    s.t. v·h_t ≤ −δ_t + s_t        t in R   (push looping tokens down along u)
         |v·h_t| ≤ ε + s_t         t in P   (leave productive tokens alone)

Solved by primal-dual hybrid gradient (Chambolle-Pock, PDLP's base algorithm). Upstream also ships a HiGHS
solver; the pipeline only ever used this one, so it is the one kept.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Spec:
    delta_R: np.ndarray             # required suppression per R token (> 0 pushes the u-write down)
    eps: float                      # productive band
    C: float = 100.0                # slack penalty
    cost: np.ndarray | None = None  # per-neuron L1 weights (default 1)
    support: np.ndarray | None = None
    v_max: float = 10.0


@dataclass
class Result:
    v: np.ndarray
    eps: float
    status: str
    rounds: list = field(default_factory=list)


def pdhg(H_R, H_P, spec: Spec, iters=20000, tol=1e-5, device=None, log_every=0, v0=None, omega=1.0):
    """Solve the LP exactly by PDHG.

    Eliminating the slacks gives the equivalent convex problem
        min_v  Σ_j c_j|v_j| + C Σ_R max(0, h_t·v + δ_t) + C Σ_P max(0, |h_t·v| − ε),
    i.e. min_v g(v) + f(Kv) with K = [H_R; H_P]. Only mat-vecs with K and Kᵀ are needed, so all tokens fit at once
    on GPU/MPS, and the L1 prox (soft threshold) returns an exactly sparse v.
    Duals y_R in [0, C], y_P in [−C, C] are the LP's constraint multipliers.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    f32 = torch.float32
    cols = np.arange(H_R.shape[1]) if spec.support is None else np.asarray(spec.support)
    K = torch.tensor(np.concatenate([H_R[:, cols], H_P[:, cols]]), dtype=f32, device=dev)
    nR = len(H_R)
    delta = torch.tensor(spec.delta_R, dtype=f32, device=dev)
    eps = float(spec.eps)
    C = float(spec.C)
    cost = torch.ones(len(cols), dtype=f32, device=dev) if spec.cost is None else torch.tensor(spec.cost[cols], dtype=f32, device=dev)
    # ||K|| by power iteration; diagonal-free step sizes with tau*sigma*||K||^2 = 0.9
    x = torch.randn(len(cols), device=dev)
    for _ in range(50):
        x = K.T @ (K @ x)
        x /= x.norm()
    L = float((K @ x).norm())
    tau, sigma = 0.95 / (L * omega), 0.95 * omega / L   # omega: PDLP's primal weight
    v = torch.zeros(len(cols), device=dev) if v0 is None else torch.tensor(v0[cols], dtype=f32, device=dev)
    vbar, y = v.clone(), torch.zeros(K.shape[0], device=dev)
    hist = []

    def objective(v):
        z = K @ v
        return float((cost * v.abs()).sum() + C * torch.relu(z[:nR] + delta).sum() + C * torch.relu(z[nR:].abs() - eps).sum())

    best = (np.inf, v.clone())
    for it in range(iters):
        w = y + sigma * (K @ vbar)
        y = torch.cat([(w[:nR] + sigma * delta).clamp(0, C),
                       torch.sign(w[nR:]) * (w[nR:].abs() - sigma * eps).clamp(min=0).clamp(max=C)])
        u = v - tau * (K.T @ y)
        vn = torch.sign(u) * (u.abs() - tau * cost).clamp(min=0)
        if spec.v_max:
            vn = vn.clamp(-spec.v_max, spec.v_max)
        vbar = 2 * vn - v
        step = float((vn - v).norm() / (vn.norm() + 1e-3))
        v = vn
        if it % 200 == 0 or it == iters - 1:
            obj = objective(v)
            if obj < best[0]:
                best = (obj, v.clone())
            hist.append((it, obj, step))
            if log_every and it % log_every == 0:
                print(f"  pdhg it={it} obj={obj:.4f} step={step:.2e} nnz={int((v != 0).sum())}", flush=True)
            if it > 2000 and len(hist) > 5 and abs(hist[-5][1] - obj) < tol * max(1.0, abs(obj)):
                break
    out = np.zeros(H_R.shape[1])
    out[cols] = best[1].cpu().numpy()
    return Result(out, eps, f"pdhg it={it}", [{"history": hist}])
