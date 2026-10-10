"""Completeness as a function of apparent magnitude, C = C(V), V = M_V + 5 log10(d / 10 pc),
fitted with the per-cluster phase likelihood (see phaselike.py) using ALL clusters at once:

    ln L = sum_i [ ln C(V_i,now) - ln <C(V_i(t))>_t ],   V_i(t) = M_V,i + DM(d_i(t)).

C(V) is a special case of C(M_V, d): it links luminosity classes (a faint cluster nearby has the
same completeness as a bright one farther away at equal V), so a single anchor - the brightest-V
bin is complete (C = 1) - fixes absolute completeness at all M_V. Extinction is not included.
"""
from __future__ import annotations

import numpy as np

from . import phaselike as PL
from .adaptive import equal_count_edges


def apparent_mag(M: np.ndarray, d_kpc: np.ndarray, A_V: np.ndarray | float = 0.0) -> np.ndarray:
    """V = M_V + 5 log10(d / 10 pc) + A_V for d in kpc (broadcasts per-cluster M and A_V over time)."""
    M = np.asarray(M, float)
    A = np.asarray(A_V, float)
    if np.ndim(d_kpc) == 2:
        if M.ndim == 1:
            M = M[:, None]
        if A.ndim == 1:
            A = A[:, None]
    return M + 5.0 * np.log10(np.maximum(d_kpc, 1e-4) * 100.0) + A


def v_edges(V_now: np.ndarray, n_min: int) -> np.ndarray:
    """Equal-count bins in observed V; outer bins are open (handled by phaselike's open last bin and clipping)."""
    return equal_count_edges(V_now, n_min, lo=-np.inf, hi=np.inf)


def fit_cv(V_t: np.ndarray, V_now: np.ndarray, edges: np.ndarray, profile: bool = True) -> dict:
    """MLE and (optionally) 68% profile intervals of C_k / C_0 (bin 0 = brightest V = anchor)."""
    F = PL.time_fractions(V_t, edges)
    kn = PL.bin_index(V_now, edges)
    fr = PL.fit(F, kn)
    ci = [PL.profile_interval(F, kn, fr, j) for j in range(len(edges) - 2)] if profile else \
        [(np.nan, np.nan)] * (len(edges) - 2)
    C = np.r_[1.0, np.exp(fr["theta"])]
    lo = np.r_[1.0, [np.exp(a) for a, _ in ci]]
    hi = np.r_[1.0, [np.exp(b) for _, b in ci]]
    return {"C": C, "lo": lo, "hi": hi, "theta": fr["theta"], "ll": fr["ll"], "at_bound": np.r_[False, fr["at_bound"]],
            "F": F, "kn": kn, "success": fr.get("success", True), "n_par": len(edges) - 2}


def cluster_q(V_t: np.ndarray, edges: np.ndarray, C: np.ndarray) -> np.ndarray:
    """q_i = <C(V_i(t))>_t (absolute under the anchor); weight = 1 / q_i."""
    return np.asarray(C)[PL.bin_index(V_t, edges)].mean(1)


def implied_class_curve(V_t, d_t, edges_V, C, d_edges) -> np.ndarray:
    """Per-distance-bin completeness implied by C(V) for a set of orbits, as the ratio
    <C 1[d bin]>_t-sum / <1[d bin]>_t-sum (normalised to the first d bin)."""
    Ct = np.asarray(C)[PL.bin_index(V_t, edges_V)]
    kd = PL.bin_index(d_t, d_edges)
    num = np.array([(Ct * (kd == k)).sum() for k in range(len(d_edges) - 1)])
    den = np.array([(kd == k).sum() for k in range(len(d_edges) - 1)])
    with np.errstate(divide="ignore", invalid="ignore"):
        c = num / den
    return c / c[0]


# ---------------------------------------------------------------------------
# Separable model C = C_V(V) * g(x), x = an obscuration proxy along the orbit (|b| or 2D E(B-V))
# ---------------------------------------------------------------------------
def _ll_sep(theta, kV, kx, kV0, kx0, nV, nx):
    """ln L = sum_i [ thV[kV0_i] + thx[kx0_i] - ln <exp(thV[kV_it] + thx[kx_it])>_t ];
    thV[0] = 0 (brightest V), thx[0] = 0 (least obscured x)."""
    thV = np.r_[0.0, theta[:nV - 1]]
    thx = np.r_[0.0, theta[nV - 1:]]
    e = np.exp(thV[kV] + thx[kx])                       # (N, T)
    den = e.mean(1)
    ll = (thV[kV0] + thx[kx0]).sum() - np.log(den).sum()
    w = e / (den[:, None] * e.shape[1])                 # per-sample responsibility, rows sum to 1
    gV = np.bincount(kV0, minlength=nV) - np.bincount(kV.ravel(), weights=w.ravel(), minlength=nV)
    gx = np.bincount(kx0, minlength=nx) - np.bincount(kx.ravel(), weights=w.ravel(), minlength=nx)
    return ll, np.r_[gV[1:], gx[1:]]


def fit_cv_sep(V_t, x_t, V_now, x_now, eV, ex) -> dict:
    """MLE of C_V (V bins eV) and g (x bins ex), both relative to their first bin."""
    from scipy.optimize import minimize
    kV, kV0 = PL.bin_index(V_t, eV), PL.bin_index(V_now, eV)
    kx, kx0 = PL.bin_index(x_t, ex), PL.bin_index(x_now, ex)
    nV, nx = len(eV) - 1, len(ex) - 1
    res = minimize(lambda t: tuple(-v for v in _ll_sep(t, kV, kx, kV0, kx0, nV, nx)), np.zeros(nV + nx - 2),
                   jac=True, method="L-BFGS-B", bounds=[PL.THETA_BOUNDS] * (nV + nx - 2))
    CV_ = np.r_[1.0, np.exp(res.x[:nV - 1])]
    g = np.r_[1.0, np.exp(res.x[nV - 1:])]
    return {"C_V": CV_, "g": g, "ll": -float(res.fun), "n_par": nV + nx - 2, "success": bool(res.success),
            "kV": kV, "kx": kx}


def cluster_q_sep(fit: dict) -> np.ndarray:
    """q_i = <C_V(V_it) g(x_it)>_t."""
    return (fit["C_V"][fit["kV"]] * fit["g"][fit["kx"]]).mean(1)



# ---------------------------------------------------------------------------
# Continuous C(V): ln C piecewise linear between nodes (node 0 = anchor, C = 1, flat brighter),
# last slope extrapolated beyond the last node and constrained <= 0. Unlike the binned model, C keeps
# falling at V fainter than any detection, so q is not inflated for faint clusters whose orbits
# spend most of their time beyond the last node (mock tests: scripts/run_cv.py, scenarios F-G).

def spline_nodes(V_now: np.ndarray, n_min: int) -> np.ndarray:
    """Nodes at the median observed V of each equal-count bin (same bins as the binned model)."""
    e = v_edges(V_now, n_min)
    k = PL.bin_index(V_now, e)
    return np.array([np.median(V_now[k == j]) for j in range(len(e) - 1)])


def spline_basis(V: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """B[..., j] with ln C(V) = B @ p, p = (t_1, ..., t_{K-2}, delta), t_{K-1} = t_{K-2} + delta, t_0 = 0."""
    V = np.asarray(V, float)
    K = len(nodes)
    if K < 2:
        raise ValueError("spline C(V) needs at least 2 nodes")
    H = np.stack([np.interp(V, nodes, np.eye(K)[j]) for j in range(1, K)], -1)   # hat functions t_1..t_{K-1}
    beyond = V > nodes[-1]
    x = np.where(beyond, (V - nodes[-1]) / (nodes[-1] - nodes[-2]), 0.0)
    H[..., -1] += x                                                               # t_{K-1} (1 + x)
    if K == 2:                                                                    # t_0 = 0: no -t_{K-2} x term;
        return H                                                                  # p = (t_1,), t_1 <= 0
    H[..., -2] -= x                                                               # - t_{K-2} x
    J = np.eye(K - 1)                                                             # t = J p
    J[-1, -2] = 1.0
    return H @ J


def lnC_spline(V: np.ndarray, nodes: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ln C(V) for parameters p (equivalent to spline_basis(V, nodes) @ p, without building the basis)."""
    p = np.asarray(p, float)
    t = np.r_[0.0, p[:-1], p[-2] + p[-1]] if len(p) > 1 else np.r_[0.0, p]
    V = np.asarray(V, float)
    s = (t[-1] - t[-2]) / (nodes[-1] - nodes[-2])
    return np.where(V > nodes[-1], t[-1] + s * (V - nodes[-1]), np.interp(V, nodes, t))


def fit_cv_spline(B_t: np.ndarray, B_now: np.ndarray, p0=None) -> dict:
    """Phase-likelihood MLE of the spline parameters, given bases B_t (N, T, P) and B_now (N, P)."""
    from scipy.optimize import minimize
    P = B_t.shape[-1]

    def nll(p):
        a = B_t @ p                                    # (N, T) ln C along orbits
        amax = a.max(1, keepdims=True)
        e = np.exp(a - amax)
        Z = e.sum(1)
        ll = (B_now @ p).sum() - (np.log(Z / a.shape[1]) + amax[:, 0]).sum()
        g = B_now.sum(0) - np.einsum("nt,ntp->p", e / Z[:, None], B_t)
        return -ll, -g
    bounds = [(-12.0, 4.0)] * (P - 1) + [(-12.0, 0.0)]  # last parameter = tail step delta <= 0
    r = minimize(nll, np.zeros(P) if p0 is None else p0, jac=True, method="L-BFGS-B", bounds=bounds)
    return {"p": r.x, "ll": -r.fun, "n_par": P, "success": bool(r.success)}


def population_detectability(M_eval: np.ndarray, mu_pop: np.ndarray, w_pop: np.ndarray, Cfun) -> np.ndarray:
    """1/V_max-style detectability of a cluster of absolute magnitude M placed on the orbit population:

        f(M) = sum_j w_j <C(M + mu_j(t))>_t / sum_j w_j,

    mu_j(t) = DM(d_j(t)) + A_V,j(t) along population orbit j (sightline extinction does not depend
    on M), w_j = 1/q_j its population weight. Assumes the orbit distribution does not depend on M_V.
    Cfun(V) returns C at apparent magnitudes V (any shape). Weight of an observed cluster i: 1 / f(M_i)."""
    W = np.asarray(w_pop, float) / np.sum(w_pop)
    Mu, inv = np.unique(np.asarray(M_eval, float), return_inverse=True)
    f = np.array([(W * Cfun(m + mu_pop).mean(1)).sum() for m in Mu])
    return f[inv]
