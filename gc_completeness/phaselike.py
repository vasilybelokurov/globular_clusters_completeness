"""Per-cluster conditional-phase likelihood for completeness C(d), and per-cluster symmetry tests.

Assumptions: completeness depends only on (M_V, d) [here: piecewise constant in d bins within an
M_V class]; each cluster's present position is a random phase of its own orbit (phase mixing,
static potential). Then, conditional on the orbit, the present phase t has density
p(t | detected, orbit) proportional to C(d(t)), so

    ln L = sum_i [ ln C(d_i,now) - ln <C(d_i(t))>_t ].

No target density enters; conditioning on each observed orbit removes the dependence on which
orbits were detected. C is identified up to one constant per class (we fix C_0 = 1 in the
nearest bin), over the distance range spanned by the clusters' orbits.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import kstest

THETA_BOUNDS = (-8.0, 4.0)   # ln(C_k / C_0)


def time_fractions(d_t: np.ndarray, d_edges: np.ndarray) -> np.ndarray:
    """F[i, k] = fraction of samples of orbit i in d bin k; the last bin is open-ended (>= d_edges[-2])."""
    e = np.asarray(d_edges, float).copy()
    e[-1] = np.inf
    k = np.clip(np.searchsorted(e, d_t, side="right") - 1, 0, len(e) - 2)
    F = np.zeros((d_t.shape[0], len(e) - 1))
    for j in range(len(e) - 1):
        F[:, j] = (k == j).mean(1)
    return F


def bin_index(d: np.ndarray, d_edges: np.ndarray) -> np.ndarray:
    e = np.asarray(d_edges, float).copy()
    e[-1] = np.inf
    return np.clip(np.searchsorted(e, d, side="right") - 1, 0, len(e) - 2)


def _ll(theta_free, F, kn, wt=None, P=None):
    """Log-likelihood and gradient. Optional cluster weights wt (N,) and soft present-bin
    probabilities P (N, K) (used for the pseudo-true value in mocks); default: hard bins kn."""
    th = np.r_[0.0, theta_free]
    ex = np.exp(th)
    den = F @ ex
    wt = np.ones(len(F)) if wt is None else wt
    if P is None:
        P = np.zeros_like(F); P[np.arange(len(F)), kn] = 1.0
    ll = (wt * (P @ th)).sum() - (wt * np.log(den)).sum()
    g = (wt[:, None] * P).sum(0) - (wt[:, None] * F * ex[None, :] / den[:, None]).sum(0)
    return ll, g[1:]


def fit(F: np.ndarray, kn: np.ndarray, wt=None, P=None) -> dict:
    """MLE of theta_k = ln(C_k/C_0), k >= 1, within THETA_BOUNDS. 'at_bound' flags estimates at the
    imposed limits (the data do not constrain them inside the allowed range)."""
    K = F.shape[1]
    if K < 2:
        return {"theta": np.zeros(0), "ll": 0.0, "at_bound": np.zeros(0, bool), "success": True}
    res = minimize(lambda t: tuple(-v for v in _ll(t, F, kn, wt, P)), np.zeros(K - 1), jac=True,
                   method="L-BFGS-B", bounds=[THETA_BOUNDS] * (K - 1))
    th = res.x
    at_b = (np.abs(th - THETA_BOUNDS[0]) < 1e-3) | (np.abs(th - THETA_BOUNDS[1]) < 1e-3)
    return {"theta": th, "ll": -float(res.fun), "at_bound": at_b, "success": bool(res.success)}


def profile_interval(F, kn, fitres, j: int, delta: float = 0.5) -> tuple[float, float]:
    """Profile-likelihood interval for theta_j (2 Delta lnL = 1 -> 68%). An endpoint equal to a
    THETA_BOUNDS limit means the interval reaches the imposed limit on that side. NaN if a fit failed."""
    K = F.shape[1]
    th_hat, llmax = fitres["theta"], fitres["ll"]
    others = [k for k in range(K - 1) if k != j]

    def prof(v):
        if not others:
            t = np.array([v])
            return _ll(t, F, kn)[0]

        def f(o):
            t = th_hat.copy(); t[others] = o; t[j] = v
            ll, g = _ll(t, F, kn)
            return -ll, -g[others]
        r = minimize(f, th_hat[others], jac=True, method="L-BFGS-B", bounds=[THETA_BOUNDS] * len(others))
        if not r.success and not np.isfinite(r.fun):
            raise RuntimeError("profile nuisance fit failed")
        return -r.fun

    if not fitres.get("success", True):
        return np.nan, np.nan
    target = lambda v: llmax - prof(v) - delta  # noqa: E731
    lo_b, hi_b = THETA_BOUNDS
    try:
        lo = brentq(target, lo_b, th_hat[j]) if target(lo_b) > 0 and th_hat[j] > lo_b + 1e-6 else lo_b
        hi = brentq(target, th_hat[j], hi_b) if target(hi_b) > 0 and th_hat[j] < hi_b - 1e-6 else hi_b
    except (RuntimeError, ValueError):
        return np.nan, np.nan
    return lo, hi


# ---------------------------------------------------------------------------
# per-cluster symmetry statistics (randomised PIT against each orbit's C-weighted phase distribution)
# ---------------------------------------------------------------------------
def orbit_statistics(xyz_t: np.ndarray) -> dict:
    """z (signed), y (signed; Galactic-longitude side) and |phi'| (azimuth from the Sun direction, 0..pi)."""
    x, y, z = xyz_t[..., 0], xyz_t[..., 1], xyz_t[..., 2]
    return {"z": z, "y": y, "absphi": np.abs(np.arctan2(y, -x))}


def pit_values(S: np.ndarray, s_now: np.ndarray, Cw: np.ndarray, rng) -> np.ndarray:
    """Randomised PIT: P_w(S < s_now) + U * P_w(S == s_now), with weights Cw (N, T) per orbit."""
    w = Cw / Cw.sum(1, keepdims=True)
    below = (w * (S < s_now[:, None])).sum(1)
    eq = (w * (S == s_now[:, None])).sum(1)
    return below + rng.random(len(s_now)) * eq


def C_along(d_t, d_edges, theta) -> np.ndarray:
    th = np.r_[0.0, theta]
    return np.exp(th)[bin_index(d_t, d_edges)]


def symmetry_ks(groups: list[dict], rng) -> dict:
    """Pool PIT values over groups (each: xyz_t, d_t, now_idx, d_edges, theta) -> KS statistic per statistic."""
    pits = {"z": [], "y": [], "absphi": []}
    for g in groups:
        st = orbit_statistics(g["xyz_t"])
        Cw = C_along(g["d_t"], g["d_edges"], g["theta"])
        ii = np.arange(len(g["now_idx"]))
        for k in pits:
            pits[k].append(pit_values(st[k], st[k][ii, g["now_idx"]], Cw, rng))
    return {k: float(kstest(np.concatenate(v), "uniform").statistic) for k, v in pits.items()}, \
        {k: np.concatenate(v) for k, v in pits.items()}


def simulate_phases(d_t, d_edges, theta, rng) -> np.ndarray:
    """Draw a present-phase index per orbit with probability proportional to C(d(t)) (null model)."""
    Cw = C_along(d_t, d_edges, theta)
    p = Cw / Cw.sum(1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return (np.cumsum(p, 1) < u).sum(1)


def pseudo_true(d_t_true: np.ndarray, C_true_t: np.ndarray, d_edges: np.ndarray) -> np.ndarray:
    """Value the phase-likelihood estimate converges to for an infinite sample from a true population
    (mock use only): maximise E[lnL] with each true orbit weighted by its detection probability
    q_i = <C_true>_t and its present-bin probabilities P_ik ~ <C_true 1[bin k]>_t / q_i."""
    F = time_fractions(d_t_true, d_edges)
    k = bin_index(d_t_true, d_edges)
    q = C_true_t.mean(1)
    P = np.stack([(C_true_t * (k == j)).mean(1) for j in range(F.shape[1])], 1) / np.maximum(q, 1e-300)[:, None]
    ok = q > 0
    return fit(F[ok], None, wt=q[ok], P=P[ok])["theta"]
