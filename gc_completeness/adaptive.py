"""Frozen equal-count (M_V, d) bins and per-bin Khoperskov completeness with support diagnostics.

Bins are chosen ONCE from the full observed sample and then frozen; bootstrap replicates and
mock comparisons are evaluated on exactly these bins (fixed estimand).
"""
from __future__ import annotations

import numpy as np

from . import khoperskov as K


def equal_count_edges(x: np.ndarray, n_min: int, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Edges at quantiles of x so that every bin holds >= n_min values (as many bins as possible).

    Ties are handled by merging any bin that ends up below n_min with its smaller neighbour.
    Outer edges are lo/hi if given, else min/max of x widened by 1e-6.
    """
    if n_min < 1:
        raise ValueError("n_min must be >= 1")
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        raise ValueError("equal_count_edges: empty input")
    nb = max(1, n // n_min)
    inner = [0.5 * (x[int(round(k * n / nb)) - 1] + x[int(round(k * n / nb))]) for k in range(1, nb)]
    a = x[0] - 1e-6 if lo is None else lo
    b = x[-1] + 1e-6 if hi is None else hi
    e = np.unique(np.array([a, *inner, b]))
    while len(e) > 2:
        c = np.histogram(x, e)[0]
        k = int(np.argmin(c))
        if c[k] >= n_min:
            break
        # remove the inner edge shared with the smaller neighbour
        if k == 0:
            e = np.delete(e, 1)
        elif k == len(c) - 1:
            e = np.delete(e, len(e) - 2)
        else:
            e = np.delete(e, k if c[k - 1] <= c[k + 1] else k + 1)
    return e


def freeze_bins(M: np.ndarray, d: np.ndarray, n_classes: int, n_min_d: int, d_hi: float) -> list[dict]:
    """Equal-count M_V classes, then equal-count d bins within each class. Returns list of
    {'M_lo','M_hi','d_edges'} (brightest class first)."""
    Me = equal_count_edges(M, max(1, len(M) // n_classes)) if n_classes > 1 else \
        np.array([M.min() - 1e-6, M.max() + 1e-6])
    out = []
    for j in range(len(Me) - 1):
        m = (M >= Me[j]) & (M < Me[j + 1])
        out.append({"M_lo": float(Me[j]), "M_hi": float(Me[j + 1]),
                    "d_edges": equal_count_edges(d[m], n_min_d, lo=0.0, hi=d_hi)})
    return out


def class_completeness(A: np.ndarray, d_t: np.ndarray, d_now: np.ndarray, Nt: np.ndarray,
                       d_edges: np.ndarray, nnls_cfg: dict, seed: int) -> dict:
    """Khoperskov NNLS for one class at unit target amplitude; completeness per d bin relative to bin 0.
    The last d bin is open-ended (d >= d_edges[-2]).

    C_rel    = (n_k / corr_k) / (n_0 / corr_0)          (plain plug-in ratio)
    C_rel_br = (n_k / corr_k) * corr_0 / (n_0 + 1)      (bias-reduced: E[1/(n_0+1)] ~ 1/mu_0 for Poisson n_0)
    """
    floor = 0.5
    wr = K.nnls_subsets(A, Nt, nnls_cfg["frac"], nnls_cfg["n_real"], seed, nnls_cfg["cell_sigma"], floor=floor)
    w = wr.w
    s = np.sqrt(Nt + floor) if nnls_cfg["cell_sigma"] == "poisson" else np.ones_like(Nt)
    wres = float(np.linalg.norm((A @ w - Nt) / s) / np.linalg.norm(Nt / s))
    e = np.asarray(d_edges, float).copy()
    e[-1] = np.inf                       # last bin open-ended (same convention as phaselike)
    corr = K.phase_hist(d_t, w, e)
    obs = np.histogram(d_now, e)[0].astype(float)
    T = d_t.shape[1]
    a = np.stack([np.histogram(d_t[i], e)[0] / T for i in range(len(w))]) if len(w) else np.zeros((0, len(obs)))
    c = a * w[:, None]
    neff = np.where((c ** 2).sum(0) > 0, c.sum(0) ** 2 / np.maximum((c ** 2).sum(0), 1e-300), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(corr > 0, obs / corr, np.nan)
        rel = r / r[0]
        rel_br = r * corr[0] / (obs[0] + 1.0)
    rel_br[0] = 1.0  # reference bin by definition; the bias reduction applies to bins k >= 1
    Ms = A / s[:, None]
    sv = np.linalg.svd(Ms, compute_uv=False)
    return {"w_unit": w, "corr_unit": corr, "obs": obs, "neff": neff, "resid_w": wres, "resid": wr.rel_resid_mean,
            "C_rel": rel, "C_rel_br": rel_br, "n_active": int((w > 0).sum()),
            "cond": float(sv[0] / sv[sv > sv[0] * 1e-12][-1]) if sv.size else np.nan,
            "rank": int((sv > sv[0] * 1e-10).sum()) if sv.size else 0}


def bootstrap_fixed(A, d_t, d_now, Nt, d_edges, nnls_cfg, n_boot, seed) -> dict:
    """Resample the class members with replacement; FIXED d_edges. Returns arrays (n_boot, n_bins)."""
    rng = np.random.default_rng(seed)
    N = len(d_now)
    rel, rel_br, corr = [], [], []
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        r = class_completeness(A[:, idx], d_t[idx], d_now[idx], Nt, d_edges, nnls_cfg, seed + b)
        rel.append(r["C_rel"]); rel_br.append(r["C_rel_br"]); corr.append(r["corr_unit"] / max(r["corr_unit"].sum(), 1e-300))
    return {"C_rel": np.array(rel), "C_rel_br": np.array(rel_br), "corr_frac": np.array(corr)}


def support_flag(neff: np.ndarray, corr_cv: np.ndarray, n_obs: np.ndarray,
                 min_neff: float = 3.0, max_cv: float = 0.5, min_obs: int = 5) -> np.ndarray:
    """True where a bin is resolved: enough effective orbits, stable corrected fraction, enough observed GCs."""
    return (neff >= min_neff) & (corr_cv <= max_cv) & (n_obs >= min_obs)
