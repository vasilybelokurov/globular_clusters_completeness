"""Faint-end estimator comparison on one sample (real or mock), with paired cluster bootstraps.

Estimators (all with the same continuous C(V) fit, cv.fit_cv_spline):
  ht      1/q_i, q_i = <C(V_i(t))>_t along cluster i's own orbit
  vmax    1/f(M_i), f = 1/q-weighted mean detectability over the orbits of bright clusters (M_V < M_bright_ref)
  kernel  1/f(M_i), population = all clusters weighted 1/q_j K(M_j - M_i), Gaussian kernel (luminosity-matched)
  midref  1/f(M_i), for faint targets the population is the intermediate class (M_bright_ref <= M_V < M_faint),
          otherwise the bright clusters
Diagnostics: R = faint(vmax) / faint(ht); effective size and largest normalised weight of the bright
population; 1/q-weighted and raw median apocentres of faint and bright clusters.
Bootstraps: 'fixed' keeps the spline nodes of the full sample; 'nodes' recomputes them from each resample.
"""
from __future__ import annotations

import numpy as np

from . import cv as CV


def make_nodes(V_now: np.ndarray, n_min: int, extra_node_pct: float | None = None) -> np.ndarray:
    nodes = CV.spline_nodes(V_now, n_min)
    if extra_node_pct is not None:
        v = np.percentile(V_now, extra_node_pct)
        if v > nodes[-1] + 0.3:
            nodes = np.r_[nodes, v]
    return nodes


def _wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(c, 0.5)])


def estimates(M, V_t, V_now, nodes, tail, fp, rapo=None, groups=None, kernel=True, p0=None):
    """All estimators and diagnostics for one (sub)sample. fp = cfg['faint'] dict."""
    M = np.asarray(M, float)
    B_t, B_now = CV.spline_basis(V_t, nodes, tail), CV.spline_basis(V_now, nodes, tail)
    f = CV.fit_cv_spline(B_t, B_now, p0 if (p0 is not None and len(p0) == B_t.shape[-1]) else None)
    q = np.exp(B_t @ f["p"]).mean(1)
    Cfun = lambda V: np.exp(CV.lnC_spline(V, nodes, f["p"], tail))
    mu = V_t - M[:, None]
    Mf, Mb = fp["M_faint"], fp["M_bright_ref"]
    faint, bright, mid = M >= Mf, M < Mb, (M >= Mb) & (M < Mf)
    out = {"p": f["p"], "success": f["success"], "ll": f["ll"]}
    w = {"ht": 1.0 / q}
    g = np.zeros(len(M), int) if groups is None else np.asarray(groups)
    fv = np.full(len(M), np.nan)
    for gv in np.unique(g):
        ing, pop = g == gv, (g == gv) & bright
        if pop.any():
            fv[ing] = CV.population_detectability(M[ing], mu[pop], 1.0 / q[pop], Cfun)
    w["vmax"] = 1.0 / fv
    fm = CV.population_detectability(M, mu[bright], 1.0 / q[bright], Cfun)
    if faint.any() and mid.any():
        fm[faint] = CV.population_detectability(M[faint], mu[mid], 1.0 / q[mid], Cfun)
    w["midref"] = 1.0 / fm
    if kernel:
        w["kernel"] = 1.0 / CV.kernel_detectability(M, M, mu, 1.0 / q, Cfun, fp["kernel_sigma"])
    for k, v in w.items():
        out[f"{k}_tot"] = float(np.nansum(v))
        out[f"{k}_faint"] = float(np.nansum(v[faint]))
    out["R"] = out["vmax_faint"] / out["ht_faint"] if out["ht_faint"] > 0 else np.nan
    wb = 1.0 / q[bright]
    out["ess_bright"] = float(wb.sum() ** 2 / (wb ** 2).sum()) if bright.any() else np.nan
    out["maxw_bright"] = float(wb.max() / wb.sum()) if bright.any() else np.nan
    if rapo is not None and faint.any() and bright.any():
        out["rapo_w_faint"] = _wmedian(rapo[faint], 1.0 / q[faint])
        out["rapo_w_bright"] = _wmedian(rapo[bright], 1.0 / q[bright])
        out["rapo_faint"], out["rapo_bright"] = float(np.median(rapo[faint])), float(np.median(rapo[bright]))
        out["dlnrapo_w"] = np.log(out["rapo_w_faint"] / out["rapo_w_bright"])
        out["dlnrapo"] = np.log(out["rapo_faint"] / out["rapo_bright"])
    out["weights"] = w
    return out


SCALARS = ("ht_tot", "ht_faint", "vmax_tot", "vmax_faint", "midref_tot", "midref_faint", "kernel_tot", "kernel_faint",
           "R", "ess_bright", "maxw_bright", "dlnrapo_w", "dlnrapo")


def run(M, V_t, V_now, fp, n_min, tail=1.0, extra_node_pct=None, rapo=None, groups=None, n_boot=0, rng=None,
        kernel=True, boot_modes=("fixed", "nodes")):
    """Point estimates plus paired bootstraps ({mode: {scalar: array}}) for each bootstrap mode."""
    nodes = make_nodes(V_now, n_min, extra_node_pct)
    est = estimates(M, V_t, V_now, nodes, tail, fp, rapo, groups, kernel)
    boots = {}
    N = len(M)
    for mode in boot_modes if n_boot else ():
        rows = []
        for _ in range(n_boot):
            idx = rng.integers(0, N, N)
            nd = nodes if mode == "fixed" else make_nodes(V_now[idx], n_min, extra_node_pct)
            try:
                e = estimates(M[idx], V_t[idx], V_now[idx], nd, tail, fp, None if rapo is None else rapo[idx],
                              None if groups is None else np.asarray(groups)[idx], kernel, est["p"])
                rows.append([e.get(k, np.nan) for k in SCALARS])
            except ValueError:            # e.g. too few distinct V for the nodes
                rows.append([np.nan] * len(SCALARS))
        a = np.array(rows, float)
        boots[mode] = {k: a[:, j] for j, k in enumerate(SCALARS)}
    return est, boots, nodes
