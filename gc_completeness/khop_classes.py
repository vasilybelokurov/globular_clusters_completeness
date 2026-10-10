"""Khoperskov et al. (2025) orbit superposition per luminosity class; completeness C(M_V, d).

Assumptions (all explicit, nothing fitted to the data except the NNLS weights):
  * completeness depends on present-day M_V and heliocentric distance d only;
  * each luminosity class follows a FIXED target shape nu(r) (T1/T2/T3), amplitude unknown;
  * orbits are phase-mixed in a static axisymmetric potential.

For class k, NNLS weights (paper: w >= 0, random-subset averaging) make the class orbits
reproduce the target shape on an (R, |z|) grid. With the amplitude unknown the weights are
known up to a constant s_k, so

    N_corr,k(d) = s_k * sum_{i in k} w_i <1[d_i(t) in d-bin]>_t,   C_k(d) = N_obs,k(d) / N_corr,k(d).

* Relative completeness C_k(d) / C_k(d_ref) is independent of s_k (anchor-free; primary result).
* Absolute values need s_k: we show them under an explicit anchor "C_k = 1 for d < d_anchor".
"""
from __future__ import annotations

import numpy as np

from . import khoperskov as K


def class_curves(A: np.ndarray, d_t: np.ndarray, d_now: np.ndarray, Nt_shape: np.ndarray,
                 d_edges: np.ndarray, nnls_cfg: dict, seed: int) -> dict:
    """Unit-amplitude reconstruction for one class.

    A : (cells, N) (R,|z|) occupancy; d_t : (N, T) heliocentric distance along orbits;
    d_now : (N,) present distance of the observed members; Nt_shape: target counts at unit amplitude.
    """
    wr = K.nnls_subsets(A, Nt_shape, nnls_cfg["frac"], nnls_cfg["n_real"], seed, nnls_cfg["cell_sigma"])
    w = wr.w
    corr_unit = K.phase_hist(d_t, w, d_edges)              # sum_i w_i <1[d_i(t) in bin]>
    obs = np.histogram(d_now, d_edges)[0].astype(float)
    # orbit support per d bin: effective number of orbits contributing
    T = d_t.shape[1]
    a = np.stack([np.histogram(d_t[i], d_edges)[0] / T for i in range(len(w))])  # (N, nb)
    c = a * w[:, None]
    neff = np.where((c ** 2).sum(0) > 0, c.sum(0) ** 2 / np.maximum((c ** 2).sum(0), 1e-300), 0.0)
    return {"w_unit": w, "corr_unit": corr_unit, "obs": obs, "neff": neff, "resid": wr.rel_resid_mean,
            "W_unit_total": float(w.sum())}


def relative_completeness(obs, corr_unit, ref_bin: int = 0):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(corr_unit > 0, obs / corr_unit, np.nan)
    return r / r[ref_bin] if np.isfinite(r[ref_bin]) and r[ref_bin] > 0 else r * np.nan


def anchored(obs, corr_unit, W_unit_total, d_edges, d_anchor: float):
    """Scale s so that C = 1 for d < d_anchor (bins with upper edge <= d_anchor)."""
    sel = d_edges[1:] <= d_anchor + 1e-9
    s = obs[sel].sum() / max(corr_unit[sel].sum(), 1e-300)
    corr = s * corr_unit
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(corr > 0, obs / corr, np.nan)
    return {"scale": float(s), "corr": corr, "C": C, "N_total": float(s * W_unit_total),
            "N_corr_binned": float(corr.sum())}
