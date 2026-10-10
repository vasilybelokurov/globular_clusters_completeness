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
