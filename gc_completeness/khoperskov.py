"""Orbit-superposition building blocks (Khoperskov et al. 2025, A&A 695, A220;
https://arxiv.org/abs/2411.15062): target counts on an (R, |z|) grid, orbit occupancy, NNLS weights.

A[k, i] is the fraction of time orbit i spends in cell k; non-negative weights w solve
sum_i w_i A[k, i] = N_target[k]. Orbits absent from the observed library cannot be recovered
(paper, Sec. 4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass
class RZGrid:
    """(R, |z|) cells in kpc; |z| folds both signs of z."""

    R_edges: np.ndarray
    z_edges: np.ndarray

    @property
    def shape(self):
        return len(self.R_edges) - 1, len(self.z_edges) - 1

    @property
    def n_cells(self):
        return int(np.prod(self.shape))

    def index(self, R, absz):
        """Flat cell index, or -1 outside the grid."""
        i = np.searchsorted(self.R_edges, R, side="right") - 1
        j = np.searchsorted(self.z_edges, absz, side="right") - 1
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        return np.where(ok, i * self.shape[1] + j, -1)


def target_counts(model, grid: RZGrid, n_sub: int = 24, q: float = 1.0) -> np.ndarray:
    """N[k] = int_cell 2 * 2 pi R nu(sqrt(R^2 + z^2/q^2)) dR d|z| (factor 2: both signs of z).

    q = 1: spherical; q < 1: oblate (flattened towards the plane)."""
    out = np.zeros(grid.shape)
    g = (np.arange(n_sub) + 0.5) / n_sub
    for i, (R0, R1) in enumerate(zip(grid.R_edges[:-1], grid.R_edges[1:])):
        R = R0 + (R1 - R0) * g
        for j, (z0, z1) in enumerate(zip(grid.z_edges[:-1], grid.z_edges[1:])):
            z = z0 + (z1 - z0) * g
            RR, ZZ = np.meshgrid(R, z, indexing="ij")
            f = 2 * 2 * np.pi * RR * model.nu(np.sqrt(RR ** 2 + (ZZ / q) ** 2))
            out[i, j] = f.mean() * (R1 - R0) * (z1 - z0)
    return out.ravel()


def occupancy_rz(xyz: np.ndarray, grid: RZGrid) -> np.ndarray:
    """A[k, i] = fraction of the samples of orbit i in cell k; samples outside the grid are dropped."""
    N, T, _ = xyz.shape
    R = np.hypot(xyz[..., 0], xyz[..., 1])
    z = np.abs(xyz[..., 2])
    k = grid.index(R, z)
    A = np.zeros((grid.n_cells, N))
    cols = np.broadcast_to(np.arange(N)[:, None], (N, T))
    ok = k >= 0
    np.add.at(A, (k[ok], cols[ok]), 1.0 / T)
    return A


@dataclass
class WeightResult:
    w: np.ndarray            # mean weights over realisations
    w_real: np.ndarray       # (n_real, N) weights per realisation (0 if the orbit is not in the subset)
    rel_resid: np.ndarray    # (n_real,) ||A w - N|| / ||N|| per realisation
    rel_resid_mean: float    # for the averaged weights


def nnls_subsets(A: np.ndarray, N_target: np.ndarray, frac: float = 1.0, n_real: int = 1,
                 seed: int = 0, cell_sigma: str = "poisson", floor: float = 0.5) -> WeightResult:
    """Fit sum_i w_i A[:, i] = N_target with w >= 0; with frac < 1, fit random orbit subsets
    and average (paper Sec. 3.1). cell_sigma='poisson' weights residuals by 1/sqrt(N_target + floor)."""
    rng = np.random.default_rng(seed)
    K, N = A.shape
    s = np.sqrt(N_target + floor) if cell_sigma == "poisson" else np.ones(K)
    Ms, ys = A / s[:, None], N_target / s
    m = max(1, int(round(frac * N)))
    W = np.zeros((n_real, N))
    rr = np.zeros(n_real)
    for r in range(n_real):
        idx = rng.choice(N, m, replace=False) if m < N else np.arange(N)
        w_sub, _ = nnls(Ms[:, idx], ys, maxiter=50 * N)
        W[r, idx] = w_sub
        rr[r] = np.linalg.norm(A @ W[r] - N_target) / np.linalg.norm(N_target)
    w = W.mean(0)
    return WeightResult(w=w, w_real=W, rel_resid=rr,
                        rel_resid_mean=float(np.linalg.norm(A @ w - N_target) / np.linalg.norm(N_target)))


def phase_hist(X: np.ndarray, w: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """sum_i w_i <1[X_i(t) in bin]>_t for (N, T) orbit samples; samples outside edges are dropped."""
    N, T = X.shape
    return np.histogram(X.ravel(), bins=edges, weights=np.repeat(w / T, T))[0]
