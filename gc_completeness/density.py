"""Spherical GC number-density shapes (amplitude nu0; r in kpc)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrokenPowerLaw:
    """nu(r) = nu0 ((r + rc)/rb)^(-alpha) (1 + r/rb)^(alpha - beta).

    alpha: inner log-slope (rc << r << rb); beta: outer log-slope (r >> rb); rb, rc in kpc.
    """

    alpha: float = 2.0
    beta: float = 4.0
    rb: float = 5.0
    rc: float = 0.3
    nu0: float = 1.0

    def nu(self, r):
        r = np.asarray(r, dtype=float)
        return self.nu0 * ((r + self.rc) / self.rb) ** (-self.alpha) * (1 + r / self.rb) ** (self.alpha - self.beta)


@dataclass
class PowerLaw:
    """nu(r) = nu0 (r + rc)^(-gamma)."""

    gamma: float = 3.5
    rc: float = 0.05
    nu0: float = 1.0

    def nu(self, r):
        r = np.asarray(r, dtype=float)
        return self.nu0 * (r + self.rc) ** (-self.gamma)


def make_model(spec: dict):
    """Build a model from {'type': 'broken_power_law' | 'power_law', **params}."""
    spec = dict(spec)
    typ = spec.pop("type", "broken_power_law")
    if typ == "broken_power_law":
        return BrokenPowerLaw(**spec)
    if typ == "power_law":
        return PowerLaw(**spec)
    raise ValueError(f"Unknown density model type '{typ}'")


def shell_counts(model, edges: np.ndarray, n_sub: int = 200) -> np.ndarray:
    """N[j] = int 4 pi r^2 nu(r) dr over each shell (trapezoid in ln r; linear for a shell from r=0)."""
    edges = np.asarray(edges, dtype=float)
    out = np.empty(len(edges) - 1)
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if a <= 0:
            r = np.linspace(0.0, b, 4 * n_sub)
            out[j] = np.trapezoid(4 * np.pi * r ** 2 * model.nu(r), r)
        else:
            lr = np.linspace(np.log(a), np.log(b), n_sub)
            r = np.exp(lr)
            out[j] = np.trapezoid(4 * np.pi * r ** 3 * model.nu(r), lr)
    return out
