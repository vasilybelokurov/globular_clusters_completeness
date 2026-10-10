"""Mock GC populations: draw phase-space points from an AGAMA quasi-spherical isotropic DF."""
from __future__ import annotations

import numpy as np
import agama

from .density import make_model
from .potential import ensure_units


def sample_population(pot, density_spec: dict, n: int, r_trunc: float, seed: int) -> np.ndarray:
    """Draw n phase-space points from an isotropic quasi-spherical DF for density_spec.

    The density is multiplied by exp(-(r/r_trunc)^2) to keep the population finite.
    """
    ensure_units()
    model = make_model(density_spec)

    def rho(xyz):
        r = np.sqrt(np.sum(xyz ** 2, axis=-1))
        return model.nu(r) * np.exp(-(r / r_trunc) ** 2)

    den = agama.Density(type="DensitySphericalHarmonic", density=rho, symmetry="s", lmax=0,
                        gridsizer=80, rmin=1e-3, rmax=5 * r_trunc)
    df = agama.DistributionFunction(type="quasispherical", density=den, potential=pot, beta0=0.0)
    agama.setRandomSeed(int(seed) % (2 ** 31))
    xv, _ = agama.GalaxyModel(pot, df).sample(int(n))
    return np.asarray(xv)
