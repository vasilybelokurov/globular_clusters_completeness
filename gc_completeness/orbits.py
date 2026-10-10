"""Orbit integration with AGAMA, caching and quality checks."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import agama

from .potential import ensure_units, time_unit_gyr


@dataclass
class OrbitLibrary:
    """Uniformly time-sampled orbits.

    Attributes
    ----------
    ic : (N, 6) initial conditions [kpc, km/s] (present-day phase-space points).
    times_gyr : (T,) sample times in Gyr (negative if backward integration included).
    xyz : (N, T, 3) Galactocentric positions [kpc].
    energy_rel_err : (N,) max |E(t)-E(0)|/|E(0)|.
    rperi, rapo : (N,) min/max spherical radius over the samples [kpc].
    """

    ic: np.ndarray
    times_gyr: np.ndarray
    xyz: np.ndarray
    energy_rel_err: np.ndarray
    rperi: np.ndarray
    rapo: np.ndarray

    @property
    def n_orbits(self) -> int:
        return self.xyz.shape[0]

    @property
    def r(self) -> np.ndarray:
        return np.sqrt(np.sum(self.xyz ** 2, axis=-1))

    def subset(self, idx) -> "OrbitLibrary":
        return OrbitLibrary(self.ic[idx], self.times_gyr, self.xyz[idx],
                            self.energy_rel_err[idx], self.rperi[idx], self.rapo[idx])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, ic=self.ic, times_gyr=self.times_gyr, xyz=self.xyz.astype(np.float32),
                            energy_rel_err=self.energy_rel_err, rperi=self.rperi, rapo=self.rapo)

    @classmethod
    def load(cls, path: str | Path) -> "OrbitLibrary":
        d = np.load(path)
        return cls(d["ic"], d["times_gyr"], d["xyz"].astype(float), d["energy_rel_err"],
                   d["rperi"], d["rapo"])


def _integrate(pot: agama.Potential, ic: np.ndarray, t_internal: float, n: int):
    res = agama.orbit(potential=pot, ic=ic, time=t_internal, trajsize=n, verbose=False)
    times = np.asarray(res[0, 0])
    traj = np.stack([np.asarray(r) for r in res[:, 1]])  # (N, n, 6)
    return times, traj


def integrate_orbits(pot: agama.Potential, ic: np.ndarray, integration_time_gyr: float = 5.0,
                     n_samples: int = 1000, both_directions: bool = False,
                     cache_dir: str | None = None, cache_tag: str = "") -> OrbitLibrary:
    """Integrate orbits in a static potential, sampled uniformly in time.

    In a static potential, the forward and backward halves sample the same orbit
    (time-reversal of a phase point gives the same orbit traversed backwards), so
    ``both_directions`` only doubles the time baseline; it is off by default.

    Parameters
    ----------
    pot : agama.Potential
    ic : (N, 6) initial conditions in kpc, km/s.
    integration_time_gyr : total time per direction [Gyr].
    n_samples : number of samples per direction (including t=0).
    cache_dir : if given, results are cached in an .npz keyed by a hash of inputs.
    cache_tag : string identifying the potential/config (part of the cache key).
    """
    ensure_units()
    ic = np.ascontiguousarray(np.asarray(ic, dtype=float))
    cache_path = None
    if cache_dir:
        h = hashlib.sha1()
        h.update(ic.tobytes())
        h.update(f"{integration_time_gyr}|{n_samples}|{both_directions}|{cache_tag}".encode())
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"orbits_{h.hexdigest()[:16]}.npz"
        if cache_path.exists():
            return OrbitLibrary.load(cache_path)

    tu = time_unit_gyr()
    t_int = integration_time_gyr / tu
    times, traj = _integrate(pot, ic, t_int, n_samples)
    if both_directions:
        tb, trajb = _integrate(pot, ic, -t_int, n_samples)
        times = np.concatenate([tb[:0:-1], times])
        traj = np.concatenate([trajb[:, :0:-1], traj], axis=1)

    xyz = traj[..., :3]
    E = pot.potential(xyz.reshape(-1, 3)).reshape(xyz.shape[:2]) + 0.5 * np.sum(traj[..., 3:] ** 2, -1)
    E0 = E[:, [np.argmin(np.abs(times))]]
    rel = np.max(np.abs(E - E0), axis=1) / np.abs(E0[:, 0])
    r = np.sqrt(np.sum(xyz ** 2, -1))
    lib = OrbitLibrary(ic=ic, times_gyr=times * tu, xyz=xyz, energy_rel_err=rel,
                       rperi=r.min(1), rapo=r.max(1))
    if cache_path is not None:
        lib.save(cache_path)
    return lib


def orbit_quality(lib: OrbitLibrary, max_energy_err: float = 1e-4) -> dict:
    """Summary of energy conservation and pericentre/apocentre ranges."""
    bad = np.where(lib.energy_rel_err > max_energy_err)[0]
    return {
        "n_orbits": lib.n_orbits,
        "max_energy_rel_err": float(lib.energy_rel_err.max()),
        "n_bad_energy": int(len(bad)),
        "bad_energy_idx": bad.tolist(),
        "rperi_range": [float(lib.rperi.min()), float(lib.rperi.max())],
        "rapo_range": [float(lib.rapo.min()), float(lib.rapo.max())],
    }
