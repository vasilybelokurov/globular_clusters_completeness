"""V-band extinction A_V along orbit samples from 3D/2D dust maps (via the `dustmaps` package).

Map data live under ~/data/dustmaps (set in configs/dustmapsrc.json, selected through the
DUSTMAPS_CONFIG_FNAME environment variable so the user's global ~/.dustmapsrc is untouched).

Conversions to A_V (stated assumptions):
  DECaPS 3D (Zucker et al. 2025): returns E(B-V)                    -> A_V = 3.1 E(B-V)
  Bayestar19 (Green et al. 2019): E(B-V) = 0.883 x map value         -> A_V = 3.1 E(B-V)
      (0.883 from Green+19 Eq. 30 with Schlafly & Finkbeiner 2011; dustmaps documentation)
  Marshall et al. (2006): returns A_Ks                              -> A_V = 8.8 A_Ks (Cardelli, R_V = 3.1;
      uncertain in the inner Galaxy where the near-IR law is steeper). Reliable only out to the
      farthest distance slice of each sightline (dustmaps clamps beyond it).
  SFD (2D): E(B-V)_SFD, integrated along the whole sightline        -> A_V = 2.742 E(B-V)_SFD
      (Schlafly & Finkbeiner 2011, Table 6)

Combination (per sample, first available wins): DECaPS (footprint 239 < l < 6 deg, |b| < 10 deg,
d <= 10 kpc) > Bayestar19 (flagged reliable) > Marshall (finite) > SFD.
Beyond a 3D map's reliable distance the true A_V is bracketed: LOW = the 3D value at its limit
(no more dust), HIGH = SFD (all dust of the sightline in front). Both brackets are returned.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

R_V = 3.1
AV_PER_AKS = 8.8
AV_PER_EBV_SFD = 2.742
BAYESTAR_TO_EBV = 0.883
DECAPS_DMAX = 10.0   # kpc, maximum reliable distance quoted by Zucker et al. (2025)


def setup(project_root: str | Path | None = None) -> str:
    """Point dustmaps at the project config (data in ~/data/dustmaps); must run before importing dustmaps."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    fn = root / "configs" / "dustmapsrc.json"
    os.environ["DUSTMAPS_CONFIG_FNAME"] = str(fn)
    return str(fn)


def _coords(l, b, d):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    return SkyCoord(l=np.asarray(l) * u.deg, b=np.asarray(b) * u.deg, distance=np.asarray(d) * u.kpc, frame="galactic")


def in_decaps_footprint(l, b):
    l = np.mod(l, 360.0)
    return ((l > 239.0) | (l < 6.0)) & (np.abs(b) < 10.0)


def query_maps(l, b, d, maps=("sfd", "marshall", "bayestar", "decaps"), chunk=200_000) -> dict:
    """Return dict of A_V arrays (NaN where a map gives no valid value) plus validity flags.

    l, b in deg, d in kpc; 1-D arrays of equal length.
    """
    setup()
    l, b, d = (np.asarray(x, float).ravel() for x in (l, b, d))
    n = len(l)
    out = {}
    if "sfd" in maps:
        from dustmaps.sfd import SFDQuery
        q = SFDQuery()
        out["sfd"] = np.concatenate([AV_PER_EBV_SFD * q(_coords(l[i:i + chunk], b[i:i + chunk], d[i:i + chunk]))
                                     for i in range(0, n, chunk)])
    if "marshall" in maps:
        from dustmaps.marshall import MarshallQuery
        q = MarshallQuery()
        vals, dmax = [], []
        for i in range(0, n, chunk):
            cc = _coords(l[i:i + chunk], b[i:i + chunk], d[i:i + chunk])
            vals.append(np.asarray(q(cc), float))
            j, k, ok = q._gal2idx(cc)                 # farthest distance slice of each sightline
            dm = np.full(len(ok), np.nan)
            dm[ok] = q._dist[j[ok], k[ok], q._n_dists[j[ok], k[ok]] - 1]
            dmax.append(dm)
        v, dm = np.concatenate(vals), np.concatenate(dmax)
        out["marshall_any"] = AV_PER_AKS * v          # beyond the last slice dustmaps returns the last value
        out["marshall"] = np.where(d <= dm, AV_PER_AKS * v, np.nan)
        out["marshall_dmax"] = dm
    if "bayestar" in maps:
        from dustmaps.bayestar import BayestarQuery
        q = BayestarQuery(version="bayestar2019", max_samples=1)
        vals, rel = [], []
        for i in range(0, n, chunk):
            r, f = q(_coords(l[i:i + chunk], b[i:i + chunk], d[i:i + chunk]), mode="median", return_flags=True)
            vals.append(np.asarray(r, float)); rel.append(np.asarray(f["reliable_dist"], bool))
        v, rel = np.concatenate(vals), np.concatenate(rel)
        out["bayestar"] = np.where(rel, R_V * BAYESTAR_TO_EBV * v, np.nan)
        out["bayestar_any"] = R_V * BAYESTAR_TO_EBV * v          # unflagged value (for the LOW bracket)
    if "decaps" in maps:
        from dustmaps.decaps import DECaPSQueryLite
        q = DECaPSQueryLite(mean_only=True)
        foot = in_decaps_footprint(l, b)
        v = np.full(n, np.nan)
        idx = np.where(foot)[0]
        dq = np.minimum(d[idx], DECAPS_DMAX)                     # value at <= 10 kpc
        for i in range(0, len(idx), chunk):
            sl = idx[i:i + chunk]
            v[sl] = np.asarray(q(_coords(l[sl], b[sl], dq[i:i + chunk])), float)
        out["decaps_at_dmax"] = R_V * v                           # E(B-V) -> A_V; capped distance
        out["decaps"] = np.where(foot & (d <= DECAPS_DMAX), R_V * v, np.nan)
    return out


def combine(q: dict, d: np.ndarray) -> dict:
    """Priority combination and LOW/HIGH brackets beyond 3D-map reach (see module docstring)."""
    n = len(d)
    best = np.full(n, np.nan)
    src = np.full(n, "", dtype=object)
    for name in ("decaps", "bayestar", "marshall"):
        if name in q:
            m = np.isnan(best) & np.isfinite(q[name])
            best[m] = q[name][m]
            src[m] = name
    # samples with no reliable 3D value: LOW = last 3D value available (DECaPS at 10 kpc, Bayestar any), HIGH = SFD
    no3d = np.isnan(best)
    low = best.copy()
    for name in ("decaps_at_dmax", "marshall_any", "bayestar_any"):
        if name in q:
            m = np.isnan(low) & np.isfinite(q[name])
            low[m] = q[name][m]
    sfd = q["sfd"]
    low = np.where(np.isnan(low), sfd, low)       # no 3D information at all -> SFD
    high = np.where(no3d, sfd, best)
    # a 3D value cannot exceed the sightline total by construction; keep LOW <= HIGH
    low = np.minimum(low, high)
    best_low = np.where(no3d, low, best)
    src[no3d & np.isfinite(sfd)] = "sfd/bracket"
    return {"A_low": best_low, "A_high": high, "source": src}


def orbit_extinction(l_t, b_t, d_t, cache_dir: str | Path, tag: str) -> dict:
    """A_V brackets for all orbit samples (N, T), cached on (tag, map list, conversions)."""
    key = hashlib.sha1(f"{tag}|{R_V}|{AV_PER_AKS}|{AV_PER_EBV_SFD}|{BAYESTAR_TO_EBV}|{DECAPS_DMAX}|v2".encode()).hexdigest()[:16]
    fn = Path(cache_dir) / f"dust_{key}.npz"
    if fn.exists():
        z = np.load(fn, allow_pickle=True)
        return {k: z[k] for k in z.files}
    shape = np.shape(d_t)
    q = query_maps(np.ravel(l_t), np.ravel(b_t), np.ravel(d_t))
    c = combine(q, np.ravel(d_t))
    res = {"A_low": c["A_low"].reshape(shape), "A_high": c["A_high"].reshape(shape),
           "source": c["source"].reshape(shape)}
    for k, v in q.items():
        res[f"raw_{k}"] = np.asarray(v).reshape(shape)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fn, **res)
    return res


def disc_extinction(l, b, d, a0: float = 1.0, h_R: float = 3.0, h_z: float = 0.1, R0: float = 8.122,
                    z_sun: float = 0.0208, n_steps: int = 64, A_max: float = 60.0) -> np.ndarray:
    """Analytic exponential dust disc for MOCKS: A_V = a0 int_0^d exp(-(R-R0)/h_R - |z|/h_z) ds [mag].

    a0 is A_V per kpc in the plane at the solar circle. Approximate geometry (Sun at (R0, 0, z_sun))."""
    l = np.radians(np.asarray(l, float)); b = np.radians(np.asarray(b, float)); dd = np.asarray(d, float)
    shape = dd.shape
    l, b, dd = l.ravel(), b.ravel(), dd.ravel()
    out = np.empty_like(dd)
    f = (np.arange(n_steps) + 0.5) / n_steps
    for i in range(0, len(dd), 100_000):
        sl = slice(i, i + 100_000)
        s = dd[sl, None] * f[None, :]
        cb = np.cos(b[sl])[:, None]
        x = R0 - s * cb * np.cos(l[sl])[:, None]
        y = s * cb * np.sin(l[sl])[:, None]
        z = z_sun + s * np.sin(b[sl])[:, None]
        rho = np.exp(-(np.hypot(x, y) - R0) / h_R - np.abs(z) / h_z)
        out[sl] = a0 * dd[sl] * rho.mean(1)
    return np.minimum(out, A_max).reshape(shape)
