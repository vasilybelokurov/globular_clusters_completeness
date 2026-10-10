"""Coordinate transformations (astropy-based, with a fast exact affine inverse).

Convention: astropy ``Galactocentric`` frame (Sun at x ~ -R0, Galactic rotation
towards +y). Positions in kpc, velocities in km/s.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import astropy.units as u
import astropy.coordinates as coord


@dataclass
class GalacticParams:
    """Solar position/motion. Defaults = astropy v4.0 Galactocentric parameters.

    R0 = 8.122 kpc (GRAVITY 2018, https://doi.org/10.1051/0004-6361/201833718),
    z_sun = 20.8 pc (Bennett & Bovy 2019, https://doi.org/10.1093/mnras/sty2813),
    v_sun = (12.9, 245.6, 7.78) km/s (Drimmel & Poggio 2018,
    https://doi.org/10.3847/2515-5172/aaef8b). See
    https://docs.astropy.org/en/stable/api/astropy.coordinates.Galactocentric.html
    """

    R0: float = 8.122
    z_sun: float = 0.0208
    v_sun: tuple[float, float, float] = (12.9, 245.6, 7.78)
    roll_deg: float = 0.0
    references: list[str] = field(default_factory=lambda: [
        "GRAVITY 2018 A&A 615 L15 (R0)", "Bennett & Bovy 2019 MNRAS 482 1417 (z_sun)",
        "Drimmel & Poggio 2018 RNAAS 2 210 (v_sun)"])

    @classmethod
    def from_dict(cls, d: dict | None) -> "GalacticParams":
        d = dict(d or {})
        if "v_sun" in d:
            d["v_sun"] = tuple(d["v_sun"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    def frame(self) -> coord.Galactocentric:
        return coord.Galactocentric(
            galcen_distance=self.R0 * u.kpc,
            z_sun=self.z_sun * u.kpc,
            galcen_v_sun=coord.CartesianDifferential(list(self.v_sun) * u.km / u.s),
            roll=self.roll_deg * u.deg,
        )


def icrs_to_galactocentric(ra, dec, dist, pmra, pmdec, vlos, params: GalacticParams) -> np.ndarray:
    """Heliocentric observables -> Galactocentric phase space.

    Parameters
    ----------
    ra, dec : deg; dist : kpc; pmra (mu_alpha*cos dec), pmdec : mas/yr; vlos : km/s.

    Returns
    -------
    (N, 6) array [x, y, z, vx, vy, vz] in kpc and km/s.
    """
    c = coord.SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg,
                       distance=np.asarray(dist) * u.kpc,
                       pm_ra_cosdec=np.asarray(pmra) * u.mas / u.yr,
                       pm_dec=np.asarray(pmdec) * u.mas / u.yr,
                       radial_velocity=np.asarray(vlos) * u.km / u.s, frame="icrs")
    g = c.transform_to(params.frame())
    return np.column_stack([g.x.to_value(u.kpc), g.y.to_value(u.kpc), g.z.to_value(u.kpc),
                            g.v_x.to_value(u.km / u.s), g.v_y.to_value(u.km / u.s),
                            g.v_z.to_value(u.km / u.s)])


def icrs_to_galactic_lbd(ra, dec, dist):
    """ICRS (deg, deg, kpc) -> Galactic (l, b) in deg and distance in kpc."""
    c = coord.SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg, frame="icrs")
    gal = c.galactic
    return gal.l.deg, gal.b.deg, np.asarray(dist, dtype=float)


class GalactocentricToGalactic:
    """Exact affine map from Galactocentric Cartesian to heliocentric Galactic (l, b, d).

    The astropy transformation Galactocentric -> Galactic is a rotation plus a
    translation; we evaluate it once with astropy and then apply it with numpy,
    which is ~100x faster for the 1e5-1e6 orbit samples used here.
    """

    def __init__(self, params: GalacticParams):
        frame = params.frame()
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        gc = coord.SkyCoord(x=pts[:, 0] * u.kpc, y=pts[:, 1] * u.kpc, z=pts[:, 2] * u.kpc,
                            frame=frame, representation_type="cartesian")
        gal = gc.transform_to(coord.Galactic())
        cart = gal.cartesian.xyz.to_value(u.kpc).T  # (4, 3)
        self.offset = cart[0]
        self.matrix = (cart[1:] - cart[0]).T  # columns = images of unit vectors

    def __call__(self, xyz: np.ndarray):
        """Return l [deg, 0..360), b [deg], d [kpc] for (..., 3) Galactocentric positions."""
        h = np.asarray(xyz) @ self.matrix.T + self.offset
        d = np.sqrt(np.sum(h * h, axis=-1))
        l = np.degrees(np.arctan2(h[..., 1], h[..., 0])) % 360.0
        b = np.degrees(np.arcsin(np.clip(h[..., 2] / np.maximum(d, 1e-12), -1, 1)))
        return l, b, d


def galactic_lbd_to_galactocentric(l, b, d, params: GalacticParams) -> np.ndarray:
    """Galactic (deg, deg, kpc) -> Galactocentric Cartesian positions (kpc), via astropy."""
    c = coord.SkyCoord(l=np.asarray(l) * u.deg, b=np.asarray(b) * u.deg,
                       distance=np.asarray(d) * u.kpc, frame="galactic")
    g = c.transform_to(params.frame())
    return np.column_stack([g.x.to_value(u.kpc), g.y.to_value(u.kpc), g.z.to_value(u.kpc)])
