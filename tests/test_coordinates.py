import numpy as np
import astropy.units as u
import astropy.coordinates as coord

from gc_completeness.coordinates import (GalacticParams, GalactocentricToGalactic,
                                         icrs_to_galactocentric, galactic_lbd_to_galactocentric)


def test_sun_position_and_velocity():
    p = GalacticParams()
    # an object at the Sun's position with zero heliocentric velocity has the Sun's phase space
    xv = icrs_to_galactocentric([0.0], [0.0], [1e-9], [0.0], [0.0], [0.0], p)[0]
    assert abs(np.sqrt(xv[0] ** 2 + xv[2] ** 2) - p.R0) < 1e-6
    assert abs(xv[2] - p.z_sun) < 1e-6
    assert np.allclose(xv[3:], p.v_sun, atol=1e-6)


def test_fast_inverse_matches_astropy():
    p = GalacticParams()
    rng = np.random.default_rng(1)
    xyz = rng.normal(scale=20, size=(500, 3))
    l, b, d = GalactocentricToGalactic(p)(xyz)
    g = coord.SkyCoord(x=xyz[:, 0] * u.kpc, y=xyz[:, 1] * u.kpc, z=xyz[:, 2] * u.kpc, frame=p.frame(),
                       representation_type="cartesian").transform_to(coord.Galactic())
    assert np.allclose(d, g.distance.kpc, atol=1e-9)
    assert np.allclose(b, g.b.deg, atol=1e-8)
    dl = (l - g.l.deg + 180) % 360 - 180
    assert np.max(np.abs(dl * np.cos(np.radians(b)))) < 1e-8


def test_roundtrip_lbd():
    p = GalacticParams()
    l, b, d = np.array([10.0, 200.0]), np.array([-30.0, 60.0]), np.array([5.0, 40.0])
    xyz = galactic_lbd_to_galactocentric(l, b, d, p)
    l2, b2, d2 = GalactocentricToGalactic(p)(xyz)
    assert np.allclose([l2, b2, d2], [l, b, d], atol=1e-8)
