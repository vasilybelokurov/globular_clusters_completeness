import numpy as np

from gc_completeness.density import BrokenPowerLaw, PowerLaw, shell_counts


def test_power_law_shell_integral_analytic():
    m = PowerLaw(gamma=2.5, rc=0.0, nu0=3.0)
    edges = np.array([1.0, 2.0, 10.0])
    N = shell_counts(m, edges)
    # int 4 pi r^2 * 3 r^-2.5 dr = 12 pi * 2 [sqrt(b) - sqrt(a)]
    exact = 24 * np.pi * (np.sqrt(edges[1:]) - np.sqrt(edges[:-1]))
    assert np.allclose(N, exact, rtol=1e-4)


def test_core_bin_from_zero():
    m = PowerLaw(gamma=0.0, rc=0.0, nu0=1.0)  # uniform density
    N = shell_counts(m, np.array([0.0, 2.0]))
    assert np.isclose(N[0], 4 / 3 * np.pi * 8, rtol=1e-5)


def test_broken_power_law_slopes():
    m = BrokenPowerLaw(alpha=1.0, beta=4.0, rb=10.0, rc=0.001)
    def slope(r):
        return (np.log(m.nu(r * 1.01)) - np.log(m.nu(r / 1.01))) / (2 * np.log(1.01))
    assert abs(slope(0.1) + 1.0) < 0.05
    assert abs(slope(1e4) + 4.0) < 0.01
