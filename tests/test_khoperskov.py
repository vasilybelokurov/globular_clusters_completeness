import numpy as np

from gc_completeness import khoperskov as K
from gc_completeness.density import PowerLaw
from gc_completeness.khop_classes import relative_completeness, anchored


def test_target_counts_uniform_density():
    # uniform density nu=1 in a cylinder R<2, |z|<1 has volume 2*pi*R^2*1 = 8*pi
    grid = K.RZGrid(np.array([0, 1, 2.0]), np.array([0, 0.5, 1.0]))
    N = K.target_counts(PowerLaw(gamma=0.0, rc=0.0), grid)
    assert np.isclose(N.sum(), 8 * np.pi, rtol=1e-6)


def test_occupancy_fractions_and_outside():
    grid = K.RZGrid(np.array([0, 5, 10.0]), np.array([0, 5.0]))
    xyz = np.zeros((2, 4, 3))
    xyz[0, :, 0] = [1, 2, 6, 7]          # half the time in each R cell
    xyz[1, :, 0] = [1, 1, 20, 20]        # half the time outside the grid
    A = K.occupancy_rz(xyz, grid)
    assert np.allclose(A[:, 0], [0.5, 0.5])
    assert np.allclose(A[:, 1], [0.5, 0.0])


def test_nnls_exact_recovery_and_scaling():
    rng = np.random.default_rng(0)
    A = rng.uniform(size=(40, 8)); A /= A.sum(0)
    w = rng.uniform(1, 5, 8)
    r1 = K.nnls_subsets(A, A @ w, cell_sigma="none")
    assert np.allclose(A @ r1.w, A @ w, atol=1e-8)
    r3 = K.nnls_subsets(A, 3 * (A @ w), cell_sigma="none")   # homogeneous in the target amplitude
    assert np.allclose(r3.w, 3 * r1.w, rtol=1e-6)


def test_phase_hist():
    X = np.array([[1.0, 3.0], [3.0, 3.0]])
    h = K.phase_hist(X, np.array([2.0, 1.0]), np.array([0, 2, 4.0]))
    assert np.allclose(h, [1.0, 2.0])


def test_relative_and_anchored_completeness():
    edges = np.array([0, 6, 10, 20.0])
    obs = np.array([10.0, 20.0, 5.0])
    corr_unit = np.array([5.0, 20.0, 10.0])      # arbitrary amplitude
    rel = relative_completeness(obs, corr_unit)
    assert np.allclose(rel, [1.0, 0.5, 0.25])
    an = anchored(obs, corr_unit, W_unit_total=40.0, d_edges=edges, d_anchor=6)
    assert np.isclose(an["scale"], 2.0) and np.allclose(an["C"], [1.0, 0.5, 0.25])
    assert np.isclose(an["N_total"], 80.0)


def test_equal_count_edges():
    from gc_completeness.adaptive import equal_count_edges
    x = np.arange(1, 46, dtype=float)
    e = equal_count_edges(x, 15, lo=0.0, hi=100.0)
    assert len(e) == 4 and e[0] == 0 and e[-1] == 100
    assert np.all(np.histogram(x, e)[0] >= 15)


def test_equal_count_edges_with_ties():
    from gc_completeness.adaptive import equal_count_edges
    x = np.r_[np.full(20, 5.0), np.arange(20, 40, dtype=float)]   # 20 identical values
    e = equal_count_edges(x, 15)
    assert np.all(np.histogram(x, e)[0] >= 15)


def test_freeze_bins_classes_partition_sample():
    from gc_completeness.adaptive import freeze_bins
    rng = np.random.default_rng(0)
    M, d = rng.normal(-7, 1, 165), rng.lognormal(2.3, 0.7, 165)
    bins = freeze_bins(M, d, 3, 15, 150.0)
    n = [((M >= b["M_lo"]) & (M < b["M_hi"])).sum() for b in bins]
    assert sum(n) == 165 and min(n) >= 50
    for b in bins:
        m = (M >= b["M_lo"]) & (M < b["M_hi"])
        assert np.all(np.histogram(d[m], b["d_edges"])[0] >= 15)


def test_bias_reduced_ratio():
    from gc_completeness.adaptive import class_completeness
    # one orbit per bin, exact target: corr = weights; check C_rel and C_rel_br formulas
    A = np.eye(2); Nt = np.array([10.0, 10.0])
    d_t = np.array([[1.0] * 4, [20.0] * 4]); d_now = np.r_[np.full(4, 1.0), np.full(2, 20.0)]
    r = class_completeness(np.c_[A[:, [0]].repeat(4, 1), A[:, [1]].repeat(2, 1)],
                           np.r_[d_t[[0]].repeat(4, 0), d_t[[1]].repeat(2, 0)], d_now, Nt,
                           np.array([0, 10, 150.0]), {"frac": 1.0, "n_real": 1, "cell_sigma": "none"}, 0)
    # corr per bin = 10 each; obs = 4, 2 -> plain 0.5, br = (2/10)*10/(4+1) = 0.4
    assert np.allclose(r["C_rel"], [1.0, 0.5]) and np.allclose(r["C_rel_br"][1], 0.4)


def _toy_orbits(n, rng, T=400):
    # radial-ish orbits whose heliocentric distance sweeps a range; present phase drawn with p ~ C(d)
    lo, hi = rng.uniform(2, 10, n), rng.uniform(15, 60, n)
    ph = rng.uniform(0, 2 * np.pi, n)[:, None]
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)[None, :]
    return lo[:, None] + (hi - lo)[:, None] * 0.5 * (1 - np.cos(t + ph))


def test_phase_likelihood_recovers_C():
    from gc_completeness import phaselike as PL
    rng = np.random.default_rng(1)
    d_t = _toy_orbits(3000, rng)
    edges = np.array([0, 10, 25, 150.0])
    theta_true = np.log([0.5, 0.15])
    idx = PL.simulate_phases(d_t, edges, theta_true, rng)
    F = PL.time_fractions(d_t, edges)
    fr = PL.fit(F, PL.bin_index(d_t[np.arange(len(idx)), idx], edges))
    assert np.allclose(fr["theta"], theta_true, atol=0.1)
    lo, hi = PL.profile_interval(F, PL.bin_index(d_t[np.arange(len(idx)), idx], edges), fr, 0)
    assert lo < theta_true[0] + 0.1 and hi > theta_true[0] - 0.1


def test_pit_uniform_under_null():
    from gc_completeness import phaselike as PL
    from scipy.stats import kstest
    rng = np.random.default_rng(2)
    d_t = _toy_orbits(2000, rng)
    edges = np.array([0, 10, 25, 150.0]); th = np.log([0.5, 0.2])
    idx = PL.simulate_phases(d_t, edges, th, rng)
    S = d_t                                  # use distance itself as the statistic
    pit = PL.pit_values(S, S[np.arange(len(idx)), idx], PL.C_along(d_t, edges, th), rng)
    assert kstest(pit, "uniform").pvalue > 0.01


def test_phase_ll_gradient_matches_finite_difference():
    from scipy.optimize import check_grad
    from gc_completeness import phaselike as PL
    rng = np.random.default_rng(3)
    d_t = _toy_orbits(200, rng)
    edges = np.array([0, 10, 25, 40, 150.0])
    F = PL.time_fractions(d_t, edges)
    kn = PL.bin_index(d_t[:, 0], edges)
    th = rng.normal(0, 0.5, 3)
    err = check_grad(lambda t: PL._ll(t, F, kn)[0], lambda t: PL._ll(t, F, kn)[1], th)
    assert err < 1e-5 * max(1.0, np.linalg.norm(PL._ll(th, F, kn)[1]))


def test_profile_interval_at_bound():
    from gc_completeness import phaselike as PL
    # no cluster is ever observed in bin 1 although orbits visit it -> MLE at the lower bound
    d_t = np.tile(np.r_[np.full(50, 5.0), np.full(50, 20.0)], (40, 1))
    edges = np.array([0, 10, 150.0])
    F = PL.time_fractions(d_t, edges)
    kn = np.zeros(40, int)
    fr = PL.fit(F, kn)
    assert fr["at_bound"][0]
    lo, hi = PL.profile_interval(F, kn, fr, 0)
    assert lo == PL.THETA_BOUNDS[0] and hi < 0


def test_last_bin_open_ended_consistent():
    from gc_completeness.adaptive import class_completeness
    from gc_completeness import phaselike as PL
    # samples beyond the configured outer edge (150) must land in the last bin in both estimators
    d_t = np.array([[5.0, 5.0], [200.0, 200.0]])
    d_now = np.array([5.0, 200.0])
    A = np.eye(2)
    r = class_completeness(A, d_t, d_now, np.array([1.0, 1.0]), np.array([0, 10, 150.0]),
                           {"frac": 1.0, "n_real": 1, "cell_sigma": "none"}, 0)
    assert r["obs"][1] == 1 and r["corr_unit"][1] > 0
    assert PL.bin_index(np.array([200.0]), np.array([0, 10, 150.0]))[0] == 1


def test_empty_reference_bin_gives_nan_not_crash():
    from gc_completeness.adaptive import class_completeness
    d_t = np.array([[20.0, 20.0], [30.0, 30.0]])
    r = class_completeness(np.eye(2), d_t, np.array([20.0, 30.0]), np.array([1.0, 1.0]),
                           np.array([0, 10, 150.0]), {"frac": 1.0, "n_real": 1, "cell_sigma": "none"}, 0)
    assert np.isnan(r["C_rel"][1])


def test_pseudo_true_equals_truth_for_piecewise_constant_C():
    from gc_completeness import phaselike as PL
    rng = np.random.default_rng(4)
    d_t = _toy_orbits(500, rng)
    edges = np.array([0, 10, 25, 150.0])
    th = np.log([0.6, 0.25])
    Ct = PL.C_along(d_t, edges, th)
    assert np.allclose(PL.pseudo_true(d_t, Ct, edges), th, atol=1e-3)


def test_cv_recovers_C_of_V_and_corrected_total():
    from gc_completeness import cv as CV, phaselike as PL
    rng = np.random.default_rng(5)
    n = 4000
    d_t = _toy_orbits(n, rng)                       # heliocentric distance along orbits
    M = rng.normal(-7.4, 1.2, n)
    V_t = CV.apparent_mag(M, d_t)
    Ct = 1 / (1 + np.exp((V_t - 13.0) / 0.5))       # true C(V)
    det = rng.random(n) < Ct.mean(1)                # detected with prob q_i
    # present phase of detected ones ~ C along the orbit
    p = Ct[det] / Ct[det].sum(1, keepdims=True)
    idx = (np.cumsum(p, 1) < rng.random(det.sum())[:, None]).sum(1)
    V_now = V_t[det][np.arange(det.sum()), idx]
    e = CV.v_edges(V_now, 200)
    f = CV.fit_cv(V_t[det], V_now, e)
    q = CV.cluster_q(V_t[det], e, f["C"])
    # brightest bin is ~complete in truth, so the corrected total should recover n within ~10%
    assert abs((1 / q).sum() / n - 1) < 0.1
