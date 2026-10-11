#!/usr/bin/env python
"""Completeness C(V) (apparent magnitude) from the per-cluster phase likelihood; absolute under
the anchor "brightest-V bin complete"; corrected M_V and [Fe/H] distributions.

Usage
-----
    python scripts/run_cv.py --mode mock [--n-real 100] [--workers 8]
    python scripts/run_cv.py --mode real

Outputs: outputs/cv/*.csv|json, plots/fig60-62_cv_*.png
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gc_completeness import phaselike as PL  # noqa: E402
from gc_completeness import cv as CV  # noqa: E402
from gc_completeness.adaptive import freeze_bins  # noqa: E402
from gc_completeness.catalog import load_catalog  # noqa: E402
from gc_completeness.coordinates import GalacticParams, GalactocentricToGalactic, icrs_to_galactocentric  # noqa: E402
from gc_completeness.orbits import integrate_orbits  # noqa: E402
from gc_completeness.plotting import plt, C, INK, MUTED  # noqa: E402
from gc_completeness.potential import build_potential, potential_label  # noqa: E402
from gc_completeness.mocks import sample_population  # noqa: E402
from gc_completeness import dust as DU  # noqa: E402

MV_EDGES = np.arange(-11, 0.01, 0.75)


def boot_hist(x, res, edges, mask_fn=None):
    """Bootstrap histograms of x with the resampled clusters and their refitted weights."""
    out = []
    for idx, w in zip(res["boot_idx"], res["boot_w"]):
        xx = x[idx]
        ok = np.isfinite(xx) if mask_fn is None else (np.isfinite(xx) & mask_fn(xx))
        out.append(np.histogram(xx[ok], edges, weights=w[ok])[0])
    return np.array(out)


def boot_sum(x, res, mask_fn=None):
    return np.array([w[(mask_fn(x[idx]) if mask_fn else np.ones(len(idx), bool))].sum()
                     for idx, w in zip(res["boot_idx"], res["boot_w"])])


def per_class_free_ll(M, d_t, d_now, n_classes, n_min_d, d_hi):
    """Sum of log-likelihoods of free per-class C(d) fits (frozen equal-count bins) and their #params."""
    bins = freeze_bins(M, d_now, n_classes, n_min_d, d_hi)
    ll, k = 0.0, 0
    curves = []
    for b in bins:
        m = (M >= b["M_lo"]) & (M < b["M_hi"])
        F = PL.time_fractions(d_t[m], b["d_edges"]); kn = PL.bin_index(d_now[m], b["d_edges"])
        fr = PL.fit(F, kn)
        ll += fr["ll"]; k += len(b["d_edges"]) - 2
        curves.append(dict(b, m=m, C=np.r_[1.0, np.exp(fr["theta"])]))
    return ll, k, curves


def analyse_sep(M, d_t, d_now, x_t, x_now, cfg, n_boot, rng):
    """C = C_V(V) * g(x); x is an obscuration proxy (larger = more obscured); 3 equal-count x bins,
    first x bin (least obscured) and brightest V bin anchored at 1."""
    from gc_completeness.adaptive import equal_count_edges
    V_t, V_now = CV.apparent_mag(M, d_t), CV.apparent_mag(M, d_now)
    eV = CV.v_edges(V_now, cfg["n_min_v"])
    ex = equal_count_edges(x_now, max(1, len(x_now) // cfg.get("n_x_bins", 3)), lo=-np.inf, hi=np.inf)
    f = CV.fit_cv_sep(V_t, x_t, V_now, x_now, eV, ex)
    w = 1.0 / CV.cluster_q_sep(f)
    N = len(M); boot_idx, boot_w = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        fb = CV.fit_cv_sep(V_t[idx], x_t[idx], V_now[idx], x_now[idx], eV, ex)
        boot_idx.append(idx); boot_w.append(1.0 / CV.cluster_q_sep(fb))
    return dict(fit=f, w=w, eV=eV, ex=ex, aic=2 * f["n_par"] - 2 * f["ll"],
                boot_idx=np.array(boot_idx, int).reshape(n_boot, N), boot_w=np.array(boot_w).reshape(n_boot, N))


def analyse_A(M, d_t, d_now, cfg, n_boot, rng, A_t):
    """As analyse(), with extinction varying along each orbit: A_t (N, T); present = column 0
    for real data (orbits start at the present position). d_now must equal d_t[:, 0] here."""
    return analyse(M, d_t, d_now, cfg, n_boot, rng, A_V=A_t, A_now=A_t[:, 0])


def analyse(M, d_t, d_now, cfg, n_boot, rng, A_V=0.0, A_now=None):
    V_t = CV.apparent_mag(M, d_t, A_V)
    V_now = CV.apparent_mag(M, d_now, A_V if A_now is None else A_now)
    e = CV.v_edges(V_now, cfg["n_min_v"])
    f = CV.fit_cv(V_t, V_now, e)
    q = CV.cluster_q(V_t, e, f["C"])
    llf, kf, curves = per_class_free_ll(M, d_t, d_now, cfg["n_classes"], cfg["n_min_d"], cfg["d_hi"])
    aic_cv, aic_free = 2 * f["n_par"] - 2 * f["ll"], 2 * kf - 2 * llf
    boot_idx, boot_w, boot_C = [], [], []
    N = len(M)
    for _ in range(n_boot):   # cluster bootstrap: resample clusters, refit C(V) (fixed V edges), and
        idx = rng.integers(0, N, N)            # weight the RESAMPLED clusters with the refitted C
        fb = CV.fit_cv(V_t[idx], V_now[idx], e, profile=False)
        boot_idx.append(idx); boot_w.append(1.0 / CV.cluster_q(V_t[idx], e, fb["C"])); boot_C.append(fb["C"])
    return dict(M=np.asarray(M, float), V_t=V_t, V_now=V_now, edges=e, fit=f, q=q, w=1.0 / q,
                boot_idx=np.array(boot_idx, int).reshape(n_boot, N), boot_w=np.array(boot_w).reshape(n_boot, N),
                boot_C=np.array(boot_C).reshape(n_boot, len(e) - 1),
                aic_cv=aic_cv, aic_free=aic_free, ll_cv=f["ll"], ll_free=llf, k_cv=f["n_par"], k_free=kf, curves=curves)


def vmax_weights(res, M_ref=-8.0):
    """Never-detectable-orbit correction (1/V_max style) with the BINNED C(V) of an analyse() result.

    Orbit population = clusters with M_V < M_ref, weighted 1/q_j; each observed cluster i gets weight
    1/f(M_i) (CV.population_detectability). Assumes orbits independent of M_V. Kept for comparison:
    the binned C is constant beyond the last V bin, which inflates f at faint M (see analyse_spline)."""
    M, e = res["M"], res["edges"]
    mu = res["V_t"] - M[:, None]                       # DM + A_V along each orbit
    pop = M < M_ref
    Cf = lambda C: (lambda V: np.asarray(C)[PL.bin_index(V, e)])
    w = 1.0 / CV.population_detectability(M, mu[pop], res["w"][pop], Cf(res["fit"]["C"]))
    bw = []
    for idx, Cb, wb in zip(res["boot_idx"], res["boot_C"], res["boot_w"]):
        pb = M[idx] < M_ref
        bw.append(1.0 / CV.population_detectability(M[idx], mu[idx][pb], wb[pb], Cf(Cb)))
    return dict(w=w, boot_idx=res["boot_idx"], boot_w=np.array(bw).reshape(res["boot_w"].shape), n_pop=int(pop.sum()))


def analyse_spline(M, V_t, V_now, cfg, n_boot, rng, M_ref=-8.0, groups=None):
    """Continuous C(V) (CV.fit_cv_spline) with two corrected-count estimators:
    w_ht = 1/q_i (own orbit) and w = 1/f(M_i) (1/V_max over the orbit population of clusters with
    M_V < M_ref; corrects for faint clusters on orbits that never enter the detectable volume).
    groups (optional, per cluster): 1/V_max uses as orbit population only the bright clusters of the
    same group (e.g. metallicity class), i.e. orbits independent of M_V within each group.
    Cluster bootstrap: resample, refit, re-select the population, recompute both weights."""
    M = np.asarray(M, float)
    nodes = CV.spline_nodes(V_now, cfg["n_min_v"])
    B_t, B_now = CV.spline_basis(V_t, nodes), CV.spline_basis(V_now, nodes)
    mu = V_t - M[:, None]

    def one(idx, p0=None):
        f = CV.fit_cv_spline(B_t[idx], B_now[idx], p0)
        q = np.exp(B_t[idx] @ f["p"]).mean(1)
        Cfun = lambda V: np.exp(CV.lnC_spline(V, nodes, f["p"]))
        g = np.zeros(len(idx), int) if groups is None else np.asarray(groups)[idx]
        fM = np.full(len(idx), np.nan)
        for gv in np.unique(g):
            ing = g == gv
            pop = ing & (M[idx] < M_ref)
            if pop.any():
                fM[ing] = CV.population_detectability(M[idx][ing], mu[idx][pop], 1.0 / q[pop], Cfun)
        return f, 1.0 / q, 1.0 / fM
    N = len(M)
    f, w_ht, w = one(np.arange(N))
    bi, bw, bh = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        _, h, v = one(idx, f["p"])
        bi.append(idx); bh.append(h); bw.append(v)
    sh = (n_boot, N)
    return dict(nodes=nodes, fit=f, w=w, w_ht=w_ht, aic=2 * f["n_par"] - 2 * f["ll"], n_pop=int((M < M_ref).sum()),
                boot_idx=np.array(bi, int).reshape(sh), boot_w=np.array(bw).reshape(sh), boot_w_ht=np.array(bh).reshape(sh))


def phase_pit(d_t, d_now, lnC_t):
    """u_i = C-weighted fraction of orbit time closer to the Sun than now; U(0,1) if the model is right."""
    Ct = np.exp(lnC_t - lnC_t.max(1, keepdims=True))
    return (Ct * (d_t < d_now[:, None])).sum(1) / Ct.sum(1)


def sky_region(l, b, window):
    """Region label per sample: 1 inside the inner window |l| < L, |b| < B (deg), else 0 (anchor region)."""
    L, B = window
    lw = np.abs(np.mod(np.asarray(l, float) + 180.0, 360.0) - 180.0)
    return ((lw < L) & (np.abs(b) < B)).astype(int)


def make_region_model(kind, V_now, reg_now, n_reg, cfg):
    """kind: 'sep' = separate spline per region (CV.RegionalC); 'contrast' = shared spline + constant
    ln-contrast per region; 'contrast_slope' = shared spline + contrast linear in V (CV.ContrastC)."""
    if kind == "sep":
        return CV.RegionalC(CV.region_nodes(V_now, reg_now, n_reg, cfg.get("n_min_v_region", cfg["n_min_v"])))
    nodes = CV.spline_nodes(V_now, cfg["n_min_v"])
    return CV.ContrastC(nodes, n_reg, slope=(kind == "contrast_slope"))


def analyse_region(M, V_t, V_now, reg_t, reg_now, cfg, n_boot, rng, M_ref=-8.0, kind="sep"):
    """Direction-dependent continuous C (see make_region_model); region 0 carries the anchor.
    Weights: w_ht = 1/q_i (own orbit, region-dependent C along it) and w = 1/f(M_i), 1/V_max over the
    orbit population M_V < M_ref, with f(M) = sum_j w_j <C(M + mu_j(t), k_j(t))>_t / sum_j w_j."""
    M = np.asarray(M, float)
    n_reg = int(max(reg_t.max(), reg_now.max())) + 1
    model = make_region_model(kind, V_now, reg_now, n_reg, cfg)
    B_t, B_now = model.basis(V_t, reg_t), model.basis(V_now, reg_now)
    mu = V_t - M[:, None]

    def one(idx, p0=None):
        f = CV.fit_cv_spline(B_t[idx], B_now[idx], p0, bounds=model.bounds())
        q = np.exp(B_t[idx] @ f["p"]).mean(1)
        pop = M[idx] < M_ref
        W = (1.0 / q[pop]) / np.sum(1.0 / q[pop])
        mp, rp = mu[idx][pop], reg_t[idx][pop]
        Mu, inv = np.unique(M[idx], return_inverse=True)
        fM = np.array([(W * np.exp(model.lnC(m + mp, rp, f["p"])).mean(1)).sum() for m in Mu])[inv]
        return f, 1.0 / q, 1.0 / fM
    N = len(M)
    f, w_ht, w = one(np.arange(N))
    bi, bw, bh, bp = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        fb, h, v = one(idx, f["p"])
        bi.append(idx); bh.append(h); bw.append(v); bp.append(fb["p"])
    sh = (n_boot, N)
    # parameters on an optimiser bound (a tail step of exactly 0, i.e. flat C, is not counted)
    at_bound = [bool(np.isclose(x, lo) or (hi != 0.0 and np.isclose(x, hi))) for x, (lo, hi) in zip(f["p"], model.bounds())]
    return dict(model=model, fit=f, w=w, w_ht=w_ht, q=1.0 / w_ht, aic=2 * f["n_par"] - 2 * f["ll"], at_bound=at_bound,
                boot_p=np.array(bp).reshape(n_boot, model.n_par),
                boot_idx=np.array(bi, int).reshape(sh), boot_w=np.array(bw).reshape(sh), boot_w_ht=np.array(bh).reshape(sh))


# ---------------------------------------------------------------------------
_W = {}


def _init(cfg):
    base = yaml.safe_load(open(cfg["catalog_config"]))
    _W.update(cfg=cfg, pot=build_potential(base["potential"]), frame=GalacticParams.from_dict(base.get("galactocentric")))


def true_C(M, d, b, scen, mc):
    DM = 5 * np.log10(np.maximum(d, 1e-4) * 100.0)
    if scen.get("not_V"):   # luminosity-dependent but NOT a function of V alone
        Cv = 1.0 / (1.0 + np.exp((M - (scen["not_V"]["V0"] - scen["not_V"]["slope"] * DM)) / mc["dM"]))
    else:
        Cv = 1.0 / (1.0 + np.exp((M - (mc["V_lim"] - DM)) / mc["dM"]))
    if scen.get("C_shape"):   # true C(V) with a tail that is not log-linear (tests the spline extrapolation)
        V = M + DM
        lg = lambda V0, w: 1.0 / (1.0 + np.exp((V - V0) / w))
        sh = scen["C_shape"]
        if sh == "two_survey":     # shallow all-sky survey + deep partial-area survey
            Cv = 0.8 * lg(8.0, 0.3) + 0.2 * lg(13.0, 0.3)
        elif sh == "power_tail":   # slow power-law decline, no cut-off
            Cv = np.where(V < 7.0, 1.0, (1.0 + np.maximum(V - 7.0, 0.0)) ** -2.0)
        elif sh == "real_spline":  # the C(V) fitted to the real data: scen["_spline"] = (nodes, ln C at nodes, tail slope)
            nd, lc, sl = scen["_spline"]
            Cv = np.minimum(np.exp(np.where(V > nd[-1], lc[-1] + sl * (V - nd[-1]), np.interp(V, nd, lc))), 1.0)
        else:
            raise ValueError(sh)
    if "b_term" in scen:
        Cv = Cv * (1 - scen["b_term"]["depth"] * np.exp(-np.abs(b) / scen["b_term"]["b0"]))
    if "window" in scen:   # direction-dependent C inside a sky window (needs l): shallower limit and/or factor
        wd = scen["window"]
        inside = sky_region(scen["_l"], b, (wd["L"], wd["B"])) == 1
        if "V_lim_in" in wd:
            Cin = 1.0 / (1.0 + np.exp((M - (wd["V_lim_in"] - DM)) / mc["dM"]))
            Cv = np.where(inside, Cin, Cv)
        Cv = np.where(inside, Cv * wd.get("factor", 1.0), Cv)
    return Cv


def mock_one(args):
    sname, r = args
    cfg, pot, frame = _W["cfg"], _W["pot"], _W["frame"]
    mk = cfg["mock"]; scen = mk["scenarios"][sname]
    seed = cfg["seed"] + 10007 * list(mk["scenarios"]).index(sname) + r
    rng = np.random.default_rng(seed)
    ic = sample_population(pot, mk["density"], mk["n_true"], mk["r_trunc"], seed)
    mixed = np.ones(len(ic), bool)
    if "clump" in scen:
        cl = scen["clump"]
        icc = np.column_stack([np.array(cl["xyz"]) + rng.normal(0, cl["sig_x"], (cl["n"], 3)),
                               np.array(cl["vxyz"]) + rng.normal(0, cl["sig_v"], (cl["n"], 3))])
        ic = np.vstack([ic, icc]); mixed = np.r_[mixed, np.zeros(cl["n"], bool)]
    M = rng.normal(*mk["gclf"], len(ic))
    lib = integrate_orbits(pot, ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"])
    N, T = lib.xyz.shape[:2]
    l, b, d = GalactocentricToGalactic(frame)(lib.xyz)
    p = np.where(mixed, rng.integers(0, T, N), 0); ii = np.arange(N)
    A = DU.disc_extinction(l, b, d, **scen["dust"]) if "dust" in scen else np.zeros_like(d)   # true A_V along orbits
    Meff = M[:, None] + A                                  # detection depends on V + A_V
    det = rng.random(N) < true_C(Meff[ii, p], d[ii, p], b[ii, p], scen, mk["completeness"])
    sep = {}
    if mk.get("proxy_tests", False):
        xs = {"absb": (-np.abs(b), -np.abs(b[ii, p])),                      # larger x = closer to the plane
              "sightline": (lambda S: (S, S[ii, p]))(DU.disc_extinction(l, b, np.full_like(d, 100.0), **mk.get(
                  "proxy_disc", {"a0": 1.0, "h_R": 3.0, "h_z": 0.1})))}       # 2D total column (SFD-like)
        for name, (xt, xn) in xs.items():
            sep[name] = analyse_sep(M[det], d[det], d[ii, p][det], xt[det], xn[det], cfg, 0, rng)
    if "dust" in scen:   # estimator given the true A_V along each orbit
        res = analyse(M[det], d[det], d[ii, p][det], cfg, mk["n_bootstrap"], rng, A_V=A[det], A_now=A[ii, p][det])
        res_nodust = analyse(M[det], d[det], d[ii, p][det], cfg, 0, rng)
    else:
        res = analyse(M[det], d[det], d[ii, p][det], cfg, mk["n_bootstrap"], rng)
        res_nodust = None
    # truth: absolute C in each V bin (pseudo-true of the binned model, phase-mixed members) and true counts
    Vt_all = CV.apparent_mag(M, d, A)
    Ct = true_C(Meff, d, b, scen, mk["completeness"])
    th_star = PL.pseudo_true(Vt_all[mixed], Ct[mixed], res["edges"])
    C_star = np.r_[1.0, np.exp(th_star)]
    # pseudo-true values are relative to bin 0; true absolute C in bin 0 (time-averaged over true samples)
    k0 = PL.bin_index(Vt_all[mixed], res["edges"]) == 0
    C0_abs = (Ct[mixed] * k0).sum() / max(k0.sum(), 1)
    vm = vmax_weights(res, cfg.get("vmax_M_ref", -8.0))           # never-detectable-orbit correction
    sp = analyse_spline(M[det], res["V_t"], res["V_now"], cfg, mk["n_bootstrap"], rng, cfg.get("vmax_M_ref", -8.0))
    spht = dict(sp, w=sp["w_ht"], boot_w=sp["boot_w_ht"])
    vt = boot_sum(M[det], sp); vf = boot_sum(M[det], sp, lambda x: x > -6.5)
    fM = M[det] > -6.5
    tc = np.histogram(M, MV_EDGES)[0]
    cc = np.histogram(M[det], MV_EDGES, weights=res["w"])[0]
    cv_ = np.histogram(M[det], MV_EDGES, weights=sp["w"])[0]
    cb = boot_hist(M[det], res, MV_EDGES)
    oc = np.histogram(M[det], MV_EDGES)[0]
    rows = []
    for k in range(len(res["edges"]) - 1):
        rows.append(dict(scenario=sname, real=r, vbin=k, C=res["fit"]["C"][k], lo=res["fit"]["lo"][k], hi=res["fit"]["hi"][k],
                         C_true_rel=C_star[k], C0_abs=C0_abs))
    bt = boot_sum(M[det], res)
    bf = boot_sum(M[det], res, lambda x: x > -6.5)
    tot = dict(scenario=sname, real=r, N_true=int(len(M)), N_obs=int(det.sum()), N_corr=float(res["w"].sum()),
               N_corr_p16=float(np.percentile(bt, 16)), N_corr_p84=float(np.percentile(bt, 84)),
               N_corr_faint_p16=float(np.percentile(bf, 16)), N_corr_faint_p84=float(np.percentile(bf, 84)),
               N_true_faint=int((M > -6.5).sum()), N_corr_faint=float(res["w"][M[det] > -6.5].sum()),
               N_obs_faint=int((M[det] > -6.5).sum()), dAIC_cv_minus_free=res["aic_cv"] - res["aic_free"], C0_abs=C0_abs,
               N_corr_vmaxbin=float(vm["w"].sum()), N_corr_faint_vmaxbin=float(vm["w"][fM].sum()),
               N_corr_splht=float(sp["w_ht"].sum()), N_corr_faint_splht=float(sp["w_ht"][fM].sum()),
               N_corr_vmax=float(sp["w"].sum()), N_corr_faint_vmax=float(sp["w"][fM].sum()),
               N_corr_vmax_p16=float(np.percentile(vt, 16)) if len(vt) else np.nan,
               N_corr_vmax_p84=float(np.percentile(vt, 84)) if len(vt) else np.nan,
               N_corr_faint_vmax_p16=float(np.percentile(vf, 16)) if len(vf) else np.nan,
               N_corr_faint_vmax_p84=float(np.percentile(vf, 84)) if len(vf) else np.nan,
               dAIC_spline_minus_bin=sp["aic"] - res["aic_cv"], n_pop_vmax=sp["n_pop"],
               N_corr_ignoring_dust=float(res_nodust["w"].sum()) if res_nodust is not None else np.nan,
               N_corr_faint_ignoring_dust=float(res_nodust["w"][M[det] > -6.5].sum()) if res_nodust is not None else np.nan,
               dAIC_dust_minus_nodust=(2 * res["k_cv"] - 2 * res["ll_cv"]) - (2 * res_nodust["k_cv"] - 2 * res_nodust["ll_cv"])
               if res_nodust is not None else np.nan,
               **{f"N_corr_sep_{k}": float(v["w"].sum()) for k, v in sep.items()},
               **{f"N_corr_faint_sep_{k}": float(v["w"][M[det] > -6.5].sum()) for k, v in sep.items()},
               **{f"dAIC_sep_{k}_minus_cv": v["aic"] - (2 * res_nodust["k_cv"] - 2 * res_nodust["ll_cv"]
                                                      if res_nodust is not None else res["aic_cv"]) for k, v in sep.items()})
    mv = [dict(scenario=sname, real=r, mv_lo=MV_EDGES[k], true=tc[k], obs=oc[k], corr=cc[k], corr_vmax=cv_[k],
               corr_p16=np.percentile(cb[:, k], 16), corr_p84=np.percentile(cb[:, k], 84)) for k in range(len(tc))]
    return rows, tot, mv


def run_mock(cfg, out, plots, workers):
    mk = cfg["mock"]
    tasks = [(s, r) for s in mk["scenarios"] for r in range(mk["n_realisations"])]
    R, Tt, MV = [], [], []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=(cfg,)) as ex:
        for k, (r_, t_, m_) in enumerate(ex.map(mock_one, tasks)):
            R += r_; Tt.append(t_); MV += m_
            if (k + 1) % 50 == 0:
                print(f"[mock] {k + 1}/{len(tasks)}", flush=True)
    R, Tt, MV = pd.DataFrame(R), pd.DataFrame(Tt), pd.DataFrame(MV)
    R.to_csv(out / "mock_cv_bins.csv", index=False); Tt.to_csv(out / "mock_cv_totals.csv", index=False)
    MV.to_csv(out / "mock_cv_mv.csv", index=False)
    R1 = R[R.vbin > 0].copy()
    R1["ratio"] = R1.C / R1.C_true_rel
    R1["cov"] = (R1.C_true_rel >= R1.lo) & (R1.C_true_rel <= R1.hi)
    s1 = R1.groupby("scenario").agg(n=("ratio", "size"), median_ratio=("ratio", "median"), p16=("ratio", lambda x: x.quantile(.16)),
                                    p84=("ratio", lambda x: x.quantile(.84)), coverage68=("cov", "mean")).reset_index()
    Tt["tot_ratio"] = Tt.N_corr / Tt.N_true
    Tt["tot_cov"] = (Tt.N_true >= Tt.N_corr_p16) & (Tt.N_true <= Tt.N_corr_p84)
    Tt["faint_ratio"] = Tt.N_corr_faint / Tt.N_true_faint
    Tt["faint_cov"] = (Tt.N_true_faint >= Tt.N_corr_faint_p16) & (Tt.N_true_faint <= Tt.N_corr_faint_p84)
    s2 = Tt.groupby("scenario").agg(N_true=("N_true", "mean"), N_obs=("N_obs", "mean"), tot_ratio=("tot_ratio", "median"),
                                    tot_p16=("tot_ratio", lambda x: x.quantile(.16)), tot_p84=("tot_ratio", lambda x: x.quantile(.84)),
                                    tot_coverage68=("tot_cov", "mean"), faint_obs_frac=("N_obs_faint", "sum"),
                                    faint_ratio=("faint_ratio", "median"), faint_coverage68=("faint_cov", "mean"),
                                    C0_abs=("C0_abs", "median"),
                                    frac_dAIC_favours_free=("dAIC_cv_minus_free", lambda x: (x > 2).mean())).reset_index()
    s2["faint_obs_frac"] = Tt.groupby("scenario").N_obs_faint.sum().values / Tt.groupby("scenario").N_true_faint.sum().values
    s2["tot_ratio_ignoring_dust"] = Tt.groupby("scenario").apply(lambda g: np.median(g.N_corr_ignoring_dust / g.N_true)).values
    s2["faint_ratio_ignoring_dust"] = Tt.groupby("scenario").apply(lambda g: np.median(g.N_corr_faint_ignoring_dust / g.N_true_faint)).values
    for k in ("absb", "sightline"):
        if f"N_corr_sep_{k}" in Tt:
            s2[f"tot_ratio_sep_{k}"] = Tt.groupby("scenario").apply(lambda g: np.median(g[f"N_corr_sep_{k}"] / g.N_true)).values
            s2[f"faint_ratio_sep_{k}"] = Tt.groupby("scenario").apply(lambda g: np.median(g[f"N_corr_faint_sep_{k}"] / g.N_true_faint)).values
            s2[f"frac_AIC_prefers_sep_{k}"] = Tt.groupby("scenario").apply(lambda g: np.mean(g[f"dAIC_sep_{k}_minus_cv"] < 0)).values
    s2["frac_AIC_prefers_true_dust"] = Tt.groupby("scenario").apply(lambda g: np.mean(g.dAIC_dust_minus_nodust < 0)
                                                                     if g.dAIC_dust_minus_nodust.notna().any() else np.nan).values
    for k in ("vmaxbin", "splht", "vmax"):
        s2[f"tot_ratio_{k}"] = Tt.groupby("scenario").apply(lambda g: np.median(g[f"N_corr_{k}"] / g.N_true)).values
        s2[f"faint_ratio_{k}"] = Tt.groupby("scenario").apply(lambda g: np.median(g[f"N_corr_faint_{k}"] / g.N_true_faint)).values
    s2["tot_coverage68_vmax"] = Tt.groupby("scenario").apply(
        lambda g: np.mean((g.N_true >= g.N_corr_vmax_p16) & (g.N_true <= g.N_corr_vmax_p84))).values
    s2["faint_coverage68_vmax"] = Tt.groupby("scenario").apply(
        lambda g: np.mean((g.N_true_faint >= g.N_corr_faint_vmax_p16) & (g.N_true_faint <= g.N_corr_faint_vmax_p84))).values
    s2["tot_p16_vmax"] = Tt.groupby("scenario").apply(lambda g: np.percentile(g.N_corr_vmax / g.N_true, 16)).values
    s2["tot_p84_vmax"] = Tt.groupby("scenario").apply(lambda g: np.percentile(g.N_corr_vmax / g.N_true, 84)).values
    s2["n_pop_vmax"] = Tt.groupby("scenario").n_pop_vmax.median().values
    s1.to_csv(out / "mock_cv_summary_C.csv", index=False); s2.to_csv(out / "mock_cv_summary_totals.csv", index=False)
    with pd.option_context("display.width", 220):
        print(s1.round(3).to_string(index=False)); print(s2.round(3).to_string(index=False))
    scen = list(mk["scenarios"])
    fig, axes = plt.subplots(1, len(scen) + 1, figsize=(4.4 * (len(scen) + 1), 3.8))
    for k, s in enumerate(scen):
        g = R1[R1.scenario == s]
        axes[0].plot(np.full(len(g), k) + np.random.default_rng(k).uniform(-.25, .25, len(g)), g.ratio, "o", ms=2, alpha=.3, color=C[k])
        axes[0].plot([k - .3, k + .3], [g.ratio.median()] * 2, color=INK, lw=2)
        axes[0].text(k, 0.12, f"cov {g['cov'].mean():.2f}", ha="center", fontsize=7)
    axes[0].set_yscale("log"); axes[0].set_ylim(0.1, 10); axes[0].axhline(1, color=MUTED)
    axes[0].set_xticks(range(len(scen)), scen, fontsize=7, rotation=15); axes[0].set_ylabel("recovered / true C(V)/C(brightest)")
    for k, s in enumerate(scen):
        ax = axes[k + 1]
        g = MV[MV.scenario == s].groupby("mv_lo")[["true", "obs", "corr"]].mean()
        e = np.r_[g.index.values, g.index.values[-1] + 0.75]
        ax.stairs(g.true.values, e, color=INK, lw=2, label="true (mean)")
        ax.stairs(g.obs.values, e, color=MUTED, lw=1.5, ls="--", label="observed")
        ax.stairs(g["corr"].values, e, color=C[1], lw=2, label="corrected")
        ax.invert_xaxis(); ax.set_xlabel(r"$M_V$"); ax.set_title(s, fontsize=8, loc="left")
        rt = s2[s2.scenario == s].iloc[0]
        ax.text(0.03, 0.9, f"total: corr/true={rt.tot_ratio:.2f}\nfaint (M_V>-6.5): {rt.faint_ratio:.2f}", transform=ax.transAxes, fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("Mock validation of C(V): completeness recovery and corrected M_V distribution", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(plots / "fig60_cv_mock.png"); plt.close(fig)


# ---------------------------------------------------------------------------
REGION_KINDS = ("single", "contrast", "contrast_slope", "sep", "single_coarse", "sep_fine")


def mock_region_one(args):
    """One realisation for the direction-dependent C tests: single spline vs regional models."""
    sname, r = args
    cfg, pot, frame = _W["cfg"], _W["pot"], _W["frame"]
    mk = cfg["mock"]; mr = cfg["mock_region"]
    scen = dict(mr["scenarios"][sname])
    seed = cfg["seed"] + 7919 * list(mr["scenarios"]).index(sname) + r + 500000
    rng = np.random.default_rng(seed)
    ic = sample_population(pot, mk["density"], mk["n_true"], mk["r_trunc"], seed)
    mixed = np.ones(len(ic), bool)
    if "clump" in scen:
        cl = scen["clump"]
        icc = np.column_stack([np.array(cl["xyz"]) + rng.normal(0, cl["sig_x"], (cl["n"], 3)),
                               np.array(cl["vxyz"]) + rng.normal(0, cl["sig_v"], (cl["n"], 3))])
        ic = np.vstack([ic, icc]); mixed = np.r_[mixed, np.zeros(cl["n"], bool)]
    M = rng.normal(*mk["gclf"], len(ic))
    lib = integrate_orbits(pot, ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"])
    N, T = lib.xyz.shape[:2]
    l, b, d = GalactocentricToGalactic(frame)(lib.xyz)
    p = np.where(mixed, rng.integers(0, T, N), 0); ii = np.arange(N)
    A = DU.disc_extinction(l, b, d, **scen["dust"]) if "dust" in scen else np.zeros_like(d)
    scen["_l"] = l[ii, p]
    det = rng.random(N) < true_C(M + A[ii, p], d[ii, p], b[ii, p], scen, mk["completeness"])
    reg_t = sky_region(l, b, tuple(mr["window"]))
    Md = M[det]
    V_t = CV.apparent_mag(Md, d[det], A[det])
    V_now = CV.apparent_mag(Md, d[ii, p][det], A[ii, p][det])
    rt, rn = reg_t[det], reg_t[ii, p][det]
    fM = Md > -6.5
    row = dict(scenario=sname, real=r, N_true=int(N), N_obs=int(det.sum()), N_true_faint=int((M > -6.5).sum()),
               N_obs_inner_now=int(rn.sum()))
    Mref = cfg.get("vmax_M_ref", -8.0)
    for kind in mr.get("kinds", REGION_KINDS):
        try:
            if kind in ("single", "single_coarse"):   # coarse: node count matched to one region of 'sep'
                cf = cfg if kind == "single" else dict(cfg, n_min_v=mr["n_min_coarse"])
                res = analyse_spline(Md, V_t, V_now, cf, mr["n_bootstrap"], rng, Mref)
                aic, nb = res["aic"], 0
            elif kind == "sep_fine":                  # separate splines with the single spline's bin size per region
                res = analyse_region(Md, V_t, V_now, rt, rn, dict(cfg, n_min_v_region=mr["n_min_fine"]),
                                     mr["n_bootstrap"], rng, Mref, kind="sep")
                aic, nb = res["aic"], int(np.sum(res["at_bound"]))
            else:
                res = analyse_region(Md, V_t, V_now, rt, rn, cfg, mr["n_bootstrap"], rng, Mref, kind=kind)
                aic, nb = res["aic"], int(np.sum(res["at_bound"]))
            for est, w, bw in (("ht", res["w_ht"], res["boot_w_ht"]), ("vmax", res["w"], res["boot_w"])):
                bt = np.array([x.sum() for x in bw]); bf = np.array([x[fM[ix]].sum() for x, ix in zip(bw, res["boot_idx"])])
                row.update({f"{kind}_{est}_tot": w.sum(), f"{kind}_{est}_faint": w[fM].sum(),
                            f"{kind}_{est}_tot_p16": np.percentile(bt, 16), f"{kind}_{est}_tot_p84": np.percentile(bt, 84),
                            f"{kind}_{est}_faint_p16": np.percentile(bf, 16), f"{kind}_{est}_faint_p84": np.percentile(bf, 84)})
            row[f"{kind}_aic"] = aic; row[f"{kind}_nbound"] = nb
        except Exception as exc:   # e.g. too few clusters in a region at present
            row[f"{kind}_error"] = repr(exc)[:120]
    return row


def run_mock_region(cfg, out, workers):
    mr = cfg["mock_region"]
    tasks = [(s, r) for s in mr["scenarios"] for r in range(mr["n_realisations"])]
    rows = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=(cfg,)) as ex:
        for k, row in enumerate(ex.map(mock_region_one, tasks)):
            rows.append(row)
            if (k + 1) % 50 == 0:
                print(f"[mock_region] {k + 1}/{len(tasks)}", flush=True)
    Tt = pd.DataFrame(rows)
    tag = mr.get("tag", "")
    Tt.to_csv(out / f"mock_region_totals{tag}.csv", index=False)
    summ = []
    for sname, g in Tt.groupby("scenario", sort=False):
        d = dict(scenario=sname, N_obs=g.N_obs.mean(), N_inner_now=g.N_obs_inner_now.mean())
        for kind in [k for k in REGION_KINDS if f"{k}_ht_tot" in g]:
            for est in ("ht", "vmax"):
                c = f"{kind}_{est}"
                if f"{c}_tot" not in g:
                    continue
                ok = g[f"{c}_tot"].notna()
                d[f"{c}_tot"] = np.median(g[f"{c}_tot"][ok] / g.N_true[ok])
                d[f"{c}_faint"] = np.median(g[f"{c}_faint"][ok] / g.N_true_faint[ok])
                d[f"{c}_cov"] = np.mean((g.N_true[ok] >= g[f"{c}_tot_p16"][ok]) & (g.N_true[ok] <= g[f"{c}_tot_p84"][ok]))
            if kind != "single" and f"{kind}_aic" in g:
                dA = g[f"{kind}_aic"] - g["single_aic"]
                d[f"{kind}_AICpref"] = np.mean(dA[dA.notna()] < 0)
                d[f"{kind}_bound"] = np.mean(g[f"{kind}_nbound"] > 0)
            if f"{kind}_error" in g:
                d[f"{kind}_nerr"] = int(g[f"{kind}_error"].notna().sum())
        summ.append(d)
    S = pd.DataFrame(summ)
    S.to_csv(out / f"mock_region_summary{tag}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(S.round(2).to_string(index=False))


# ---------------------------------------------------------------------------
def run_real(cfg, out, plots):
    _init(cfg)
    base = yaml.safe_load(open(cfg["catalog_config"]))
    df, _ = load_catalog(base["catalog"])
    c = df[(df["status"] == "confirmed") & df["has_6d"]].reset_index(drop=True)
    ic = icrs_to_galactocentric(c["ra"], c["dec"], c["distance"], c["pmra"], c["pmdec"], c["vlos"], _W["frame"])
    lib = integrate_orbits(_W["pot"], ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"],
                           cache_dir=cfg["cache_dir"], cache_tag=potential_label(base["potential"]))
    _, _, d_t = GalactocentricToGalactic(_W["frame"])(lib.xyz)
    M = c["M_V"].values.astype(float); feh = c["feh"].values.astype(float)
    d_now = c["distance"].values.astype(float)
    ext = cfg.get("extinction", "none")
    if ext == "present_ebv":   # A_V = R_V E(B-V) at the PRESENT position, held fixed along the orbit
        ebv = c["ebv"].values.astype(float)
        if not np.all(np.isfinite(ebv)):
            raise ValueError("E(B-V) missing for some clusters")
        A_V = cfg.get("R_V", 3.1) * ebv
    elif ext in ("map_low", "map_high"):   # A_V from 3D/2D dust maps at every orbit sample
        l_t, b_t, _ = GalactocentricToGalactic(_W["frame"])(lib.xyz)
        dz = DU.orbit_extinction(l_t, b_t, d_t, cfg["cache_dir"],
                                 tag=f"{potential_label(base['potential'])}|{lib.xyz.shape}|{float(lib.xyz.sum()):.6e}")
        A_V = dz["A_low"] if ext == "map_low" else dz["A_high"]   # (N, T); column 0 = present position
    else:
        A_V = np.zeros(len(c))
    rng = np.random.default_rng(cfg["seed"] + 3)
    if np.ndim(A_V) == 2:
        res = analyse_A(M, d_t, d_now, cfg, cfg["n_bootstrap"], rng, A_V)
    else:
        res = analyse(M, d_t, d_now, cfg, cfg["n_bootstrap"], rng, A_V=A_V)
    e, f = res["edges"], res["fit"]
    tab = pd.DataFrame(dict(V_lo=e[:-1], V_hi=e[1:], N_obs=np.bincount(f["kn"], minlength=len(e) - 1),
                            C=f["C"], C_lo=f["lo"], C_hi=f["hi"], at_bound=f["at_bound"]))
    tag = "" if ext == "none" else f"_{ext}"
    tab.to_csv(out / f"real_cv{tag}.csv", index=False)
    bt, bf = boot_sum(M, res), boot_sum(M, res, lambda x: x > -6.5)
    summ = {"extinction": ext, "N_obs": int(len(M)), "N_corr": float(res["w"].sum()),
            "N_corr_p16": float(np.percentile(bt, 16)), "N_corr_p84": float(np.percentile(bt, 84)),
            "faint_obs": int((M > -6.5).sum()), "faint_corr": float(res["w"][M > -6.5].sum()),
            "faint_corr_p16": float(np.percentile(bf, 16)), "faint_corr_p84": float(np.percentile(bf, 84)),
            "lnL_CV": res["ll_cv"], "k_CV": res["k_cv"], "lnL_free_classC(d)": res["ll_free"], "k_free": res["k_free"],
            "AIC_CV": res["aic_cv"], "AIC_free": res["aic_free"], "anchor": "brightest-V bin complete (C=1)"}
    # obscuration proxies: C = C_V(V) * g(x), x = -|b| or 2D SFD E(B-V) along the orbit (no extinction in V)
    if ext == "none" and cfg.get("proxy_tests_real", True):
        l_t, b_t, _ = GalactocentricToGalactic(_W["frame"])(lib.xyz)
        sfd_t = DU.query_maps(l_t.ravel(), b_t.ravel(), d_t.ravel(), maps=("sfd",))["sfd"].reshape(d_t.shape) / DU.AV_PER_EBV_SFD
        proxies = {"absb": -np.abs(b_t), "ebv_sfd": sfd_t}
        summ["proxies"] = {}
        for name, xt in proxies.items():
            r = analyse_sep(M, d_t, d_now, xt, xt[:, 0], cfg, cfg["n_bootstrap"], np.random.default_rng(cfg["seed"] + 9))
            bt = np.array([w.sum() for w in r["boot_w"]])
            bf = np.array([w[M[idx] > -6.5].sum() for idx, w in zip(r["boot_idx"], r["boot_w"])])
            summ["proxies"][name] = {"AIC": r["aic"], "C_V": r["fit"]["C_V"].tolist(), "g": r["fit"]["g"].tolist(),
                                     "x_edges": r["ex"].tolist(), "N_corr": float(r["w"].sum()),
                                     "N_corr_p16": float(np.percentile(bt, 16)), "N_corr_p84": float(np.percentile(bt, 84)),
                                     "faint_corr": float(r["w"][M > -6.5].sum()),
                                     "faint_corr_p16": float(np.percentile(bf, 16)), "faint_corr_p84": float(np.percentile(bf, 84))}
            print(f"[real] proxy {name}: AIC {r['aic']:.1f}  g={np.round(r['fit']['g'], 2)}  N_corr {r['w'].sum():.0f} "
                  f"[{np.percentile(bt, 16):.0f}-{np.percentile(bt, 84):.0f}]  faint {r['w'][M > -6.5].sum():.0f}")
    # continuous C(V) with extrapolated tail; 1/q (own orbit) and 1/V_max (never-detectable-orbit correction)
    sp = analyse_spline(M, res["V_t"], res["V_now"], cfg, cfg["n_bootstrap"], np.random.default_rng(cfg["seed"] + 11),
                        cfg.get("vmax_M_ref", -8.0))
    spht = dict(sp, w=sp["w_ht"], boot_w=sp["boot_w_ht"])
    fehc = np.where(np.isfinite(feh), (feh >= -1).astype(int), -1)        # 1/V_max per metallicity class
    spf = analyse_spline(M, res["V_t"], res["V_now"], cfg, cfg["n_bootstrap"], np.random.default_rng(cfg["seed"] + 13),
                         cfg.get("vmax_M_ref", -8.0), groups=fehc)
    for cn, cv_ in (("feh<-1", 0), ("feh>=-1", 1)):
        ok = fehc == cv_
        b_ = np.array([np.nansum(w[fehc[idx] == cv_]) for idx, w in zip(spf["boot_idx"], spf["boot_w"])])
        b2 = np.array([w[fehc[idx] == cv_].sum() for idx, w in zip(sp["boot_idx"], sp["boot_w"])])
        b3 = np.array([w[fehc[idx] == cv_].sum() for idx, w in zip(sp["boot_idx"], sp["boot_w_ht"])])
        summ.setdefault("spline", {})
        summ["spline"][f"vmax_split_{cn}"] = [float(spf["w"][ok].sum()), float(np.percentile(b_, 16)), float(np.percentile(b_, 84))]
        summ["spline"][f"vmax_{cn}"] = [float(sp["w"][ok].sum()), float(np.percentile(b2, 16)), float(np.percentile(b2, 84))]
        summ["spline"][f"ht_{cn}"] = [float(sp["w_ht"][ok].sum()), float(np.percentile(b3, 16)), float(np.percentile(b3, 84))]
        summ["spline"][f"n_pop_{cn}"] = int((ok & (M < cfg.get("vmax_M_ref", -8.0))).sum())
        print(f"[real] {cn}: obs {ok.sum()}  1/q {summ['spline'][f'ht_{cn}'][0]:.0f}  1/V_max(all pop) {summ['spline'][f'vmax_{cn}'][0]:.0f}"
              f"  1/V_max(own class pop) {summ['spline'][f'vmax_split_{cn}'][0]:.0f} {np.round(summ['spline'][f'vmax_split_{cn}'][1:], 0)}")
    classes = {"all": lambda x: np.ones(len(x), bool), "MV<-8": lambda x: x < -8, "-8<=MV<-6.5": lambda x: (x >= -8) & (x < -6.5),
               "MV>=-6.5": lambda x: x >= -6.5}
    summ["spline"] = summ.get("spline", {}) | {"nodes_V": sp["nodes"].tolist(),
                      "C_nodes": np.exp(CV.lnC_spline(sp["nodes"], sp["nodes"], sp["fit"]["p"])).tolist(),
                      "tail_dlnC_dV": float((CV.lnC_spline(sp["nodes"][-1] + 1, sp["nodes"], sp["fit"]["p"])
                                             - CV.lnC_spline(sp["nodes"][-1], sp["nodes"], sp["fit"]["p"]))),
                      "lnL": sp["fit"]["ll"], "k": sp["fit"]["n_par"], "AIC": sp["aic"], "fit_success": sp["fit"]["success"],
                      "vmax_M_ref": cfg.get("vmax_M_ref", -8.0), "n_pop": sp["n_pop"]}
    for lab_, r_, key in (("own-orbit 1/q", spht, "ht"), ("1/V_max", sp, "vmax")):
        for cn, fn in classes.items():
            b_ = boot_sum(M, r_, fn)
            summ["spline"][f"{key}_{cn}"] = [float(r_["w"][fn(M)].sum()), float(np.percentile(b_, 16)), float(np.percentile(b_, 84))]
        print(f"[real] spline C(V), {lab_}: " + ", ".join(f"{cn} {v[0]:.0f} [{v[1]:.0f}-{v[2]:.0f}]"
                                                        for cn, v in ((cn, summ['spline'][f'{key}_{cn}']) for cn in classes)))
    print(f"[real] spline AIC {sp['aic']:.1f} (binned {res['aic_cv']:.1f}); C at nodes {np.round(summ['spline']['C_nodes'], 3)}")
    # direction-dependent C: region 1 = inner window (fixed before fitting), region 0 = rest (anchor)
    from scipy.stats import kstest
    l_r, b_r, _ = GalactocentricToGalactic(_W["frame"])(lib.xyz)
    reg_t = sky_region(l_r, b_r, tuple(cfg.get("region_window", [30.0, 15.0])))
    reg_now = reg_t[:, 0]
    fr = reg_t.mean(1)
    summ["region"] = {"window_deg": cfg.get("region_window", [30.0, 15.0]), "n_now_inner": int(reg_now.sum()),
                      "n_orbits_always_inner": int((fr == 1).sum()), "n_orbits_never_inner": int((fr == 0).sum()),
                      "n_orbits_crossing_0.1_0.9": int(((fr > 0.1) & (fr < 0.9)).sum())}
    lnC_single = CV.lnC_spline(res["V_t"], sp["nodes"], sp["fit"]["p"])
    u = phase_pit(d_t, d_now, lnC_single)
    summ["region"]["single"] = {"AIC": sp["aic"], "PIT_KS_p": {f"region{k}": float(kstest(u[reg_now == k], "uniform").pvalue) for k in (0, 1)},
                                "PIT_median": {f"region{k}": float(np.median(u[reg_now == k])) for k in (0, 1)}}
    reg_w, reg_boot = {}, {}
    for kind in ("contrast", "contrast_slope", "sep"):
        rr = analyse_region(M, res["V_t"], res["V_now"], reg_t, reg_now, cfg, cfg["n_bootstrap"],
                            np.random.default_rng(cfg["seed"] + 17), cfg.get("vmax_M_ref", -8.0), kind=kind)
        mdl, p_ = rr["model"], rr["fit"]["p"]
        Vg = np.arange(4.0, 17.01, 0.5)
        lc = {k: mdl.lnC(Vg, np.full(len(Vg), k), p_) for k in (0, 1)}
        bl = np.array([mdl.lnC(Vg, np.ones(len(Vg), int), pb) - mdl.lnC(Vg, np.zeros(len(Vg), int), pb) for pb in rr["boot_p"]])
        u = phase_pit(d_t, d_now, mdl.lnC(res["V_t"], reg_t, p_))
        e_ = {"AIC": rr["aic"], "dAIC_vs_single": rr["aic"] - sp["aic"], "k": mdl.n_par, "lnL": rr["fit"]["ll"],
              "fit_success": rr["fit"]["success"], "n_at_bound": int(np.sum(rr["at_bound"])), "p": p_.tolist(),
              "V_grid": Vg.tolist(), "C_outer": np.exp(lc[0]).tolist(), "C_inner": np.exp(lc[1]).tolist(),
              "ln_inner_minus_outer_p16": np.percentile(bl, 16, 0).tolist(), "ln_inner_minus_outer_p84": np.percentile(bl, 84, 0).tolist(),
              "max_C": float(np.exp(max(lc[0].max(), lc[1].max()))),
              "PIT_KS_p": {f"region{k}": float(kstest(u[reg_now == k], "uniform").pvalue) for k in (0, 1)},
              "PIT_median": {f"region{k}": float(np.median(u[reg_now == k])) for k in (0, 1)}}
        rht = dict(rr, w=rr["w_ht"], boot_w=rr["boot_w_ht"])
        for lab_, r_, key in (("1/q", rht, "ht"), ("1/V_max", rr, "vmax")):
            for cn, fn_ in classes.items():
                b_ = boot_sum(M, r_, fn_)
                e_[f"{key}_{cn}"] = [float(r_["w"][fn_(M)].sum()), float(np.percentile(b_, 16)), float(np.percentile(b_, 84))]
            for cn, cv_ in (("feh<-1", 0), ("feh>=-1", 1)):
                b_ = boot_sum(feh, r_, (lambda x, cv_=cv_: np.isfinite(x) & ((x >= -1) == bool(cv_))))
                okf = np.isfinite(feh) & ((feh >= -1) == bool(cv_))
                e_[f"{key}_{cn}"] = [float(r_["w"][okf].sum()), float(np.percentile(b_, 16)), float(np.percentile(b_, 84))]
            for k in (0, 1):
                e_[f"{key}_now_region{k}"] = float(r_["w"][reg_now == k].sum())
            print(f"[real] region {kind} {lab_}: " + ", ".join(f"{cn} {e_[f'{key}_{cn}'][0]:.0f} [{e_[f'{key}_{cn}'][1]:.0f}-{e_[f'{key}_{cn}'][2]:.0f}]"
                                                             for cn in classes))
        print(f"[real] region {kind}: AIC {rr['aic']:.1f} (single spline {sp['aic']:.1f}); bound hits {e_['n_at_bound']}; "
              f"PIT KS p {e_['PIT_KS_p']}; single PIT KS p {summ['region']['single']['PIT_KS_p']}")
        summ["region"][kind] = e_
        reg_w[f"w_reg_{kind}"], reg_w[f"w_reg_{kind}_vmax"] = rr["w_ht"], rr["w"]
        reg_boot[f"{kind}_idx"], reg_boot[f"{kind}_w"], reg_boot[f"{kind}_w_vmax"] = rr["boot_idx"], rr["boot_w_ht"], rr["boot_w"]
    np.savez_compressed(out / f"real_cv_region_boot{tag}.npz", **reg_boot)
    json.dump(summ, open(out / f"real_cv_summary{tag}.json", "w"), indent=2)
    pd.DataFrame({"name": c["name"], "M_V": M, "feh": feh, "d": d_now, "region_now": reg_now, **reg_w,
                  "A_V_now": A_V[:, 0] if np.ndim(A_V) == 2 else A_V, "V_now": res["V_now"], "q": res["q"],
                  "w": res["w"], "w_spline": sp["w_ht"], "w_vmax": sp["w"], "w_vmax_fehsplit": spf["w"]}).to_csv(out / f"real_cv_weights{tag}.csv", index=False)
    np.savez_compressed(out / f"real_cv_boot{tag}.npz", boot_idx=res["boot_idx"], boot_w=res["boot_w"],
                        boot_idx_spline=sp["boot_idx"], boot_w_spline=sp["boot_w_ht"], boot_w_vmax=sp["boot_w"],
                        boot_idx_fehsplit=spf["boot_idx"], boot_w_vmax_fehsplit=spf["boot_w"])
    with pd.option_context("display.width", 200):
        print(tab.round(3).to_string(index=False)); print(json.dumps(summ, indent=1))
    # fig61: C(V) + implied per-class C(d) vs free fits
    fig, axes = plt.subplots(1, 1 + len(res["curves"]), figsize=(4.6 * (1 + len(res["curves"])), 3.8))
    vc = np.where(np.isfinite(tab.V_lo) & np.isfinite(tab.V_hi), 0.5 * (tab.V_lo + tab.V_hi),
                  np.where(np.isfinite(tab.V_hi), tab.V_hi - 0.5, tab.V_lo + 0.5))
    axes[0].errorbar(vc, tab.C, yerr=[tab.C - tab.C_lo, tab.C_hi - tab.C], fmt="s", color=INK, capsize=0)
    for _, r in tab.iterrows():
        axes[0].text(vc[_], 0.012, f"{r.N_obs}", ha="center", fontsize=7, color=MUTED)
    axes[0].set_yscale("log"); axes[0].set_ylim(0.01, 3); axes[0].axhline(1, color=MUTED)
    axes[0].set_xlabel("apparent V" + {"none": " (no extinction)", "present_ebv": " incl. A_V = 3.1 E(B-V)_present",
                                       "map_low": " incl. map A_V (low bracket)",
                                       "map_high": " incl. map A_V (high bracket)"}[ext]); axes[0].set_ylabel("C(V) (brightest bin = 1)")
    axes[0].set_title(f"C(V); ΔAIC(C(V) − free per-class C(d)) = {res['aic_cv'] - res['aic_free']:+.1f}", fontsize=8, loc="left")
    for j, cu in enumerate(res["curves"]):
        ax = axes[j + 1]
        de = cu["d_edges"]; m = cu["m"]
        dc = np.sqrt(np.maximum(de[:-1], 0.5) * np.minimum(de[1:], 150))
        imp = CV.implied_class_curve(res["V_t"][m], d_t[m], e, f["C"], de)
        ax.plot(dc, cu["C"], "s", color=INK, label="free per-class fit")
        ax.plot(dc * 1.08, imp, "o", color=C[1], label="implied by C(V)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.05, 5); ax.axhline(1, color=MUTED)
        ax.set_title(f"{cu['M_lo']:.2f} ≤ M_V < {cu['M_hi']:.2f}", fontsize=8, loc="left"); ax.set_xlabel("d [kpc]")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(plots / f"fig61_cv_real_C{tag}.png"); plt.close(fig)
    # fig62: corrected M_V and [Fe/H]
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1], hspace=0.06, wspace=0.25)
    for col, (x, edges, lab) in enumerate([(M, MV_EDGES, r"$M_V$ [mag]"), (feh, np.arange(-2.6, 0.21, 0.2), "[Fe/H]")]):
        ok = np.isfinite(x)
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        obs = np.histogram(x[ok], edges)[0]
        hb = np.histogram(x[ok], edges, weights=res["w"][ok])[0]
        hbb = boot_hist(x, res, edges)
        lo, hi = np.percentile(hbb, 16, 0), np.percentile(hbb, 84, 0)
        ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step="post", color=C[1], alpha=0.25, lw=0)
        ax.stairs(hb, edges, color=C[1], lw=2, label=f"corrected (N={hb.sum():.0f}; band: 68% bootstrap)")
        ax.errorbar(0.5 * (edges[1:] + edges[:-1]), obs, yerr=np.sqrt(obs), fmt="none", ecolor=INK, lw=1)
        ax.stairs(obs, edges, color=INK, lw=1.8, label=f"observed (N={obs.sum()})")
        ax.set_ylim(bottom=0); ax.tick_params(labelbottom=False); ax.legend(fontsize=8, loc="upper left"); ax.set_ylabel("GCs per bin")
        okb = obs >= 2
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.stairs(np.where(okb, hb / obs, np.nan), edges, color=C[1], lw=2)
            axr.fill_between(edges, np.r_[np.where(okb, lo / obs, np.nan), np.nan], np.r_[np.where(okb, hi / obs, np.nan), np.nan],
                             step="post", color=C[1], alpha=0.25, lw=0)
        axr.axhline(1, color=MUTED); axr.set_ylim(0, None); axr.set_xlabel(lab); axr.set_ylabel("corrected / observed")
        for a in (ax, axr):
            a.set_xlim(edges[0], edges[-1])
            if col == 0:
                a.invert_xaxis()
    fig.suptitle("Observed vs completeness-corrected GCs with C(V) (phase likelihood, all clusters; anchor: brightest-V bin "
                 "complete; " + {"none": "no extinction", "present_ebv": "A_V = 3.1 E(B-V) at present position",
                                 "map_low": "map A_V along orbits, low bracket",
                                 "map_high": "map A_V along orbits, high bracket"}[ext] + ")",
                 fontsize=9.5, x=0.01, ha="left")
    fig.savefig(plots / f"fig62_cv_corrected_MV_FeH{tag}.png", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cv.yaml")
    ap.add_argument("--mode", choices=["mock", "mock_region", "real"], required=True)
    ap.add_argument("--n-real", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extinction", choices=["none", "present_ebv", "map_low", "map_high"], default=None)
    ap.add_argument("--region-control", action="store_true", help="mock_region: node-count controls only")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.n_real:
        cfg["mock"]["n_realisations"] = cfg["mock_region"]["n_realisations"] = a.n_real
    if a.n_boot is not None:
        cfg["mock"]["n_bootstrap"] = cfg["mock_region"]["n_bootstrap"] = cfg["n_bootstrap"] = a.n_boot
    if a.region_control:
        cfg["mock_region"].update(kinds=["single", "single_coarse", "sep_fine"], tag="_control")
    if a.extinction:
        cfg["extinction"] = a.extinction
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    plots = Path(cfg["plots_dir"]); plots.mkdir(parents=True, exist_ok=True)
    if a.mode == "mock":
        run_mock(cfg, out, plots, a.workers)
    elif a.mode == "mock_region":
        run_mock_region(cfg, out, a.workers)
    else:
        run_real(cfg, out, plots)


if __name__ == "__main__":
    main()
