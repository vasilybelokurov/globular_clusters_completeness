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


def analyse(M, d_t, d_now, cfg, n_boot, rng, A_V=0.0):
    V_t = CV.apparent_mag(M, d_t, A_V)
    V_now = CV.apparent_mag(M, d_now, A_V)
    e = CV.v_edges(V_now, cfg["n_min_v"])
    f = CV.fit_cv(V_t, V_now, e)
    q = CV.cluster_q(V_t, e, f["C"])
    llf, kf, curves = per_class_free_ll(M, d_t, d_now, cfg["n_classes"], cfg["n_min_d"], cfg["d_hi"])
    aic_cv, aic_free = 2 * f["n_par"] - 2 * f["ll"], 2 * kf - 2 * llf
    boot_idx, boot_w = [], []
    N = len(M)
    for _ in range(n_boot):   # cluster bootstrap: resample clusters, refit C(V) (fixed V edges), and
        idx = rng.integers(0, N, N)            # weight the RESAMPLED clusters with the refitted C
        fb = CV.fit_cv(V_t[idx], V_now[idx], e, profile=False)
        boot_idx.append(idx); boot_w.append(1.0 / CV.cluster_q(V_t[idx], e, fb["C"]))
    return dict(V_t=V_t, V_now=V_now, edges=e, fit=f, q=q, w=1.0 / q,
                boot_idx=np.array(boot_idx, int).reshape(n_boot, N), boot_w=np.array(boot_w).reshape(n_boot, N),
                aic_cv=aic_cv, aic_free=aic_free, ll_cv=f["ll"], ll_free=llf, k_cv=f["n_par"], k_free=kf, curves=curves)


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
    if "b_term" in scen:
        Cv = Cv * (1 - scen["b_term"]["depth"] * np.exp(-np.abs(b) / scen["b_term"]["b0"]))
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
    _, b, d = GalactocentricToGalactic(frame)(lib.xyz)
    p = np.where(mixed, rng.integers(0, T, N), 0); ii = np.arange(N)
    det = rng.random(N) < true_C(M, d[ii, p], b[ii, p], scen, mk["completeness"])
    res = analyse(M[det], d[det], d[ii, p][det], cfg, mk["n_bootstrap"], rng)
    # truth: absolute C in each V bin (pseudo-true of the binned model, phase-mixed members) and true counts
    Vt_all = CV.apparent_mag(M, d)
    Ct = true_C(M[:, None], d, b, scen, mk["completeness"])
    th_star = PL.pseudo_true(Vt_all[mixed], Ct[mixed], res["edges"])
    C_star = np.r_[1.0, np.exp(th_star)]
    # pseudo-true values are relative to bin 0; true absolute C in bin 0 (time-averaged over true samples)
    k0 = PL.bin_index(Vt_all[mixed], res["edges"]) == 0
    C0_abs = (Ct[mixed] * k0).sum() / max(k0.sum(), 1)
    tc = np.histogram(M, MV_EDGES)[0]
    cc = np.histogram(M[det], MV_EDGES, weights=res["w"])[0]
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
               N_obs_faint=int((M[det] > -6.5).sum()), dAIC_cv_minus_free=res["aic_cv"] - res["aic_free"], C0_abs=C0_abs)
    mv = [dict(scenario=sname, real=r, mv_lo=MV_EDGES[k], true=tc[k], obs=oc[k], corr=cc[k],
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
    else:
        A_V = np.zeros(len(c))
    rng = np.random.default_rng(cfg["seed"] + 3)
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
    json.dump(summ, open(out / f"real_cv_summary{tag}.json", "w"), indent=2)
    pd.DataFrame({"name": c["name"], "M_V": M, "feh": feh, "d": d_now, "A_V": A_V, "V_now": res["V_now"], "q": res["q"],
                  "w": res["w"]}).to_csv(out / f"real_cv_weights{tag}.csv", index=False)
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
    axes[0].set_xlabel("apparent V" + (" (no extinction)" if ext == "none" else " incl. A_V = 3.1 E(B-V)_present")); axes[0].set_ylabel("C(V) (brightest bin = 1)")
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
                 "complete; " + ("no extinction" if ext == "none" else "A_V = 3.1 E(B-V) at present position") + ")",
                 fontsize=9.5, x=0.01, ha="left")
    fig.savefig(plots / f"fig62_cv_corrected_MV_FeH{tag}.png", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cv.yaml")
    ap.add_argument("--mode", choices=["mock", "real"], required=True)
    ap.add_argument("--n-real", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extinction", choices=["none", "present_ebv"], default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.n_real:
        cfg["mock"]["n_realisations"] = a.n_real
    if a.n_boot is not None:
        cfg["mock"]["n_bootstrap"] = cfg["n_bootstrap"] = a.n_boot
    if a.extinction:
        cfg["extinction"] = a.extinction
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    plots = Path(cfg["plots_dir"]); plots.mkdir(parents=True, exist_ok=True)
    (run_mock(cfg, out, plots, a.workers) if a.mode == "mock" else run_real(cfg, out, plots))


if __name__ == "__main__":
    main()
