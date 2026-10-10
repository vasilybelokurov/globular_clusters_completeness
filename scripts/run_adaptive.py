#!/usr/bin/env python
"""Frozen equal-count (M_V, d) bins; completeness from (a) Khoperskov orbit superposition with fixed
targets and (b) the target-free per-cluster phase likelihood; per-cluster symmetry tests calibrated
by a refitting null simulation.

Usage
-----
    python scripts/run_adaptive.py --mode mock [--n-real 100] [--workers 8]
    python scripts/run_adaptive.py --mode real

Outputs: outputs/adaptive/*.csv|json, plots/fig50-54_adaptive_*.png
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
from gc_completeness import khoperskov as K  # noqa: E402
from gc_completeness import phaselike as PL  # noqa: E402
from gc_completeness.adaptive import freeze_bins, class_completeness, bootstrap_fixed, support_flag  # noqa: E402
from gc_completeness.catalog import load_catalog  # noqa: E402
from gc_completeness.coordinates import GalacticParams, GalactocentricToGalactic, icrs_to_galactocentric  # noqa: E402
from gc_completeness.density import make_model  # noqa: E402
from gc_completeness.orbits import integrate_orbits  # noqa: E402
from gc_completeness.plotting import plt, C, INK, MUTED  # noqa: E402
from gc_completeness.potential import build_potential, potential_label  # noqa: E402
from gc_completeness.mocks import sample_population  # noqa: E402


def shapes_for(targets, grid):
    return {t: K.target_counts(make_model(v["model"]), grid, q=v.get("q", 1.0)) for t, v in targets.items()}


# ---------------------------------------------------------------------------
# core analysis on one sample with frozen bins
# ---------------------------------------------------------------------------
def analyse(S, bins, shapes, cfg, seed, boot_targets, n_boot):
    """S: dict xyz_t (N,T,3), d_t (N,T), A (cells,N), M (N,), d_now (N,), now_idx (N,).
    Returns per-bin rows and the per-class phase-likelihood fits (for symmetry tests)."""
    rows, groups = [], []
    for j, b in enumerate(bins):
        m = (S["M"] >= b["M_lo"]) & (S["M"] < b["M_hi"])
        de = b["d_edges"]
        nb = len(de) - 1
        F = PL.time_fractions(S["d_t"][m], de)
        kn = PL.bin_index(S["d_now"][m], de)
        ph = PL.fit(F, kn)
        ph_ci = [PL.profile_interval(F, kn, ph, k) for k in range(nb - 1)]
        groups.append(dict(xyz_t=S["xyz_t"][m], d_t=S["d_t"][m], now_idx=S["now_idx"][m], d_edges=de,
                           theta=ph["theta"], F=F))
        kh = {}
        for t, Nt in shapes.items():
            r = class_completeness(S["A"][:, m], S["d_t"][m], S["d_now"][m], Nt, de, cfg["nnls"], seed)
            if t in boot_targets and n_boot > 0:
                bt = bootstrap_fixed(S["A"][:, m], S["d_t"][m], S["d_now"][m], Nt, de, cfg["nnls"], n_boot, seed + 11)
                r["boot"] = bt
            kh[t] = r
        for k in range(nb):
            row = dict(cls=j, M_lo=b["M_lo"], M_hi=b["M_hi"], bin=k, d_lo=de[k], d_hi=de[k + 1], N_class=int(m.sum()),
                       N_obs=int((kn == k).sum()),
                       phase_C=1.0 if k == 0 else float(np.exp(ph["theta"][k - 1])),
                       phase_lo=1.0 if k == 0 else float(np.exp(ph_ci[k - 1][0])),
                       phase_hi=1.0 if k == 0 else float(np.exp(ph_ci[k - 1][1])),
                       phase_at_bound=False if k == 0 else bool(ph["at_bound"][k - 1]),
                       phase_success=bool(ph.get("success", True)))
            for t, r in kh.items():
                row[f"{t}_C"] = r["C_rel"][k]
                row[f"{t}_Cbr"] = r["C_rel_br"][k]
                row[f"{t}_neff"] = r["neff"][k]
                row[f"{t}_resid_w"] = r["resid_w"]
                row[f"{t}_cond"] = r["cond"]
                row[f"{t}_rank"] = r["rank"]
                row[f"{t}_n_active"] = r["n_active"]
                if "boot" in r:
                    bt = r["boot"]
                    for key, arr in (("C", bt["C_rel"]), ("Cbr", bt["C_rel_br"])):
                        v = arr[:, k][np.isfinite(arr[:, k])]
                        row[f"{t}_{key}_p16"] = float(np.percentile(v, 16)) if len(v) else np.nan
                        row[f"{t}_{key}_p84"] = float(np.percentile(v, 84)) if len(v) else np.nan
                    cf = bt["corr_frac"][:, k]
                    cv = float(np.std(cf) / np.mean(cf)) if np.mean(cf) > 0 else np.inf
                    row[f"{t}_corr_cv"] = cv
                    row[f"{t}_resolved"] = bool(support_flag(np.array([r["neff"][k]]), np.array([cv]),
                                                             np.array([row["N_obs"]]))[0])
            rows.append(row)
    return rows, groups


def symmetry_test(groups, n_null, rng):
    """Observed pooled KS statistics vs a null where present phases are redrawn from the fitted C
    and C is REFITTED for every simulation (accounts for estimating C on the same data)."""
    obs, pits = PL.symmetry_ks(groups, rng)
    null = {k: [] for k in obs}
    for _ in range(n_null):
        sim = []
        for g in groups:
            idx = PL.simulate_phases(g["d_t"], g["d_edges"], g["theta"], rng)
            d_now = g["d_t"][np.arange(len(idx)), idx]
            th = PL.fit(g["F"], PL.bin_index(d_now, g["d_edges"]))["theta"]
            sim.append(dict(g, now_idx=idx, theta=th))
        s, _ = PL.symmetry_ks(sim, rng)
        for k in s:
            null[k].append(s[k])
    pv = {k: float((1 + (np.array(null[k]) >= obs[k]).sum()) / (n_null + 1)) for k in obs}
    return obs, pv, pits


# ---------------------------------------------------------------------------
# mocks
# ---------------------------------------------------------------------------
_W = {}


def _init_worker(cfg):
    base = yaml.safe_load(open(cfg["catalog_config"]))
    _W["pot"] = build_potential(base["potential"])
    _W["frame"] = GalacticParams.from_dict(base.get("galactocentric"))
    _W["grid"] = K.RZGrid(np.array(cfg["grid"]["R_edges"], float), np.array(cfg["grid"]["z_edges"], float))
    _W["shapes"] = shapes_for(cfg["targets"], _W["grid"])
    _W["cfg"] = cfg


def true_C(M, d, b, mc, scen):
    DM = 5 * np.log10(np.maximum(d, 1e-4) * 100.0)
    Cv = 1.0 / (1.0 + np.exp((M - (mc["V_lim"] - DM)) / mc["dM"]))
    if "b_term" in scen:
        Cv = Cv * (1 - scen["b_term"]["depth"] * np.exp(-np.abs(b) / scen["b_term"]["b0"]))
    return Cv


def mock_realisation(args):
    sname, r = args
    cfg, pot, frame, grid, shapes = _W["cfg"], _W["pot"], _W["frame"], _W["grid"], _W["shapes"]
    mk = cfg["mock"]; scen = mk["scenarios"][sname]
    seed = cfg["seed"] + 1000 * list(mk["scenarios"]).index(sname) + r
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
    p = np.where(mixed, rng.integers(0, T, N), 0)
    ii = np.arange(N)
    det = rng.random(N) < true_C(M, d[ii, p], b[ii, p], mk["completeness"], scen)
    A = K.occupancy_rz(lib.xyz, grid)
    S = dict(xyz_t=lib.xyz[det], d_t=d[det], A=A[:, det], M=M[det], d_now=d[ii, p][det], now_idx=p[det])
    bins = freeze_bins(S["M"], S["d_now"], cfg["n_classes"], cfg["n_min_d"], cfg["d_hi"])
    rows, groups = analyse(S, bins, shapes, cfg, seed, boot_targets=["T2_broken_pl"], n_boot=mk["n_bootstrap"])
    Ct = true_C(M[:, None], d, b, mk["completeness"], scen)
    th_star = {}
    for j, bb in enumerate(bins):   # estimator-specific truth for the phase likelihood (pseudo-true value)
        mm = (M >= bb["M_lo"]) & (M < bb["M_hi"]) & mixed
        th_star[j] = PL.pseudo_true(d[mm], Ct[mm], bb["d_edges"])
    for row in rows:
        mm = (M >= row["M_lo"]) & (M < row["M_hi"])
        dd = d[mm]
        e = bins[row["cls"]]["d_edges"].copy(); e[-1] = np.inf
        inb = (dd >= e[row["bin"]]) & (dd < e[row["bin"] + 1])
        in0 = (dd >= e[0]) & (dd < e[1])
        c_k = (Ct[mm] * inb).sum() / max(inb.sum(), 1)
        c_0 = (Ct[mm] * in0).sum() / max(in0.sum(), 1)
        row.update(scenario=sname, real=r, C_true_rel=c_k / c_0 if c_0 > 0 else np.nan,
                   C_true_phase=1.0 if row["bin"] == 0 else float(np.exp(th_star[row["cls"]][row["bin"] - 1])))
    ks, pv, _ = symmetry_test(groups, mk["n_null"], rng)
    sym = dict(scenario=sname, real=r, **{f"KS_{k}": v for k, v in ks.items()}, **{f"p_{k}": v for k, v in pv.items()})
    return rows, sym


def run_mock(cfg, out, plots, workers):
    mk = cfg["mock"]
    tasks = [(s, r) for s in mk["scenarios"] for r in range(mk["n_realisations"])]
    rows, syms = [], []
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_init_worker,
                                 initargs=(cfg,)) as ex:
            for k, (rr, ss) in enumerate(ex.map(mock_realisation, tasks)):
                rows += rr; syms.append(ss)
                if (k + 1) % 25 == 0:
                    print(f"[mock] {k + 1}/{len(tasks)}", flush=True)
    else:
        _init_worker(cfg)
        for t in tasks:
            rr, ss = mock_realisation(t); rows += rr; syms.append(ss)
    df = pd.DataFrame(rows); sym = pd.DataFrame(syms)
    df.to_csv(out / "mock_bins.csv", index=False); sym.to_csv(out / "mock_symmetry.csv", index=False)
    summarise_mock(cfg, df, sym, out, plots)


def summarise_mock(cfg, df, sym, out, plots):
    d1 = df[df.bin > 0].copy()
    est = {"phase": ("phase_C", "phase_lo", "phase_hi", "C_true_phase"),
           "khop_T2_plain": ("T2_broken_pl_C", "T2_broken_pl_C_p16", "T2_broken_pl_C_p84", "C_true_rel"),
           "khop_T2_br": ("T2_broken_pl_Cbr", "T2_broken_pl_Cbr_p16", "T2_broken_pl_Cbr_p84", "C_true_rel")}
    rows = []
    for s, g in d1.groupby("scenario"):
        for name, (c, lo, hi, tru) in est.items():
            ratio = g[c] / g[tru]
            cov = ((g[tru] >= g[lo]) & (g[tru] <= g[hi])).mean()
            rows.append(dict(scenario=s, estimator=name, n=len(g), median_ratio=ratio.median(),
                             p16=ratio.quantile(0.16), p84=ratio.quantile(0.84),
                             mean_log_ratio=np.log(ratio[ratio > 0]).mean(), coverage68=cov,
                             median_C_true_rel=g[tru].median()))
        res = g[g["T2_broken_pl_resolved"].astype(bool)] if "T2_broken_pl_resolved" in g else g
        r2 = res.T2_broken_pl_Cbr / res.C_true_rel
        rows.append(dict(scenario=s, estimator="khop_T2_br_resolved_only", n=len(res), median_ratio=r2.median(),
                         p16=r2.quantile(0.16), p84=r2.quantile(0.84), mean_log_ratio=np.log(r2[r2 > 0]).mean(),
                         coverage68=((res.C_true_rel >= res.T2_broken_pl_Cbr_p16) & (res.C_true_rel <= res.T2_broken_pl_Cbr_p84)).mean(),
                         median_C_true_rel=res.C_true_rel.median()))
    sm = pd.DataFrame(rows); sm.to_csv(out / "mock_summary_estimators.csv", index=False)
    ss = sym.groupby("scenario")[[c for c in sym.columns if c.startswith("p_")]].apply(lambda x: (x < 0.05).mean()).reset_index()
    ss.to_csv(out / "mock_summary_symmetry.csv", index=False)
    tr = df[df.bin == 0].groupby(["scenario", "real", "cls"])[[f"{t}_resid_w" for t in cfg["targets"]]].first().reset_index()
    tr.to_csv(out / "mock_target_residuals.csv", index=False)
    with pd.option_context("display.width", 200):
        print(sm.round(3).to_string(index=False)); print(ss.round(3).to_string(index=False))
        print(tr.groupby("scenario")[[f"{t}_resid_w" for t in cfg["targets"]]].describe(percentiles=[.16, .5, .84]).round(3).T.to_string())
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    scen = list(cfg["mock"]["scenarios"])
    for k, name in enumerate(["phase", "khop_T2_plain", "khop_T2_br"]):
        for i, s in enumerate(scen):
            g = d1[d1.scenario == s]
            c, tru = est[name][0], est[name][3]
            v = g[c] / g[tru]
            x = i + (k - 1) * 0.25
            axes[0].plot(x + np.random.default_rng(k + 3 * i).uniform(-0.08, 0.08, len(v)), v, "o", ms=2, alpha=0.3, color=C[k],
                         label=name if i == 0 else None)
            axes[0].plot([x - 0.1, x + 0.1], [v.median()] * 2, color=INK, lw=2)
            cv = sm[(sm.scenario == s) & (sm.estimator == name)].coverage68.iloc[0]
            axes[0].text(x, 0.08, f"{cv:.2f}", ha="center", fontsize=7, color=C[k])
    axes[0].set_yscale("log"); axes[0].set_ylim(0.05, 20); axes[0].axhline(1, color=MUTED)
    axes[0].set_xticks(range(len(scen)), scen, fontsize=8); axes[0].set_ylabel("recovered / true C(d)/C(nearest)")
    axes[0].legend(fontsize=7); axes[0].set_title("bias & scatter (numbers: 68% coverage)", fontsize=9, loc="left")
    for k, st in enumerate(["p_z", "p_y", "p_absphi"]):
        axes[1].plot(range(len(scen)), [ss[ss.scenario == s][st].iloc[0] for s in scen], "o-", color=C[k], label=st[2:])
    axes[1].axhline(0.05, color=MUTED, ls=":"); axes[1].set_ylim(0, 1); axes[1].set_xticks(range(len(scen)), scen, fontsize=8)
    axes[1].set_ylabel("fraction of mocks with p<0.05"); axes[1].legend(fontsize=7)
    axes[1].set_title("per-cluster symmetry tests (null refits C)", fontsize=9, loc="left")
    g = tr[tr.scenario == scen[0]]
    for k, t in enumerate(cfg["targets"]):
        axes[2].hist(g[f"{t}_resid_w"], bins=20, histtype="step", lw=2, color=C[k], label=t)
    axes[2].set_xlabel("weighted NNLS residual (mock truth = T2)"); axes[2].legend(fontsize=7)
    axes[2].set_title("target-fit residual distributions", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(plots / "fig50_adaptive_mock.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------
def load_real(cfg, pot, frame, grid):
    base = yaml.safe_load(open(cfg["catalog_config"]))
    df, _ = load_catalog(base["catalog"])
    c = df[(df["status"] == "confirmed") & df["has_6d"]].reset_index(drop=True)
    from astropy.io import fits
    import os
    raw = fits.open(os.path.expanduser(base["catalog"]["path"]))[1].data
    lab = dict(zip([n.strip() for n in raw["NAME"]], [p.strip() for p in raw["PROG_MA"]]))
    c["prog"] = c["name"].map(lab).fillna("")
    ic = icrs_to_galactocentric(c["ra"], c["dec"], c["distance"], c["pmra"], c["pmdec"], c["vlos"], frame)
    lib = integrate_orbits(pot, ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"],
                           cache_dir=cfg["cache_dir"], cache_tag=potential_label(base["potential"]))
    _, _, d_t = GalactocentricToGalactic(frame)(lib.xyz)
    S = dict(xyz_t=lib.xyz, d_t=d_t, A=K.occupancy_rz(lib.xyz, grid), M=c["M_V"].values.astype(float),
             d_now=c["distance"].values.astype(float), now_idx=np.zeros(len(c), int))  # t=0 sample = present
    return c, S


def run_real(cfg, out, plots):
    _init_worker(cfg)
    grid, shapes = _W["grid"], _W["shapes"]
    c, S = load_real(cfg, _W["pot"], _W["frame"], grid)
    checks = {"N": int(len(c)), "Sgr_included": int((c.prog == "Sag").sum()),
              "frac_obs_d_gt_150": float((S["d_now"] > cfg["d_hi"]).mean()),
              "frac_orbit_samples_d_gt_150": float((S["d_t"] > cfg["d_hi"]).mean()),
              "frac_orbit_samples_outside_grid": float(max(0.0, 1 - S["A"].sum() / S["A"].shape[1]))}
    print("[real] checks:", checks)
    rng = np.random.default_rng(cfg["seed"] + 5)
    bins = freeze_bins(S["M"], S["d_now"], cfg["n_classes"], cfg["n_min_d"], cfg["d_hi"])
    rows, groups = analyse(S, bins, shapes, cfg, cfg["seed"], boot_targets=list(shapes), n_boot=cfg["n_bootstrap"])
    tab = pd.DataFrame(rows); tab.to_csv(out / "real_bins.csv", index=False)
    ks, pv, pits = symmetry_test(groups, cfg["n_null_real"], rng)
    sym = {"KS": ks, "p_null_refit": pv, "n_null": cfg["n_null_real"]}
    # populations
    prows, psym = [], {}
    for pname, pc in cfg["populations"].items():
        m = c.prog.isin(pc["labels"]).values
        sub = {k: (v[:, m] if k == "A" else v[m]) for k, v in S.items()}
        pb = freeze_bins(sub["M"], sub["d_now"], cfg["population_n_classes"], cfg["n_min_d"], cfg["d_hi"])
        r_, g_ = analyse(sub, pb, shapes_for(pc["targets"], grid), cfg, cfg["seed"], list(pc["targets"]), cfg["n_bootstrap"])
        for x in r_:
            x["population"] = pname
        prows += r_
        kp, pp, _ = symmetry_test(g_, cfg["n_null_real"], rng)
        psym[pname] = {"N": int(m.sum()), "KS": kp, "p_null_refit": pp}
    pdf = pd.DataFrame(prows); pdf.to_csv(out / "real_populations_bins.csv", index=False)
    json.dump({"checks": checks, "symmetry_all": sym, "symmetry_populations": psym}, open(out / "real_summary.json", "w"), indent=2)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        cols = ["cls", "M_lo", "M_hi", "d_lo", "d_hi", "N_obs", "phase_C", "phase_lo", "phase_hi", "phase_at_bound"]
        for t in shapes:
            cols += [f"{t}_Cbr", f"{t}_Cbr_p16", f"{t}_Cbr_p84", f"{t}_resolved", f"{t}_resid_w"]
        print(tab[cols].round(2).to_string(index=False))
        print(pdf[["population", "d_lo", "d_hi", "N_obs", "phase_C", "phase_lo", "phase_hi"] +
                  [x for x in pdf.columns if x.endswith("_Cbr") or x.endswith("_resid_w")]].round(2).to_string(index=False))
    print("[real] symmetry:", json.dumps(sym), "\n[real] populations symmetry:", json.dumps(psym))
    plot_real(cfg, tab, pdf, pits, sym, plots, out)


def plot_real(cfg, tab, pdf, pits, sym, plots, out):
    def panel(ax, g, targets, title):
        x0 = np.sqrt(np.maximum(g.d_lo, 0.5) * np.minimum(g.d_hi, 150))
        ax.errorbar(x0, g.phase_C, yerr=[g.phase_C - g.phase_lo, g.phase_hi - g.phase_C], fmt="s", color=INK, ms=5,
                    label="phase likelihood (target-free), 68% profile", capsize=0, lw=1.5)
        for k, t in enumerate(targets):
            x = x0 * (1 + 0.07 * (k + 1))
            res = g[f"{t}_resolved"].astype(bool).values
            ax.errorbar(x, g[f"{t}_Cbr"], yerr=[np.clip(g[f"{t}_Cbr"] - g[f"{t}_Cbr_p16"], 0, None),
                                                np.clip(g[f"{t}_Cbr_p84"] - g[f"{t}_Cbr"], 0, None)],
                        fmt="none", ecolor=C[k], lw=1, capsize=0)
            ax.plot(x[res], g[f"{t}_Cbr"].values[res], "o", color=C[k], ms=5, label=f"Khoperskov {t} (bias-reduced)")
            ax.plot(x[~res], g[f"{t}_Cbr"].values[~res], "o", mfc="none", mec=C[k], ms=5)
        for _, r in g.iterrows():
            ax.hlines(r.phase_C, max(r.d_lo, 0.5), min(r.d_hi, 150), color=MUTED, lw=0.8)
            ax.text(np.sqrt(max(r.d_lo, .5) * min(r.d_hi, 150)), 0.06, f"{r.N_obs}", ha="center", fontsize=7, color=MUTED)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.05, 10); ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel("heliocentric d [kpc] (numbers: N_obs; open: unresolved)")

    ncls = tab.cls.nunique()
    fig, axes = plt.subplots(1, ncls, figsize=(5.2 * ncls, 4), sharey=True, squeeze=False)
    for j in range(ncls):
        g = tab[tab.cls == j]
        panel(axes[0, j], g, [t for t in cfg["targets"] if t != "T1_power_law"],
              f"{g.M_lo.iloc[0]:.2f} ≤ M_V < {g.M_hi.iloc[0]:.2f}")
    axes[0, 0].set_ylabel("C(d) / C(nearest bin)"); axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Real data: completeness in frozen equal-count bins (T1 omitted: orbit library cannot fit it)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(plots / "fig51_adaptive_real_C.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, k, lab in zip(axes, ["z", "y", "absphi"], ["z", "y (Galactic-longitude side)", "|azimuth from Sun direction|"]):
        ax.hist(pits[k], bins=10, range=(0, 1), color=C[0], alpha=0.7)
        ax.axhline(len(pits[k]) / 10, color=INK, lw=1)
        ax.set_title(f"PIT of {lab}: KS={sym['KS'][k]:.3f}, p={sym['p_null_refit'][k]:.3f}", fontsize=9, loc="left")
        ax.set_xlabel("PIT value (uniform if model holds)")
    fig.tight_layout(); fig.savefig(plots / "fig52_adaptive_real_symmetry.png"); plt.close(fig)
    pops = list(cfg["populations"])
    fig, axes = plt.subplots(1, len(pops), figsize=(5.6 * len(pops), 4), sharey=True, squeeze=False)
    for j, pn in enumerate(pops):
        g = pdf[pdf.population == pn]
        panel(axes[0, j], g, list(cfg["populations"][pn]["targets"]), f"{pn} (N={g.N_class.iloc[0]})")
    axes[0, 0].set_ylabel("C(d) / C(nearest bin)"); axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(plots / "fig53_adaptive_real_populations.png"); plt.close(fig)
    # target residuals vs mock calibration (mock truth T2)
    mtr = out / "mock_target_residuals.csv"
    if mtr.exists():
        mt = pd.read_csv(mtr); mt = mt[mt.scenario == list(cfg["mock"]["scenarios"])[0]]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for k, t in enumerate(cfg["targets"]):
            ax.hist(mt[f"{t}_resid_w"], bins=25, histtype="step", lw=1.5, color=C[k], label=f"mock (truth T2), fit {t}")
            for v in tab[tab.bin == 0][f"{t}_resid_w"]:
                ax.axvline(v, color=C[k], lw=2, ls="--")
        ax.set_xlabel("weighted NNLS residual"); ax.legend(fontsize=7)
        ax.set_title("Target ranking: real classes (dashed) vs mock residual distributions", fontsize=9, loc="left")
        fig.tight_layout(); fig.savefig(plots / "fig54_adaptive_target_residuals.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/adaptive.yaml")
    ap.add_argument("--mode", choices=["mock", "real"], required=True)
    ap.add_argument("--n-real", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=None, help="override bootstrap replicates (mock and real)")
    ap.add_argument("--n-null", type=int, default=None, help="override null simulations (mock and real)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.n_real:
        cfg["mock"]["n_realisations"] = a.n_real
    if a.n_boot is not None:
        cfg["mock"]["n_bootstrap"] = cfg["n_bootstrap"] = a.n_boot
    if a.n_null is not None:
        cfg["mock"]["n_null"] = cfg["n_null_real"] = a.n_null
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    plots = Path(cfg["plots_dir"]); plots.mkdir(parents=True, exist_ok=True)
    if a.mode == "mock":
        run_mock(cfg, out, plots, a.workers)
    else:
        run_real(cfg, out, plots)


if __name__ == "__main__":
    main()
