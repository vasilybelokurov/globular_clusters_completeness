#!/usr/bin/env python
"""Khoperskov orbit superposition per luminosity class -> completeness C(M_V, d).

Usage
-----
    python scripts/run_khop_classes.py --mode mock   # validation (true C known only to the generator)
    python scripts/run_khop_classes.py --mode real

Outputs: outputs/khop_classes/*.csv|json, plots/fig4x_khopcls_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gc_completeness import khoperskov as K  # noqa: E402
from gc_completeness.khop_classes import class_curves, relative_completeness, anchored  # noqa: E402
from gc_completeness.catalog import load_catalog  # noqa: E402
from gc_completeness.coordinates import GalacticParams, GalactocentricToGalactic, icrs_to_galactocentric  # noqa: E402
from gc_completeness.density import make_model  # noqa: E402
from gc_completeness.orbits import integrate_orbits  # noqa: E402
from gc_completeness.plotting import plt, C, INK, MUTED  # noqa: E402
from gc_completeness.potential import build_potential, potential_label  # noqa: E402
from gc_completeness.mocks import sample_population  # noqa: E402


def target_shapes(cfg, grid):
    return {t: K.target_counts(make_model(spec), grid) for t, spec in cfg["targets"].items()}


def class_of(M, classes):
    k = np.full(len(M), -1)
    for j, c in enumerate(classes):
        k[(M >= c["M_lo"]) & (M < c["M_hi"])] = j
    return k


def analyse(A, d_t, d_now, kcls, shapes, cfg, seed):
    """Per class, per target: unit reconstruction. Returns nested dict res[class][target]."""
    de = np.array(cfg["d_edges"], float)
    out = {}
    for j, c in enumerate(cfg["classes"]):
        m = kcls == j
        out[c["name"]] = {t: class_curves(A[:, m], d_t[m], d_now[m], Nt, de, cfg["nnls"], seed)
                          for t, Nt in shapes.items()}
    return out


# ---------------------------------------------------------------------------
def true_C(M, d, b, mc, scen):
    DM = 5 * np.log10(np.maximum(d, 1e-4) * 100.0)
    C = 1.0 / (1.0 + np.exp((M - (mc["V_lim"] - DM)) / mc["dM"]))
    if "b_term" in scen:
        C = C * (1 - scen["b_term"]["depth"] * np.exp(-np.abs(b) / scen["b_term"]["b0"]))
    return C


def run_mock(cfg, base, pot, frame, grid, shapes, out, plots):
    mk = cfg["mock"]
    de = np.array(cfg["d_edges"], float)
    to_lbd = GalactocentricToGalactic(frame)
    cls = cfg["classes"]
    rows, curves = [], {}
    for sname, scen in mk["scenarios"].items():
        for r in range(mk["n_realisations"]):
            seed = cfg["seed"] + 1000 * list(mk["scenarios"]).index(sname) + r
            rng = np.random.default_rng(seed)
            ic = sample_population(pot, mk["density"], mk["n_true"], mk["r_trunc"], seed)
            M = rng.normal(*mk["gclf"], len(ic))
            if "faint_density" in scen:  # replace faint members by draws from the compact density
                faint = M >= cls[-1]["M_lo"]
                ic2 = sample_population(pot, scen["faint_density"], int(faint.sum()), mk["r_trunc"], seed + 7)
                ic[faint] = ic2
            lib = integrate_orbits(pot, ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"])
            N, T = lib.xyz.shape[:2]
            l, b, d = to_lbd(lib.xyz)
            p = rng.integers(0, T, N)
            ii = np.arange(N)
            Cnow = true_C(M, d[ii, p], b[ii, p], mk["completeness"], scen)
            det = rng.random(N) < Cnow
            kc = class_of(M, cls)
            # truth per class and d bin (time-averaged): intrinsic counts and expected completeness
            Ct = true_C(M[:, None], d, b, mk["completeness"], scen)
            A = K.occupancy_rz(lib.xyz, grid)
            res = analyse(A[:, det], d[det], d[ii, p][det], kc[det], shapes, cfg, seed)
            for j, c in enumerate(cls):
                m = kc == j
                N_true_d = K.phase_hist(d[m], np.ones(m.sum()), de)
                exp_obs_d = np.histogram(d[m].ravel(), de, weights=(Ct[m] / T).ravel())[0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    C_true = np.where(N_true_d > 0, exp_obs_d / N_true_d, np.nan)
                C_true_rel = C_true / C_true[0]
                for t in shapes:
                    rr = res[c["name"]][t]
                    rel = relative_completeness(rr["obs"], rr["corr_unit"])
                    row = dict(scenario=sname, real=r, cls=c["name"], target=t, N_true=float(m.sum()),
                               N_obs=float(rr["obs"].sum()), resid=rr["resid"])
                    for a in cfg["anchors_kpc"]:
                        an = anchored(rr["obs"], rr["corr_unit"], rr["W_unit_total"], de, a)
                        row[f"N_anchor{a}"] = an["N_total"]
                    rows.append(row)
                    key = (sname, c["name"], t)
                    curves.setdefault(key, []).append((rel, C_true_rel, C_true))
        print(f"[mock] {sname} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "mock_realisations.csv", index=False)
    summ = []
    for (s, c, t), g in df.groupby(["scenario", "cls", "target"]):
        d0 = dict(scenario=s, cls=c, target=t, N_true=g.N_true.mean(), N_obs=g.N_obs.mean())
        for a in cfg["anchors_kpc"]:
            rel = g[f"N_anchor{a}"] / g.N_true
            d0[f"Nrec/Ntrue_anchor{a}"] = rel.median()
            d0[f"Nrec/Ntrue_anchor{a}_p16"] = rel.quantile(0.16)
            d0[f"Nrec/Ntrue_anchor{a}_p84"] = rel.quantile(0.84)
        cv = curves[(s, c, t)]
        rel = np.array([x[0] for x in cv]); tru = np.array([x[1] for x in cv])
        with np.errstate(all="ignore"):
            d0["median_rel_C_ratio(rec/true)"] = list(np.round(np.nanmedian(rel / tru, 0), 2))
        summ.append(d0)
    sm = pd.DataFrame(summ)
    sm.to_csv(out / "mock_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sm.round(2).to_string(index=False))
    # figure: relative completeness recovered vs true, per scenario x class, target T2 (true) and T1/T3
    scen_names = list(mk["scenarios"]); dc = np.sqrt(np.maximum(de[:-1], 1) * de[1:])
    fig, axes = plt.subplots(len(scen_names), len(cls), figsize=(4.2 * len(cls), 3.2 * len(scen_names)),
                             sharex=True, squeeze=False)
    for i, s in enumerate(scen_names):
        for j, c in enumerate(cls):
            ax = axes[i, j]
            tru = np.nanmedian(np.array([x[1] for x in curves[(s, c["name"], list(shapes)[0])]]), 0)
            ax.plot(dc, tru, "k-", lw=2.5, label="true relative C(d)")
            for k, t in enumerate(shapes):
                rel = np.array([x[0] for x in curves[(s, c["name"], t)]])
                med = np.nanmedian(rel, 0)
                lo, hi = np.nanpercentile(rel, 16, 0), np.nanpercentile(rel, 84, 0)
                ax.plot(dc * (1 + 0.04 * (k - 1)), med, "o-", color=C[k], ms=4, label=t)
                ax.vlines(dc * (1 + 0.04 * (k - 1)), lo, hi, color=C[k], lw=1)
            ax.set_xscale("log"); ax.set_ylim(0, 1.6); ax.axhline(1, color=MUTED, lw=0.8)
            ax.set_title(f"{s} | {c['name']}", fontsize=9, loc="left")
            if i == len(scen_names) - 1:
                ax.set_xlabel("heliocentric d [kpc]")
            if j == 0:
                ax.set_ylabel("C(d) / C(first bin)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Mock validation of relative completeness per luminosity class (median, 16-84% over realisations); "
                 "mock truth density = T2", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(plots / "fig40_khopcls_mock_relC.png"); plt.close(fig)


# ---------------------------------------------------------------------------
def run_real(cfg, base, pot, frame, grid, shapes, out, plots):
    df, _ = load_catalog(base["catalog"])
    c = df[(df["status"] == "confirmed") & df["has_6d"]].reset_index(drop=True)
    M = c["M_V"].values.astype(float)
    ic = icrs_to_galactocentric(c["ra"], c["dec"], c["distance"], c["pmra"], c["pmdec"], c["vlos"], frame)
    lib = integrate_orbits(pot, ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"],
                           cache_dir=cfg["cache_dir"], cache_tag=potential_label(base["potential"]))
    _, _, d_t = GalactocentricToGalactic(frame)(lib.xyz)
    d_now = c["distance"].values.astype(float)
    A = K.occupancy_rz(lib.xyz, grid)
    kc = class_of(M, cfg["classes"])
    de = np.array(cfg["d_edges"], float)
    print("[real] class sizes:", {cl["name"]: int((kc == j).sum()) for j, cl in enumerate(cfg["classes"])})
    res = analyse(A, d_t, d_now, kc, shapes, cfg, cfg["seed"])
    # bootstrap: resample members within each class (orbits and present distances together)
    rng = np.random.default_rng(cfg["seed"] + 1)
    boot = {cl["name"]: {t: [] for t in shapes} for cl in cfg["classes"]}
    for bi in range(cfg["n_bootstrap"]):
        idx = np.concatenate([rng.choice(np.where(kc == j)[0], (kc == j).sum(), replace=True)
                              for j in range(len(cfg["classes"]))])
        rb = analyse(A[:, idx], d_t[idx], d_now[idx], kc[idx], shapes, cfg, cfg["seed"] + bi)
        for cl in cfg["classes"]:
            for t in shapes:
                boot[cl["name"]][t].append(rb[cl["name"]][t])
    rows = []
    for cl in cfg["classes"]:
        n = cl["name"]
        for t in shapes:
            r0 = res[n][t]
            rel = relative_completeness(r0["obs"], r0["corr_unit"])
            relb = np.array([relative_completeness(b["obs"], b["corr_unit"]) for b in boot[n][t]])
            for k in range(len(de) - 1):
                row = dict(cls=n, target=t, d_lo=de[k], d_hi=de[k + 1], N_obs=r0["obs"][k], neff=r0["neff"][k],
                           C_rel=rel[k], C_rel_p16=np.nanpercentile(relb[:, k], 16), C_rel_p84=np.nanpercentile(relb[:, k], 84),
                           nnls_resid=r0["resid"])
                if n in cfg["absolute_for"]:
                    for a in cfg["anchors_kpc"]:
                        an = anchored(r0["obs"], r0["corr_unit"], r0["W_unit_total"], de, a)
                        anb = [anchored(b["obs"], b["corr_unit"], b["W_unit_total"], de, a) for b in boot[n][t]]
                        row[f"C_anchor{a}"] = an["C"][k]
                        row[f"Ncorr_anchor{a}"] = an["corr"][k]
                        row[f"Ncorr_anchor{a}_p16"] = np.nanpercentile([x["corr"][k] for x in anb], 16)
                        row[f"Ncorr_anchor{a}_p84"] = np.nanpercentile([x["corr"][k] for x in anb], 84)
                        row[f"Ntot_anchor{a}"] = an["N_total"]
                        row[f"Ntot_anchor{a}_p16"] = np.nanpercentile([x["N_total"] for x in anb], 16)
                        row[f"Ntot_anchor{a}_p84"] = np.nanpercentile([x["N_total"] for x in anb], 84)
                rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "real_completeness.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["cls", "target", "d_lo", "d_hi", "N_obs", "neff", "C_rel", "C_rel_p16", "C_rel_p84"]
        print(tab[cols].round(2).to_string(index=False))
        tot = tab.groupby(["cls", "target"]).first().reset_index()
        tc = ["cls", "target"] + [x for x in tab.columns if x.startswith("Ntot")]
        print(tot[tc].round(1).to_string(index=False))
    plot_real(cfg, tab, shapes, plots)


def plot_real(cfg, tab, shapes, plots):
    de = np.array(cfg["d_edges"], float); dc = np.sqrt(np.maximum(de[:-1], 1) * de[1:])
    cls = [c["name"] for c in cfg["classes"]]
    fig, axes = plt.subplots(1, len(cls), figsize=(4.6 * len(cls), 3.6), sharey=True)
    for j, n in enumerate(cls):
        ax = axes[j]
        for k, t in enumerate(shapes):
            g = tab[(tab.cls == n) & (tab.target == t)]
            x = dc * (1 + 0.05 * (k - 1))
            ax.plot(x, g.C_rel, "o-", color=C[k], ms=4, label=t)
            ax.vlines(x, g.C_rel_p16, g.C_rel_p84, color=C[k], lw=1)
        g0 = tab[(tab.cls == n) & (tab.target == list(shapes)[0])]
        for x, nn in zip(dc, g0.N_obs):
            ax.text(x, 0.03, f"{nn:.0f}", ha="center", fontsize=7, color=MUTED)
        ax.set_xscale("log"); ax.set_ylim(0, 2.0); ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_xlabel("heliocentric d [kpc]  (numbers: observed GCs)")
        ax.set_title(f"{n}: completeness relative to d<{de[1]:g} kpc", fontsize=9, loc="left")
    axes[0].set_ylabel("C(d) / C(nearest bin)"); axes[0].legend(fontsize=7)
    fig.suptitle("Real data: relative completeness per luminosity class (anchor-free); bars = 68% bootstrap",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(plots / "fig41_khopcls_real_relC.png"); plt.close(fig)
    # absolute (anchored) corrected counts, bright & mid
    ab = cfg["absolute_for"]
    fig, axes = plt.subplots(1, len(ab), figsize=(5.5 * len(ab), 3.8), squeeze=False)
    for j, n in enumerate(ab):
        ax = axes[0, j]
        g0 = tab[(tab.cls == n) & (tab.target == list(shapes)[0])]
        ax.stairs(g0.N_obs.values, de, color=INK, lw=2, label="observed")
        for k, t in enumerate(shapes):
            for a, ls in zip(cfg["anchors_kpc"], ["-", "--"]):
                g = tab[(tab.cls == n) & (tab.target == t)]
                ax.stairs(g[f"Ncorr_anchor{a}"].values, de, color=C[k], lw=1.6, ls=ls,
                          label=f"{t}, C=1 at d<{a}" if j == 0 else None)
        ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, de[-1])
        ax.set_xlabel("heliocentric d [kpc]"); ax.set_ylabel("GCs per d bin")
        ax.set_title(f"{n}: observed vs completeness-corrected (anchored)", fontsize=9, loc="left")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(plots / "fig42_khopcls_real_corrected.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/khop_classes.yaml")
    ap.add_argument("--mode", choices=["mock", "real"], required=True)
    ap.add_argument("--n-real", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.n_real:
        cfg["mock"]["n_realisations"] = a.n_real
    base = yaml.safe_load(open(cfg["catalog_config"]))
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    plots = Path(cfg["plots_dir"]); plots.mkdir(parents=True, exist_ok=True)
    frame = GalacticParams.from_dict(base.get("galactocentric"))
    pot = build_potential(base["potential"])
    grid = K.RZGrid(np.array(cfg["grid"]["R_edges"], float), np.array(cfg["grid"]["z_edges"], float))
    shapes = target_shapes(cfg, grid)
    (run_mock if a.mode == "mock" else run_real)(cfg, base, pot, frame, grid, shapes, out, plots)


if __name__ == "__main__":
    main()
