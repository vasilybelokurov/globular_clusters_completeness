#!/usr/bin/env python
"""Print-size figures for writeup/gc_completeness.tex (6.5 in wide, 8-9 pt fonts) from SAVED outputs.

Reads outputs/adaptive/*.csv|json and outputs/cv/*.csv|json; orbits come from the cache. Only the
quantities that were not saved (per-cluster PIT values, C(V) bootstrap replicates) are recomputed,
with the same seeds as the original runs. Writes plots/paper_*.png.

Usage: python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gc_completeness import cv as CV  # noqa: E402
from gc_completeness import phaselike as PL  # noqa: E402
from gc_completeness.catalog import load_catalog  # noqa: E402
from gc_completeness.coordinates import GalacticParams, GalactocentricToGalactic, icrs_to_galactocentric  # noqa: E402
from gc_completeness.orbits import integrate_orbits  # noqa: E402
from gc_completeness.plotting import plt, C, INK, MUTED  # noqa: E402
from gc_completeness.potential import build_potential, potential_label  # noqa: E402

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7, "lines.linewidth": 1.5, "savefig.dpi": 300,
                     "figure.dpi": 100})
W = 6.5  # inches = \textwidth
OUT = Path("plots")
AD, CVD = Path("outputs/adaptive"), Path("outputs/cv")
SCEN_LABEL = {"A_axisymmetric": "A: holds", "B_latitude": "B: latitude", "C_sgr_clump": "C: Sgr clump",
              "A_C_of_V": "A: C(V) true", "D_not_V": "D: not C(V)"}
MV_EDGES = np.arange(-11, 0.01, 0.75)


def save(fig, name):
    from matplotlib.ticker import NullFormatter
    for ax in fig.axes:   # no minor-tick labels on log axes
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_formatter(NullFormatter())
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
def load_real():
    cfg = yaml.safe_load(open("configs/cv.yaml"))
    base = yaml.safe_load(open(cfg["catalog_config"]))
    frame = GalacticParams.from_dict(base.get("galactocentric"))
    df, _ = load_catalog(base["catalog"])
    c = df[(df["status"] == "confirmed") & df["has_6d"]].reset_index(drop=True)
    ic = icrs_to_galactocentric(c["ra"], c["dec"], c["distance"], c["pmra"], c["pmdec"], c["vlos"], frame)
    lib = integrate_orbits(build_potential(base["potential"]), ic, cfg["orbits"]["integration_time_gyr"],
                           cfg["orbits"]["n_samples"], cache_dir=cfg["cache_dir"], cache_tag=potential_label(base["potential"]))
    _, _, d_t = GalactocentricToGalactic(frame)(lib.xyz)
    from astropy.io import fits
    import os
    raw = fits.open(os.path.expanduser(base["catalog"]["path"]))[1].data
    lab = dict(zip([n.strip() for n in raw["NAME"]], [p.strip() for p in raw["PROG_MA"]]))
    c["prog"] = c["name"].map(lab).fillna("")
    return cfg, c, lib, d_t


# ---------------------------------------------------------------------------
def fig_mock_classes():
    b = pd.read_csv(AD / "mock_bins.csv"); b = b[b.bin > 0]
    sym = pd.read_csv(AD / "mock_symmetry.csv")
    tr = pd.read_csv(AD / "mock_target_residuals.csv")
    est = [("phase_C", "phase_lo", "phase_hi", "C_true_phase", "phase likelihood"),
           ("T2_broken_pl_C", "T2_broken_pl_C_p16", "T2_broken_pl_C_p84", "C_true_rel", "target, plain"),
           ("T2_broken_pl_Cbr", "T2_broken_pl_Cbr_p16", "T2_broken_pl_Cbr_p84", "C_true_rel", "target, bias-red.")]
    scen = ["A_axisymmetric", "B_latitude", "C_sgr_clump"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    for k, (c, lo, hi, tru, lab) in enumerate(est):
        for i, s in enumerate(scen):
            g = b[b.scenario == s]
            r = g[c] / g[tru]
            cov = ((g[tru] >= g[lo]) & (g[tru] <= g[hi])).mean()
            x = i + (k - 1) * 0.25
            ax.errorbar(x, r.median(), yerr=[[r.median() - r.quantile(.16)], [r.quantile(.84) - r.median()]],
                        fmt="o", color=C[k], ms=4, capsize=0, label=lab if i == 0 else None)
            ax.text(x, 0.42, f"{cov:.2f}", ha="center", fontsize=6.5, color=C[k], rotation=90)
    ax.axhline(1, color=MUTED, lw=0.8); ax.set_yscale("log"); ax.set_ylim(0.38, 2.6)
    ax.set_yticks([0.5, 1, 2], ["0.5", "1", "2"])
    ax.set_xticks(range(3), ["A", "B", "C"]); ax.set_ylabel("recovered / true")
    ax.legend(loc="upper left", fontsize=6.5); ax.set_title("(a) bias and coverage", loc="left")
    ax = axes[1]
    for k, st in enumerate(["p_z", "p_y", "p_absphi"]):
        rates = [(sym[sym.scenario == s][st] < 0.05).mean() for s in scen]
        ax.plot(range(3), rates, "o-", color=C[k], ms=4, label={"p_z": "z", "p_y": "y", "p_absphi": "|φ′|"}[st])
    ax.axhline(0.05, color=MUTED, ls=":", lw=1); ax.set_ylim(0, 0.4)
    ax.set_xticks(range(3), [SCEN_LABEL[s].split(":")[0] for s in scen]); ax.set_ylabel("fraction with p < 0.05")
    ax.legend(); ax.set_title("(b) symmetry tests", loc="left")
    ax = axes[2]
    g = tr[tr.scenario == "A_axisymmetric"]
    for k, (t, lab) in enumerate([("T1_power_law", "fit T1"), ("T2_broken_pl", "fit T2 (true)"), ("T3_cored", "fit T3")]):
        ax.hist(g[f"{t}_resid_w"], bins=np.linspace(0, 1, 26), histtype="step", lw=1.5, color=C[k], label=lab)
    ax.set_xlabel("weighted NNLS residual"); ax.set_ylabel("mocks"); ax.legend(loc="upper right", fontsize=6.5,
                                                                             handlelength=1.2)
    ax.set_title("(c) target-fit residuals", loc="left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "paper_mock_classes.png")


def fig_mock_cv():
    R = pd.read_csv(CVD / "mock_cv_bins.csv"); R = R[R.vbin > 0]
    MV = pd.read_csv(CVD / "mock_cv_mv.csv"); Tt = pd.read_csv(CVD / "mock_cv_totals.csv")
    scen = ["A_C_of_V", "B_latitude", "C_sgr_clump", "D_not_V"]
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.2))
    ax = axes[0, 0]
    for i, s in enumerate(scen):
        g = R[R.scenario == s]
        r = g.C / g.C_true_rel
        cov = ((g.C_true_rel >= g.lo) & (g.C_true_rel <= g.hi)).mean()
        ax.errorbar(i, r.median(), yerr=[[r.median() - r.quantile(.16)], [r.quantile(.84) - r.median()]], fmt="s",
                    color=INK, ms=4, capsize=0)
        ax.text(i, 0.45, f"{cov:.2f}", ha="center", fontsize=6.5)
    ax.axhline(1, color=MUTED, lw=0.8); ax.set_yscale("log"); ax.set_ylim(0.4, 2.5)
    ax.set_yticks([0.5, 1, 2], ["0.5", "1", "2"])
    ax.set_xticks(range(4), ["A", "B", "C", "D"]); ax.set_ylabel("C(V): recovered / true")
    ax.set_title("(a) C(V) bias", loc="left")
    for j, s in enumerate(scen):
        ax = axes.ravel()[j + 1]
        g = MV[MV.scenario == s].groupby("mv_lo")[["true", "obs", "corr"]].mean()
        e = np.r_[g.index.values, g.index.values[-1] + 0.75]
        ax.stairs(g.true.values, e, color=INK, lw=1.6, label="true")
        ax.stairs(g.obs.values, e, color=MUTED, lw=1.2, ls="--", label="observed")
        ax.stairs(g["corr"].values, e, color=C[1], lw=1.6, label="corrected")
        ax.invert_xaxis(); ax.set_xlim(-2, -11); ax.set_xlabel(r"$M_V$")
        t = Tt[Tt.scenario == s]
        ax.set_title(f"({'bcde'[j]}) {SCEN_LABEL[s]}", loc="left")
        ax.text(0.03, 0.95, f"total {np.median(t.N_corr / t.N_true):.2f}\nfaint {np.median(t.N_corr_faint / t.N_true_faint):.2f}",
                transform=ax.transAxes, va="top", fontsize=7)
        if j in (0, 2):
            ax.set_ylabel("GCs per bin (mean)")
    h, l = axes[0, 1].get_legend_handles_labels()
    axes[1, 2].axis("off")
    axes[1, 2].legend(h, l, loc="upper left", fontsize=7.5, frameon=False)
    axes[1, 2].text(0.0, 0.45, "Numbers: corrected / true\n(median over mocks);\n'faint' = $M_V > -6.5$",
                    transform=axes[1, 2].transAxes, fontsize=7.5, va="top")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "paper_mock_cv.png")


def class_panel(ax, g, targets, title, show_legend=False):
    x0 = np.sqrt(np.maximum(g.d_lo, 0.5) * np.minimum(g.d_hi, 150))
    ax.errorbar(x0, g.phase_C, yerr=[g.phase_C - g.phase_lo, g.phase_hi - g.phase_C], fmt="s", color=INK, ms=4,
                capsize=0, lw=1.3, label="phase likelihood")
    for k, (t, lab) in enumerate(targets):
        x = x0 * (1 + 0.09 * (k + 1))
        ax.errorbar(x, g[f"{t}_Cbr"], yerr=[np.clip(g[f"{t}_Cbr"] - g[f"{t}_Cbr_p16"], 0, None),
                                            np.clip(g[f"{t}_Cbr_p84"] - g[f"{t}_Cbr"], 0, None)],
                    fmt="o", color=C[k], ms=3.5, capsize=0, lw=1, label=lab)
    for _, r in g.iterrows():
        ax.hlines(r.phase_C, max(r.d_lo, 0.5), min(r.d_hi, 150), color=MUTED, lw=0.8)
        ax.text(np.sqrt(max(r.d_lo, .5) * min(r.d_hi, 150)), 0.115, f"{r.N_obs}", ha="center", fontsize=7, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.1, 4); ax.set_xlim(0.5, 150)
    ax.set_yticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"]); ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_title(title, loc="left"); ax.set_xlabel("heliocentric $d$ [kpc]")
    if show_legend:
        ax.legend(loc="upper left", fontsize=6.5)


def fig_real_classes():
    t = pd.read_csv(AD / "real_bins.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.4), sharey=True)
    for j in range(3):
        g = t[t.cls == j]
        class_panel(axes[j], g, [("T2_broken_pl", "target T2"), ("T3_cored", "target T3")],
                    f"({'abc'[j]}) ${g.M_lo.iloc[0]:.1f} \\leq M_V < {g.M_hi.iloc[0]:.1f}$", show_legend=(j == 0))
    axes[0].set_ylabel("C(d) / C(nearest bin)")
    fig.tight_layout(w_pad=0.5)
    save(fig, "paper_real_classes.png")


def fig_real_populations():
    p = pd.read_csv(AD / "real_populations_bins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for j, (pn, targets) in enumerate([("in_situ", [("T2_q0.5", "target T2, q=0.5"), ("T2_q0.8", "target T2, q=0.8")]),
                                       ("accreted", [("T2_q1", "target T2"), ("T3_q1", "target T3")])]):
        g = p[p.population == pn]
        class_panel(axes[j], g, targets, f"({'ab'[j]}) {pn.replace('_', '-')} (N={g.N_class.iloc[0]})", show_legend=True)
    axes[0].set_ylabel("C(d) / C(nearest bin)")
    fig.tight_layout(w_pad=0.5)
    save(fig, "paper_real_populations.png")


def fig_real_symmetry(c, lib, d_t):
    """Recompute pooled PIT values (same fitted C as saved; randomisation seed as in the original run)."""
    t = pd.read_csv(AD / "real_bins.csv")
    summ = json.load(open(AD / "real_summary.json"))["symmetry_all"]
    M = c["M_V"].values.astype(float)
    groups = []
    for cls, g in t.groupby("cls"):
        g = g.sort_values("bin")
        m = (M >= g.M_lo.iloc[0]) & (M < g.M_hi.iloc[0])
        groups.append(dict(xyz_t=lib.xyz[m], d_t=d_t[m], now_idx=np.zeros(m.sum(), int),
                           d_edges=np.r_[g.d_lo.values, g.d_hi.values[-1]], theta=np.log(g.phase_C.values[1:])))
    _, pits = PL.symmetry_ks(groups, np.random.default_rng(20261010 + 5))
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.0), sharey=True)
    for ax, k, lab in zip(axes, ["z", "y", "absphi"], ["$z$", "$y$ (longitude side)", "$|\\phi'|$ (azimuth from Sun)"]):
        ax.hist(pits[k], bins=10, range=(0, 1), color=C[0], alpha=0.75)
        ax.axhline(len(pits[k]) / 10, color=INK, lw=1)
        ax.set_title(f"{lab}: p = {summ['p_null_refit'][k]:.3f}", loc="left")
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("clusters")
    fig.tight_layout(w_pad=0.5)
    save(fig, "paper_real_symmetry.png")


def fig_real_cv(cfg, c, d_t):
    tab = pd.read_csv(CVD / "real_cv.csv")
    t = pd.read_csv(AD / "real_bins.csv")
    M = c["M_V"].values.astype(float)
    edges = np.r_[tab.V_lo.values, tab.V_hi.values[-1]]
    V_t = CV.apparent_mag(M, d_t)
    summ = json.load(open(CVD / "real_cv_summary.json"))
    fig = plt.figure(figsize=(W, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    vc = np.where(np.isfinite(tab.V_lo) & np.isfinite(tab.V_hi), 0.5 * (tab.V_lo + tab.V_hi),
                  np.where(np.isfinite(tab.V_hi), tab.V_hi - 0.5, tab.V_lo + 0.5))
    ax.errorbar(vc, tab.C, yerr=[tab.C - tab.C_lo, tab.C_hi - tab.C], fmt="s", color=INK, ms=4, capsize=0)
    lo_e = np.where(np.isfinite(tab.V_lo), tab.V_lo, vc - 0.5); hi_e = np.where(np.isfinite(tab.V_hi), tab.V_hi, vc + 0.5)
    ax.hlines(tab.C, lo_e, hi_e, color=MUTED, lw=0.8)
    for x, n in zip(vc, tab.N_obs):
        ax.text(x, 0.055, f"{n}", ha="center", fontsize=7, color=MUTED)
    ax.set_yscale("log"); ax.set_ylim(0.045, 2.5); ax.set_yticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
    ax.axhline(1, color=MUTED, lw=0.8); ax.set_xlabel("apparent $V$ (no extinction)"); ax.set_ylabel("C(V)")
    ax.set_title(f"(a) C(V), brightest bin = 1; numbers: GCs per bin; "
                 f"AIC: C(V) {summ['AIC_CV']:.1f}, free per-class {summ['AIC_free']:.1f}", loc="left")
    for j, (cls, g) in enumerate(t.groupby("cls")):
        g = g.sort_values("bin")
        m = (M >= g.M_lo.iloc[0]) & (M < g.M_hi.iloc[0])
        de = np.r_[g.d_lo.values, g.d_hi.values[-1]]
        imp = CV.implied_class_curve(V_t[m], d_t[m], edges, tab.C.values, de)
        dc = np.sqrt(np.maximum(de[:-1], 0.5) * np.minimum(de[1:], 150))
        a = fig.add_subplot(gs[1, j])
        a.plot(dc, g.phase_C, "s", color=INK, ms=4, label="free per-class")
        a.plot(dc * 1.1, imp, "o", color=C[1], ms=4, label="implied by C(V)")
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(0.2, 2.5); a.set_xlim(0.8, 150)
        a.set_yticks([0.3, 0.5, 1, 2], ["0.3", "0.5", "1", "2"]); a.axhline(1, color=MUTED, lw=0.8)
        a.set_title(f"({'bcd'[j]}) ${g.M_lo.iloc[0]:.1f} \\leq M_V < {g.M_hi.iloc[0]:.1f}$", loc="left")
        a.set_xlabel("$d$ [kpc]")
        if j == 0:
            a.set_ylabel("C(d) / C(nearest)"); a.legend(fontsize=6.5, loc="lower left")
    save(fig, "paper_real_cv.png")


def fig_corrected(cfg, c, d_t, ext, lib=None):
    """Corrected M_V and [Fe/H]; bootstrap replicates regenerated exactly as in run_cv.py (same seed and order)."""
    tag = "" if ext == "none" else f"_{ext}"
    tab = pd.read_csv(CVD / f"real_cv{tag}.csv")
    M = c["M_V"].values.astype(float); feh = c["feh"].values.astype(float); d_now = c["distance"].values.astype(float)
    if ext in ("map_low", "map_high"):   # cached A_V along orbits (same tag as run_cv.py)
        from gc_completeness import dust as DU
        from gc_completeness.potential import potential_label
        base = yaml.safe_load(open(cfg["catalog_config"]))
        l_t, b_t, _ = GalactocentricToGalactic(GalacticParams.from_dict(base.get("galactocentric")))(lib.xyz)
        dz = DU.orbit_extinction(l_t, b_t, d_t, cfg["cache_dir"],
                                 tag=f"{potential_label(base['potential'])}|{lib.xyz.shape}|{float(lib.xyz.sum()):.6e}")
        A_t = dz["A_low"] if ext == "map_low" else dz["A_high"]
        V_t, V_now = CV.apparent_mag(M, d_t, A_t), CV.apparent_mag(M, d_now, A_t[:, 0])
    else:
        A_V = 3.1 * c["ebv"].values.astype(float) if ext == "present_ebv" else np.zeros(len(c))
        V_t, V_now = CV.apparent_mag(M, d_t, A_V), CV.apparent_mag(M, d_now, A_V)
    edges = np.r_[tab.V_lo.values, tab.V_hi.values[-1]]
    w = 1.0 / CV.cluster_q(V_t, edges, tab.C.values)
    rng = np.random.default_rng(cfg["seed"] + 3)
    N = len(M); boots = []
    for _ in range(cfg["n_bootstrap"]):
        idx = rng.integers(0, N, N)
        fb = CV.fit_cv(V_t[idx], V_now[idx], edges, profile=False)
        boots.append((idx, 1.0 / CV.cluster_q(V_t[idx], edges, fb["C"])))
    fig = plt.figure(figsize=(W, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.08, wspace=0.28)
    for col, (x, e, lab) in enumerate([(M, MV_EDGES, r"$M_V$ [mag]"), (feh, np.arange(-2.6, 0.21, 0.2), "[Fe/H]")]):
        ok = np.isfinite(x)
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        obs = np.histogram(x[ok], e)[0]
        hb = np.histogram(x[ok], e, weights=w[ok])[0]
        hbb = []
        for idx, wb in boots:
            xx = x[idx]; o = np.isfinite(xx)
            hbb.append(np.histogram(xx[o], e, weights=wb[o])[0])
        lo, hi = np.percentile(hbb, 16, 0), np.percentile(hbb, 84, 0)
        ax.fill_between(e, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step="post", color=C[1], alpha=0.25, lw=0)
        ax.stairs(hb, e, color=C[1], lw=1.6, label=f"corrected ({hb.sum():.0f})")
        ax.errorbar(0.5 * (e[1:] + e[:-1]), obs, yerr=np.sqrt(obs), fmt="none", ecolor=INK, lw=0.8)
        ax.stairs(obs, e, color=INK, lw=1.4, label=f"observed ({obs.sum()})")
        ax.set_ylim(bottom=0); ax.tick_params(labelbottom=False); ax.legend(loc="upper left")
        ax.set_ylabel("GCs per bin"); ax.set_title(f"({'ab'[col]})", loc="left")
        okb = obs >= 2
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.stairs(np.where(okb, hb / obs, np.nan), e, color=C[1], lw=1.6)
            axr.fill_between(e, np.r_[np.where(okb, lo / obs, np.nan), np.nan], np.r_[np.where(okb, hi / obs, np.nan), np.nan],
                             step="post", color=C[1], alpha=0.25, lw=0)
        axr.axhline(1, color=MUTED, lw=0.8); axr.set_ylim(0, 12 if col == 0 else 7); axr.set_xlabel(lab)
        axr.set_ylabel("corr./obs.")
        for a in (ax, axr):
            a.set_xlim(e[0], e[-1])
            if col == 0:
                a.set_xlim(-1.5, -11)
    save(fig, f"paper_corrected{tag}.png")


def fig_corrected_vmax(ext):
    """Corrected M_V and [Fe/H] with the continuous C(V): 1/V_max (band: 68% bootstrap) and own-orbit 1/q;
    for [Fe/H] the 1/V_max orbit population is the bright clusters of the same class ([Fe/H] < -1 or >= -1),
    read from the saved run_cv.py outputs (real_cv_weights{tag}.csv, real_cv_boot{tag}.npz)."""
    tag = "" if ext == "none" else f"_{ext}"
    t = pd.read_csv(CVD / f"real_cv_weights{tag}.csv")
    z = np.load(CVD / f"real_cv_boot{tag}.npz")
    M, feh = t.M_V.values, t.feh.values
    fig = plt.figure(figsize=(W, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.08, wspace=0.28)
    for col, (x, e, lab) in enumerate([(M, MV_EDGES, r"$M_V$ [mag]"), (feh, np.arange(-2.6, 0.21, 0.2), "[Fe/H]")]):
        ok = np.isfinite(x)
        wv, bidx, bwv = ((t.w_vmax.values, z["boot_idx_spline"], z["boot_w_vmax"]) if col == 0 else
                         (t.w_vmax_fehsplit.values, z["boot_idx_fehsplit"], z["boot_w_vmax_fehsplit"]))
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        obs = np.histogram(x[ok], e)[0]
        hv = np.histogram(x[ok], e, weights=wv[ok])[0]
        hq = np.histogram(x[ok], e, weights=t.w_spline.values[ok])[0]
        hbb = []
        for idx, wb in zip(bidx, bwv):
            xx = x[idx]; o = np.isfinite(xx) & np.isfinite(wb)
            hbb.append(np.histogram(xx[o], e, weights=wb[o])[0])
        lo, hi = np.percentile(hbb, 16, 0), np.percentile(hbb, 84, 0)
        ax.fill_between(e, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step="post", color=C[1], alpha=0.25, lw=0)
        ax.stairs(hv, e, color=C[1], lw=1.6, label=f"corrected, $1/V_{{\\rm max}}$" + (" (per class)" if col else "") + f" ({hv.sum():.0f})")
        ax.stairs(hq, e, color=C[2], lw=1.2, ls="--", label=f"corrected, own orbit $1/q$ ({hq.sum():.0f})")
        ax.errorbar(0.5 * (e[1:] + e[:-1]), obs, yerr=np.sqrt(obs), fmt="none", ecolor=INK, lw=0.8)
        ax.stairs(obs, e, color=INK, lw=1.4, label=f"observed ({obs.sum()})")
        ax.set_ylim(0, 1.5 * max(hi.max(), hq.max())); ax.tick_params(labelbottom=False); ax.legend(loc="upper left")
        ax.set_ylabel("GCs per bin"); ax.set_title(f"({'ab'[col]})", loc="left")
        okb = obs >= 2
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.stairs(np.where(okb, hv / obs, np.nan), e, color=C[1], lw=1.6)
            axr.stairs(np.where(okb, hq / obs, np.nan), e, color=C[2], lw=1.2, ls="--")
            axr.fill_between(e, np.r_[np.where(okb, lo / obs, np.nan), np.nan], np.r_[np.where(okb, hi / obs, np.nan), np.nan],
                             step="post", color=C[1], alpha=0.25, lw=0)
        axr.axhline(1, color=MUTED, lw=0.8); axr.set_ylim(0, 12 if col == 0 else 7); axr.set_xlabel(lab)
        axr.set_ylabel("corr./obs.")
        for a in (ax, axr):
            a.set_xlim(e[0], e[-1])
            if col == 0:
                a.set_xlim(-1.5, -11)
    save(fig, f"paper_corrected_vmax{tag}.png")


def fig_region():
    """Direction-dependent C (region 1 = inner window |l|<30, |b|<15 deg) vs the single C(V), without dust
    and with map extinction (high bracket); read from real_cv_summary{tag}.json, real_cv_weights{tag}.csv
    and real_cv_region_boot{tag}.npz (scripts/run_cv.py --mode real)."""
    import json
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.4), gridspec_kw=dict(hspace=0.42, wspace=0.28))
    ymax = 0.0
    for row, (ext, lab) in enumerate((("none", "no extinction"), ("map_high", "3D-map $A_V$ (high bracket)"))):
        tag = "" if ext == "none" else f"_{ext}"
        sm = json.load(open(CVD / f"real_cv_summary{tag}.json"))
        sp, rg = sm["spline"], sm["region"]
        t = pd.read_csv(CVD / f"real_cv_weights{tag}.csv")
        z = np.load(CVD / f"real_cv_region_boot{tag}.npz")
        ax = axes[row, 0]
        nd, Cn = np.array(sp["nodes_V"]), np.array(sp["C_nodes"])
        Vg = np.array(rg["sep"]["V_grid"])
        lnc = np.where(Vg > nd[-1], np.log(Cn[-1]) + sp["tail_dlnC_dV"] * (Vg - nd[-1]), np.interp(Vg, nd, np.log(Cn)))
        ax.plot(Vg, np.exp(lnc), color=INK, lw=1.4, label=f"single $C(V)$")
        for kind, ls in (("sep", "-"), ("contrast", ":")):
            e = rg[kind]
            ax.plot(Vg, e["C_outer"], color=C[0], ls=ls, lw=1.3, label=f"{'separate' if kind == 'sep' else 'contrast'}: outer (ΔAIC {e['dAIC_vs_single']:+.1f})")
            ax.plot(Vg, e["C_inner"], color=C[1], ls=ls, lw=1.3, label=f"{'separate' if kind == 'sep' else 'contrast'}: inner window")
        ax.set_yscale("log"); ax.set_ylim(0.01, 2); ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_xlabel("apparent $V$" + (" + $A_V$" if ext != "none" else "") + " [mag]"); ax.set_ylabel("C (anchor = 1)")
        ax.set_title(f"({'ac'[row]}) {lab}", loc="left"); ax.legend(fontsize=6, loc="lower left")
        ax = axes[row, 1]
        M = t.M_V.values; e_ = MV_EDGES
        obs = np.histogram(M, e_)[0]
        hb = [np.histogram(M[idx], e_, weights=wb)[0] for idx, wb in zip(z["sep_idx"], z["sep_w"])]
        lo, hi = np.percentile(hb, 16, 0), np.percentile(hb, 84, 0)
        ax.fill_between(e_, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step="post", color=C[1], alpha=0.25, lw=0)
        for col, ls, c_, nm in (("w_reg_sep", "-", C[1], "separate"), ("w_reg_contrast", ":", C[1], "contrast"), ("w_spline", "--", INK, "single $C(V)$")):
            h = np.histogram(M, e_, weights=t[col].values)[0]
            ax.stairs(h, e_, color=c_, ls=ls, lw=1.4, label=f"{nm}, $1/q$ ({h.sum():.0f})")
        ax.stairs(obs, e_, color=INK, lw=1.0, alpha=0.6, label=f"observed ({obs.sum()})")
        ymax = max(ymax, 1.3 * max(hi.max(), obs.max()))
        ax.set_xlim(-1.5, -11); ax.set_xlabel(r"$M_V$ [mag]"); ax.set_ylabel("GCs per bin")
        ax.set_title(f"({'bd'[row]}) corrected $M_V$, {lab}", loc="left"); ax.legend(fontsize=6, loc="upper left")
    for row in (0, 1):                                   # same y-scale in both rows
        axes[row, 1].set_ylim(0, ymax)
    save(fig, "paper_region.png")


def sys_envelope(h, alts):
    """Per-bin systematic range in ln: for each budget component, the most negative and most positive
    ln(h_alt / h) over its variants (zero if all variants lie on one side), then quadrature over components.
    h: baseline counts (any shape); alts: {component: [counts of each variant]}."""
    dn, up = np.zeros(np.shape(h)), np.zeros(np.shape(h))
    with np.errstate(divide="ignore", invalid="ignore"):
        for comp, hs in alts.items():
            r = np.log(np.array(hs) / h)
            r = np.where(np.isfinite(r), r, 0.0)
            dn += np.minimum(r.min(0), 0) ** 2
            up += np.maximum(r.max(0), 0) ** 2
    return np.sqrt(dn), np.sqrt(up)


def fig_final():
    """Final model (sec:final): continuous C(V), map A_V (high bracket), own-orbit 1/q. Dark band: 68% cluster
    bootstrap (sampling). Light band: sampling and the systematic envelope (sys_envelope over the alternative
    models of tab:budget, evaluated in each bin) in quadrature. Reads real_cv_weights_map_high.csv,
    real_cv_boot_map_high.npz and final_alt_weights.csv (scripts/final_systematics.py)."""
    t = pd.read_csv(CVD / "real_cv_weights_map_high.csv")
    z = np.load(CVD / "real_cv_boot_map_high.npz")
    A = pd.read_csv(CVD / "final_alt_weights.csv")
    assert (A.name.values == t.name.values).all() and np.allclose(A.base, t.w_spline, rtol=1e-6)
    comps = {}
    for col in A.columns[4:]:
        comps.setdefault(col.split(":")[0], []).append(col)
    w, bidx, bw = t.w_spline.values, z["boot_idx_spline"], z["boot_w_spline"]
    M = t.M_V.values; faint = M >= -6.5

    def total_err(sel):
        tb = np.array([wb[sel[idx]].sum() for idx, wb in zip(bidx, bw)])
        h0 = w[sel].sum()
        s_lo, s_hi = np.log(h0 / np.percentile(tb, 16)), np.log(np.percentile(tb, 84) / h0)
        y_lo, y_hi = sys_envelope(h0, {k: [A[v].values[sel].sum() for v in vs] for k, vs in comps.items()})
        return h0, np.percentile(tb, [16, 84]), (s_lo, s_hi), (float(y_lo), float(y_hi)), (np.hypot(s_lo, y_lo), np.hypot(s_hi, y_hi))
    res = {}
    for nm, sel in (("total", np.ones(len(w), bool)), ("faint", faint)):
        h0, pb, s, y, T = total_err(sel)
        res[nm] = (h0, pb, T)
        print(f"[final] {nm}: {h0:.0f}; sampling {pb[0]:.0f}-{pb[1]:.0f} (ln {-s[0]:+.3f}, {s[1]:+.3f}); "
              f"systematic ln {-y[0]:+.3f}, {y[1]:+.3f}; total ln {-T[0]:+.3f}, {T[1]:+.3f} -> "
              f"{h0 * np.exp(-T[0]):.0f}-{h0 * np.exp(T[1]):.0f}")
    N0, _, (T_lo, T_hi) = res["total"]
    fig = plt.figure(figsize=(W, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.08, wspace=0.28)
    for col, (x, e, lab) in enumerate([(M, MV_EDGES, r"$M_V$ [mag]"),
                                       (t.feh.values, np.arange(-2.6, 0.21, 0.2), "[Fe/H]")]):
        ok = np.isfinite(x)
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        obs = np.histogram(x[ok], e)[0]
        h = np.histogram(x[ok], e, weights=w[ok])[0]
        hb = []
        for idx, wb in zip(bidx, bw):
            xx = x[idx]; o = np.isfinite(xx)
            hb.append(np.histogram(xx[o], e, weights=wb[o])[0])
        lo, hi = np.percentile(hb, 16, 0), np.percentile(hb, 84, 0)
        y_lo, y_hi = sys_envelope(h, {k: [np.histogram(x[ok], e, weights=A[v].values[ok])[0] for v in vs]
                                      for k, vs in comps.items()})
        with np.errstate(divide="ignore", invalid="ignore"):
            dlo = np.where(h > 0, np.log(np.where(lo > 0, h / lo, np.inf)), 0)
            dhi = np.where(h > 0, np.log(hi / h), 0)
        Tlo, Thi = h * np.exp(-np.hypot(dlo, y_lo)), h * np.exp(np.hypot(dhi, y_hi))
        st = lambda a: np.r_[a, a[-1]]
        okb = obs >= 2
        for a_, den in ((ax, 1.0), (axr, np.where(okb, obs, np.nan))):
            with np.errstate(divide="ignore", invalid="ignore"):
                a_.fill_between(e, st(Tlo / den), st(Thi / den), step="post", color=C[1], alpha=0.15, lw=0,
                                label="68%, sampling + systematics" if a_ is ax else None)
                a_.fill_between(e, st(lo / den), st(hi / den), step="post", color=C[1], alpha=0.35, lw=0,
                                label="68%, sampling" if a_ is ax else None)
                a_.stairs(h / den, e, color=C[1], lw=1.6, baseline=None,
                          label=(f"corrected ({N0:.0f}$^{{+{N0 * (np.exp(T_hi) - 1):.0f}}}_{{-{N0 * (1 - np.exp(-T_lo)):.0f}}}$)"
                                 if col == 0 else f"corrected ({h.sum():.0f}; [Fe/H] known)") if a_ is ax else None)
        ax.errorbar(0.5 * (e[1:] + e[:-1]), obs, yerr=np.sqrt(obs), fmt="none", ecolor=INK, lw=0.8)
        ax.stairs(obs, e, color=INK, lw=1.4, label=f"observed ({obs.sum()})")
        if col == 0:
            hf, _, (f_lo, f_hi) = res["faint"]
            ax.axvline(-6.5, ymax=0.6, color=MUTED, lw=0.8, ls=":")
            ax.text(-1.8, 0.56, f"$M_V>-6.5$:\n{hf:.0f}$^{{+{hf * (np.exp(f_hi) - 1):.0f}}}_{{-{hf * (1 - np.exp(-f_lo)):.0f}}}$",
                    transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=7, color=INK)
        ax.set_ylim(0, 1.25 * np.nanmax(Thi)); ax.tick_params(labelbottom=False)
        ax.legend(loc="upper left", fontsize=6.5)
        ax.set_ylabel("GCs per bin"); ax.set_title(f"({'ab'[col]})", loc="left")
        axr.axhline(1, color=MUTED, lw=0.8); axr.set_ylim(0, 12 if col == 0 else 7); axr.set_xlabel(lab)
        axr.set_ylabel("corr./obs.")
        for a in (ax, axr):
            a.set_xlim((-1.5, -11) if col == 0 else (e[0], e[-1]))
        if col == 0:
            print("[final] per-bin M_V systematic ln down/up:", np.round(y_lo, 2).tolist(), np.round(y_hi, 2).tolist())
    save(fig, "paper_final.png")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--final":
        fig_final()
        return
    if len(sys.argv) > 2 and sys.argv[1] == "--vmax":   # single figure from saved outputs
        fig_corrected_vmax(sys.argv[2])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--region":
        fig_region()
        return
    fig_mock_classes()
    fig_mock_cv()
    fig_real_classes()
    fig_real_populations()
    cfg, c, lib, d_t = load_real()
    fig_real_symmetry(c, lib, d_t)
    fig_real_cv(cfg, c, d_t)
    fig_corrected(cfg, c, d_t, "none")
    fig_corrected(cfg, c, d_t, "present_ebv")
    fig_corrected(cfg, c, d_t, "map_high", lib)
    fig_corrected_vmax("map_high")


if __name__ == "__main__":
    main()
