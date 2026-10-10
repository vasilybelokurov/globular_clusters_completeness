#!/usr/bin/env python
"""Observed vs completeness-corrected M_V and [Fe/H] distributions, using the phase-likelihood
completeness from outputs/adaptive/real_bins.csv (run_adaptive.py --mode real must have been run).

Correction: each cluster gets weight 1/q_i, q_i = <C_class(d_i(t))>_t (its class's piecewise C
averaged along its own orbit). The phase likelihood gives C only RELATIVE to the nearest distance
bin of each M_V class, so the plot ASSUMES that bin is complete (C = 1) in every class. The
[Fe/H] shape within a class does not depend on this; the relative heights of M_V classes do.
Band: weights recomputed with all C at their 68% profile lower / upper limits.

Usage: python scripts/plot_corrected_mv_feh.py   (reads orbits from the cache; no pipeline rerun)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gc_completeness.catalog import load_catalog  # noqa: E402
from gc_completeness.coordinates import GalacticParams, GalactocentricToGalactic, icrs_to_galactocentric  # noqa: E402
from gc_completeness.orbits import integrate_orbits  # noqa: E402
from gc_completeness.plotting import plt, C, INK, MUTED  # noqa: E402
from gc_completeness.potential import build_potential, potential_label  # noqa: E402
from gc_completeness import phaselike as PL  # noqa: E402


def main():
    cfg = yaml.safe_load(open("configs/adaptive.yaml"))
    base = yaml.safe_load(open(cfg["catalog_config"]))
    frame = GalacticParams.from_dict(base.get("galactocentric"))
    df, _ = load_catalog(base["catalog"])
    c = df[(df["status"] == "confirmed") & df["has_6d"]].reset_index(drop=True)
    ic = icrs_to_galactocentric(c["ra"], c["dec"], c["distance"], c["pmra"], c["pmdec"], c["vlos"], frame)
    lib = integrate_orbits(build_potential(base["potential"]), ic, cfg["orbits"]["integration_time_gyr"],
                           cfg["orbits"]["n_samples"], cache_dir=cfg["cache_dir"], cache_tag=potential_label(base["potential"]))
    _, _, d_t = GalactocentricToGalactic(frame)(lib.xyz)
    M = c["M_V"].values.astype(float)
    feh = c["feh"].values.astype(float)
    tab = pd.read_csv(Path(cfg["output_dir"]) / "real_bins.csv")

    w = {k: np.full(len(c), np.nan) for k in ("best", "lo", "hi")}
    for cls, g in tab.groupby("cls"):
        g = g.sort_values("bin")
        m = (M >= g.M_lo.iloc[0]) & (M < g.M_hi.iloc[0])
        edges = np.r_[g.d_lo.values, g.d_hi.values[-1]]
        k = PL.bin_index(d_t[m], edges)
        for key, col in (("best", "phase_C"), ("lo", "phase_lo"), ("hi", "phase_hi")):
            Cb = g[col].values
            q = Cb[k].mean(1)
            w[key][m] = 1.0 / q
    assert np.all(np.isfinite(w["best"])), "every cluster must belong to a class"
    # low C -> high weight: the 'hi' C limits give the lower weight bound and vice versa
    w_lo, w_hi = w["hi"], w["lo"]

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1], hspace=0.06, wspace=0.25)
    panels = [("M_V", M, np.arange(-11, 0.01, 0.75), r"$M_V$ [mag]"),
              ("[Fe/H]", feh, np.arange(-2.6, 0.21, 0.2), "[Fe/H]")]
    print(f"N observed = {len(c)};  corrected total = {w['best'].sum():.1f}  "
          f"(range {w_lo.sum():.1f}-{w_hi.sum():.1f}), assuming the nearest d bin of each class is complete")
    for col, (name, x, e, lab) in enumerate(panels):
        ok = np.isfinite(x)
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        obs = np.histogram(x[ok], e)[0]
        hb = np.histogram(x[ok], e, weights=w["best"][ok])[0]
        hl = np.histogram(x[ok], e, weights=w_lo[ok])[0]
        hh = np.histogram(x[ok], e, weights=w_hi[ok])[0]
        ax.fill_between(e, np.r_[hl, hl[-1]], np.r_[hh, hh[-1]], step="post", color=C[1], alpha=0.25, lw=0)
        ax.stairs(hb, e, color=C[1], lw=2, label=f"corrected (N={hb.sum():.0f}; band: 68% C limits)")
        xc = 0.5 * (e[1:] + e[:-1])
        ax.errorbar(xc, obs, yerr=np.sqrt(obs), fmt="none", ecolor=INK, lw=1)
        ax.stairs(obs, e, color=INK, lw=1.8, label=f"observed (N={obs.sum()})")
        ax.set_ylim(bottom=0); ax.tick_params(labelbottom=False); ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("GCs per bin")
        with np.errstate(divide="ignore", invalid="ignore"):
            r, rl, rh = hb / obs, hl / obs, hh / obs
        okb = obs >= 2
        axr.stairs(np.where(okb, r, np.nan), e, color=C[1], lw=2)
        axr.fill_between(e, np.r_[np.where(okb, rl, np.nan), np.nan], np.r_[np.where(okb, rh, np.nan), np.nan],
                         step="post", color=C[1], alpha=0.25, lw=0)
        axr.axhline(1, color=MUTED, lw=1); axr.set_ylim(0, None); axr.set_xlabel(lab)
        axr.set_ylabel("corrected / observed")
        for a in (ax, axr):
            a.set_xlim(e[0], e[-1])
        if name == "M_V":
            for a in (ax, axr):
                a.invert_xaxis()
                for mb in tab.groupby("cls").M_lo.first().values[1:]:
                    a.axvline(mb, color=MUTED, ls=":", lw=1)
    fig.suptitle("Observed vs completeness-corrected GCs (phase-likelihood C(M_V, d); assumes the nearest distance bin "
                 "of each M_V class is complete; dotted: class boundaries)", fontsize=9.5, x=0.01, ha="left")
    out = Path(cfg["plots_dir"]) / "fig55_corrected_MV_FeH.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
