#!/usr/bin/env python
"""Why does the two-region C (separate splines) leave most of the no-dust halo excess in place?

Diagnostics on the real data (continuous C, 1/q weights), without dust and with map A_V (high bracket):
 1. Score decomposition: each cluster's contribution to d lnL / d lambda, where lambda scales the
    outer-region ln C curve (lambda > 1 = steeper). Zero in total at the MLE; grouped by present
    region, orbit crossing, halo membership and off-window dust along the orbit.
 2. Three regions: inner window | low latitude outside it (|b| < 15, |l| >= 30) | rest (|b| >= 15).
 3. Leave-one-out influence of each cluster on the summed halo weight (d > 15 kpc) and on C_out(V=14).

Usage:  python scripts/region_diagnostics.py
Output: outputs/cv/region_diagnostics.json, outputs/cv/region_loo.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv as R  # noqa: E402
import faint_audit as FAU  # noqa: E402
from gc_completeness import cv as CV  # noqa: E402

D_HALO = 15.0


def fit(model, V_t, V_now, reg_t, reg_now, p0=None):
    B_t, B_now = model.basis(V_t, reg_t), model.basis(V_now, reg_now)
    f = CV.fit_cv_spline(B_t, B_now, p0, bounds=model.bounds())
    q = np.exp(B_t @ f["p"]).mean(1)
    return f, q, B_t, B_now


def lnC_out(model, p, V):
    return float(model.lnC(np.array([V]), np.array([0]), p)[0])


def main():
    cfg = yaml.safe_load(open("configs/cv.yaml"))
    c, lib, d_t, dz = FAU.load(cfg)
    M = c["M_V"].values.astype(float)
    l_t, b_t, _ = R.GalactocentricToGalactic(R._W["frame"])(lib.xyz)
    win = R.sky_region(l_t, b_t, (30.0, 15.0))
    reg2 = win.copy()
    reg2[(win == 0) & (np.abs(b_t) < 15.0)] = 2                 # low latitude outside the window
    halo = c["distance"].values > D_HALO
    A_map = dz["A_high"]
    off_dust = ((A_map > 1) & (win == 0)).sum(1) / np.maximum((win == 0).sum(1), 1)   # off-window dusty fraction
    frac_in = win.mean(1)
    out = {}
    for ext in ("none", "map_high"):
        A = 0.0 if ext == "none" else A_map
        V_t = CV.apparent_mag(M, d_t, A); V_now = V_t[:, 0]
        res = {}
        # --- two-region separate splines (as in run_cv)
        m2 = CV.RegionalC(CV.region_nodes(V_now, win[:, 0], 2, cfg["n_min_v"]))
        f2, q2, B_t, B_now = fit(m2, V_t, V_now, win, win[:, 0])
        # 1. score decomposition along lambda (scales the outer-region parameters)
        sl = slice(m2.slices[0], m2.slices[1])
        a = B_t @ f2["p"]; e = np.exp(a - a.max(1, keepdims=True)); P = e / e.sum(1, keepdims=True)
        g = B_now - np.einsum("nt,ntp->np", P, B_t)                # per-cluster gradient of ln L_i
        s_i = g[:, sl] @ f2["p"][sl]                                 # d lnL_i / d lambda at lambda = 1
        groups = {
            "now outer, orbit always outer": (win[:, 0] == 0) & (frac_in == 0),
            "now outer, orbit crosses": (win[:, 0] == 0) & (frac_in > 0),
            "now inner, orbit crosses": (win[:, 0] == 1) & (frac_in < 1),
            "now inner, orbit always inner": (win[:, 0] == 1) & (frac_in == 1),
            "halo d>15": halo,
            "off-window dusty fraction > 0.2": off_dust > 0.2,
        }
        res["score_total"] = float(s_i.sum())
        res["score_by_group"] = {k: [int(m.sum()), float(s_i[m].sum()), float(s_i[m][s_i[m] > 0].sum()),
                                     float(s_i[m][s_i[m] < 0].sum())] for k, m in groups.items()}
        top = np.argsort(s_i)
        res["score_top_steepening"] = [(c["name"][i], round(float(s_i[i]), 2)) for i in top[::-1][:8]]
        res["score_top_flattening"] = [(c["name"][i], round(float(s_i[i]), 2)) for i in top[:8]]
        res["two_region"] = {"AIC": 2 * m2.n_par - 2 * f2["ll"], "total": float((1 / q2).sum()),
                             "halo": float((1 / q2)[halo].sum()), "C_out_14": float(np.exp(lnC_out(m2, f2["p"], 14.0))),
                             "outer_nodes": m2.nodes[0].tolist()}
        # 2. three regions
        try:
            m3 = CV.RegionalC(CV.region_nodes(V_now, reg2[:, 0], 3, cfg["n_min_v"]))
            f3, q3, _, _ = fit(m3, V_t, V_now, reg2, reg2[:, 0])
            res["three_region"] = {"AIC": 2 * m3.n_par - 2 * f3["ll"], "total": float((1 / q3).sum()),
                                   "halo": float((1 / q3)[halo].sum()),
                                   "C_highlat_14": float(np.exp(lnC_out(m3, f3["p"], 14.0))),
                                   "n_now": [int((reg2[:, 0] == k).sum()) for k in range(3)],
                                   "nodes": [n.tolist() for n in m3.nodes]}
        except ValueError as exc:
            res["three_region"] = {"error": repr(exc)}
        # 3. leave-one-out influence (nodes kept fixed, as in the bootstrap)
        rows = []
        base_halo = (1 / q2)[halo]
        for i in range(len(M)):
            k = np.arange(len(M)) != i
            fi, qi, _, _ = fit(m2, V_t[k], V_now[k], win[k], win[k, 0], f2["p"])
            h_ref = (1 / q2)[k & halo].sum()                       # halo sum of the same clusters, full fit
            rows.append(dict(name=c["name"][i], d=c["distance"].values[i], M_V=M[i], now_inner=int(win[i, 0]),
                             frac_in=frac_in[i], halo=bool(halo[i]),
                             d_halo=float((1 / qi)[halo[k]].sum() - h_ref),
                             d_lnC_out_14=float(lnC_out(m2, fi["p"], 14.0) - lnC_out(m2, f2["p"], 14.0))))
        L = pd.DataFrame(rows).sort_values("d_halo")
        L.to_csv(Path(cfg["output_dir"]) / f"region_loo_{ext}.csv", index=False)
        res["loo_largest_halo_drop"] = L.head(8)[["name", "d", "now_inner", "frac_in", "d_halo", "d_lnC_out_14"]].round(3).values.tolist()
        res["loo_largest_halo_rise"] = L.tail(5)[["name", "d", "now_inner", "frac_in", "d_halo", "d_lnC_out_14"]].round(3).values.tolist()
        res["loo_sum_abs_top10_share"] = float(np.sort(np.abs(L.d_halo))[::-1][:10].sum() / np.abs(L.d_halo).sum())
        out[ext] = res
        print(f"\n=== {ext} ===")
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("loo")}, indent=1, default=str))
        print("LOO: removing these lowers the halo weight most:")
        print(L.head(8).round(3).to_string(index=False))
    json.dump(out, open(Path(cfg["output_dir"]) / "region_diagnostics.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
