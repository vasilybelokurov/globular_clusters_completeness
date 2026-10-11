#!/usr/bin/env python
"""Per-cluster weights of every alternative model in the error budget (tab:budget), on the same 165 clusters,
so that the systematic spread can be evaluated bin by bin (fig:final), not only for the total.

Baseline: continuous C(V), map A_V (high bracket), own-orbit 1/q. Alternatives, grouped by component:
  weighting  : 1/V_max (bright population), 1/V_max per [Fe/H] class, midref, kernel 1/V_max
  tail       : spline tail variants T1-T3 (configs/cv.yaml faint.tail_variants)
  direction  : two regions, shared curve + contrast; two regions, separate curves; three regions, separate
  map        : map A_V low bracket
Usage:  python scripts/final_systematics.py
Output: outputs/cv/final_alt_weights.csv (columns: name, M_V, feh, base, <component>:<variant>)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv as R  # noqa: E402
import faint_audit as FAU  # noqa: E402
import region_diagnostics as RD  # noqa: E402
from gc_completeness import cv as CV  # noqa: E402
from gc_completeness import faint as FA  # noqa: E402


def main():
    cfg = yaml.safe_load(open("configs/cv.yaml")); fp = cfg["faint"]
    c, lib, d_t, dz = FAU.load(cfg)
    M = c["M_V"].values.astype(float); feh = c["feh"].values.astype(float)
    fehc = np.where(np.isfinite(feh), (feh >= -1).astype(int), -1)
    out = pd.DataFrame(dict(name=c["name"], M_V=M, feh=feh))

    def weights(A, tail=1.0, extra=None, groups=None, kernel=False):
        V_t = CV.apparent_mag(M, d_t, A)
        nodes = FA.make_nodes(V_t[:, 0], cfg["n_min_v"], extra)
        return FA.estimates(M, V_t, V_t[:, 0], nodes, tail, fp, groups=groups, kernel=kernel)["weights"]

    w0 = weights(dz["A_high"], kernel=True)
    out["base"] = w0["ht"]
    out["weighting:vmax"], out["weighting:midref"], out["weighting:kernel"] = w0["vmax"], w0["midref"], w0["kernel"]
    out["weighting:vmax_feh"] = weights(dz["A_high"], groups=fehc)["vmax"]
    for tname, tv in fp["tail_variants"].items():
        if tname != "T0_baseline":
            out[f"tail:{tname}"] = weights(dz["A_high"], tv.get("tail", 1.0), tv.get("extra_node_pct"))["ht"]
    out["map:low"] = weights(dz["A_low"])["ht"]
    # direction-dependent C, as in run_cv.py (two regions) and region_diagnostics.py (three regions)
    V_t = CV.apparent_mag(M, d_t, dz["A_high"]); V_now = V_t[:, 0]
    l_t, b_t, _ = R.GalactocentricToGalactic(R._W["frame"])(lib.xyz)
    win = R.sky_region(l_t, b_t, tuple(cfg["region_window"]))
    for kind in ("contrast", "sep"):
        m = R.make_region_model(kind, V_now, win[:, 0], 2, cfg)
        out[f"direction:{kind}"] = 1.0 / RD.fit(m, V_t, V_now, win, win[:, 0])[1]
    reg3 = win.copy(); reg3[(win == 0) & (np.abs(b_t) < 15.0)] = 2
    m3 = CV.RegionalC(CV.region_nodes(V_now, reg3[:, 0], 3, cfg["n_min_v"]))
    out["direction:three"] = 1.0 / RD.fit(m3, V_t, V_now, reg3, reg3[:, 0])[1]
    p = Path(cfg["output_dir"]) / "final_alt_weights.csv"
    out.to_csv(p, index=False)
    print(out.drop(columns=["name", "M_V", "feh"]).sum().round(1).to_string())
    print("wrote", p)


if __name__ == "__main__":
    main()
