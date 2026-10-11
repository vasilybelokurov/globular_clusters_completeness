#!/usr/bin/env python
"""Real data: sensitivity audit of the faint-end ratio R = faint(1/V_max) / faint(1/q) and the
apocentre diagnostic (steps 2-3 of the faint-end plan; definitions in configs/cv.yaml 'faint').

Variants: extinction (none, map_low, map_high) x spline tail (T0-T3) x bright population (all, per [Fe/H]).
Paired cluster bootstraps: 'fixed' nodes for every variant, plus 'nodes' (node placement resampled) for T0.

Usage:  python scripts/faint_audit.py [--n-boot 200]
Output: outputs/cv/faint_audit.csv, outputs/cv/faint_audit.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv as R  # noqa: E402
from gc_completeness import cv as CV  # noqa: E402
from gc_completeness import faint as FA  # noqa: E402
from gc_completeness import dust as DU  # noqa: E402


def load(cfg):
    R._init(cfg)
    base = yaml.safe_load(open(cfg["catalog_config"]))
    df, _ = R.load_catalog(base["catalog"])
    c = df[(df["status"] == "confirmed") & df["has_6d"]].reset_index(drop=True)
    ic = R.icrs_to_galactocentric(c["ra"], c["dec"], c["distance"], c["pmra"], c["pmdec"], c["vlos"], R._W["frame"])
    lib = R.integrate_orbits(R._W["pot"], ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"],
                             cache_dir=cfg["cache_dir"], cache_tag=R.potential_label(base["potential"]))
    l_t, b_t, d_t = R.GalactocentricToGalactic(R._W["frame"])(lib.xyz)
    dz = DU.orbit_extinction(l_t, b_t, d_t, cfg["cache_dir"],
                             tag=f"{R.potential_label(base['potential'])}|{lib.xyz.shape}|{float(lib.xyz.sum()):.6e}")
    return c, lib, d_t, dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cv.yaml")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); fp = cfg["faint"]
    nb = a.n_boot if a.n_boot is not None else fp["n_boot"]
    c, lib, d_t, dz = load(cfg)
    M = c["M_V"].values.astype(float); feh = c["feh"].values.astype(float)
    fehc = np.where(np.isfinite(feh), (feh >= -1).astype(int), -1)
    rows, summ = [], {}
    for ext in ("none", "map_low", "map_high"):
        A = 0.0 if ext == "none" else (dz["A_low"] if ext == "map_low" else dz["A_high"])
        V_t = CV.apparent_mag(M, d_t, A)
        V_now = V_t[:, 0]                                   # orbits start at the present position
        for tname, tv in fp["tail_variants"].items():
            for grp, groups in (("all", None), ("feh", fehc)):
                base = tname == "T0_baseline"
                rng = np.random.default_rng(cfg["seed"] + 101)
                est, boots, nodes = FA.run(M, V_t, V_now, fp, cfg["n_min_v"], tv.get("tail", 1.0), tv.get("extra_node_pct"),
                                           lib.rapo, groups, nb, rng, kernel=base and grp == "all",
                                           boot_modes=("fixed", "nodes") if base else ("fixed",))
                row = dict(extinction=ext, tail=tname, population=grp, n_nodes=len(nodes), fit_success=est["success"],
                           lnL=est["ll"], **{k: est.get(k, np.nan) for k in FA.SCALARS},
                           rapo_w_faint=est.get("rapo_w_faint"), rapo_w_bright=est.get("rapo_w_bright"),
                           rapo_faint=est.get("rapo_faint"), rapo_bright=est.get("rapo_bright"))
                for mode, b in boots.items():
                    for k, v in b.items():
                        ok = np.isfinite(v)
                        if ok.sum() > 10:
                            row[f"{k}_{mode}_p16"], row[f"{k}_{mode}_p84"] = np.percentile(v[ok], [16, 84])
                    row[f"R_{mode}_P_lt1"] = float(np.mean(b["R"][np.isfinite(b["R"])] < 1))
                    row[f"dlnrapo_w_{mode}_P_gt0"] = float(np.mean(b["dlnrapo_w"][np.isfinite(b["dlnrapo_w"])] > 0))
                rows.append(row)
                print(f"[audit] {ext:8s} {tname:14s} {grp:3s}: R {est['R']:.2f} "
                      f"[{row.get('R_fixed_p16', np.nan):.2f}-{row.get('R_fixed_p84', np.nan):.2f}]"
                      + (f" nodes-boot [{row.get('R_nodes_p16', np.nan):.2f}-{row.get('R_nodes_p84', np.nan):.2f}]" if base else "")
                      + f"  faint 1/q {est['ht_faint']:.0f}  vmax {est['vmax_faint']:.0f}  midref {est['midref_faint']:.0f}"
                      + (f"  kernel {est['kernel_faint']:.0f}" if "kernel_faint" in est else "")
                      + f"  ESS {est['ess_bright']:.1f} maxw {est['maxw_bright']:.2f}", flush=True)
    T = pd.DataFrame(rows)
    out = Path(cfg["output_dir"])
    T.to_csv(out / "faint_audit.csv", index=False)
    json.dump({"n_boot": nb, "faint_def": f"M_V >= {fp['M_faint']}", "bright_ref": f"M_V < {fp['M_bright_ref']}",
               "rows": T.replace({np.nan: None}).to_dict("records")}, open(out / "faint_audit.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(T[["extinction", "tail", "population", "n_nodes", "R", "R_fixed_p16", "R_fixed_p84", "ht_faint", "vmax_faint",
                 "ht_tot", "vmax_tot", "ess_bright", "maxw_bright", "dlnrapo_w", "dlnrapo"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
