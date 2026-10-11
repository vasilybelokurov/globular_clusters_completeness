#!/usr/bin/env python
"""Export the completeness product consumed by the GC IMF project (gcimf); see HANDOFF.md.

One frozen file holds everything gcimf needs: the per-cluster table, the orbit library sampled along
time (sky track, A_V brackets, sky-region labels, pericentre/apocentre) and every completeness model
of the error budget, each written in one common form:

    region k of variant v:  ln C(V) = interp(V; nodes_k, lnC_k)       for V <= nodes_k[-1] (flat brighter)
                                     = lnC_k[-1] + slope_k (V - nodes_k[-1])   beyond the last node,

with the sky region of every orbit sample given by the variant's region label array, and V = M_V + DM(d) + A_V
with the variant's A_V bracket. The baseline also carries its 200 cluster-bootstrap refits in the same form.

Self-check (the file is written only if it passes): for every variant, the per-cluster weights 1/q rebuilt
from the exported arrays alone reproduce the weights saved by run_cv.py / final_systematics.py, and the
bootstrap refits reproduce the saved bootstrap weights.

Usage:  python scripts/export_imf_product.py [--allow-dirty]
Output: products/completeness_product_v1.npz, products/completeness_product_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_cv as R  # noqa: E402
import faint_audit as FAU  # noqa: E402
from gc_completeness import cv as CV  # noqa: E402
from gc_completeness import dust as DU  # noqa: E402
from gc_completeness import faint as FA  # noqa: E402
from gc_completeness.catalog import _crossmatch, read_baumgardt_combined  # noqa: E402

PRODUCT = "completeness_product_v1"
TOL_SAME_PATH = 1e-9      # weights rebuilt along the same code path as the saved ones
TOL_CROSS_PATH = 1e-6     # final_systematics.py uses V_now = V(t=0) instead of the catalogue distance (~1e-7 mag)


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
            "remote": run("remote", "get-url", "origin")}


def piecewise(lnC_fn, nodes_list) -> list[dict]:
    """Common form of a model: per region, ln C at its nodes and the slope beyond the last node."""
    out = []
    for k, nd in enumerate(nodes_list):
        nd = np.asarray(nd, float)
        t = lnC_fn(nd, np.full(len(nd), k))
        end = lnC_fn(np.array([nd[-1], nd[-1] + 1.0]), np.array([k, k]))
        out.append({"nodes": nd, "lnC": t, "slope": float(end[1] - end[0])})
    return out


def eval_piecewise(regions: list[dict], V: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """ln C from the common form (the same few lines gcimf uses)."""
    out = np.zeros(np.shape(V))
    for k, r in enumerate(regions):
        m = reg == k
        n, t = r["nodes"], r["lnC"]
        out[m] = np.where(V[m] > n[-1], t[-1] + r["slope"] * (V[m] - n[-1]), np.interp(V[m], n, t))
    return out


def check(name, got, ref, tol, log):
    err = float(np.max(np.abs(np.asarray(got) / np.asarray(ref) - 1.0)))
    ok = err < tol
    log.append({"check": name, "max_rel_err": err, "tol": tol, "pass": ok})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:55s} max rel err {err:.2e} (tol {tol:.0e})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true", help="export from a working tree with uncommitted changes")
    a = ap.parse_args()
    git = git_state()
    if git["dirty"] and not a.allow_dirty:
        sys.exit("Working tree has uncommitted changes; commit first (or pass --allow-dirty for a test export).")

    cfg = yaml.safe_load(open(ROOT / "configs/cv.yaml"))
    base = yaml.safe_load(open(ROOT / cfg["catalog_config"]))
    c, lib, d_t, dz = FAU.load(cfg)                      # same loaders as run_cv.py / final_systematics.py
    frame = R._W["frame"]
    l_t, b_t, _ = R.GalactocentricToGalactic(frame)(lib.xyz)
    N, T = d_t.shape
    M = c["M_V"].values.astype(float)
    d_now = c["distance"].values.astype(float)
    window = tuple(cfg["region_window"])
    reg2 = R.sky_region(l_t, b_t, window)
    reg3 = reg2.copy()
    reg3[(reg2 == 0) & (np.abs(b_t) < 15.0)] = 2          # as region_diagnostics.py / final_systematics.py
    reg1 = np.zeros_like(reg2)
    A = {"map_high": dz["A_high"], "map_low": dz["A_low"], "none": np.zeros((N, T))}

    # ---- variants, each built exactly as in the script whose saved weights it is checked against ----
    variants, refs = {}, {}
    wcsv = {e: pd.read_csv(ROOT / f"outputs/cv/real_cv_weights{'' if e == 'none' else '_' + e}.csv")
            for e in ("map_high", "map_low", "none")}
    alt = pd.read_csv(ROOT / "outputs/cv/final_alt_weights.csv")
    for e in ("map_high", "map_low", "none"):
        if not np.array_equal(wcsv[e]["name"].values, c["name"].values):
            sys.exit(f"cluster order differs from real_cv_weights ({e})")

    def single_runcv(ext):
        V_t = CV.apparent_mag(M, d_t, A[ext]); V_now = CV.apparent_mag(M, d_now, A[ext][:, 0])
        nodes = CV.spline_nodes(V_now, cfg["n_min_v"])
        f = CV.fit_cv_spline(CV.spline_basis(V_t, nodes), CV.spline_basis(V_now, nodes))
        return piecewise(lambda V, k: CV.lnC_spline(V, nodes, f["p"]), [nodes]), f, nodes, V_t, V_now

    desc = {}
    for ext, key in (("map_high", "baseline"), ("map_low", "map_low"), ("none", "no_extinction")):
        regs, f, nodes, V_t, V_now = single_runcv(ext)
        variants[key] = dict(regions=regs, extinction=ext, region_labels="none")
        refs[key] = [("real_cv_weights" + ("" if ext == "none" else "_" + ext) + ".csv:w_spline",
                      wcsv[ext]["w_spline"].values, TOL_SAME_PATH)]
        if ext == "map_high":
            base_fit, base_nodes, base_Vt, base_Vnow = f, nodes, V_t, V_now
    refs["baseline"].append(("final_alt_weights.csv:base", alt["base"].values, TOL_CROSS_PATH))
    refs["map_low"].append(("final_alt_weights.csv:map:low", alt["map:low"].values, TOL_CROSS_PATH))
    desc.update(baseline="ADOPTED. Continuous C(V), 3D-map A_V high bracket, one sky region (write-up Sec. final).",
                map_low="Systematic: as baseline with the A_V low bracket.",
                no_extinction="NOT adopted (disfavoured, Delta AIC ~ 16): continuous C(V) without extinction.")

    # tail variants (final_systematics.py: V_now = V_t[:, 0], FA.make_nodes)
    V_t = CV.apparent_mag(M, d_t, A["map_high"])
    for tname, tv in cfg["faint"]["tail_variants"].items():
        if tname == "T0_baseline":
            continue
        tail, extra = tv.get("tail", 1.0), tv.get("extra_node_pct")
        nodes = FA.make_nodes(V_t[:, 0], cfg["n_min_v"], extra)
        f = CV.fit_cv_spline(CV.spline_basis(V_t, nodes, tail), CV.spline_basis(V_t[:, 0], nodes, tail))
        key = f"tail_{tname}"
        variants[key] = dict(regions=piecewise(lambda V, k, n=nodes, p=f["p"], t=tail: CV.lnC_spline(V, n, p, t), [nodes]),
                             extinction="map_high", region_labels="none")
        refs[key] = [(f"final_alt_weights.csv:tail:{tname}", alt[f"tail:{tname}"].values, TOL_CROSS_PATH)]
        desc[key] = f"Systematic: faint-end tail variant {tname} {tv} (configs/cv.yaml faint.tail_variants)."

    # direction-dependent C (run_cv.py analyse_region: V_now from the catalogue distance)
    for kind in ("contrast", "sep"):
        mdl = R.make_region_model(kind, base_Vnow, reg2[:, 0], 2, cfg)
        f = CV.fit_cv_spline(mdl.basis(base_Vt, reg2), mdl.basis(base_Vnow, reg2[:, 0]), bounds=mdl.bounds())
        nl = [mdl.nodes, mdl.nodes] if kind == "contrast" else mdl.nodes
        key = f"direction_{kind}"
        variants[key] = dict(regions=piecewise(lambda V, k, m=mdl, p=f["p"]: m.lnC(V, k, p), nl),
                             extinction="map_high", region_labels="reg2")
        refs[key] = [(f"real_cv_weights_map_high.csv:w_reg_{kind}", wcsv["map_high"][f"w_reg_{kind}"].values, TOL_SAME_PATH),
                     (f"final_alt_weights.csv:direction:{kind}", alt[f"direction:{kind}"].values, TOL_CROSS_PATH)]
    desc["direction_contrast"] = ("Systematic: two sky regions (inner window |l|<30, |b|<15 = region 1), shared curve "
                                  "plus a constant ln-contrast in region 1.")
    desc["direction_sep"] = "Systematic: two sky regions, separate curves (region 0 anchored)."
    m3 = CV.RegionalC(CV.region_nodes(V_t[:, 0], reg3[:, 0], 3, cfg["n_min_v"]))
    f3 = CV.fit_cv_spline(m3.basis(V_t, reg3), m3.basis(V_t[:, 0], reg3[:, 0]), bounds=m3.bounds())
    variants["direction_three"] = dict(regions=piecewise(lambda V, k: m3.lnC(V, k, f3["p"]), m3.nodes),
                                       extinction="map_high", region_labels="reg3")
    refs["direction_three"] = [("final_alt_weights.csv:direction:three", alt["direction:three"].values, TOL_CROSS_PATH)]
    desc["direction_three"] = ("Systematic, NOT adopted in the write-up (19 clusters now in region 2): three regions = "
                               "inner window | |b|<15 outside it | rest; separate curves.")

    # ---- baseline bootstrap: replay run_cv.py analyse_spline (same RNG stream, nodes fixed, warm start) ----
    rng = np.random.default_rng(cfg["seed"] + 11)
    B_t, B_now = CV.spline_basis(base_Vt, base_nodes), CV.spline_basis(base_Vnow, base_nodes)
    boot_idx, boot_lnC, boot_slope, boot_w = [], [], [], []
    for _ in range(cfg["n_bootstrap"]):
        idx = rng.integers(0, N, N)
        fb = CV.fit_cv_spline(B_t[idx], B_now[idx], base_fit["p"])
        r_ = piecewise(lambda V, k, p=fb["p"]: CV.lnC_spline(V, base_nodes, p), [base_nodes])[0]
        boot_idx.append(idx); boot_lnC.append(r_["lnC"]); boot_slope.append(r_["slope"])
        boot_w.append(1.0 / np.exp(B_t[idx] @ fb["p"]).mean(1))
    saved = np.load(ROOT / "outputs/cv/real_cv_boot_map_high.npz")

    # ---- self-check: everything rebuilt from the exported arrays only ----
    print("Self-check (weights rebuilt from the exported arrays):")
    labels = {"none": reg1, "reg2": reg2, "reg3": reg3}
    log, ok = [], True
    for key, v in variants.items():
        V = CV.apparent_mag(M, d_t, A[v["extinction"]])
        q = np.exp(eval_piecewise(v["regions"], V, labels[v["region_labels"]])).mean(1)
        v["q"] = q
        for name, ref, tol in refs[key]:
            ok &= check(f"{key} vs {name}", 1.0 / q, ref, tol, log)
    ok &= check("bootstrap indices vs real_cv_boot_map_high.npz", np.array(boot_idx) + 1,
                saved["boot_idx_spline"] + 1, 1e-15, log)
    reb = []
    for idx, t, s in zip(boot_idx, boot_lnC, boot_slope):
        regs = [{"nodes": base_nodes, "lnC": t, "slope": s}]
        reb.append(1.0 / np.exp(eval_piecewise(regs, base_Vt[idx], reg1[idx])).mean(1))
    ok &= check("bootstrap weights vs real_cv_boot_map_high.npz", np.array(reb), saved["boot_w_spline"], TOL_SAME_PATH, log)
    tot = np.array([w.sum() for w in reb])
    print(f"  baseline total 1/q = {(1.0 / variants['baseline']['q']).sum():.1f}; "
          f"bootstrap 16-84%: {np.percentile(tot, 16):.0f}-{np.percentile(tot, 84):.0f} (write-up: 432, 321-628)")
    if not ok:
        sys.exit("Self-check failed: product NOT written.")

    # ---- per-cluster table (M/L_V from Baumgardt's combined table, matched as in catalog.py) ----
    sup = base["catalog"]["supplementary"]
    bm = read_baumgardt_combined(ROOT / sup["path"])
    j = _crossmatch(c["ra"].values, c["dec"].values, bm["RA"].values, bm["DEC"].values, sup["match_radius_arcsec"])
    if np.any(j < 0):
        sys.exit("some clusters have no Baumgardt match")
    mass_b, ml_v = bm["Mass"].values[j], bm["M/L_V"].values[j]
    if not np.allclose(4.83 - 2.5 * np.log10(mass_b / ml_v), M, atol=1e-9):
        sys.exit("M_V does not follow from Baumgardt mass and M/L_V")

    # ---- write ----
    out_dir = ROOT / "products"; out_dir.mkdir(exist_ok=True)
    arrays = {
        "name": c["name"].values.astype(str), "ra": c["ra"].values.astype(float), "dec": c["dec"].values.astype(float),
        "l": c["l"].values.astype(float), "b": c["b"].values.astype(float), "distance_kpc": d_now,
        "M_V": M, "mass_msun": mass_b, "ml_v": ml_v, "mass_fits_msun": c["mass"].values.astype(float),
        "feh": c["feh"].values.astype(float), "age_gyr": c["age"].values.astype(float),
        "r_h_pc": c["r_h"].values.astype(float),
        "orbit_ic": lib.ic, "orbit_times_gyr": lib.times_gyr, "orbit_xyz_kpc": lib.xyz.astype(np.float32),
        "orbit_l_deg": l_t.astype(np.float32), "orbit_b_deg": b_t.astype(np.float32), "orbit_d_kpc": d_t,
        "orbit_A_V_high": A["map_high"], "orbit_A_V_low": A["map_low"],
        "orbit_reg2": reg2.astype(np.int8), "orbit_reg3": reg3.astype(np.int8),
        "orbit_rperi_kpc": lib.rperi, "orbit_rapo_kpc": lib.rapo, "orbit_energy_rel_err": lib.energy_rel_err,
        "baseline_boot_idx": np.array(boot_idx), "baseline_boot_lnC": np.array(boot_lnC),
        "baseline_boot_slope": np.array(boot_slope),
    }
    for key, v in variants.items():
        arrays[f"{key}__q"] = v["q"]
        for k, r_ in enumerate(v["regions"]):
            arrays[f"{key}__r{k}_nodes"] = r_["nodes"]
            arrays[f"{key}__r{k}_lnC"] = r_["lnC"]
            arrays[f"{key}__r{k}_slope"] = np.array(r_["slope"])
    npz = out_dir / f"{PRODUCT}.npz"
    np.savez_compressed(npz, **arrays)
    meta = {
        "product": PRODUCT,
        "source": git, "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "n_clusters": int(N), "n_orbit_samples": int(T),
        "clusters": "confirmed GCs with 6D data (configs/default.yaml catalog), in the order of real_cv_weights_*.csv",
        "anchor": "C = 1 at V brighter than the first node of region 0 (brightest clusters complete)",
        "apparent_magnitude": "V = M_V + 5 log10(d / 10 pc) + A_V(t); M_V = 4.83 - 2.5 log10(mass_msun / ml_v)",
        "lnC_form": ("region k: ln C = interp(V; nodes, lnC) for V <= nodes[-1] (flat brighter than nodes[0]); "
                     "lnC[-1] + slope (V - nodes[-1]) beyond"),
        "region_labels": {"none": "all samples region 0",
                          "reg2": f"orbit_reg2: 1 inside |l|<{window[0]}, |b|<{window[1]} deg, else 0",
                          "reg3": "orbit_reg3: as reg2, plus 2 = |b|<15 deg outside the window"},
        "variants": {k: {"extinction": v["extinction"], "region_labels": v["region_labels"],
                         "n_regions": len(v["regions"]), "description": desc[k],
                         "total_inverse_q": float((1.0 / v["q"]).sum())} for k, v in variants.items()},
        "extinction": {"map_high": "orbit_A_V_high", "map_low": "orbit_A_V_low", "none": "0"},
        "bootstrap": ("baseline only: baseline_boot_lnC (B, n_nodes) at baseline__r0_nodes and baseline_boot_slope (B,), "
                      "cluster resamples baseline_boot_idx; fixed nodes; replay of run_cv.py analyse_spline"),
        "orbits": {"potential": R.potential_label(base["potential"]), "galactocentric_frame": frame.to_dict(),
                   "integration_time_gyr": cfg["orbits"]["integration_time_gyr"], "n_samples": cfg["orbits"]["n_samples"],
                   "rperi_rapo": "min / max galactocentric radius over the samples",
                   "column_0": "present position"},
        "A_V_conversions": {"R_V": DU.R_V, "AV_per_AKs": DU.AV_PER_AKS, "AV_per_EBV_SFD": DU.AV_PER_EBV_SFD,
                            "bayestar_to_EBV": DU.BAYESTAR_TO_EBV, "decaps_dmax_kpc": DU.DECAPS_DMAX},
        "self_check": log,
        "configs": {"cv": cfg, "default": base},
    }
    json.dump(meta, open(out_dir / f"{PRODUCT}.json", "w"), indent=1, default=str)
    print(f"wrote {npz} ({npz.stat().st_size / 1e6:.1f} MB) and {PRODUCT}.json; source commit {git['commit'][:8]}"
          f"{' (DIRTY)' if git['dirty'] else ''}")


if __name__ == "__main__":
    main()
