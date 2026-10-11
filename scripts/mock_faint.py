#!/usr/bin/env python
"""Calibration/validation mocks for the faint-end estimators (steps 4-5 of the faint-end plan).

Each realisation draws M_V first and samples each luminosity class (bright < -8 <= mid < -6.5 <= faint)
from its own isotropic DF, with break radius rb x s^0, s^0.5, s^1 for bright, mid and faint (s = 1: null,
orbits independent of M_V). Detection, C(V) fit and estimators exactly as for the real data
(gc_completeness/faint.py). Realisations 0..n_calib-1 are the calibration set, the rest validation.

Usage:  python scripts/mock_faint.py [--workers 12] [--n-calib 100 --n-valid 100 --n-boot 100]
        python scripts/mock_faint.py --summarise          (calibrate on calib set, evaluate on valid set)
Output: outputs/cv/mock_faint_rows.csv, mock_faint_calibration.csv, mock_faint_validation.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
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

EST = ("ht", "vmax", "midref", "kernel")


def one(args):
    sname, r = args
    cfg, pot, frame = R._W["cfg"], R._W["pot"], R._W["frame"]
    fp, mk = cfg["faint"], cfg["mock"]
    fm = fp["mock"]; scen = dict(fm["scenarios"][sname])
    if scen.get("C_shape") == "real_spline":
        scen["_spline"] = R._W["real_spline"]
    seed = cfg["seed"] + 104729 * list(fm["scenarios"]).index(sname) + r + 900000
    rng = np.random.default_rng(seed)
    n = fm["n_true"]
    M = rng.normal(*fm.get("gclf", mk["gclf"]), n)
    cls = np.digitize(M, fm["classes_M"])                       # 0 bright, 1 mid, 2 faint
    ic = np.zeros((n, 6))
    for c in range(3):
        m = cls == c
        if m.any():
            dens = dict(mk["density"], rb=mk["density"]["rb"] * scen["s"] ** (c / 2.0))
            ic[m] = R.sample_population(pot, dens, int(m.sum()), mk["r_trunc"], seed * 3 + c)
    lib = R.integrate_orbits(pot, ic, cfg["orbits"]["integration_time_gyr"], cfg["orbits"]["n_samples"])
    N, T = lib.xyz.shape[:2]
    l, b, d = R.GalactocentricToGalactic(frame)(lib.xyz)
    p = rng.integers(0, T, N); ii = np.arange(N)
    A = DU.disc_extinction(l, b, d, **scen["dust"]) if "dust" in scen else np.zeros_like(d)
    det = rng.random(N) < R.true_C(M + A[ii, p], d[ii, p], b[ii, p], scen, mk["completeness"])
    Md = M[det]
    V_t = CV.apparent_mag(Md, d[det], A[det])
    V_now = CV.apparent_mag(Md, d[ii, p][det], A[ii, p][det])
    row = dict(scenario=sname, real=r, set="calib" if r < fm["n_calib"] else "valid", N_true=n,
               N_true_faint=int((M >= fp["M_faint"]).sum()), N_obs=int(det.sum()), N_obs_faint=int((Md >= fp["M_faint"]).sum()),
               rapo_true_faint=float(np.median(lib.rapo[cls == 2])), rapo_true_bright=float(np.median(lib.rapo[cls == 0])))
    try:
        est, boots, nodes = FA.run(Md, V_t, V_now, fp, cfg["n_min_v"], rapo=lib.rapo[det], n_boot=fm["n_boot"], rng=rng)
    except ValueError as exc:
        row["error"] = repr(exc)[:120]
        return row
    row.update({k: est.get(k, np.nan) for k in FA.SCALARS}, success=est["success"])
    for mode, bb in boots.items():
        for e in EST:
            for part in ("tot", "faint"):
                v = bb[f"{e}_{part}"]; v = v[np.isfinite(v) & (v > 0)]
                if len(v) > 10:
                    row[f"{e}_{part}_{mode}_p16"], row[f"{e}_{part}_{mode}_p84"] = np.percentile(v, [16, 84])
                    row[f"{e}_{part}_{mode}_sdln"] = float(np.std(np.log(v)))
    return row


def _init(cfg):
    R._init(cfg)
    import json
    sp = json.load(open(Path(cfg["output_dir"]) / cfg["faint"]["mock"]["real_spline_from"]))["spline"]
    R._W["real_spline"] = (np.array(sp["nodes_V"]), np.log(np.array(sp["C_nodes"])), sp["tail_dlnC_dV"])


def simulate(cfg, workers):
    fm = cfg["faint"]["mock"]
    tasks = [(s, r) for s in fm["scenarios"] for r in range(fm["n_calib"] + fm["n_valid"])]
    rows = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=(cfg,)) as ex:
        for k, row in enumerate(ex.map(one, tasks)):
            rows.append(row)
            if (k + 1) % 100 == 0:
                print(f"[mock_faint] {k + 1}/{len(tasks)}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(Path(cfg["output_dir"]) / "mock_faint_rows.csv", index=False)
    return T


def summarise(cfg, T):
    """Calibrate interval methods on the calib set; evaluate everything on the valid set."""
    fp = cfg["faint"]; out = Path(cfg["output_dir"])
    T = T[T.get("error", pd.Series(np.nan, index=T.index)).isna()] if "error" in T else T
    truth = {"tot": "N_true", "faint": "N_true_faint"}
    # inflation factor k: interval point x exp(+-k sdln) covering 68% on the calib set, pooled over scenarios
    C = T[T.set == "calib"]
    kf = []
    for e in EST:
        for part in ("tot", "faint"):
            for mode in ("fixed", "nodes"):
                sd = C.get(f"{e}_{part}_{mode}_sdln")
                if sd is None:
                    continue
                z = np.abs(np.log(C[f"{e}_{part}"] / C[truth[part]])) / sd
                z = z[np.isfinite(z)]
                kf.append(dict(estimator=e, part=part, mode=mode, k68=float(np.quantile(z, 0.68)), n=len(z)))
    K = pd.DataFrame(kf, columns=["estimator", "part", "mode", "k68", "n"])
    K.to_csv(out / "mock_faint_calibration.csv", index=False)
    V = T[T.set == "valid"]
    res = []
    for s, g in V.groupby("scenario", sort=False):
        d = dict(scenario=s, n=len(g), N_obs=g.N_obs.mean(), N_obs_faint=g.N_obs_faint.mean(),
                 R_median=g.R.median(), P_R_lt_real=float(np.mean(g.R < fp["R_real"])),
                 dlnrapo_w_median=g.dlnrapo_w.median(), dlnrapo_median=g.dlnrapo.median(),
                 dlnrapo_true=np.median(np.log(g.rapo_true_faint / g.rapo_true_bright)))
        for e in EST:
            for part in ("tot", "faint"):
                x = np.log(g[f"{e}_{part}"] / g[truth[part]])
                x = x[np.isfinite(x)]
                d[f"{e}_{part}_medln"], d[f"{e}_{part}_rmsln"] = float(np.median(x)), float(np.sqrt(np.mean(x ** 2)))
                for mode in ("fixed", "nodes"):
                    lo, hi = g.get(f"{e}_{part}_{mode}_p16"), g.get(f"{e}_{part}_{mode}_p84")
                    if lo is not None:
                        d[f"{e}_{part}_cov_{mode}"] = float(np.mean((g[truth[part]] >= lo) & (g[truth[part]] <= hi)))
                    kk = K[(K.estimator == e) & (K.part == part) & (K["mode"] == mode)]
                    sd = g.get(f"{e}_{part}_{mode}_sdln")
                    if len(kk) and sd is not None:
                        z = np.abs(np.log(g[f"{e}_{part}"] / g[truth[part]])) / sd
                        d[f"{e}_{part}_cov_{mode}_infl"] = float(np.mean(z <= kk.k68.values[0]))
        res.append(d)
    S = pd.DataFrame(res)
    S.to_csv(out / "mock_faint_validation.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(K.round(2).to_string(index=False))
        print(S[["scenario", "N_obs", "N_obs_faint", "R_median", "P_R_lt_real", "dlnrapo_w_median", "dlnrapo_median", "dlnrapo_true"]].round(2).to_string(index=False))
        for part in ("tot", "faint"):
            cols = ["scenario"] + [f"{e}_{part}_{x}" for e in EST for x in ("medln", "rmsln", "cov_fixed", "cov_nodes", "cov_nodes_infl")]
            print(S[[c for c in cols if c in S]].round(2).to_string(index=False))
    return K, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cv.yaml")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-calib", type=int); ap.add_argument("--n-valid", type=int); ap.add_argument("--n-boot", type=int)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    fm = cfg["faint"]["mock"]
    for k, v in (("n_calib", a.n_calib), ("n_valid", a.n_valid), ("n_boot", a.n_boot)):
        if v is not None:
            fm[k] = v
    T = pd.read_csv(Path(cfg["output_dir"]) / "mock_faint_rows.csv") if a.summarise else simulate(cfg, a.workers)
    summarise(cfg, T)


if __name__ == "__main__":
    main()
