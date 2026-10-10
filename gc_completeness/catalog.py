"""Catalogue ingestion and bookkeeping of GC subsets.

Standardised columns produced by :func:`load_catalog`::

    name, ra, dec, distance [kpc], pmra, pmdec [mas/yr], vlos [km/s],
    *_error columns (if available), M_V, r_h [pc], feh, ebv, mass [Msun],
    l, b [deg], status ('confirmed' | 'candidate'), has_6d (bool)

No measurement is invented: missing values stay NaN. Clusters without full 6D
information are kept in the table (has_6d=False) and in observed counts, but are
excluded from the orbit library.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from .coordinates import icrs_to_galactic_lbd

REQUIRED = ["name", "ra", "dec", "distance", "pmra", "pmdec", "vlos"]
SIXD = ["ra", "dec", "distance", "pmra", "pmdec", "vlos"]
M_V_SUN = 4.83  # absolute V magnitude of the Sun (Willmer 2018 gives 4.81; difference negligible here)

BAUMGARDT_COLUMNS = [
    "Cluster", "RA", "DEC", "R_Sun", "DRSun", "R_GC", "DRGC", "N_RV", "N_PM", "Mass", "DM",
    "V", "Delta_V", "M/L_V", "DM/L", "rc", "rh_l", "rh_m", "rt", "rho_c", "rho_h_m", "sig_c",
    "sig_h_m", "lg_Trh", "lg_Mini", "T_Diss", "M_Low", "M_High", "MF", "Delta_MF", "sig0",
    "vesc", "etac", "etah", "A_Rot", "Delta_AR", "P_Rot",
]


def _read_table(path: Path) -> pd.DataFrame:
    suf = "".join(path.suffixes).lower()
    if suf.endswith((".fits", ".fit", ".fits.gz")):
        with fits.open(path) as h:
            data = h[1].data
            out = {}
            for c in data.columns.names:
                col = np.asarray(data[c])
                if col.dtype.kind in "SU":
                    out[c] = np.char.strip(col.astype(str))
                else:
                    out[c] = col.astype(col.dtype.newbyteorder("="))
            return pd.DataFrame(out)
    return pd.read_csv(path)


def read_baumgardt_combined(path: str | Path) -> pd.DataFrame:
    """Read Baumgardt's ``combined_table.txt`` (https://people.smp.uq.edu.au/HolgerBaumgardt/globular/).

    Rows with fewer fields lack the trailing rotation columns; those are NaN.
    """
    rows = []
    with open(os.path.expanduser(str(path))) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            vals = [parts[0]] + [float(x) for x in parts[1:]]
            vals += [np.nan] * (len(BAUMGARDT_COLUMNS) - len(vals))
            rows.append(vals)
    return pd.DataFrame(rows, columns=BAUMGARDT_COLUMNS)


def _crossmatch(ra1, dec1, ra2, dec2, radius_arcsec: float):
    """Return index into catalogue 2 for each row of 1 (-1 if no match within radius)."""
    c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg)
    c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg)
    idx, sep, _ = c1.match_to_catalog_sky(c2)
    idx = np.where(sep.arcsec <= radius_arcsec, idx, -1)
    return idx


def load_catalog(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Load and standardise the GC catalogue.

    Parameters
    ----------
    cfg : dict
        'path', 'columns' (standard_name -> file column), optional 'classification_column',
        'candidate_names', 'supplementary' (Baumgardt combined table) settings.

    Returns
    -------
    df : DataFrame with standardised columns and subset flags.
    meta : dict with provenance and subset counts.
    """
    path = Path(os.path.expanduser(cfg["path"]))
    raw = _read_table(path)
    colmap = cfg.get("columns", {})
    df = pd.DataFrame()
    for std, src in colmap.items():
        if src is None:
            continue
        if src not in raw.columns:
            raise KeyError(f"Column '{src}' (for '{std}') not in {path}")
        df[std] = raw[src].values
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing after mapping: {missing}")
    df["name"] = df["name"].astype(str).str.strip()
    for col, sentinels in (cfg.get("missing_values") or {}).items():  # e.g. FEH=-99, AGE in {-999, 0}
        if col in df.columns:
            df[col] = df[col].astype(float).where(~df[col].isin(sentinels), np.nan)
    df["source"] = "main"

    meta = {"catalog_path": str(path), "n_main_rows": len(df)}

    # --- supplementary table: M_V, r_h, and confirmed clusters absent from main table ---
    sup = cfg.get("supplementary") or {}
    if sup.get("path"):
        bm = read_baumgardt_combined(sup["path"])
        # M_V from dynamical mass and M/L_V: L_V = M / (M/L_V); M_V = M_V_sun - 2.5 log10 L_V
        with np.errstate(divide="ignore", invalid="ignore"):
            bm["M_V"] = M_V_SUN - 2.5 * np.log10(bm["Mass"] / bm["M/L_V"])
        idx = _crossmatch(df["ra"].values, df["dec"].values, bm["RA"].values, bm["DEC"].values,
                          sup.get("match_radius_arcsec", 60.0))
        ok = idx >= 0
        for std, src in (("M_V", "M_V"), ("r_h_light", "rh_l")):
            vals = np.full(len(df), np.nan)
            vals[ok] = bm[src].values[idx[ok]]
            if std not in df.columns:
                df[std] = vals
            else:
                df[std] = np.where(np.isfinite(df[std]), df[std], vals)
        df["baumgardt_name"] = np.where(ok, bm["Cluster"].values[np.clip(idx, 0, None)], "")
        meta["n_main_matched_to_supplementary"] = int(ok.sum())
        meta["main_unmatched"] = df.loc[~ok, "name"].tolist()
        if sup.get("add_unmatched", True):
            matched = set(idx[ok].tolist())
            extra = bm.loc[[j for j in range(len(bm)) if j not in matched]]
            add = pd.DataFrame({
                "name": extra["Cluster"].values, "ra": extra["RA"].values,
                "dec": extra["DEC"].values, "distance": extra["R_Sun"].values,
                "M_V": extra["M_V"].values, "r_h_light": extra["rh_l"].values,
                "mass": extra["Mass"].values,
            })
            for c in ["pmra", "pmdec", "vlos"]:
                add[c] = np.nan  # not taken from the main table: no 6D -> not in orbit library
            add["source"] = "supplementary_only"
            add = add.reindex(columns=list(df.columns) + [c for c in add.columns if c not in df.columns])
            df = pd.concat([df, add.astype({c: df[c].dtype for c in df.columns
                                            if c in add and df[c].dtype.kind == "f"})], ignore_index=True)
            meta["n_supplementary_only"] = len(add)
            meta["supplementary_only_names"] = add["name"].tolist()
        meta["supplementary_path"] = str(sup["path"])

    # --- classification ---
    cls_col = cfg.get("classification_column")
    if cls_col and cls_col in raw.columns:
        stat = np.full(len(df), "confirmed", dtype=object)
        stat[: len(raw)] = np.where(raw[cls_col].astype(str).str.lower().str.contains("cand"),
                                    "candidate", "confirmed")
        df["status"] = stat
    else:
        df["status"] = "confirmed"
    cand = set(cfg.get("candidate_names") or [])
    if cand:
        df.loc[df["name"].isin(cand) | df.get("baumgardt_name", pd.Series(dtype=str)).isin(cand),
               "status"] = "candidate"

    df["has_6d"] = np.all(np.isfinite(df[SIXD].astype(float).values), axis=1)
    l, b, _ = icrs_to_galactic_lbd(df["ra"].values, df["dec"].values, df["distance"].values)
    df["l"], df["b"] = l, b

    meta.update({
        "n_total_rows": len(df),
        "n_confirmed": int((df["status"] == "confirmed").sum()),
        "n_candidate": int((df["status"] == "candidate").sum()),
        "n_6d": int(df["has_6d"].sum()),
        "n_confirmed_6d": int(((df["status"] == "confirmed") & df["has_6d"]).sum()),
        "n_confirmed_no6d": int(((df["status"] == "confirmed") & ~df["has_6d"]).sum()),
    })
    return df, meta
