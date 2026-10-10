"""Galactic potential construction (AGAMA)."""
from __future__ import annotations

import os
from pathlib import Path

import agama

_UNITS_SET = False

# 4-component model used across the user's existing projects
# (uma3/plot_dillamore_fig8.py, "from gse_visualisation.ipynb").
MW4_COMPONENTS = [
    dict(type="miyamotonagai", mass=6.8e10, scaleradius=3.0, scaleheight=0.28),  # disc
    dict(type="dehnen", mass=5.00e9, scaleradius=1.0),   # bulge
    dict(type="dehnen", mass=1.71e9, scaleradius=0.07),  # nucleus
    dict(type="nfw", mass=5.4e11, scaleradius=15.62),    # halo
]


def ensure_units() -> None:
    """Set AGAMA units to kpc, Msun, km/s (time unit = 0.9778 Gyr)."""
    global _UNITS_SET
    if not _UNITS_SET:
        agama.setUnits(length=1, mass=1, velocity=1)
        _UNITS_SET = True


def time_unit_gyr() -> float:
    """AGAMA internal time unit in Gyr."""
    ensure_units()
    t = agama.getUnits()["time"]  # Myr; an astropy Quantity when astropy is loaded
    if hasattr(t, "to_value"):
        return float(t.to_value("Gyr"))
    return float(t) / 1e3


def build_potential(cfg: dict) -> agama.Potential:
    """Build an AGAMA potential from a config dict.

    cfg['kind'] options
    -------------------
    'mw4'        : 4-component Miyamoto-Nagai + 2 Dehnen + NFW model (MW4_COMPONENTS).
    'agama_data' : cfg['file'] is an .ini shipped in agama/data (e.g. 'McMillan17.ini').
    'ini'        : cfg['file'] is a path to an AGAMA .ini file.
    'components' : cfg['components'] is a list of AGAMA component dicts.
    """
    ensure_units()
    kind = cfg.get("kind", "mw4")
    if kind == "mw4":
        return agama.Potential(*MW4_COMPONENTS)
    if kind == "components":
        return agama.Potential(*cfg["components"])
    if kind == "agama_data":
        path = Path(os.path.dirname(agama.__file__)) / "data" / cfg["file"]
        return agama.Potential(str(path))
    if kind == "ini":
        path = Path(os.path.expanduser(cfg["file"]))
        if not path.exists():
            raise FileNotFoundError(f"Potential file not found: {path}")
        return agama.Potential(str(path))
    raise ValueError(f"Unknown potential kind '{kind}'")


def potential_label(cfg: dict) -> str:
    kind = cfg.get("kind", "mw4")
    if kind in ("agama_data", "ini"):
        return f"{kind}:{Path(cfg['file']).name}"
    if kind == "components":
        return "components:" + ",".join(c["type"] for c in cfg["components"])
    return kind
