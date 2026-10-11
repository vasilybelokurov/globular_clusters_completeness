# Hand-off to the GC IMF project (gcimf)

The completeness measured here is consumed by the GC initial-mass-function project (`../gcimf`) through
**one frozen file**, not through imports of this code or its caches.

## Product

`products/completeness_product_v1.npz` + `.json` (metadata), written by

```bash
source ~/Work/venvs/.venv/bin/activate
python scripts/export_imf_product.py        # refuses to run on a dirty tree; stamps the commit
```

| Content | Arrays |
|---|---|
| 165 clusters (confirmed, 6D), order of `outputs/cv/real_cv_weights_*.csv` | `name, ra, dec, l, b, distance_kpc, M_V, mass_msun, ml_v, mass_fits_msun, feh, age_gyr, r_h_pc` |
| Orbit library (MW4 potential, 5 Gyr, 500 samples; column 0 = present) | `orbit_xyz_kpc, orbit_l_deg, orbit_b_deg, orbit_d_kpc, orbit_rperi_kpc, orbit_rapo_kpc, orbit_ic, orbit_times_gyr, orbit_energy_rel_err` |
| A_V along orbits (3D maps, brackets) | `orbit_A_V_high, orbit_A_V_low` |
| Sky regions per sample | `orbit_reg2` (inner window), `orbit_reg3` (+ low latitude) |
| Completeness models, common form | `<variant>__r<k>_nodes, __r<k>_lnC, __r<k>_slope, <variant>__q` |
| Baseline bootstrap (200 refits) | `baseline_boot_lnC, baseline_boot_slope, baseline_boot_idx` |

Common form of every model, region k: ln C(V) = linear interpolation through (nodes, lnC), flat brighter
than the first node, and lnC[-1] + slope (V - nodes[-1]) beyond the last node; V = M_V + DM(d) + A_V.

Variants: `baseline` (adopted: continuous C(V), maps high), `map_low`, `tail_T1/T2/T3`,
`direction_contrast`, `direction_sep`, `direction_three` (not adopted), `no_extinction` (not adopted).

## Self-check

The export rebuilds every variant's per-cluster weights 1/q from the exported arrays alone and compares
them with the saved `real_cv_weights_*.csv` (tolerance 1e-9) and `final_alt_weights.csv` (1e-6; that script
uses V at t = 0 rather than the catalogue distance). The bootstrap replays the RNG stream of `run_cv.py`
and reproduces `real_cv_boot_map_high.npz`. The file is not written if any check fails. gcimf repeats the
baseline check in its own tests.

## Rules for the consumer

- Real and mock clusters use the **library** orbits: a = (r_peri + r_apo)/2, e = (r_apo - r_peri)/(r_apo + r_peri)
  from `orbit_rperi_kpc / orbit_rapo_kpc`. Baumgardt's orbit table (other potential) differs by 0.77-1.22 in a
  (5-95%) and up to +0.3 in e.
- A change of C here means a new product version (`_v2`), never an edit of v1.
- Known caveat passed on: the adopted C gives the most luminous clusters (M_V < -8) a summed 1/q of 66 for 40
  observed, including halo clusters (NGC 2419: 1/q = 1.97); the direction variants do not change this.
