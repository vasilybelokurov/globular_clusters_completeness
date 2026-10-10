# What has been coded: GC completeness via orbit superposition

## Goal

Measure the completeness of the Milky Way globular-cluster (GC) catalogue and reconstruct
completeness-corrected GC counts. The method follows the orbit-superposition idea of Khoperskov
et al. 2025 (A&A 695, A220, [arXiv:2411.15062](https://arxiv.org/abs/2411.15062)).

## Assumptions (set by the PI; no others are intended)

1. Completeness C depends only on present-day absolute magnitude M_V and heliocentric distance d.
2. The target density has a fixed shape. There are three spherical options, none fitted to the data:
   - T1: power law ν ∝ (r + 0.05)^−3.5;
   - T2: broken power law, slopes 2 → 4, r_b = 5 kpc, r_c = 0.3 kpc;
   - T3: cored, slopes 0 → 4, r_b = 3 kpc.
3. GC orbits are phase-mixed in a static axisymmetric potential.

## Data and shared infrastructure (`gc_completeness/`)

| Module | Content |
|---|---|
| `catalog.py` | Loads `~/data/catalogues/gc_catalog_updated.fits` (165 GCs with 6D data and errors). M_V comes from Baumgardt's combined table (`data/baumgardt_combined_table.txt`) as M_V = 4.83 − 2.5 log10(Mass / (M/L_V)). |
| `coordinates.py` | astropy Galactocentric frame, v4.0 defaults: R0 = 8.122 kpc, z_sun = 20.8 pc, v_sun = (12.9, 245.6, 7.78) km/s. |
| `potential.py` | AGAMA, units kpc, Msun, km/s. 4-component potential: Miyamoto–Nagai disc, two Dehnen spheroids, NFW halo. |
| `orbits.py` | Orbit integration in the static potential: 5 Gyr, 500 uniform time samples. |
| `khoperskov.py` | (R, \|z\|) grid; target counts per cell; occupancy A[cell, orbit] = time fraction; NNLS (`nnls_subsets`). |
| `khop_classes.py` | Per-class reconstruction and completeness (algorithm below). |

The driver is `scripts/run_khop_classes.py`, the configuration `configs/khop_classes.yaml`, and the
mock populations come from `gc_completeness/mocks.py`.

## Algorithm (per luminosity class)

There are three classes: bright M_V ≤ −7 (90 GCs), mid −7 < M_V ≤ −6 (43), faint M_V > −6 (32).
Only GCs with 6D data are used.

1. Compute the occupancy A of the class's orbits on the (R, |z|) grid (R and |z| up to 150 kpc).
2. Find NNLS weights w ≥ 0 such that A·w ≈ N_target(T) at unit amplitude. Cell residuals are
   weighted by 1/√(N_target + 0.5), and all class orbits are fitted in one solve.
3. Corrected counts per heliocentric-distance bin (edges 0, 6, 10, 15, 25, 50, 150 kpc):
   N_corr(d) = Σ_i w_i ⟨1[d_i(t) ∈ bin]⟩_t, with d evaluated along each orbit.
4. Completeness C(d) = N_obs(d) / N_corr(d), where N_obs counts present-day positions. It is known
   up to the unknown target amplitude, so it is reported relative to the first bin (d < 6 kpc).
5. Absolute values are given only under an explicit anchor: C = 1 summed over the bins with
   d < 6 kpc, or with d < 10 kpc. This is done for the bright and mid classes.
6. Uncertainty: bootstrap of members within each class (200 replicates).

## Mock validation (`--mode mock`)

- **Set-up.**
  - True density: T2, drawn from an AGAMA quasi-spherical DF.
  - Luminosities: Gaussian GCLF N(−7.4, 1.2).
  - True completeness: C = 1 / (1 + exp((M_V − (10 − DM(d))) / 0.5)), known only to the generator.
  - 30 realisations per scenario, about 250 GCs each.
- **Scenarios.**
  - S1: separable truth.
  - S2: the faint class follows a more compact density.
  - S3: C also depends on Galactic latitude.

Reconstructed / true total, median over realisations, with the correct target T2:

| Scenario | bright, d < 10 anchor | mid, d < 10 | faint, d < 10 | bright, d < 6 anchor |
|---|---|---|---|---|
| S1 | 0.93 | 0.80 | 0.63 | 1.05 (16–84%: 0.75–1.35) |
| S2 | 0.94 | 0.79 | 0.92 | 1.01 |
| S3 | 0.69 | 0.69 | 0.52 | 0.94 |

- **Wrong targets (T1 or T3 against a T2 truth):** 0.45–3.1 × truth.
- **Missing distance bins:** in S1 the mid and faint classes get zero corrected counts in the
  most distant bins. No observed orbits reach those bins.

## Real-data results (`--mode real`)

Outputs are `outputs/khop_classes/real_completeness.csv`, `plots/fig41_khopcls_real_relC.png` and
`plots/fig42_khopcls_real_corrected.png`.

Relative completeness C(d) / C(d < 6 kpc) by heliocentric-distance bin [kpc]. N_obs is the number
of observed GCs in each bin.

| class | target | 6–10 | 10–15 | 15–25 | 25–50 | 50–150 |
|---|---|---|---|---|---|---|
| bright (N_obs 14/39/23/6/7/1) | T1 | 0.13 | 0.55 | 0.27 | 0.40 | 0.08 |
| bright | T2 | 0.45 | 0.53 | 0.19 | 0.31 | 0.08 |
| bright | T3 | 0.69 | 0.50 | 0.15 | 0.23 | 0.05 |
| mid (11/20/2/8/1/1) | T2 | 0.27 | 0.06 | 0.29 | 0.06 | 0.09 |
| faint (2/8/3/6/6/7) | T2 | 0.62 | 0.47 | 1.00 | 1.70 | 2.04 |

The faint class gives values above 1 at large distances under all three targets (T1 up to 1.85,
T3 up to 1.39).

Absolute total of bright GCs (sum of weights): T1 69, T2 111, T3 154 with the 10 kpc anchor;
T1 409, T2 212, T3 205 with the 6 kpc anchor. NNLS relative residuals for the bright class are
0.16 (T2) and 0.20 (T3).

## Known limitations of the implementation

- Orbit occupancy outside the grid is dropped, and orbit samples beyond 150 kpc fall outside the
  distance bins.
- The bootstrap covers sampling noise only; it does not propagate uncertainty in the potential,
  the target or the orbit integration.
- Unit tests (`tests/test_khoperskov.py`) cover the grid, target counts, occupancy, NNLS and the completeness formulas, but not the driver script end to end.

## Open questions for review

1. Are the algorithm and code a correct implementation of the Khoperskov idea under the three
   assumptions above?
2. Within these assumptions, what can make relative completeness fall to about 0.1–0.3 for bright
   GCs at 15–150 kpc?
3. Within these assumptions, what can make relative completeness exceed 1 for faint GCs at
   d > 15 kpc?
4. Are the absolute numbers interpretable, given that they change by up to 6× with target and anchor?
5. What is the most defensible next step that adds no new assumptions?

## Adaptive (M_V, d) bins: two estimators and symmetry tests

Code: `adaptive.py`, `phaselike.py`, `scripts/run_adaptive.py`, `configs/adaptive.yaml`.

### Bins

- 3 equal-count M_V classes. Within each class, equal-count distance bins of at least 15 GCs,
  merged where tied values would leave a bin short.
- Bins are chosen **once** on the observed sample and then frozen. Bootstraps and mock
  comparisons use exactly these bins.
- The last distance bin is open-ended in both estimators. In the real data no observed cluster and
  no orbit sample lies beyond 150 kpc.

### Estimator 1: Khoperskov

- Fixed target (T1, T2, T3; flattened T2 with q = 0.5 or 0.8 for in-situ clusters) and NNLS
  weights; C = N_obs / N_corr relative to the nearest bin.
- Two versions: the plain ratio, and a bias-reduced ratio (n₀ + 1 in the reference count).
- Intervals: fixed-bin bootstrap with 200 replicates.
- Per-bin diagnostics: effective number of orbits, weighted residual, SVD conditioning and rank,
  and stability of the corrected counts across bootstraps. A bin is flagged unresolved if
  N_eff < 3, the bootstrap CV > 0.5, or N_obs < 5.

### Estimator 2: per-cluster phase likelihood (target-free)

- ln L = Σᵢ [ln C(d_i,now) − ln ⟨C(d_i(t))⟩_t], with C piecewise constant within each M_V class.
- Assumptions: C = C(M_V, d) and phase mixing; no target density.
- Intervals: 68% profile-likelihood.

### Symmetry tests

- For each cluster, a randomised PIT of z, y (Galactic-longitude side) and |azimuth from the Sun
  direction|, measured against its own orbit's C-weighted phase distribution.
- The PIT values are pooled and tested with a single KS statistic.
- p-values are calibrated by null simulations that redraw phases from the fitted C and refit C
  (+1-corrected). 500 simulations for the real data, 100 per mock realisation.

### Mock validation

100 realisations × 3 scenarios. True density T2; true C = logistic in apparent magnitude, V_lim = 8.
Each estimator is compared with its own correct truth:

- **phase likelihood:** the value it converges to for an infinite sample;
- **Khoperskov:** the true detected / intrinsic counts per bin.

Recovered / true, median (16–84%), and coverage of the nominal 68% intervals:

| Estimator | A: assumptions hold | B: C also depends on latitude | C: Sgr-like clump |
|---|---|---|---|
| Phase likelihood | 0.99 (0.68–1.41); 0.67 | 0.98 (0.71–1.52); 0.71 | 0.97 (0.68–1.34); 0.71 |
| Khoperskov T2 (correct target), plain | 1.29; 0.48 | 1.30; 0.54 | 1.39; 0.45 |
| Khoperskov T2, bias-reduced | 1.23; 0.50 | 1.22; 0.58 | 1.32; 0.47 |

- **Symmetry tests:** rejection at p < 0.05 is 6–13% under the null (the azimuth test is mildly
  anti-conservative at 13%) and 2–19% under the alternatives, so they have little power. The
  azimuth test reaches p ≤ 0.01 in 2% of null mocks.
- **Target ranking by weighted residual fails:** with T2 as the truth, the median residual is 0.41
  for a T1 fit and 0.44 for T2.

### Real data (`outputs/adaptive/real_*.csv`, `real_summary.json`; `plots/fig50`–`54`)

**Phase likelihood**, C(d) / C(nearest bin), 68% profile intervals:

| Class (M_V range) | Second distance bin | Third distance bin |
|---|---|---|
| −10.54 to −7.58 | 8.3–11.1 kpc: 1.22 [0.88, 1.71] | 11.1–150 kpc: 0.63 [0.41, 0.95] |
| −7.58 to −6.50 | 7.4–9.5 kpc: 0.75 [0.52, 1.07] | 9.5–150 kpc: 0.41 [0.28, 0.61] |
| −6.50 to −0.96 | 7.8–21.5 kpc: 0.44 [0.31, 0.62] | 21.5–150 kpc: 0.56 [0.27, 1.22] |

**Populations, phase likelihood:**

- in-situ: 0.31 (5.7–7.6 kpc), 0.26 (7.6–8.8 kpc), 0.18 (8.8–150 kpc);
- accreted: 0.74 (10–18 kpc), 0.91 (18–27 kpc), 0.27 (27–150 kpc).

The Khoperskov estimates lie within the phase-likelihood intervals for T2 in most bins. For the
accreted clusters Khoperskov gives values above 1 beyond 18 kpc (T2: 2.26 and 1.45), while the
phase likelihood stays below 1.

**Symmetry, p-values (null simulations with refitting):**

| Sample | z | y | \|azimuth\| |
|---|---|---|---|
| All clusters | 0.46 | 0.04 | 0.006 |
| In-situ | 0.04 | 0.29 | 0.006 |
| Accreted | 0.51 | 0.49 | 0.75 |

## Completeness as a function of apparent magnitude, C(V) (`cv.py`, `scripts/run_cv.py`, `configs/cv.yaml`)

### Method

- Per-cluster phase likelihood fitted to all 165 clusters at once, with C piecewise constant in
  8 equal-count bins of V = M_V + 5 log10(d / 10 pc). No extinction.
- Anchor: the brightest-V bin (V < 6.0) is complete.
- Corrected counts: weight 1/q_i, q_i = ⟨C(V_i(t))⟩_t.
- Uncertainties: profile intervals for C; cluster bootstrap (200 replicates) for corrected counts.

### Mock validation (100 realisations per scenario; true C = logistic in V with V_lim = 8)

| Scenario | C(V) shape, recovered/true (coverage) | Corrected total / true (16–84%) | Faint (M_V > −6.5) corrected / true | Faint observed / true |
|---|---|---|---|---|
| A: C(V) true | 0.92 (0.69) | 0.80 (0.60–1.07) | 0.44 | 0.15 |
| B: latitude term | 1.08 | 0.56 | 0.27 | 0.12 |
| C: Sgr-like clump | 1.01 | 0.71 | 0.42 | 0.15 |
| D: C not a function of V | 1.08 | 0.87 | 0.73 | 0.64 |

- The corrected totals are LOW, because faint GCs on orbits that never enter the detectable volume
  are missing from the orbit library and cannot be reweighted.
- ΔAIC > 2 favours the free per-class C(d) fits in only 31% of D mocks (13–24% in the others), so
  the model check has little power.

### Real data (`outputs/cv/real_cv*.csv|json`, `plots/fig61`, `plots/fig62`)

| V bin | < 6.0 | 6.0–6.6 | 6.6–7.2 | 7.2–7.6 | 7.6–8.2 | 8.2–9.2 | 9.2–11.0 | > 11.0 |
|---|---|---|---|---|---|---|---|---|
| C | 1 (anchor) | 1.17 | 0.56 | 0.64 | 0.41 | 0.25 | 0.21 | 0.14 [0.07–0.30] |

- **Corrected totals:** 483 GCs [68%: 335–765] from 165 observed; faint (M_V > −6.5) 267
  [168–473] from 55 observed (cluster bootstrap after the fix that resamples, refits and counts
  the same clusters).
- **Model comparison:** AIC(C(V)) = −6.0 against −1.5 for the free per-class C(d) fits.

## Extinction (`dust.py`, `run_cv.py --extinction ...`, `configs/dustmapsrc.json`)

### Method

- **3D maps along orbits** (`--extinction map_low|map_high`). A_V is evaluated at every one of the 165×500
  orbit samples with `dustmaps`; map data live in `~/data/dustmaps`, selected through
  `configs/dustmapsrc.json`. Maps by priority:
  - DECaPS 3D mean (Zucker+2025; 239° < l < 6°, |b| < 10°, d ≤ 10 kpc), A_V = 3.1 E(B−V);
  - Bayestar19 (where flagged reliable), A_V = 3.1 × 0.883 × value;
  - Marshall+2006 (within each sightline's last distance slice), A_V = 8.8 A_Ks;
  - SFD, A_V = 2.742 E(B−V)_SFD.
- **Brackets.** Beyond a map's reliable reach, the LOW bracket uses the last 3D value and the HIGH
  bracket the SFD total of the sightline. Results are cached in `outputs/cache/dust_<hash>.npz`.
- **Source fractions over orbit samples:** DECaPS 24%, Marshall 13%, Bayestar19 8%, SFD/bracket 55%.
  At present positions the maps give 5–8% less A_V than 3.1 E(B−V)_catalogue (median ratio 0.92
  for LOW, 0.95 for HIGH, over 144 clusters with catalogue A_V > 0.1).
- **Present reddening held fixed** (`--extinction present_ebv`): a test only, since it is unphysical.
- **Proxies** (`cv.fit_cv_sep`): C = C_V(V)·g(x), with x = −|b| or the 2D SFD E(B−V) at each orbit
  sample, in 3 equal-count x bins, the least obscured anchored to 1. They are fitted on the real data
  (stored in `real_cv_summary.json`) and on mocks (`proxy_tests: true`). Mock scenario E is an
  exponential dust disc (`dust.disc_extinction`, a0 = 1 mag/kpc, h_R = 3 kpc, h_z = 0.1 kpc).

### Mocks (corrected total / true, median of 100; faint in brackets)

| Scenario | C(V) | C_V·g(\|b\|) | C_V·g(2D column) | AIC prefers proxy |
|---|---|---|---|---|
| A | 0.80 (0.44) | 0.83 (0.43) | 0.82 (0.41) | 15% / 12% |
| B | 0.56 (0.27) | 0.64 (0.31) | 0.65 (0.31) | 30% / 30% |
| E, dust ignored | 0.51 (0.16) | 0.68 (0.19) | 0.69 (0.19) | 69% / 71% |
| E, true A_V supplied | 0.61 (0.18) | – | – | true dust preferred in 98% |

### Real data

| Model | AIC | Corrected total [68%] | Faint (M_V > −6.5) |
|---|---|---|---|
| No extinction | −6.0 | 483 [335–765] | 267 [168–473] |
| Present E(B−V) fixed | +1.3 | 689 [457–1172] | 337 [203–651] |
| Proxy g(SFD), g = [1, 1.26, 0.75] | −7.5 | 462 [346–668] | 251 [169–403] |
| Proxy g(\|b\|), g = [1, 0.94, 0.49] | −10.4 | 520 [400–753] | 269 [184–432] |
| 3D maps, LOW | −13.7 | 455 [354–634] | 218 [158–327] |
| 3D maps, HIGH (best) | −14.4 | 411 [319–566] | 197 [140–291] |

- **Correction factor by metallicity with maps:** [Fe/H] < −1 goes from ×3.1 to ×2.4, while
  [Fe/H] ≥ −1 stays at ×2.6.
- **Figure:** `plots/paper_corrected_map_high.png`.
