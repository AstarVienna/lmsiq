# Extraction analysis: lmsiq to the METIS pipeline

What of lmsiq can be extracted into the METIS data-reduction pipeline (`pymetis`) LMS recipes as QC parameters or diagnostic plots/reports.

Analysis date 2026-09-17, by the METIS pipeline team (A* Vienna). The master copy lives in the pipeline team's project notes; this page is the version shipped with the lmsiq documentation, edited only to drop machine-local paths. Question asked: how much of Alistair Glasse's `lmsiq` package
(LMS image-quality and AIT performance-test analysis) can be extracted and added to the
LMS recipes in `METIS_Pipeline/metisp/pymetis/src/pymetis` as QC parameters or diagnostic
plots/reports.

Sources read: this repository at `1547ea4` (2026-09-07, "LMS-OPT-01 v1.41"; the
AstarVienna repo is a fork of `AGlasse/lmsiq` and is level with it),
`METIS_Pipeline` main at `4d5a033c` (2026-09-14), `METIS_DRLD` master,
`irdb/METIS`, and the AIT performance-test mirror. All statements about code are
from reading it, not running it. Line numbers refer to those checkouts.

---

## 1. Verdict in one page

**Roughly 15 % of lmsiq by volume is science content that transfers; the rest is scaffolding
for Zemax design analysis and for driving a toy simulator.** Of the ~10 000 lines, about
1 500 lines carry algorithms that are meaningful on real or ScopeSim-simulated LMS detector
frames. None of it can be imported as-is; every candidate needs to be rewritten as a pure
function because the package has hidden class-level state, prints instead of returning,
calls `plt.show()` inside compute loops, and reads hard-coded relative paths.

What transfers splits into three tiers:

| Tier | What | Where it lands | Effort |
|---|---|---|---|
| **1. Scalar QC, existing recipes** | Slice-edge positions and inter-slice gaps, illuminated FOV and aspect ratio, laser-line FWHM → spectral resolving power against MET-2745 / MET-3739, inter-detector gap and relative rotation of each detector pair, dark statistics | `metis_ifu_distortion`, `metis_ifu_wavecal`, `metis_det_dark` | 2–4 weeks total |
| **2. Table products and diagnostics** | Slice map image, per-slice trace tilt table, relative spectral response image, enslitted-energy curves and along/across-slice FWHM of a point source, trace-overlay and profile plots | New table extensions on `IFU_DISTORTION_TABLE` / `IFU_WAVECAL_TAB`; a new ADARI report plugin | 4–8 weeks |
| **3. New infrastructure** | The E-TNT-ATC-MET-1058 distortion transform (4×4 matrices per slice and order, forward and inverse, focal plane ↔ detector), its prism/echelle-angle dependence, and the pinhole (iso-alpha) × laser (iso-lambda) intersection method that measures it | A new `IFU_DISTORTION_TABLE` schema or separate product, new `drl` module, pinhole-frame handling in `metis_ifu_distortion` | 2–3 months, and a design decision with the DRLD owners |

**Not extractable**: the `lmsiq` program proper (Zemax perfect/design/Monte-Carlo PSF-stack
ingestion, MC statistics, relative "Strehl" against a perfect model), the detector
diffusion/IPC forward model, ToySim, and the LMS-OPT-03/04/09 modules, which are stubs that
do not run.

Two facts shape everything else:

1. **The pipeline produces no plots or reports at all today.** `adari_core` is a declared
   dependency and the EDPS config sets `genreport_path=genreport`, but no METIS report
   plugin exists. Any "diagnostic plot" from lmsiq needs that plugin to be created first.
2. **The pipeline's LMS geometry model is per-slice 1-D** (`y = P(x)` mid-line plus a tilt
   polynomial of degree ≤ 2). It has no focal-plane ↔ detector transform, no along-slice
   spatial solution ξ(x, y), no knowledge of the prism or echelle angle, and no
   nominal/extended distinction. lmsiq's distortion machinery is built around exactly
   those things, which is why it is Tier 3 rather than Tier 1.

---

## 2. What lmsiq is

Four "programs" (README), each a script that runs on import with `src/` as the working
directory. Git history: 58 commits 2022-12 → 2026-09, single author (STFC UK ATC).
No tests, no CI, no data or notebooks in the repo (`/data/` is git-ignored; every input
lives in sibling `../data`, `../config`, `../output` trees that are **absent from the clone**).

| Program | Entry | Purpose | State at HEAD |
|---|---|---|---|
| `lmsiq` | `lmsiq.py` → `lmsiq_cuber.Cuber.build` | Ingest Zemax PSF stacks (perfect / as-designed / Monte-Carlo), compute LSF FWHM, resolving power, enslitted energy, along/across-slice FWHM, relative Strehl vs wavelength and field | Broken: calls `Filer.set_configuration` with two arguments where the signature now takes three (`lmsiq.py:88`, `lms_filer.py:27`). Active 2023–2024, then superseded in the author's attention. |
| `lmsdist` | `lmsdist.py` | Fit the E-TNT-ATC-MET-1058 distortion transforms from Zemax ray traces; fit their dependence on prism and echelle angle; write `lms_efp_mfp_*.fits` | Runs (given the Zemax CSVs). Docstring: "This is the data product we provide to ScopeSim and the pipeline." |
| `lmssim` | `lmssim.py` → `lmssim_toy.Toy` or `lmssim_scope.Scope` | Generate simulated performance-test frames, either with a bespoke transform-driven "ToySim" or via ScopeSim `wcu_lms` | ToySim path works; ScopeSim path uses configuration keys the observation map never produces and would fail. |
| `lmsaiv` | `lmsaiv.py` → `lmsaiv_opt_tests.OptTests.run` | Analysis of AIT tests LMS-OPT-01 (FOV, slices, RRF), LMS-OPT-02 (distortion, traces), stubs for 03/04/09 | OPT-01 works and is the subject of the last ten commits; OPT-02 runs only on ToySim output; 03/04/09 do not run. |

The AIT mirror (`metis-ait-performance-testing-mirror/LMS/LMS_OPT/lms_opt_01_fov.ipynb`,
as-run HTML pushed 2026-09-16) is the live consumer: it imports `lmsaiv_opt01`,
`lmsaiv_opt_tools`, `lms_filer`, `lms_mosaic` from a sibling lmsiq checkout and analyses
data in `data/test_toysim`.

### 2.1 Relationship to the irdb and ScopeSim

The irdb's `METIS/TRACE_LMS.fits` is a product of the `lmsdist` lineage. Its `WCAL`
extension is authored "Alistair Glasse" (wavelength → echelle angle per order, linear), and
its `Polynomial coefficients` extension (37 632 rows = 28 slices × 21 orders × 4 matrices ×
16 terms) holds the A, B, AI, BI transform terms as cubic polynomials in echelle angle.
ScopeSim's `MetisLMSSpectralTrace.get_matrices` evaluates exactly that. So **every
simulated LMS frame the pipeline has ever seen was generated from Glasse's transform
model**, and the pipeline could read the same file as a reference transform for a
"measured vs design" QC without any new data delivery. The current lmsiq fits a richer
form (bivariate in prism and echelle angle, 10 coefficients per term) and writes a
different FITS layout (`lms_efp_mfp_*.fits`: one `SLICE_n` table per slice with columns
`A, B, AI, BI`, float32). The two layouts would need reconciling.

### 2.2 Constants that disagree with the irdb

| Quantity | lmsiq `lms_globals.py` | irdb `METIS_LMS.yaml` / `FPA_metis_lms_layout.dat` |
|---|---|---|
| Along-slice pixel scale | 8.7 mas (`alpha_pix`, l.34) | 8.2 mas (`pixel_scale: 0.0082`) |
| Slice width | 20.7 mas | 20.7 mas |
| Detector gap in mosaic | 3.0 mm (`det_gap`, l.122) | 2.23 mm (from `x_cen = ±19.547` mm) |

lmsiq's own `lms_detector.py` uses 2.23 mm, so the 3.0 mm in `Globals` is an internal
inconsistency that feeds the affine transforms and the FOV calculation. Anything ported
must take these from the pipeline's instrument description, not from lmsiq.

---

## 3. The pipeline side, as it stands

### 3.1 QC mechanism

A QC parameter is a subclass of `QcParameter` (`engine/qc/parameter.py`) declared inside
the recipe's `Impl.Qc` class, with `_name_template` (e.g. `"QC IFU WAVECAL RMS"`), `_type`
(`int`/`float`), `_unit`, `_description_template`. The recipe instantiates it with a plain
Python scalar, `collect_qc_parameters()` turns the set into a `cpl.core.PropertyList`, and
the recipe appends that to a product header by hand. Constraints that matter here:

- Scalars only in practice. `list` maps to `cpl.core.Type.ARRAY` but nothing uses it.
- Strict `isinstance`: `numpy.int64` fails for `int`; wrap with `int()`.
- Indexed names (`POLYCOEFF{n}`) are declared in places but never instantiated; there is
  no working pattern for one keyword per slice or per coefficient.
- **Per-slice or per-detector vectors go into a table product, not QC.** The precedent is
  `IFU_WAVECAL_TAB`, added specifically to carry per-slice coefficients, RMS and a
  fallback flag that the scalar QC could not hold.
- Uncomputable QC values are omitted, not filled with sentinels.

### 3.2 State of the IFU recipes

| Recipe | Real? | QC attached today | DRLD QC not yet attached |
|---|---|---|---|
| `metis_ifu_distortion` | yes: stacks continuum (mask `open`) frames, `drl.trace.trace`, `measure_trace_edges`, writes `IFU_DISTORTION_TABLE` (`slice_nb, trace_nb, column_range, pos, bottom, top` per detector) | `QC IFU DISTORT NTRACES` (not in DRLD) | `DISTORT RMS` (computed as `Trace.residual`, never exported), `DISTORT FWHM` (`measure_trace_fwhm` exists, never called), `DISTORT NSPOTS`. Pinhole frames are ignored (TODO at l.204-206); `PINHOLE_TABLE` input dropped. |
| `metis_ifu_wavecal` | yes: per-slice line detection, tilt fit (degree ≤ 2), 1-D wavelength solution, `IFU_WAVECAL` map + `IFU_WAVECAL_TAB` | `NLINES, RMS, PEAK CNTS, LINE WIDTH` | none; but individual `Line` objects (position, FWHM, height) are not persisted |
| `metis_ifu_rsrf` | partial: background, Planck division at hard-coded 800 K, crude 1-D extraction | raw `QC IFU RSRF NBADPIX` property, bypassing the QC classes | – |
| `metis_ifu_reduce` | skeleton (averages raws) | none computed | `STD FWHM`, `STD ELLIP`, `REDUCE NBADPIX/MEANBKG/MEANSTRY` |
| `calibrate`, `telluric`, `postprocess` | skeletons | none computed | all |

Header keywords the IFU recipes read: `INS OPTI20 POSNAME`, `INS WLEN CEN`,
`SEQ/INS WCU LASERn WLEN`. Nothing about prism/echelle angle (`INS.OPTI6/7/8` in the DRLD
operations templates) or nominal vs extended mode.

### 3.3 DRL modules relevant to the mapping

- `drl/trace.py`: slice tracing, `Trace` dataclass (`pos`, `bottom`, `top`, `height`,
  `residual`; `slice` is "simply set equal to m", i.e. **no stable physical slice
  numbering**), `measure_trace_edges`, `measure_trace_fwhm`, `compute_heights`.
- `drl/wavecal.py`: `SliceSolution` (wavelength coefficients, tilt coefficients, `rms`,
  `lines`, `fallback`), `build_wavelength_map`.
- `drl/lines.py`: `detect_lines`, `fit_gaussian` with baseline, smoothing deconvolved
  from width. This already does what lmsiq's `_fit_gaussian` does, better.
- `drl/rectify.py`: `rectify_trace`, `build_cube` (not yet wired into any recipe).
- `engine/core/functions/polyfit2d.py`: bivariate least squares, currently used only for
  the tilt.
- Tests: 322 pytest functions, DRL tests on synthetic 2048² frames, no FITS fixtures for
  full-size LMS data.

### 3.4 Reporting

Nothing. Comment in `drl/trace.py:34`: "All plotting and interactive cluster merging is
removed. A CPL recipe must never block on standard input." The ESO route is an ADARI
`genreport` plugin package keyed on `PRO.CATG`; EDPS is already configured to call it.

---

## 4. Mapping lmsiq → pipeline, by function

### 4.1 Tier 1: scalar QC parameters into existing recipes

| # | lmsiq source | Computes | Target recipe / QC name (proposed) | What the recipe already has | Port effort |
|---|---|---|---|---|---|
| 1 | `lmsaiv_opt01._find_slices` (l.147-257), `_find_gaps` (l.84-115) | Sub-pixel slice edges from 50 %-of-peak crossings of 9-column mean profiles at 4 columns per detector; inter-slice gap in pixels per slice pair (design value 20 px) | `metis_ifu_distortion`: `QC IFU DISTORT GAP MIN`, `GAP MEAN`, `GAP RMS` (pixels) | `measure_trace_edges` finds the same half-max edges over 9 column bands and `compute_heights` the spacing. Only the gap statistic and its comparison to design are new. | S |
| 2 | `lmsaiv_opt01._make_slice_map` + `_calculate_fov` (l.260-304), `flood_stats` (l.437-445) | Illuminated FOV α × β in arcsec, aspect ratio; requirement METIS-3667 (> 0.5 arcsec², aspect between 1:1 and 2:1) | `metis_ifu_distortion`: `QC IFU DISTORT FOV ALPHA`, `FOV BETA` (arcsec), `NSLICES` | Needs slice count and pixel scale from the instrument description (8.2 mas, 20.7 mas), not from lmsiq. Note lmsiq divides by 2 because each slice appears on two detectors; the pipeline's per-detector loop needs the same care. | S |
| 3 | `lmsaiv_opt02._find_detector_gaps` (l.380-458) | From laser lines seen on both detectors of a pair: inter-detector gap in pixels and the relative rotation θ₁₂, θ₃₄ ± error | `metis_ifu_wavecal`: `QC IFU WAVECAL DETGAP 12`, `DETGAP 34` (pixels), `DETROT 12`, `DETROT 34` (deg) | Wavecal has per-slice `SliceSolution` per detector with identified laser lines and wavelengths. Crossing detector 1→2 requires matching slices across detectors, which the pipeline cannot do today (no physical slice ID). | M, blocked on slice identity |
| 4 | `lmsiq_analyse.lsf` + `Cuber.plot:380` SRP formula; thresholds in `lmsiq_plot.series` (l.398-415) | Spectral resolving power R = λ / (FWHM_pix · dλ/dpix); MET-2745 (R ≥ 1e5, 0.85e5 above 4.8 µm) and MET-3739 (FWHM ≤ 2.0 pix) | `metis_ifu_wavecal`: `QC IFU WAVECAL RESPOW` (median R), `RESPOW MIN` | Wavecal already has `Line.fwhm` per line and the dispersion from the solution; `LINE WIDTH` is the median FWHM. Only the division and thresholds are new. | S |
| 5 | `lmsaiv_opt02._find_beta_phase` (l.329-377) | Across-slice position of a point source from a Gaussian fit of per-slice flux vs slice number | `metis_ifu_distortion` (pinhole frames) or `metis_ifu_reduce` (STD): `QC IFU STD BETA PHASE` | Requires per-slice fluxes of a point source, i.e. pinhole-frame handling that the distortion recipe currently skips. | S once pinholes are handled |
| 6 | `lmsaiv_opt_tools.dark_stats` (l.45-69) | Per-detector median, σ, dark current el/s, read noise el using DIT, NDIT, gain | `metis_det_dark` | Already covered by `QC DARK MEAN/MEDIAN/RMS` etc. Nothing to add except the unit conversion to el/s and electrons, which is trivial. | none |

Items 1, 2 and 4 are the immediate wins. They need no new inputs, no new products, and the
DRL already computes the underlying quantities. Item 2 is the only one that delivers a
requirement check (METIS-3667) the pipeline does not otherwise cover.

### 4.2 Tier 2: table products and diagnostic outputs

| # | lmsiq source | Computes | Proposed home | Effort |
|---|---|---|---|---|
| 7 | `_make_slice_map` (l.260-287) | Image with pixel value = slice number, 0 = unilluminated, from quadratic edge fits vs column | New image extension on `IFU_DIST_REDUCED` or a `SLICE_MAP` column set. Also useful downstream: `metis_ifu_rsrf` and `reduce` currently rebuild slice bands from the trace table each time. | S |
| 8 | `_get_trace_coordinates` + cubic fit with 2σ clipping + `_find_thetas` (opt02 l.548-661, opt_tools l.83-98) | Gaussian-centroided trace positions, cubic fit, tilt angle θ ± error per slice | Extra columns `theta`, `theta_err`, `residual` on `IFU_DISTORTION_TABLE`. `Trace.residual` already exists and is the DRLD's `DISTORT RMS`. | S |
| 9 | `_find_rrf` (l.118-144) | Relative response = flood / Planck(T) per pixel | `metis_ifu_rsrf` does this already, with a real wavelength map instead of lmsiq's admitted "very approximate" linear dispersion. Nothing to take except the idea of storing the 2-D RRF image, which `MASTER_FLAT_IFU` already is. | none |
| 10 | `lmsiq_analyse.eed` (l.294-381), `find_ee_axis_references` (l.384-408) | Enslitted energy vs aperture half-width along and across slice; EE at reference half-width 2 px × max(1, λ/3.7 µm) | `metis_ifu_reduce` STD branch, on a reconstructed cube plane: `QC IFU STD EE` scalar plus an EE-curve table. Depends on `drl.rectify.build_cube` being wired in first. | M |
| 11 | `lmsiq_analyse.lsf` on a cube plane (`Cuber.build` l.200-231) | Along-slice FWHM (pixels) and across-slice FWHM (slices) of a point source | `metis_ifu_reduce`: the DRLD's `QC IFU STD FWHM` and `STD ELLIP`, currently absent. | M, same dependency |
| 12 | `lmsiq_analyse.exact_rectangular` (l.547-595) | Fractional-pixel rectangular aperture sum | Helper in `drl`; used by 10 and 11. Pure numpy. | S |
| 13 | `lmsaiv_plot.mosaic/profiles/det_traces/slice_gaps/histograms`, `lmsdist_trace.plot_scatter/plot_fit_maps` | Four-detector overview with trace overlays; column profiles with detected edges; gap vs slice; residual scatter and quiver maps | ADARI report plugin, one page per `PRO.CATG` (`IFU_DISTORTION_TABLE`, `IFU_WAVECAL_TAB`, `MASTER_DARK_IFU`). Every function must be rewritten to take arrays and return a `Figure`. | M for the plugin skeleton, S per plot |

### 4.3 Tier 3: the distortion transform

This is the intellectually valuable part of lmsiq and the part with no pipeline
counterpart.

**Model** (`lmsdist_util.py`): wavelength enters as phase `φ = λ·n / (2·d)` with
`d = 18.2 µm × 4.05` (immersed germanium grating). For each (slice, echelle order) a 4×4
matrix pair maps `(φ, α)` → `(x_mfp, y_mfp)` as
`x = Σ A[i,j] α^i φ^j`, `y = Σ B[i,j] α^i φ^j` (cubic, upper triangle), with inverse
matrices AI, BI. A per-detector affine (`Transform.create_affines`) maps mosaic
millimetres to pixels. The matrices themselves are fitted as bivariate polynomials in
(prism angle, echelle angle) so a transform can be built for any setting
(`lmsdist_polyfit.PolyFit`, 10 coefficients per term). Solver is SVD least squares
(`solve_svd_distortion`, "converted from Ronayette's IDL, see MIRI calibration").

**Measurement** (`lmsaiv_opt02`): iso-alpha traces from pinhole frames (WCU pinhole + BB,
known field position from `CFO CHOP X/Y`), iso-lambda traces from tunable-laser flood
frames (known λ). Fixed-point intersection of the two trace families yields fiducial
points `(det, row, col) ↔ (α, β, λ)` (`_find_trace_intersections`, l.664-726), which are
written to a CSV and fitted by the same SVD solver. Comparing fitted to Zemax matrices
term by term gives a "measured vs design" fractional difference table.

**Why it does not drop in:**

- The pipeline has nothing to hold it. `IFU_DISTORTION_TABLE` is per-slice `y = P(x)`.
  The DRLD says the table holds "distortion coefficients" without specifying columns, and
  describes the goal as `ξ = f_i(x,y)`, `λ = g_i(x,y)` "low-order polynomials", so a
  4×4-matrix-per-slice-per-order layout is compatible in spirit but is a schema decision
  to be made with the DRLD owners.
- The recipe has no prism/echelle angle awareness and no order identification. The
  transform is indexed by echelle order; the pipeline's traces are indexed by bottom-up
  rank on one detector.
- The measurement needs the pinhole frames the recipe currently discards, plus a
  cross-detector slice identity (same problem as Tier 1 item 3).
- lmsiq's header contract for this (`HIERARCH AIT PRI_ANG`, `AIT ECH_ANG`,
  `ACHG CFO CHOP X/Y`, `ACHG LASER WAVE`, `ACHG REF_ECH_ORD`) is written only by ToySim.
  ScopeSim output and, presumably, real ICS headers use `INS.OPTI7/8` and the
  `SEQ WCU LASERn WLEN` keys the wavecal recipe already reads. A keyword mapping is
  required before any of Opt02 runs on pipeline data.

**What is still worth taking now, at library level** (`drl/lms_transform.py`, say):

| Piece | Source | Why |
|---|---|---|
| `waves_to_phases`, `apply_svd_distortion`, `solve_svd_distortion` | `lmsdist_util.py` l.45-90, 720-774 | Pure numpy. Gives the pipeline the ability to *read and evaluate* the irdb transform and, later, to fit it. |
| `Transform` reader/writer for `lms_efp_mfp_*.fits`, and a reader for the irdb `TRACE_LMS.fits` polynomial layout | `lms_transform.py` l.69-150 | Lets a recipe predict where slice n, order m, wavelength λ falls on the detector for the header's setting. That prediction is the "model" in the DRLD's own definition of `QC IFU DISTORT RMS`: "RMS deviation between measured position and model". Today the pipeline measures that residual against its own polynomial, not against a model. |
| `PolyFit.wave_to_config`, `make_fit_matrix` | `lmsdist_polyfit.py` l.29-228 | λ → (prism angle, echelle angle, order) and back. Would let `metis_ifu_wavecal` derive its first-guess dispersion per slice and order from the setting instead of `INS WLEN CEN ± 0.036 µm` per detector with a hard-coded fallback. |
| `find_wavelength_bounds` | `lmsdist_trace.py` l.130-150 | Per-slice λ_min/λ_max for a setting; a natural `IFU_WAVECAL_TAB` column and a check against the header. |

Doing this gives two new DRLD-conformant QC values almost for free once the reader exists:
`QC IFU DISTORT RMS` measured against the design transform, and a wavecal "predicted vs
measured central wavelength" per detector.

### 4.4 Not extractable

- `lmsiq_image_manager`, `lmsiq_cuber`, `lmsiq_summariser`, `lmsiq_fitsio`, `lmsiq_phase`:
  Zemax folder conventions, perfect/design/Monte-Carlo bookkeeping, pixel-phase study by
  synthetic shifting of model PSFs.
- `find_strehls`: peak-to-total of a 1-D Gaussian normalised to a perfect-model image in
  slot 0. Not a Strehl and undefined without the model.
- `lms_ipc`, `lms_detector.detect`, `lmssim_*`: forward models. ScopeSim already covers
  this, and `Detector.detect` draws Poisson noise from √signal, which is wrong.
- `lmsaiv_opt03/04/09`: no working algorithm. `lms_wcal`: legacy, superseded by
  `PolyFit` and by the irdb `WCAL` table. `lms_baffle`, `lms_test`, `lmsiq_test`: scripts
  for one-off studies, some referencing attributes that no longer exist.

---

## 5. Recommended shape of the work

1. **Immediate (Tier 1, items 1, 2, 4; plus exporting the two DRLD QC the DRL already
   computes).** Add `QC IFU DISTORT RMS` and `DISTORT FWHM` from `Trace.residual` and
   `measure_trace_fwhm`; add gap and FOV statistics from the existing edge measurements;
   add resolving power to wavecal. No new inputs, no schema change, ~1 week each including
   tests against the synthetic-frame fixtures. This closes three DRLD gaps and adds the
   METIS-3667 and MET-2745 checks that AIT will ask for.
2. **Report plugin skeleton.** Create the ADARI `genreport` package with one page each for
   `IFU_DISTORTION_TABLE` and `IFU_WAVECAL_TAB`: four-detector mosaic with trace overlay,
   edge profiles, gap vs slice, line FWHM vs wavelength with the MET-2745 line. Rewrite
   the lmsiq plot functions as `Figure`-returning helpers. This is where the AIT team's
   notebook output becomes a pipeline product.
3. **Slice identity.** Before anything in Tier 3 or item 3, the pipeline needs to label
   traces with a physical slice number consistent across detectors 1/2 and 3/4. lmsiq's
   `slice_order` convention (detectors 1/2 carry slices 15–28, 3/4 carry 1–14, in the
   nominal mode) plus the irdb aperture list is enough to do this deterministically once
   the recipe knows which detector it is on. This is a small change with large leverage.
4. **Transform library (Tier 3, library level).** Port the pure numpy transform code and
   write a reader for the irdb `TRACE_LMS.fits`. Use it first only to *predict*, giving
   model-based `DISTORT RMS` and a setting-aware wavecal first guess. Fitting measured
   transforms from pinhole and laser frames comes after and needs the DRLD schema decision
   and a header-keyword contract with the ICS/AIT teams.
5. **Point-source metrics (Tier 2, items 10–11)** wait for `metis_ifu_reduce` to build
   cubes.

---

## 6. Caveats and open questions

- **Authorship and licence.** lmsiq has no licence file. It is STFC/UK ATC work by
  Alistair Glasse. Porting algorithms into the ESO-delivered pipeline needs his agreement
  and attribution; the cleanest route is a co-authored `drl` module rather than copying.
- **Two sources of truth for constants.** lmsiq's 8.7 mas and 3.0 mm differ from the irdb
  (8.2 mas, 2.23 mm). Which is current should be checked with the author; E-REP-ATC-MET-1003
  is cited by both.
- **Header keyword contract.** lmsiq's AIT/ACHG keywords vs the DRLD's `INS.OPTI6/7/8`
  and `SEQ WCU LASERn WLEN`. Neither the pipeline nor ScopeSim writes the lmsiq keys.
- **Slice numbering in extended mode.** lmsiq's `Globals.fslice_no_range` (12–14) and
  `Opt01.slice_order` (11–13) disagree; the pipeline has no extended mode at all.
- **Float32 matrices.** `lms_efp_mfp_*.fits` stores transform terms as `E` (float32). For
  cubic terms over 2048 pixels this loses precision at the 0.1 px level the DRLD asks for.
  Any pipeline product should use `D`.
- **The DRLD does not know about the AIT tests.** No reference to LMS-OPT-xx, lmsiq, or
  E-TNT-ATC-MET-1058 anywhere in the LaTeX. The QC list for `metis_ifu_distortion` (RMS,
  FWHM, NSPOTS) is pinhole-spot language, matching lmsiq's Opt02 rather than the
  continuum-trace approach the recipe currently takes. Adding gap, FOV and transform QC
  means a DRLD change request.

---

## 7. Effort summary

| Work package | Content | Estimate | Blocks on |
|---|---|---|---|
| A | Export `DISTORT RMS`, `DISTORT FWHM`; add gap and FOV QC; add resolving power QC | 2–3 weeks | nothing |
| B | ADARI report plugin with distortion + wavecal pages, lmsiq plots rewritten | 3–4 weeks | nothing (parallel to A) |
| C | Physical slice numbering across detectors; extra `IFU_DISTORTION_TABLE` columns (θ, residual) | 1–2 weeks | nothing |
| D | Transform library: evaluate irdb/lmsdist transforms, model-based `DISTORT RMS`, setting-aware wavecal first guess, inter-detector gap/rotation QC | 4–6 weeks | C; header keyword contract |
| E | Pinhole-frame handling in `metis_ifu_distortion`, iso-alpha × iso-lambda fiducials, transform fitting, measured-vs-design QC | 6–8 weeks | D; DRLD schema decision; agreement with author |
| F | STD point-source metrics (EE, FWHM, ellipticity) | 2–3 weeks | `metis_ifu_reduce` cube building |

Totals: A+B+C ≈ 2 months delivers everything in Tiers 1 and 2 that is unblocked.
D+E is the distortion transform proper and is a project in its own right.
