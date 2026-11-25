# The four programs

All four programs are plain Python scripts in `src/`. They import each other by
bare module name (`from lms_globals import Globals`), so they must be run **with
`src/` as the working directory** (or with `src/` on `sys.path`, as the AIT
notebooks do with `sys.path.insert(0, '../src')`). `setup.py` declares
`packages=['src']` but the code is not importable as `lmsiq.<module>`.

Every program expects its inputs under `../data/` and `../config/` and writes to
`../output/`, all relative to `src/`. **None of that data is in the repository**
(`/data/` is git-ignored). See {doc}`data_layout` for what has to be present.

Requirements: Python 3.10 or later (`match`/`case` in `lmsdist_trace`,
dictionary union in `lmsdist_util`), numpy, scipy, astropy, matplotlib,
photutils; ScopeSim and synphot for `lmssim`. `requirements.txt` in the repo
root is a frozen environment dump (numpy 2.3, ScopeSim 0.11.2) with many
unrelated packages.

## `lmsiq`: image quality from Zemax PSF stacks

```
cd src
python lmsiq.py
```

`lmsiq.py` selects a Zemax data set by a ten-digit *date stamp*
(`iq_date_stamp = '2024073000'` at the top of the file) and drives
{mod}`lmsiq_cuber` to:

1. load, for every (configuration = wavelength setting, field point, defocus,
   slice) the stack `[perfect, as-designed, MC_0000, MC_0001, ...]` of 128 x 128
   Zemax PSF images (4.5 um pixels, four times oversampled with respect to the
   18 um H2RG pixel);
2. apply the detector diffusion / inter-pixel-capacitance kernel
   ({mod}`lms_ipc`) and rebin to detector pixels ({mod}`lms_detector`);
3. measure on the target slice the spectral line-spread function and its FWHM
   ({meth}`lmsiq_analyse.Analyse.lsf`), the local dispersion from the
   `lmsdist` ray trace, and hence the spectral resolving power;
4. assemble a pseudo alpha x beta cube across neighbouring slices and measure
   enslitted energy ({meth}`lmsiq_analyse.Analyse.eed`), along- and
   across-slice FWHM and a relative Strehl
   ({meth}`lmsiq_analyse.Analyse.find_strehls`);
5. plot everything against wavelength per field, with the requirement lines
   MET-2745 (resolving power) and MET-3739 (spectral FWHM) overlaid
   ({meth}`lmsiq_plot.Plot.series`).

Status at the documented commit (`9b6143a`): the two-argument
`Filer.set_configuration` calls that stopped the driver at `1547ea4` have been
fixed upstream (`is_ait_data=False` is now passed). Whether it runs end to end
has not been checked here, because the Zemax PSF data are not in the repository.
The image-quality code was last substantively touched in February 2026 and has
been superseded in the author's attention by `lmsaiv`.

## `lmsdist`: distortion transforms

```
cd src
python lmsdist.py
```

The run configuration is hard-coded at the top of {func}`lmsdist.run`; at
`9b6143a` it processes the **extended** (spectral-IFU) mode Zemax data
(`opticon = Globals.extended`, `is_ait_data = False`).

Reads Zemax ray-trace CSV files (one per prism/echelle setting) and for every
(slice, echelle order) fits a pair of 4 x 4 polynomial matrices mapping
(phase, alpha) in the entrance focal plane to (x, y) in the detector mosaic
plane, plus the inverse pair ({meth}`lmsdist_util.Util.solve_svd_distortion`).
Wavelength enters as the grating *phase* `lambda * order / (2 d)`. The matrix
terms are then fitted as bivariate polynomials in prism and echelle angle
({class}`lmsdist_polyfit.PolyFit`) so that a transform can be constructed for
any setting, and the boresight wavelength as a function of the two angles is
inverted to give `wave_to_config`.

Products, written to `../output/distortion/<opticon>/...`:

- `lms_efp_mfp_<nom|ext>_pa<...>_ea<...>_v<date>.fits`: one binary table per
  slice (`SLICE_n`) with columns `A, B, AI, BI` (16 terms each).
- `lms_dist_efp_mfp_<nom>_fit_parameters_v<date>.fits`: the angle-dependence
  fits.
- `lms_dist_mfp_dfp_v<date>.fits`: the per-detector affine transforms.
- pickles of the ray traces and an index of available transforms.

The irdb's `METIS/TRACE_LMS.fits`, used by ScopeSim to lay out LMS spectra, is
an earlier form of this product (matrix terms as cubics in echelle angle only,
in a single `Polynomial coefficients` table; and the linear `WCAL` table
authored by Glasse).

## `lmssim`: simulated test frames

```
cd src
python lmssim.py
```

Reads the Polarion export `../config/METIS_Performance_Test_Sequence_LMS.csv`
through {class}`lms_obs_map.ObsMap` and generates four-detector FITS frames for
every step of one test (`test_name = 'lms_opt_02'` at `9b6143a`); ToySim writes
`lms_nexp` exposures per step, each with independent detector noise. Two back
ends:

- **ToySim** ({class}`lmssim_toy.Toy`): a geometric forward model. For each
  slice transform it maps every detector pixel back to (wavelength, alpha),
  interpolates the source spectrum (WCU black body, ambient and CFO masks,
  lasers, or the Leiden sky), applies a blaze envelope, convolves with a Zemax
  PSF, adds pinholes where the test calls for them, and applies a simple
  detector noise model. It writes the `HIERARCH AIT ...` and `HIERARCH ACHG ...`
  keywords that `lmsaiv_opt02` depends on.
- **ScopeSim** ({class}`lmssim_scope.Scope`): drives the METIS `wcu_lms` mode.
  The configuration keys it reads are not the ones `ObsMap` produces, so this
  path is stale.

## `lmsaiv`: AIT performance-test analysis

```
cd src
python lmsaiv.py
```

`lmsaiv.py` points {class}`lms_filer.Filer` at `../data/test_toysim`, loads the
`as_built` pickle (or starts an empty one), looks up one analysis in the
registry of {mod}`lmsaiv_opt_tests` with `OptTests.get_test`, calls it as
`method(title, as_built)` and pickles the returned object. The analysis selected
at `9b6143a` is `lms_opt_04_ghost`; edit `cap_name` to choose another.

| Key | Class | AIT test | Status |
|---|---|---|---|
| `lms_rad_01_dark` | {class}`lmsaiv_rad01.Rad01` | LMS-RAD-01 dark current and read noise | works; prints a per-detector table (moved here from LMS-OPT-01) |
| `lms_rad_04_pix_flat` | {class}`lmsaiv_rad04.Rad04` | LMS-RAD-04 pixel flat field (pixel / 5 x 5 box mean) | runs with a built-in synthetic gain pattern as self test; returns `None`, which then overwrites the `as_built` pickle |
| `lms_opt_01_fov` | {class}`lmsaiv_opt01.Opt01` | LMS-OPT-01 field of view, slice edges and gaps, relative response | works; the subject of the AIT team's notebook |
| `lms_opt_02_dist` | {class}`lmsaiv_opt02.Opt02` | LMS-OPT-02 iso-alpha / iso-lambda traces, detector gaps and rotation, fiducial points for the distortion fit | runs only on ToySim headers; finds the `Opt01` slice map by name match (the key mismatch at `1547ea4` is fixed) |
| `lms_opt_03_psf` | {class}`lmsaiv_opt03.Opt03` | LMS-OPT-03 monochromatic PSF | stub: prints "Analysis not yet available" |
| `lms_opt_04_ghost` | {class}`lmsaiv_opt04.Opt04` | ghosts | stub: prints "Analysis not yet available" |
| `lms_opt_09_sky` | {class}`lmsaiv_opt09.Opt09` | Leiden sky | draft, fails with a TypeError |

Frames are handled as {class}`lms_mosaic.Mosaic` objects (primary header plus
four detector HDUs) and selected by substrings of their file names. The only
persistent product of the chain is the `as_built` pickle at
`../output/asbuilt/asbuilt_toysim.pkl`; everything else is printed or plotted.
Every plot function ends in `plt.show()` and blocks.

The AIT team's notebook
`metis-ait-performance-testing-mirror/LMS/LMS_OPT/lms_opt_01_fov.ipynb` is the
live consumer of this program.
