# Data layout, file formats and header contracts

Nothing under this heading is in the repository. Paths are relative to `src/`,
which must be the working directory.

## Directory tree expected

```
lmsiq/
  src/                          the code (this repo)
  data/                         git-ignored
    iq/<opticon>/<date_stamp>/  Zemax PSF stacks for the lmsiq program
      lms_<stamp>_config<NNN>_field<NNN>_defoc<NNN>um/
        <one .txt parameter file>
        lms_..._spat<SS>_spec<P>_detperf.fits     perfect model
        lms_..._spat<SS>_spec<P>_detdesi.fits     as-designed model
        lms_..._spat<SS>_spec<P>_det<MMMM>.fits   Monte-Carlo instance MMMM
        lms_..._spat<SS>_spec<P>_sli*.fits        slicer-plane images (skipped)
    iq/<opticon>/<date_stamp>_sim/                PSF copy read by ToySim (lmssim_model.load_psf_dict)
    distortion/<nominal|extended>/<date>/          Zemax ray-trace CSVs (lmsdist)
    distortion_ait/<opticon>/ait/                  LMS_ait_dist_coord.csv written by lmsaiv_opt02
    sky/leiden/LBL_A10_w3000_R0120000_ALL_Leiden_LM_R.fits
    baffle/APP_METIS_PSF.fits
    test_toysim/                                    frames written by lmssim (ToySim)
    test_scopesim/                                  frames written by lmssim (ScopeSim)
  config/
    METIS_Performance_Test_Sequence_LMS.csv         Polarion test-sequence export
    sim_template.fits                                header template for ToySim
  output/                        created on demand by Filer.get_folder
    iq/<opticon>/<date_stamp>/{profiles/{ee,lsf}, cube/{fits,pkl,series/{png,csv}}, phase, test}
    distortion/<opticon>/<date>/{fits, ...}, svd_fits_index_*.pkl
    asbuilt/asbuilt_toysim.pkl
  docs/lb_ps_131124.txt          cited as the source of the EFP plate scale; absent
src/inst_pkgs/METIS/wcu/*.dat    irdb WCU focal-plane mask files (fp_mask_pinhole_lm.dat, fp_mask_grid_lm.dat)
```

Date stamps hard-coded in `lms_globals.Globals.model_configurations`:
`20240109` and `20260112` (distortion, nominal / extended), `2024073000` and
`2024032401` (iq). `lmsiq.py` carries its own, partly different, table.

## Detector-frame FITS layout (`Mosaic`)

Primary HDU plus four image HDUs, one per H2RG detector.

| Location | Keyword | Meaning | Written by |
|---|---|---|---|
| primary | `ORIGIN` | `'SCOPESIM'` triggers re-ordering of the extensions with `Globals.mos_idx = {2:0, 1:1, 3:2, 4:3}`; `'TOYSIM'` otherwise | ScopeSim / ToySim |
| extension | `ID` | detector number 1..4 (string) | both |
| extension | `EXTNAME` | `DET<n>.DATA` | both |
| extension | `CRVAL1D`, `CRVAL2D` | detector centre in the mosaic plane, mm (+-19.547) | both |
| extension | `X_CEN`, `Y_CEN` | same, read by `Opt01` | ToySim |
| extension | `HIERARCH ESO DET3 CHIP GAIN` | el/ADU, read by `Rad01.dark_stats` | ToySim (2.0) |
| extension | `HIERARCH AIT PIXEL_PITCH` | mm | ToySim |
| primary | `HIERARCH ESO DET DIT`, `NDIT` | exposure | template |
| primary | `HIERARCH ESO INS WLEN CEN` | central wavelength, um | template |
| primary | `HIERARCH AIT OPTICON`, `AIT PRI_ANG`, `AIT ECH_ANG` | nominal/extended, prism and echelle angle; `OPTICON` and `ECH_ANG` also name the `Opt01` slice map | ToySim only |
| primary | `HIERARCH ACHG REF_ECH_ORD` | echelle order of the setting; part of the slice-map name | ToySim only |
| primary | `HIERARCH ACHG CFO CHOP X`, `Y` | pinhole position in the CFO focal plane, mm | ToySim only |
| primary | `HIERARCH ACHG LASER WAVE` | tunable-laser wavelength, um | ToySim, and added by `lmssim_scope` |

Detector geometry, as the code sees it: detectors 1 and 2 (top row) carry slices
15 to 28, detectors 3 and 4 (bottom row) slices 1 to 14, in the nominal mode;
detectors 1 and 3 (left) are at shorter wavelength than 2 and 4 within a slice.
The mosaic gap is 2.23 mm in `lms_detector.Detector` and in the irdb, but
3.0 mm in `lms_globals.Globals.det_gap`, which is what the affine transforms
use.

**The `AIT` and `ACHG` keywords are a ToySim-only contract.** ScopeSim output and
the METIS pipeline use `INS.OPTI*`, `INS WLEN CEN` and `SEQ WCU LASERn WLEN`.
`lmsaiv_opt02` cannot run on ScopeSim frames until these are mapped.

## File-name tags

Frames are selected by substrings of their file names, built by
`lms_obs_map.ObsMap.get_configuration`:

```
<test>_<step>_<nom|ext>_<kind>_<obs:03d>.fits
kind = dark | sky_ | flat_bb_ | bb_iso_alpha_a<p|n>NNN_b<p|n>NNN_ | las_iso_lambda_ | las_psf_ | woff_<p|m>NNNnm
```

e.g. `lmsaiv_rad01` reads `['lms_rad_01', '_dark']`, `lmsaiv_rad04` reads
`['lms_rad_04', '_flat', '_nom']` and `lmsaiv_opt01` reads
`['lms_opt_01', 'flat_bb', 'step9']`. A `sky_` kind is produced when the
`wcu_per_arm` column is `out`.

## The `as_built` pickle

`../output/asbuilt/asbuilt_toysim.pkl` is a dict loaded and saved by
`lmsaiv.py` around every analysis. LMS-OPT-01 adds one slice map per
configuration group under the map's own name,
`slice_map_<nom|ext>_eo<order>_ea<p|n><NNNNN>` (`NNNNN` = 1000 x |echelle angle|);
LMS-OPT-02 picks the last key containing `slice_map_nom` and adds
`alpha_traces`, `lambda_traces`, `affines` and `dist_coord`.

## Zemax ray-trace CSV (`lmsdist_trace.RayTrace._read_csv`)

Header lines `Echelle angle: <deg>`, `Prism angle: <deg>`, `Spectral order: <n>`,
then a column-name line. Nominal columns: `sp_slice, order, slice, wavelength,
fp2_x, fp2_y, slicer_x, slicer_y, ifu_x, ifu_y, slit_x, slit_y, det_x, det_y,
chief_ray_vignetted`. Extended mode adds `fp1_x, fp1_y, sp_slicer_x,
sp_slicer_y`. The AIT variant written by `lmsaiv_opt02._print_dist_coord` has
`slice_no, spifu_no, ech_ord, efp_w, efp_x, efp_y, mfp_x, mfp_y, cf_vig`.

## Transform FITS (`lms_filer.Filer.write_svd_transforms`)

`lms_efp_mfp<_nom|_ext>_pa#####_ea<p|n>#####_v<date>.fits`

- primary header: `OPTICON`, `PRI_ANG`, `ECH_ANG`, `N_MATS = 4`, `MAT_ORD = 4`,
  then an empty card `W` (added upstream in `9b6143a`, purpose not recorded)
- one `BinTableHDU` per slice, `EXTNAME = SLICE_<n>`, header `SLICE_NO`,
  `ECH_ORD`, `W_MIN`, `W_MAX`; columns `A`, `B`, `AI`, `BI`, each 16 values
  (a flattened 4 x 4 matrix), stored as `E` (float32).

The polynomial form is, with `x` the grating phase and `y` the along-slice
coordinate alpha in mm,

```
x_out = sum_{i,j} A[i, j] * y**i * x**j        (i + j <= 3)
y_out = sum_{i,j} B[i, j] * y**i * x**j
```

and `AI`, `BI` are the inverse maps from mosaic (x, y) back to (phase, alpha).

## Zemax PSF parameter file (`lmsiq_image_manager.ImageManager._read_param_file`)

`key: value` lines; recognised keys `wavelength`, `prism angle`,
`grating angle`, `grating order`, `Pixel size (micron)`. No FITS header
keyword is read from the PSF images themselves; all metadata comes from the
folder and file names and this file.
