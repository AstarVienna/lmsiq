#!/usr/bin/env python
"""
LMS-OPT-02 analysis: iso-alpha / iso-lambda traces and distortion coordinates (lmsaiv program).

Implements AIT performance test LMS-OPT-02 for the ``lmsaiv`` program
(entry ``Opt02.dist``, registered as 'lms_opt_02_dist' in
``lmsaiv_opt_tests.OptTests``).  Pinhole + black-body frames give
iso-alpha traces (constant along-slice field position, roughly along
detector rows) and tunable-laser flood frames give iso-lambda traces
(constant wavelength, roughly along columns).  Their intersections are
fiducial points ``(det_no, row, col) <-> (efp_x, efp_y, efp_w)`` which
are written to a CSV for the ``lmsdist`` SVD distortion fit, and the
iso-lambda traces also yield the inter-detector gap and relative rotation
of detector pairs 1/2 and 3/4.

**Input data**: FITS mosaics in ``Filer.test_data_folder`` whose names
contain ['lms_opt_02', 'nom'] plus:

- ['flat', ``step1_``] (nominal) or ``step38_`` (extended): black-body flat
  used as the iso-alpha background;
- 'iso_alpha': pinhole + black-body frames;
- 'iso_lambda': tunable-laser flood frames (background = per-pixel
  median of the series, ``OptTools.median_subtract``).

Also required: a nominal slice map from LMS-OPT-01 in the pickle at
``Globals.as_built_file``, and Zemax / AIT SVD transform FITS files read
through ``Filer.read_svd_transforms`` from the folders selected by
``Filer.set_configuration('distortion', opticon, is_ait_data)``.

Primary header keywords read: 'HIERARCH ACHG CFO CHOP X', 'HIERARCH ACHG
CFO CHOP Y' (pinhole offset; appears to be entrance focal plane mm),
'HIERARCH ACHG REF_ECH_ORD', 'HIERARCH ACHG LASER WAVE' (micron),
'HIERARCH AIT OPTICON', 'HIERARCH AIT PRI_ANG', 'HIERARCH AIT ECH_ANG'
(degrees).  Per HDU: 'ID' (detector number, as a string).  All of these
are written only by ``lmssim_toy``; ScopeSim output and real ICS headers
do not carry them.

**Output**:

- ``as_built['alpha_traces']``, ``as_built['lambda_traces']`` (lists of
  trace dicts, see ``_extract_det_traces``), ``as_built['affines']``,
  ``as_built['dist_coord']`` (see ``_find_trace_intersections``);
  pickled to ``Globals.as_built_file``.
- CSV 'LMS_ait_dist_coord.csv' in ``filer.ray_trace_folder``
  ('../data/distortion_ait/nominal/<data_set>/').
- Printed tables and blocking figures (``Plot.mosaic``,
  ``Plot.gap_data``, ``Plot.det_traces``).

Notes
-----
- Inconsistency: the slice map is found by sub-string search for
  'slice_map_nom' over the ``as_built`` keys and the last match wins;
  LMS-OPT-01 stores one nominal map per step group, so which one is used
  depends on dictionary order.
- Bug: if no slice map matches, a message is printed but the analysis
  continues with ``slice_map = None`` and fails in
  ``_build_slice_row_dict`` (AttributeError).
- Bug: ``lmsdist.run()`` (the ``calculate_transforms`` step) has
  ``is_ait_data = False`` hard-coded, so it refits the Zemax ray traces
  rather than the AIT CSV written here; the AIT transforms compared
  against Zemax are therefore never produced by this chain.
- Bug: the ``test_transforms`` block calls ``Util.slice_to_efp_y(slice_no,
  beta_phase, opticon)`` with three arguments while the function takes
  two, and discards the forward-projected points.
- Inconsistency: ``_extract_det_traces`` indexes the slice map with
  ``Globals.mos_idx[det_no]`` while ``Filer.read_mosaic`` stores ToySim
  HDUs at ``det_no - 1``; for ToySim data detectors 1 and 2 appear to be
  paired with each other's slice map.
- Open question: the analysis is controlled by hard-coded booleans in
  ``dist`` (``misalign_detector``, ``find_iso_alphas``,
  ``find_iso_lambdas``, ``calculate_transforms``, ``test_transforms``)
  rather than keyword arguments.
- Clean-up: stray ``print(0)`` and ``print(rows, gaps)`` debug output;
  ``_print_dist_coord`` writes its CSV by redirecting ``sys.stdout``.

@author: Alistair Glasse
"""
import math
import sys
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

import lmsdist
from lmsdist_util import Util
from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer
from lms_transform import Transform


class Opt02:
    """LMS-OPT-02 distortion analysis (static methods).

    Entry point ``dist``; helpers extract traces, fit them, find the
    inter-detector gaps and intersect the trace families.
    """


    def __init__(self):
        """Do nothing; all methods are static."""
        return

    @staticmethod
    def dist(title, as_built, **kwargs):
        """Run the LMS-OPT-02 trace / distortion analysis.

        Steps, gated by hard-coded booleans (defaults in parentheses):

        1. ``filer.set_configuration('distortion', nominal, is_ait_data=True)``
           and take the slice map: the last ``as_built`` key containing
           ``'slice_map_' + opticon[0:3]`` (i.e. 'slice_map_nom').
        2. ``misalign_detector`` (False): rotate detector 2 with
           ``OptTools.transform_detector_image`` to test the recovery of a
           misalignment (offset (0, 0) and angle 0.0, i.e. a no-op).
        3. ``find_iso_alphas`` (False): subtract the ['flat', ``step1_``] flat from
           each 'iso_alpha' pinhole frame, extract traces with ``snr_cut=1.``,
           reduce them to one PSF trace per detector with ``_find_beta_phase``,
           add 'mean_theta_disp' (mean tilt over 16 column samples, degrees) and
           store ``as_built['alpha_traces']``.
        4. ``find_iso_lambdas`` (True): median-subtract the 'iso_lambda' series,
           read 'HIERARCH ACHG LASER WAVE' per frame and extract traces with
           ``snr_cut=10``; store ``as_built['lambda_traces']``.
        5. ``_find_detector_gaps`` on the lambda traces; design affines from
           ``Transform.create_affines`` (all offsets zero); ``Plot.det_traces``
           per LMS configuration; ``_find_trace_intersections``;
           ``_print_dist_coord`` to stdout and to CSV; store
           ``as_built['affines']`` and ``as_built['dist_coord']``.
        6. For each configuration read Zemax (``is_ait_data=False``) and AIT
           (``is_ait_data=True``) SVD transforms, match them with
           ``Util.find_matching_transforms`` and print the fractional matrix
           term differences (AIT - ZMX) / ZMX (debug 'low').
        7. ``calculate_transforms`` (True): ``lmsdist.run()``.
        8. ``test_transforms`` (True): forward-project the centre of each matched
           slice with ``Util.efp_to_mfp`` through both transforms.

        Parameters
        ----------
        title : str
            Registry title; unused.
        as_built : dict
            'As built' dictionary; must contain a key containing
            'slice_map_nom' (from LMS-OPT-01) and, when
            ``find_iso_alphas`` is False, 'alpha_traces' from a previous run.
        **kwargs
            ``overwrite_as_built_traces`` : bool, default True.  Intended to
            append to rather than replace the archived trace lists.

        Returns
        -------
        dict
            ``as_built`` updated with 'alpha_traces', 'lambda_traces', 'affines'
            and 'dist_coord'.  Intermediate pickles are also written after steps
            3 and 4.

        Raises
        ------
        KeyError
            If 'alpha_traces' / 'lambda_traces' are missing from ``as_built``
            when the corresponding extraction is disabled.
        AttributeError
            If no slice map is found (``slice_map`` stays None).

        Notes
        -----
        - Bug: with ``overwrite_as_built_traces=False`` the archived list is
          extended in place but then overwritten by the new list, so the option
          has no effect.
        - Bug: ``matching_transforms`` is only defined inside the loop over
          ``dist_coord``; if no intersections were found the ``test_transforms``
          block raises NameError.
        - Bug: ``Util.slice_to_efp_y`` is called with three arguments (it takes
          ``fslice_no, phase``) and with the encoded slice number, and the
          round-trip results ``mfp_pts_ait`` / ``mfp_pts_zmx`` are never compared
          or returned.
        - Bug: the AIT transforms are read (step 6) before ``lmsdist.run()``
          (step 7) is called, and ``lmsdist.run`` fits Zemax data anyway
          (``is_ait_data = False`` hard-coded), so no AIT transform FITS files
          are produced by this chain.
        - Inconsistency: ``sig_mosaic.subtract(bgd_mosaic)`` modifies
          ``sig_mosaic`` in place, so the 'Signal' plot that follows already
          shows the background-subtracted image.
        - Inconsistency: ``Plot.mosaic`` is called with a ``bounds`` keyword it
          does not use.
        - Open question: the ``misalign_detector`` test uses offset (0, 0) and
          rotation 0.0, so even when enabled it does not perturb the data.
        - Open question: extended mode is not reachable (``opticon =
          Globals.nominal`` hard-coded) although the flat step for it (``step38_``)
          is defined.
        - Clean-up: ``slice_row_dict`` is computed but unused; ``Plot.det_traces``
          is called twice per configuration; stray ``print(0)``.
        """
        overwrite_as_built_traces = kwargs.get('overwrite_as_built_traces', True)

        test_name = 'lms_opt_02'
        opticon = Globals.nominal

        misalign_detector = False
        find_iso_alphas = False
        find_iso_lambdas = True
        calculate_transforms = True
        test_transforms = True

        inc_tags = [test_name, opticon[0:3]]         # Tokens to identify image files.
        filer = Filer()
        filer.set_configuration('distortion', opticon, is_ait_data=True)

        flag_text = 'Analysing'
        if find_iso_alphas:
            flag_text += ' iso-alpha'
            if find_iso_lambdas:
                flag_text += ' and'
        if find_iso_lambdas:
            flag_text += ' iso-lambda'
        if not find_iso_alphas and not find_iso_lambdas:
            flag_text = 'Calculating trace intersections only '
        else:
            flag_text += ' traces'
        print('{:s}'.format(flag_text))
        print()
        tgt_map = 'slice_map_' + opticon[0:3]
        slice_map = None
        for key in as_built:
            if tgt_map in key:
                slice_map = as_built[key]
        if slice_map is None:
            print('!! Slice map {:s} not found !!'.format(tgt_map))

        if Globals.is_debug('medium'):
            Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
        slice_row_dict = Opt02._build_slice_row_dict(slice_map)         # Dictionary of per slice row bounds

        # Set up a deliberately misaligned detector
        test_det_no, test_det_offset, test_img_rot_deg = None, None, None
        if misalign_detector:
            test_det_no = 2
            test_det_offset = 0, 0
            test_det_rot_deg = 0.0
            test_img_rot_deg = -test_det_rot_deg
            print('Deliberately misaligning detector')
            print('det_no =       {:d}'.format(test_det_no))
            print('det_offset =   {:d}, {:d} arcsec'.format(test_det_offset[0], test_det_offset[1]))
            print('det_rotation = {:5.3f} deg.'.format(test_det_rot_deg))
            print()

        # ---------- ISO-ALPHA -------------------
        # Start with data loading and background subtraction.  The iso-alphas use the WCU BB and the lm_pinhole
        # mask.  The background under the trace is then the emission spectrum of the mask (at 300 K).  We remove
        # this by moving the pinhole out of the LMS field using the CFO chopper.
        if find_iso_alphas:
            print('Derive detector rotation and offset differences from iso-alpha trace data. ')
            alpha_traces = []

            # Load background images (BB flat field)
            print('Background')
            flat_step = {Globals.nominal: 'step1_', Globals.extended: 'step38_'}[opticon]
            bgd_mosaics = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags + ['flat', flat_step])
            bgd_mosaic = bgd_mosaics[0]
            if misalign_detector:
                bgd_mosaic = OptTools.transform_detector_image(bgd_mosaics[0],
                                                               det_no=test_det_no,
                                                               xy_pix=test_det_offset,
                                                               angle=test_img_rot_deg)
            if Globals.is_debug('medium'):
                Plot.mosaic(bgd_mosaic, title='WCU lm_pinhole mask background', cmap='hot')

            sig_file_list = Filer.get_file_list(Filer.test_data_folder, inc_tags=inc_tags + ['iso_alpha'])
            for sig_file in sig_file_list:
                print()
                sig_mosaic = Filer.read_mosaic(Filer.test_data_folder, sig_file)
                print("Processing mosaic file {:s}".format(sig_mosaic.name))
                if misalign_detector:
                    sig_mosaic = OptTools.transform_detector_image(sig_mosaic,
                                                                   det_no=test_det_no,
                                                                   xy_pix=test_det_offset,
                                                                   angle=test_img_rot_deg)
                # Extract iso-alpha traces for all slices (by setting snr_cut=0.)
                tra_mosaic = sig_mosaic.subtract(bgd_mosaic)
                if Globals.is_debug('low'):
                    Plot.mosaic(sig_mosaic, title='Signal')
                if Globals.is_debug('low'):
                    Plot.mosaic(tra_mosaic, title='Signal - Bgd')

                # We find the multiple alpha traces in adjacent slices by accepting all traces above snr which
                # is low enough to avoid fit problems in un-illuminated slices, but to include enough traces to
                # calculate the PSF across-slice / beta phase.
                psf_alpha_traces = Opt02._extract_det_traces(tra_mosaic, 'iso-alpha', slice_map,
                                                             snr_cut=1.)
                # ..then fit a gaussian to all solid detections to find the single trace for the PSF centroid.
                # The slice plus beta phase should ideally(!) match the WCU derived efp_y parameter.
                # The method clips any points which are displaced by more than 'sigma_clip' times the rms
                # displacement from the best fit and then recalculates the best fit.
                alpha_trace_pair = Opt02._find_beta_phase(psf_alpha_traces, opticon, snr_cut=1.)
                if Globals.is_debug('high'):
                    print()
                    print('PSF iso-alpha traces found')
                    fmt = "{:>12s},{:>12s},{:>12s},"
                    print(fmt.format('Det. no.', 'Slice no.', 'Beta phase'))
                    fmt = "{:12d},{:12d},{:12.3f},"
                    for alpha_trace in alpha_trace_pair:
                        det_no = alpha_trace['det_no']
                        slice_no = alpha_trace['slice_no']
                        beta_phase = alpha_trace['beta_phase']
                        print(fmt.format(det_no, slice_no, beta_phase))

                if Globals.is_debug('low'):
                    overlay = {'type': 'det_traces', 'trace_type': 'iso_alpha', 'data': alpha_trace_pair}
                    Plot.mosaic(sig_mosaic, title='Background subtracted', cmap='grey', overlay=overlay)
                alpha_traces = alpha_traces + alpha_trace_pair         # 1 alpha traces per detector

            # Calculate mean angle between alpha traces and the row direction.
            if Globals.is_debug('high'):
                fmt = "{:>10s},{:>10s},{:>12s},{:>20s},"
                print(fmt.format('Det', 'Slice', 'Row     ', 'Mean rotation dispersion'))
                print(fmt.format('no.', 'no.  ', 'fiducial', 'dispersion to row'))
                print(fmt.format('-  ', '-    ', '-       ', '[degrees]'))
                fmt = "{:>10d},{:>10d},{:>12.2f},{:>20.2f},"
            for alpha_trace in alpha_traces:
                det_no = alpha_trace['det_no']
                slice_no = alpha_trace['slice_no']
                popt = alpha_trace['popt']
                row_fid = alpha_trace['v_fid']
                col_samples = np.arange(0, 2048, 128)
                gradients = Globals.polynomial(col_samples, *popt, gradient=True)
                mean_gradient = np.mean(gradients)
                # Calculate the angle between polynomials at the mosaic centre.
                deg_rad = 180./math.pi
                mean_theta_disp = -deg_rad * mean_gradient
                # Calculate the angle between the alpha trace (dispersion) and the detector row at the centre column
                if Globals.is_debug('high'):
                    print(fmt.format(det_no, slice_no, row_fid, mean_theta_disp))
                alpha_trace['mean_theta_disp'] = mean_theta_disp

            if 'alpha_traces' in as_built.keys() and not overwrite_as_built_traces:
                alpha_traces_archive = as_built['alpha_traces']
                alpha_traces_archive += alpha_traces
            as_built['alpha_traces'] = alpha_traces
            Filer.write_pickle(Globals.as_built_file, as_built)

        # -----------------------
        # ISO-LAMBDA
        if find_iso_lambdas:
            print('2. Extract iso-lambda traces to measure the intra-detector gap and the line spread function.')
            print('   The gap calculation will assume that the laser lines are spaced according to a smooth polynomial.')

            raw_mosaics = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags + ['iso_lambda'])
            mosaics = OptTools.median_subtract(raw_mosaics)

            print(0)
            lambda_traces = []
            for mosaic in mosaics:
                print()
                print("Processing mosaic file {:s}".format(mosaic.name))
                if misalign_detector:
                    mosaic = OptTools.transform_detector_image(mosaic,
                                                               det_no=test_det_no,
                                                               xy_pix=test_det_offset,
                                                               angle=test_img_rot_deg)
                lt_wave = mosaic.primary_hdr['HIERARCH ACHG LASER WAVE']
                if Globals.is_debug('low'):
                    Plot.mosaic(mosaic, title='laser_wavelength = ' + "{:10.3f}".format(lt_wave))
                # Find the gap between detectors as a function of row number.
                snr_cut = 10
                mos_lambda_traces = Opt02._extract_det_traces(mosaic, 'iso-lambda', slice_map, snr_cut,
                                                              lt_wave=lt_wave)
                if len(mos_lambda_traces) == 0:
                    continue
                if Globals.is_debug('low'):
                    overlay = {'type': 'det_traces', 'trace_type': 'iso_lambda', 'data': mos_lambda_traces}
                    row_fid = mos_lambda_traces[0]['v_fid']
                    bounds = 0, 2048, row_fid-100, row_fid+100
                    Plot.mosaic(mosaic, title='Background subtracted', cmap='grey',
                                bounds=bounds, overlay=overlay)

                lambda_traces = lambda_traces + mos_lambda_traces

            if 'lambda_traces' in as_built.keys() and not overwrite_as_built_traces:
                lambda_traces_archive = as_built['lambda_traces']
                lambda_traces_archive += lambda_traces
            as_built['lambda_traces'] = lambda_traces
            Filer.write_pickle(Globals.as_built_file, as_built)

        alpha_traces = as_built['alpha_traces']
        lambda_traces = as_built['lambda_traces']

        # Find the intra-detector gap from a list of column position v wavelength for tunable laser spectral lines.
        gap_data, det_thetas = Opt02._find_detector_gaps(lambda_traces)
        if Globals.is_debug('low'):
            Plot.gap_data(gap_data, det_thetas)

        # Generate baseline affine transforms to project detector pixel coords into the mosaic focal plane.
        # We can/ought to project the detector intersections into the mosaic focal plane using the baseline
        # Zemax derived affine transforms.  If these are discrepant due to detector misalignment, they can
        # be adjusted explicitly using 'affine term offsets' for each detector.
        affines = Transform.create_affines(theta_offsets=[0.]*4,
                                           x_mfp_org_offsets=[0.]*4, y_mfp_org_offsets=[0.]*4,
                                           x_scale_offsets=[0.]*4, y_scale_offsets=[0.]*4)

        # Find all unique LMS configurations and plot the traces
        lms_config_ids = []
        for alpha_trace in alpha_traces:
            id = alpha_trace['lms_config_id']
            if id not in lms_config_ids:
                lms_config_ids.append(id)
        for lms_config_id in lms_config_ids:
            Plot.det_traces(alpha_traces, lambda_traces, lms_config_id)

        dist_coord = Opt02._find_trace_intersections(alpha_traces, lambda_traces, affines)
        _ = Opt02._print_dist_coord(filer, dist_coord, to_csv=False)
        path = Opt02._print_dist_coord(filer, dist_coord, to_csv=True)
        print("Coordinates written to file {:s}".format(path))
        print()

        for lms_config_id in dist_coord:
            coords = dist_coord[lms_config_id]
            Plot.det_traces(alpha_traces, lambda_traces, lms_config_id)
            # Plot.det_traces(alpha_traces, lambda_traces, lms_config_id, coords=coords, bounds=(0, 600, 330, 430))

        as_built['affines'] = affines
        as_built['dist_coord'] = dist_coord

        for lms_config_id in dist_coord:
            inc_tags = [lms_config_id]
            is_ait_data = False
            filer.set_configuration('distortion', opticon, is_ait_data)
            zemax_transforms = filer.read_svd_transforms(inc_tags=inc_tags)

            is_ait_data = True
            filer.set_configuration('distortion', opticon, is_ait_data)
            ait_transforms = filer.read_svd_transforms(inc_tags=inc_tags)
            # As a measure of accuracy, calculate the fractional difference between matrix terms,
            matching_transforms = Util.find_matching_transforms(ait_transforms, zemax_transforms)
            for ait_transform, zmx_transform in matching_transforms:
                if Globals.is_debug('low'):
                    a_slice_no = ait_transform.slice_configuration['slice_no']
                    fmt = "Slice {:d} fractional difference in matrix terms (AIT - ZMX) / ZMX"
                    print(fmt.format(a_slice_no))
                ait_matrices = ait_transform.matrices
                zmx_matrices = zmx_transform.matrices
                for key in ait_matrices:
                    if Globals.is_debug('low'):
                        fmt = "Matrix {:s}"
                        print(fmt.format(key.upper()))
                    ait_matrix = ait_matrices[key]
                    zmx_matrix = zmx_matrices[key]
                    with warnings.catch_warnings(record=True):
                        f_mat = (ait_matrix - zmx_matrix) / zmx_matrix
                    nr, nc = ait_matrix.shape
                    for r in range(0, nr):
                        text = "{:>6s}{:4d}".format('Row', r)
                        for c in range(0, nc - r):
                            f = f_mat[r, c]
                            text += "{:15.3f}".format(f)
                        if Globals.is_debug('low'):
                            print(text)

        if calculate_transforms:
            lmsdist.run()

        if test_transforms:
            print()
            # Better yet, use the transforms to do a forward and back ray trace.
            beta_phase = 0.5
            for ait_transform, zmx_transform in matching_transforms:
                slice_config = ait_transform.slice_configuration
                slice_no = slice_config['slice_no']

                efp_x = 0.
                efp_y = Util.slice_to_efp_y(slice_no, beta_phase, opticon).value
                efp_w = 0.5 * (slice_config['w_min'] + slice_config['w_max'])
                efp_points = {'efp_x': np.array([efp_x]), 'efp_y': np.array([efp_y]), 'efp_w': np.array([efp_w])}
                mfp_pts_ait = Util.efp_to_mfp(ait_transform, efp_points)
                mfp_pts_zmx = Util.efp_to_mfp(zmx_transform, efp_points)
        return as_built

    @staticmethod
    def _build_slice_row_dict(slice_map):
        """Tabulate the row extent of every slice on every detector of a slice map.

        Parameters
        ----------
        slice_map : Mosaic
            Slice map from ``lmsaiv_opt01.Opt01._make_slice_map``; per-HDU 'ID'
            gives the detector number.

        Returns
        -------
        dict
            ``{det_no: {slice_no: {'row_min': int, 'row_max': int}}}`` over the
            non-zero slice numbers of each detector image.

        Notes
        -----
        - Clean-up: computed in ``dist`` but never used; ``_extract_det_traces``
          recomputes the same bounds from the slice map directly.
        """
        # _, _, hdus = slice_map
        slice_row_dict = {}
        for hdu in slice_map.hdu_list:
            image, hdr = hdu.data, hdu.header
            det_no = int(hdr['ID'])
            slice_row_dict[det_no] = {}
            slice_nos = np.unique(image)
            for slice_no in slice_nos:
                if slice_no == 0.:
                    continue
                indices = np.argwhere(image == slice_no)
                row_min = np.min(indices[:, 0])
                row_max = np.max(indices[:, 0])
                slice_row_dict[det_no][slice_no] = {'row_min': row_min, 'row_max': row_max}
        return slice_row_dict

    @staticmethod
    def _find_beta_phase(all_alpha_traces, opticon, snr_cut=1.):
        """Reduce per-slice iso-alpha traces of a pinhole image to one PSF trace per detector and pupil slice.

        For each detector (1..4) and each pupil slice number in
        ``Globals.pslice_no_range[opticon]`` the traces with that pupil slice
        (decoded with ``Util.decode_slice_no``) are collected.  If at least one
        has ``snr > snr_cut`` a Gaussian ``Globals.gauss`` is fitted to trace
        signal versus encoded slice number (initial guess: peak signal, peak
        slice, sigma 2.5 slices).  The brightest trace is copied and given::

            beta_phase = gopt[1] - slice_no_peak + 0.5

        i.e. 0.5 when the PSF is centred on the brightest slice.

        Parameters
        ----------
        all_alpha_traces : list of dict
            Traces from ``_extract_det_traces(..., 'iso-alpha', ...)``, typically
            extracted with a low ``snr_cut`` so that the wings of the PSF in
            neighbouring slices are included.
        opticon : str
            ``Globals.nominal`` or ``Globals.extended``.
        snr_cut : float, optional
            A group is analysed only if one of its traces exceeds this SNR.
            Default 1.

        Returns
        -------
        list of dict
            One trace per detector and pupil slice, a shallow copy of the
            brightest trace with an added 'beta_phase' key.  Groups whose fit
            raises RuntimeError / TypeError / ValueError are reported and skipped.

        Notes
        -----
        - Open question: in extended mode only three field slices per pupil
          slice are available, so the three-parameter Gaussian fit is exactly
          determined.
        - Open question: the fit is in encoded slice-number units
          (``100 * pslice + fslice``); within one pupil slice this is a plain
          slice index, so the phase is in slice widths.
        - Clean-up: ``dict.copy()`` is shallow; 'popt' / 'pcov' arrays are shared
          with the source trace.
        """
        alpha_trace_list = []

        for det_no in range(1, 5):
            pslice_no_range = Globals.pslice_no_range[opticon]
            for pslice_no in pslice_no_range:
                signal_list, slice_no_list, spifu_alpha_traces = [], [], []
                is_detected = False
                for alpha_trace in all_alpha_traces:
                    if det_no != alpha_trace['det_no']:
                        continue
                    trace_slice_no = alpha_trace['slice_no']
                    trace_fslice_no, trace_pslice_no = Util.decode_slice_no(trace_slice_no)
                    # trace_pslice_no = alpha_trace['pslice_no'] // 100
                    if pslice_no != trace_pslice_no:
                        continue
                    # Select all slices with the same pupil slice number
                    signal_list.append(alpha_trace['signal'])
                    slice_no_list.append(trace_slice_no)
                    spifu_alpha_traces.append(alpha_trace)
                    snr = alpha_trace['snr']
                    is_detected = True if snr > snr_cut else is_detected

                if len(signal_list) == 0 or not is_detected:
                    continue
                signals, slice_nos = np.array(signal_list), np.array(slice_no_list)
                sig_max = np.amax(np.array(signals))
                idx_max = np.argmax(signals)
                slice_no_peak = slice_nos[idx_max]
                alpha_trace = spifu_alpha_traces[idx_max].copy()
                psf_alpha_sigma = 2.5
                p0_guess = [sig_max, slice_no_peak, psf_alpha_sigma]
                is_error = False
                try:
                    gopt, gcov = curve_fit(Globals.gauss, slice_nos, signals, p0=p0_guess, maxfev=5000)
                except (RuntimeError, TypeError, ValueError):
                    print("!! Error finding beta-phase for det {:d} !!".format(det_no))
                    is_error = True
                if not is_error:
                    beta_phase = gopt[1] - slice_no_peak + 0.5       # Phase = 0.5 for source centred in slice
                    alpha_trace['beta_phase'] = beta_phase
                    alpha_trace_list.append(alpha_trace)
        return alpha_trace_list

    @staticmethod
    def _find_detector_gaps(lambda_traces, order=2):
        """Measure the column gap and relative rotation between the detectors of each pair (1/2 and 3/4).

        Iso-lambda traces are grouped by slice number; those on detectors 1 and 3
        form the 'left' (short-wavelength) set, those on 2 and 4 the 'right' set.
        For each slice with more than ``order`` lines on both sides::

            lam(col_l)  linear fit -> lam_mid_l = lam(2048)   (RH edge of left det)
            col(lam_r)  linear fit -> col_gap   = -col_r(lam_mid_l)

        so ``col_gap`` is the number of columns between the last column of the
        left detector and the first column of the right one.  Then, for each
        detector pair, ``col_gap`` versus trace row (``u_mean``) is fitted
        linearly and the slope converted to an angle::

            theta     = atan(gradient) degrees
            theta_err = sqrt(pcov[1][1]) * (180 / pi) / (1 + gradient ** 2)

        Parameters
        ----------
        lambda_traces : list of dict
            Iso-lambda traces with keys 'slice_no', 'det_no', 'v_fid' (column at
            the trace mid-row), 'efp_w' (laser wavelength), 'u_mean' (mid-row).
        order : int, optional
            Number of polynomial coefficients used in the ``curve_fit`` initial
            guess ``[0.] * order``; 2 (default) gives a linear fit.  A slice needs
            more than ``order`` lines on each side.

        Returns
        -------
        gap_data : dict
            ``{slice_no: {'lams_l', 'lams_r', 'cols_l', 'cols_r', 'col_gap',
            'u_mean', 'det_no'}}``; 'col_gap' is None where the fit was not
            possible.  Consumed by ``Plot.gap_data``.
        det_thetas : dict
            ``{'12': (theta, theta_err) or None, '34': ...}`` in degrees.

        Notes
        -----
        - Inconsistency: the parameter is named ``order`` but sets the number of
          coefficients (degree + 1).
        - Bug: the bare ``except`` clauses print 'Failed curve_fit' and then use
          the undefined ``lam_popt_l`` / ``col_popt_r``, raising TypeError /
          NameError anyway.
        - Bug: ``except OptimizeWarning`` cannot catch a warning that
          ``curve_fit`` only emits (not raises) unless warnings are turned into
          errors; a failed fit leaves ``popt`` undefined.
        - Bug: (minor) the slope is evaluated with ``Globals.polynomial(mean_gap,
          *popt, gradient=True)``, i.e. at x = mean gap instead of a row value;
          harmless for a linear fit, wrong for higher orders.
        - Open question: 'u_mean' and 'det_no' per slice are overwritten by the
          last trace seen, so the row used for the rotation fit belongs to
          whichever detector's trace came last.
        - Clean-up: stray ``print(rows, gaps)``.
        """
        gap_data = {}
        for lambda_trace in lambda_traces:
            slice_no = lambda_trace['slice_no']
            if slice_no not in gap_data:
                gap_data[slice_no] = {'lams_l': [], 'lams_r': [], 'cols_l': [], 'cols_r': [], 'col_gap': None}
            det_no = lambda_trace['det_no']
            v_fid = lambda_trace['v_fid']
            lam = lambda_trace['efp_w']
            lams_tag, cols_tag = 'lams_l' , 'cols_l'
            if det_no in [2, 4]:
                lams_tag, cols_tag = 'lams_r' , 'cols_r'
            gap_data[slice_no][lams_tag].append(lam)
            gap_data[slice_no][cols_tag].append(v_fid)
            gap_data[slice_no]['u_mean'] = lambda_trace['u_mean']
            gap_data[slice_no]['det_no'] = det_no

        for slice_no in gap_data:
            gap_sample = gap_data[slice_no]
            idx_l = np.argsort(np.array(gap_sample['lams_l']))
            idx_r = np.argsort(np.array(gap_sample['lams_r']))
            lams_l = np.array(gap_sample['lams_l'])[idx_l]
            lams_r = np.array(gap_sample['lams_r'])[idx_r]
            cols_l = np.array(gap_sample['cols_l'])[idx_l]
            cols_r = np.array(gap_sample['cols_r'])[idx_r]
            # with warnings.catch_warnings(record=True):
            if cols_l.shape[0] <= order or cols_r.shape[0] <= order:
                print('Insufficient iso-lambda traces for slice ', slice_no)
                continue
            lam_popt_l, lam_mid_l = None, None
            try:
                lam_popt_l, _ = curve_fit(Globals.polynomial, cols_l, lams_l, p0=[0.]*order)
            except:
                print('Failed curve_fit ')
            lam_mid_l = Globals.polynomial(2048., *lam_popt_l)[0]     # Wavelength of RH edge of left hand detector
            try:
                col_popt_r, _ = curve_fit(Globals.polynomial, lams_r, cols_r, p0=[0.]*order)
            except:
                print('Failed curve_fit ')
            col_gap = -1. * Globals.polynomial(lam_mid_l, *col_popt_r)[0]
            gap_sample['col_gap'] = col_gap

        # Find angle between 1,2 and 3,4 by linear fitting to the intra-detector gap as a function of row number.
        det_thetas = {}
        for det_nos in ['12', '34']:
            det_thetas[det_nos] = None
            col_gap_list, row_list = [], []
            for slice_no in gap_data:
                gap_sample = gap_data[slice_no]
                if gap_sample['col_gap'] is None:       # No gap value available
                    continue
                if str(gap_sample['det_no']) in det_nos:
                    col_gap_list.append(gap_sample['col_gap'])
                    row_list.append(gap_sample['u_mean'])
            if len(col_gap_list) == 0:
                print('Insufficient gap data for detector pair ', det_nos)
                continue

            rows = np.array(row_list)
            gaps = np.array(col_gap_list)
            mean_gap = np.mean(gaps)

            linear_guess = [mean_gap, 0.]
            try:
                print(rows, gaps)
                popt, pcov = curve_fit(Globals.polynomial, rows, gaps, p0=linear_guess)
            except OptimizeWarning:
                print('Warning: polynomial fit failed')
            gradient = Globals.polynomial(mean_gap, *popt, gradient=True)[0]
            gradient_err = np.sqrt(pcov[1][1])
            deg_rad = 180. / math.pi
            theta = math.atan(gradient) * deg_rad
            theta_err = gradient_err * deg_rad / (1 + gradient ** 2)
            det_thetas[det_nos] = theta, theta_err
        return gap_data, det_thetas

    @staticmethod
    def _extract_det_traces(mosaic, trace_type, slice_map, snr_cut,
                            lt_wave=None, sigma_clip=2.):
        """Extract one trace per illuminated slice from a background-subtracted mosaic.

        For each detector HDU the slice map
        (``slice_map.hdu_list[Globals.mos_idx[det_no]]``) gives the slices
        present.  Each slice strip (rows ``[row_min - 10, row_max + 10)``, with
        10 = ``Globals.intra_slice_gap / 2``) is collapsed along the 'u'
        direction (columns for 'iso-alpha', rows for 'iso-lambda'); the brightest
        'v' position gives::

            noise  = median over v of std along u
            bgd    = median of the strip
            signal = mean of the collapsed profile within +-5 of the peak
            snr    = (signal - bgd) / noise

        Slices with ``snr < snr_cut`` are skipped.  Otherwise
        ``_get_trace_coordinates`` samples the trace centroid at 10 (alpha) or 5
        (lambda) positions and a 4-coefficient polynomial
        ``Globals.polynomial`` v(u) is fitted with iterative ``sigma_clip``
        (default 2 sigma) rejection, one point at a time, until no point is
        rejected or fewer than four remain.  ``OptTools._find_thetas`` then adds
        the local tilt.

        Parameters
        ----------
        mosaic : Mosaic
            Background-subtracted frame.  Primary keywords read: 'HIERARCH ACHG
            CFO CHOP X' / 'Y' (-> 'efp_x', 'efp_y'), 'HIERARCH ACHG REF_ECH_ORD',
            'HIERARCH AIT OPTICON', 'HIERARCH AIT PRI_ANG', 'HIERARCH AIT
            ECH_ANG'; per HDU 'ID' (string, ``.strip()`` is applied).
        trace_type : str
            'iso-alpha' or 'iso-lambda'.
        slice_map : Mosaic
            Slice map from LMS-OPT-01.
        snr_cut : float
            Minimum SNR for a slice to be analysed.
        lt_wave : float or None, optional
            Laser wavelength (micron) stored as 'efp_w' for iso-lambda traces;
            iso-alpha traces get 'efp_w' = -1.
        sigma_clip : float, optional
            Clipping threshold in units of the rms residual.  Default 2.

        Returns
        -------
        list of dict
            One dict per accepted slice with keys 'lms_config_id'
            (``Globals.make_lms_config_id``: '_nom' or '_ext' + '_pa' + 5-digit
            prism angle + '_ea' + sign + 5-digit echelle angle), 'version'
            ('ait'), 'type', 'det_no', 'slice_no', 'snr', 'signal', 'popt',
            'pcov', 'pt_u_coords', 'pt_v_coords', 'u_mean', 'v_fid' (fitted v at
            ``u_mean``), 'efp_x', 'efp_y', 'efp_w', 'ref_ech_ord', 'opticon',
            'pri_ang', 'ech_ang', 'fits_name', 'theta', 'theta_err'.

        Notes
        -----
        - Inconsistency: ``Globals.mos_idx`` (= {2: 0, 1: 1, 3: 2, 4: 3}) is the
          ScopeSim HDU order; ``Filer.read_mosaic`` stores ToySim HDUs at
          ``det_no - 1``, so for ToySim data the slice map of detector 2 is used
          for detector 1 and vice versa (appears to work only because both carry
          the same slices at similar rows).
        - Inconsistency: ``int(hdu.header['ID'].strip())`` requires a string
          keyword; an integer 'ID' raises AttributeError.
        - Open question: only the brightest trace per slice is found, so the
          input must be a single-pinhole (or single-line) image.
        - Open question: the intercept guess ``p0_guess[0] = u_mean`` is the mean
          along-trace coordinate rather than a v value.
        - Open question: when clipping reduces the sample below four points the
          last successful ``popt`` is kept without further checks.
        - The 'Gaussian fit failed' message can never print because
          ``_get_trace_coordinates`` always returns ``is_error = False``.
        - Clean-up: the previous docstring described the mosaic as a tuple.
        """
        name, prim_hdr, hdul = mosaic.name, mosaic.primary_hdr, mosaic.hdu_list

        is_alpha = trace_type == 'iso-alpha'

        efp_w = -1. if is_alpha else lt_wave
        efp_chop_x = prim_hdr['HIERARCH ACHG CFO CHOP X']
        efp_chop_y = prim_hdr['HIERARCH ACHG CFO CHOP Y']
        efp_x, efp_y = efp_chop_x, efp_chop_y

        ref_ech_ord = prim_hdr['HIERARCH ACHG REF_ECH_ORD']
        opticon = prim_hdr['HIERARCH AIT OPTICON']
        pri_ang = prim_hdr['HIERARCH AIT PRI_ANG']
        ech_ang = prim_hdr['HIERARCH AIT ECH_ANG']

        det_traces = []
        # _, _, slice_map_hdus = slice_map
        fmt = ''
        if Globals.is_debug('high'):
            rowcol_tag = 'Row' if is_alpha else 'Column'
            sli_tag = rowcol_tag + ' in slice'
            abs_tag = rowcol_tag + ' in image'
            print()
            print("Slice by slice {:s} trace extraction".format(trace_type))
            fmt = "{:>12s},{:>12s},{:>20s},{:>20s},{:>12s},{:>12s},{:>12s},{:>12s}"
            print(fmt.format('Detector', 'Slice ', 'Brightest', 'Brightest', 'Signal', 'Bgd', 'Noise', 'SNR'))
            print(fmt.format('Number  ', 'Number',  sli_tag   , abs_tag    ,   'DN/sec', 'DN/sec', '1 sigma', '-'))
            fmt = "{:>12d},{:>12d},{:>20d},{:>20d},{:>12.2f},{:>12.2f},{:>12.2f},{:>12.1f}"

        for hdu in hdul:
            det_no = int(hdu.header['ID'].strip())
            mos_idx = Globals.mos_idx[det_no]
            image = np.array(hdu.data)

            slice_map_data = np.array(slice_map.hdu_list[mos_idx].data)
            snu = np.unique(slice_map_data)
            slice_nos = [int(s) for s in snu if s > 0.]
            # Start by extracting the data for each slice.
            for slice_no in slice_nos:
                ism = int(0.5 * Globals.intra_slice_gap)                # Intra-slice margin
                idx = np.argwhere(slice_no == slice_map_data)
                slice_row_min, slice_row_max = np.amin(idx[:, 0]) - ism, np.amax(idx[:, 0]) + ism

                # Extract slice image
                slice_image = np.array(image[slice_row_min:slice_row_max, :])
                (u_axis, v_axis) = (1, 0) if is_alpha else (0, 1)
                v_count = slice_image.shape[v_axis]

                # We locate traces by collapsing along the u direction (rows for iso-alpha, columns for iso-lambda)
                # Currently only the brightest trace in each slice is found, so the input data should be for a
                # single pinhole only.  This approach simplifies the estimation of beta_phase, the across slice
                # position of the trace.
                noise = np.median(np.std(slice_image, axis=u_axis))      # Across-trace noise measure for slice image
                bgd = np.median(slice_image)

                s_aves = np.mean(slice_image, axis=u_axis)
                v_max_signal = np.argmax(s_aves)                         # Brightest row in slice (alpha) / column (lambda)
                row_max_signal = slice_row_min + v_max_signal
                v1_sig, v2_sig = v_max_signal - 5, v_max_signal + 5      # Rows/cols to use for trace analysis
                v1_sig, v2_sig = max(0, v1_sig), min(v_count, v2_sig)
                signal = np.mean(s_aves[v1_sig: v2_sig])
                snr = (signal - bgd) / noise
                is_low_snr = snr < snr_cut
                if Globals.is_debug('high'):
                    text = fmt.format(det_no, slice_no, v_max_signal, row_max_signal, signal, bgd, noise, snr)
                    if is_low_snr:
                        text += ' Low SNR'
                    print(text)
                if is_low_snr:
                    continue
                pt_u_coords, pt_v_coords, is_error = Opt02._get_trace_coordinates(slice_image, slice_row_min,
                                                                                  v_max_signal, is_alpha)
                u_mean = np.mean(pt_u_coords)       # Initialise trace parameters (overwritten by fit)
                v_fid = np.mean(pt_v_coords)
                popt, pcov = None, None
                if is_error and Globals.is_debug('high'):
                    text = fmt.format(det_no, slice_no, v_max_signal, row_max_signal, signal, bgd, noise, snr)
                    print(text + ' Gaussian fit failed')

                p0_guess = [0.]*4
                n_func_pars = len(p0_guess)

                sigma_filter = True
                while sigma_filter:
                    n_data_points = len(pt_u_coords)
                    if n_data_points < n_func_pars:
                        sigma_filter = False
                        # print('Skipping polynomial fit')
                        continue
                    u_mean = np.mean(pt_u_coords)
                    p0_guess[0] = u_mean
                    with warnings.catch_warnings(record=True):
                        popt, pcov = curve_fit(Globals.polynomial, pt_u_coords, pt_v_coords, p0=p0_guess)
                    vs = Globals.polynomial(pt_u_coords, *popt)
                    displacements = np.array(pt_v_coords) - vs
                    clip_limit = np.std(displacements) * sigma_clip
                    indices = np.argwhere(np.abs(displacements) > clip_limit)
                    if indices.shape[0] == 0:
                        sigma_filter = False
                        # print(slice_no, n_data_points, 'No points to delete')
                    else:
                        idx = indices[0, 0]     # Remove points one at a time
                        # print(slice_no, n_data_points, pt_v_coords[idx], displacements[idx])
                        pt_u_coords = np.delete(pt_u_coords, idx, axis=0)
                        pt_v_coords = np.delete(pt_v_coords, idx, axis=0)
                    v_fid = Globals.polynomial(u_mean, *popt)[0]

                if popt is None:
                    # print(slice_no, n_data_points, 'No polynomial trace fit found')
                    continue
                lms_config = Globals.lms_config_template.copy()
                lms_config['opticon'] = opticon
                lms_config['pri_ang'] = pri_ang
                lms_config['ech_ang'] = ech_ang
                version = 'ait'
                lms_config_id = Globals.make_lms_config_id(lms_config)
                det_trace = {'lms_config_id': lms_config_id, 'version': version,
                             'type': trace_type,
                             'det_no': det_no, 'slice_no': slice_no,
                             'snr': snr, 'signal': signal,
                             'popt': popt, 'pcov': pcov, 'pt_u_coords': pt_u_coords, 'pt_v_coords': pt_v_coords,
                             'u_mean': u_mean, 'v_fid': v_fid, 'efp_x': efp_x, 'efp_y': efp_y, 'efp_w': efp_w,
                             'ref_ech_ord': ref_ech_ord, 'opticon': opticon, 'pri_ang': pri_ang, 'ech_ang': ech_ang,
                             'fits_name': name
                             }
                det_traces.append(det_trace)
        det_traces = OptTools._find_thetas(det_traces)          # Add detector rotation angle measurements
        print("- {:d} {:s} traces found with snr > {:4.1f}".format(len(det_traces), trace_type, snr_cut))
        if Globals.is_debug('high'):
            Opt02._print_det_traces(det_traces, trace_type)
        return det_traces

    @staticmethod
    def _print_det_traces(det_traces, type):
        """Print a table of trace positions and tilt angles.

        Parameters
        ----------
        det_traces : list of dict
            Traces with 'det_no', 'slice_no', 'u_mean', 'v_fid', 'theta'.
        type : str
            'iso-alpha' or 'iso-lambda'; selects the column labels.

        Returns
        -------
        None

        Notes
        -----
        - Bug: (labels only) ``'row' if type == 'iso-lambda' else 'col' +
          '_mean'`` binds as ``'row' if ... else ('col' + '_mean')``, so the
          '_mean' / '_fiducial' suffixes are only appended in the else branch.
        - Clean-up: ``type`` shadows the builtin.
        """
        print('Trace type = ', type)
        fmt = "{:>10s},{:>10s},{:>12s},{:>12s},{:>20s}"
        u_tag = 'row' if type == 'iso-lambda' else 'col' + '_mean'
        v_tag = 'col' if type == 'iso-alpha' else 'row' + '_fiducial'
        print(fmt.format('Det. no.', 'Slice no.', u_tag, v_tag, 'theta_disp / deg'))
        fmt = "{:10d},{:10d},{:12.3f},{:12.3f},{:20.3f}"
        for det_trace in det_traces:
            det_no = det_trace['det_no']
            slice_no = det_trace['slice_no']
            u_mean = det_trace['u_mean']
            v_fid = det_trace['v_fid']
            rot_angle = det_trace['theta']
            print(fmt.format(det_no, slice_no, u_mean, v_fid, rot_angle))
        return

    @staticmethod
    def _get_trace_coordinates(slice_image, rs_min, v_max, is_alpha):
        """Sample the centroid of a trace at several positions along it by Gaussian fitting.

        The slice strip is sampled at ``n_samples`` (10 for iso-alpha, 5 for
        iso-lambda) positions ``u`` spaced ``u_count // (n_samples + 1)`` apart.
        At each, the window ``u +- 5`` (along) by ``v_max +- 10`` (across) is
        averaged along u and ``Globals.gauss`` is fitted to the across-trace
        profile (initial guess: peak value, centre ``v1 + 10``, sigma 1 pixel).
        Coordinates are returned in the full detector image frame: for iso-alpha
        traces u is the column and v the row (``rs_min`` added); for iso-lambda
        traces u is the row (``rs_min`` added) and v the column.

        Parameters
        ----------
        slice_image : numpy.ndarray
            Strip of about 130 rows x 2048 columns covering one slice.
        rs_min : int
            Row of the strip's first row in the detector image.
        v_max : int
            Across-trace index (row for alpha, column for lambda) of the
            brightest position within the strip.
        is_alpha : bool
            True for iso-alpha traces.

        Returns
        -------
        pt_u_coords : list of float
            Along-trace coordinates of the successful samples.
        pt_v_coords : list of float
            Fitted centroid at each sample.
        is_error : bool
            Always False (see notes).

        Notes
        -----
        - Bug: fit failures are swallowed by a bare ``except: a = 1``; the
          sample is silently dropped and ``is_error`` is never set to True.
        - Open question: the centre guess is ``v1 + v_hw`` (the window centre)
          rather than ``v_vals[idx_max]``; near the strip edges, where the window
          is clipped, the guess is off-centre.
        - Clean-up: returns lists that the caller immediately converts with
          ``np.delete`` / ``np.mean``.
        """
        (u_off, u_axis, v_axis) = (0, 1, 0) if is_alpha else (rs_min, 0, 1)
        u_count = slice_image.shape[u_axis]
        v_count = slice_image.shape[v_axis]

        n_samples = 10 if is_alpha else 5
        u_interval = u_count // (n_samples + 1)
        u_start, u_end = u_interval, u_count - u_interval
        u_list = list(range(u_start, u_end, u_interval))
        u_hw, v_hw = 5, 10       # Sample half width in along and across trace dimensions to fit gaussian.

        v1, v2 = v_max - v_hw, v_max + v_hw  # Across trace rows/cols to use for trace analysis
        v1, v2 = max(v1, 0), min(v2, v_count)
        pt_u_coords, pt_v_coords = [], []
        for u in u_list:
            # Find the trace coordinates by fitting gaussians
            u1, u2 = u - u_hw, u + u_hw
            u1, u2 = max(u1, 0), min(u2, u_count)
            sample_image = slice_image[v1:v2, u1:u2] if is_alpha else slice_image[u1:u2, v1:v2]
            z_vals = np.mean(sample_image, axis=u_axis)
            v_vals = np.array(list(range(v1, v2)))
            idx_max = np.argmax(z_vals)
            z_max = z_vals[idx_max]
            v_sigma = 1.0
            p0_guess = [z_max, v1 + v_hw, v_sigma]
            is_error = False
            try:
                gopt, gcov = curve_fit(Globals.gauss, v_vals, z_vals, p0=p0_guess)
                v_off = rs_min if is_alpha else 0
                v_cen = gopt[1] + v_off     # Get the row/col coordinate in the image frame.
                pt_u_coords.append(float(u + u_off))
                pt_v_coords.append(v_cen)
            except:
                a = 1

        return pt_u_coords, pt_v_coords, is_error

    @staticmethod
    def _find_trace_intersections(alpha_traces, lambda_traces, affines):
        """Intersect iso-alpha and iso-lambda traces to obtain distortion fiducial points.

        For each detector, every iso-alpha trace is paired with every iso-lambda
        trace on the same detector with the same 'slice_no' and the same
        'lms_config_id'.  The intersection of ``row = a(col)`` (alpha fit) and
        ``col = l(row)`` (lambda fit) is found by fixed-point iteration from
        (1024, 1024) until both updates change by less than 0.01 pixel.  Each
        point carries the entrance focal plane coordinates ``efp_x, efp_y`` from
        the alpha trace (pinhole chop offsets) and ``efp_w`` from the lambda
        trace (laser wavelength), and is projected into the mosaic focal plane
        with ``Util.dfp_to_mfp(affines, ...)``.

        Parameters
        ----------
        alpha_traces : list of dict
            Iso-alpha traces (one per detector per pinhole frame, from
            ``_find_beta_phase``).
        lambda_traces : list of dict
            Iso-lambda traces.
        affines : object
            Per-detector affine matrices returned by ``Transform.create_affines``.

        Returns
        -------
        dict
            ``{lms_config_id: points}`` with ``points`` = {'lms_config' (opticon,
            pri_ang, ech_ang), 'ref_ech_ord', 'det_no', 'slice_no', 'row', 'col',
            'efp_x', 'efp_y', 'efp_w' (lists), 'mfp_x', 'mfp_y' (arrays)}.
            Stored as ``as_built['dist_coord']`` and written by
            ``_print_dist_coord``.

        Notes
        -----
        - Open question: the fixed-point iteration has no iteration cap; if the
          two polynomials do not attract it does not terminate.
        - Bug: ``Util.dfp_to_mfp`` documents that all points are assumed to lie
          on the detector of the first point, but ``points`` accumulates points
          from all four detectors of a configuration (outer loop over
          ``det_no``), so the mosaic coordinates of points on later detectors
          appear to use the wrong affine.
        - Clean-up: ``mfp_x`` / ``mfp_y`` are recomputed for all points after
          every new point (quadratic cost); the ``np.array(det_no) != det_no``
          comparisons are needless.
        """
        # Select traces for each slice
        util = Util()
        dist_coord = {}
        for det_no in range(1, 5):
            for alpha_trace in alpha_traces:
                # Filter out traces on other detectors
                if np.array(alpha_trace['det_no']) != det_no:
                    continue
                slice_no_alpha = alpha_trace['slice_no']
                a_config_id = alpha_trace['lms_config_id']
                # print(a_config_id)
                for lambda_trace in lambda_traces:
                    if np.array(lambda_trace['det_no']) != det_no:
                        continue
                    slice_no_lambda = lambda_trace['slice_no']
                    if slice_no_alpha != slice_no_lambda:
                        continue
                    l_config_id = lambda_trace['lms_config_id']
                    # print(l_config_id)
                    # print()
                    if a_config_id == l_config_id:
                        if a_config_id in dist_coord:           # Add points to existing LMS configuration.
                            points = dist_coord[a_config_id]
                        else:                                   # New LMS configuration
                            lms_config = Globals.lms_config_template.copy()
                            lms_config['opticon'] = alpha_trace['opticon']
                            lms_config['pri_ang'] = alpha_trace['pri_ang']
                            lms_config['ech_ang'] = alpha_trace['ech_ang']
                            ref_ech_ord = alpha_trace['ref_ech_ord']
                            points = {'lms_config': lms_config, 'ref_ech_ord': ref_ech_ord,
                                      'det_no': [], 'slice_no': [], 'row': [], 'col': [],
                                      'efp_x': [], 'efp_y': [], 'efp_w': []}
                            dist_coord[a_config_id] = points
                        a_popt = np.array(alpha_trace['popt'])
                        l_popt = np.array(lambda_trace['popt'])
                        efp_x, efp_y, efp_w = alpha_trace['efp_x'], alpha_trace['efp_y'], lambda_trace['efp_w']
                        # Use an iterative method to find the intersection (y = a(x), x = w(y))
                        x, y, dx_res, dy_res = 1024., 1024., .01, .01
                        is_converged, count = False, 0
                        while not is_converged:
                            yp, = Globals.polynomial(x, *a_popt)
                            dy = math.fabs(yp - y)
                            xp, = Globals.polynomial(yp, *l_popt)
                            dx = math.fabs(xp - x)
                            x, y = xp, yp
                            count += 1
                            is_converged = dy < dy_res and dx < dx_res
                        points['det_no'].append(det_no)
                        points['slice_no'].append(slice_no_alpha)
                        points['col'].append(x)
                        points['row'].append(y)
                        points['efp_x'].append(efp_x)
                        points['efp_y'].append(efp_y)
                        points['efp_w'].append(efp_w)
                        dfp_xy = {'dfp_x': np.array(points['col']), 'dfp_y': np.array(points['row']),
                                  'det_nos': np.array(points['det_no'])}
                        mfp_xy = util.dfp_to_mfp(affines, dfp_xy)
                        points['mfp_x'], points['mfp_y'] = mfp_xy['mfp_x'], mfp_xy['mfp_y']
        return dist_coord

    @staticmethod
    def _print_dist_coord(filer, dist_coord, to_csv=False):
        """Print the distortion fiducial points, optionally into 'LMS_ait_dist_coord.csv'.

        For each LMS configuration a header block is printed ('Spectral order:'
        for nominal mode, 'Echelle angle:', 'Prism angle:'), then a column
        header and one line per point::

            slice_no, efp_w, efp_x, efp_y, mfp_x, mfp_y

        With ``to_csv=True`` the same text is written to
        ``filer.ray_trace_folder + '/LMS_ait_dist_coord.csv'`` by temporarily
        redirecting ``sys.stdout``.

        Parameters
        ----------
        filer : Filer
            Configured with ``set_configuration('distortion', opticon,
            is_ait_data=True)`` so that ``ray_trace_folder`` points at the AIT
            data folder ('../data/distortion_ait/nominal/<data_set>/').
        dist_coord : dict
            From ``_find_trace_intersections``.
        to_csv : bool, optional
            Write to file instead of stdout.  Default False.

        Returns
        -------
        str or None
            Path of the CSV file, or None when printing to stdout.

        Notes
        -----
        - Clean-up: redirecting ``sys.stdout`` leaves it redirected if an
          exception occurs; write to the file object instead.
        - Open question: 'det_no', 'row' and 'col' are not written, and all
          configurations are concatenated in one file; whether the ``lmsdist``
          CSV reader (written for the Zemax ray-trace layout) can consume this
          file is not verified here.
        """
        path = None
        if to_csv:
            filename = 'LMS_ait_dist_coord.csv'
            path = filer.ray_trace_folder + '/' + filename
            file = open(path, 'w')
            orig_stdout = sys.stdout
            sys.stdout = file

        for config in dist_coord:
            coord = dist_coord[config]
            lms_config = coord['lms_config']
            opticon = lms_config['opticon']
            pri_ang = lms_config['pri_ang']
            ech_ang = lms_config['ech_ang']
            ref_ech_ord = coord['ref_ech_ord']

            if opticon == Globals.nominal:
                print("{:s},{:d}".format('Spectral order:', ref_ech_ord))
            print("{:s},{:10.3f}".format('Echelle angle:', ech_ang))
            print("{:s},{:10.3f}".format('Prism angle:', pri_ang))
            print()
            fmt = "{:12s},{:12s},{:12s},{:12s},{:12s},{:12s},"
            print(fmt.format('slice_no', 'efp_w', 'efp_x', 'efp_y', 'mfp_x', 'mfp_y'))
            fmt = "{:12d},{:12.3f},{:12.3f},{:12.3f},{:12.3f},{:12.3f},"
            for i in range(0, len(coord['det_no'])):
                slice_no = coord['slice_no'][i]
                efp_w = coord['efp_w'][i]
                efp_x = coord['efp_x'][i]
                efp_y = coord['efp_y'][i]
                mfp_x = coord['mfp_x'][i]
                mfp_y = coord['mfp_y'][i]
                print(fmt.format(slice_no, efp_w, efp_x, efp_y, mfp_x, mfp_y))
        if to_csv:
            file.close()
            sys.stdout = orig_stdout
        return path