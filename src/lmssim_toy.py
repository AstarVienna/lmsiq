#!/usr/bin/env python
"""ToySim: geometric forward model of LMS detector frames (``lmssim`` program).

:class:`Toy` turns each observation of an AIT test into a four-detector LMS
FITS mosaic without ScopeSim.  Instead of ScopeSim effects it uses the
``lmsdist`` products directly: per-slice SVD distortion transforms
(``lms_efp_mfp_*.fits``), the MFP -> detector affine matrices
(``lms_dist_mfp_dfp_v{date}.fits``) and, on the nearest-transform path, the
polynomial fit parameters.  Sources, masks, blaze and PSFs come from
``lmssim_model.Model``; detector noise from ``lms_detector.Detector.detect``.

Input files (relative to ``src/``; all outside the repository)
----------------------------------------------------------------
* ``../config/sim_template.fits``: header template.  Its primary header is
  reused (``ORIGIN`` etc. overwritten) and the data shape of HDU 1 sets the
  detector format.  The ``HIERARCH ESO DET DIT``, ``ESO DET NDIT`` and
  ``ESO INS WLEN CEN`` keywords that ``lmsaiv`` reads therefore come from
  this template, not from the observation.
* ``../output/distortion/{opticon}/svd_fits_index_{nom|ext}_zmx.pkl``: index
  of stored transforms keyed by configuration code (``'eo24_ean05'`` style),
  used when ``lms_cfg_code`` starts with ``'eo'``.
* ``{Filer.tf_dir}lms_efp_mfp_*.fits``, ``lms_dist_mfp_dfp_v{date}.fits`` and
  ``lms_dist_efp_mfp_*_fit_parameters_v{date}.fits`` through
  ``lms_filer.Filer`` (``set_configuration('distortion', opticon, False)``).
* Zemax PSFs, WCU mask files and the Leiden sky through ``lmssim_model``.

Output
------
``../data/test_toysim/{cfg_id}_000.fits`` per observation (the folder must
exist): primary HDU plus ``DET1.DATA`` .. ``DET4.DATA`` image extensions.

Header keywords written
-----------------------
Primary: ``ORIGIN = 'TOYSIM'``, ``HIERARCH ACHG REF_ECH_ORD``,
``HIERARCH ACHG CFO CHOP X`` / ``Y`` (EFP chop offset, mm),
``HIERARCH ACHG LASER WAVE`` (um), ``AIT OPTICON``, ``AIT PRI_ANG``,
``AIT ECH_ANG`` (from the last transform's LMS configuration), plus
everything the template carries.  Extensions: ``EXTNAME = 'DET{n}.DATA'``,
``ID`` (detector number, string), ``CRVAL1D`` / ``CRVAL2D`` and ``X_CEN`` /
``Y_CEN`` (detector centre, +-19.547 mm, strings), ``HIERARCH ESO DET3 CHIP
GAIN`` ('2.00' el/ADU, string), ``HIERARCH AIT PIXEL_PITCH`` (18.0 um).

These ``AIT`` / ``ACHG`` keywords are the only header contract that
``lmsaiv_opt02`` understands; ScopeSim does not write them (Inconsistency).

Notes
-----
* Open question: fluxes are scaled by ``lmssim_model`` ``k_toy_ss = 0.00177``
  "to transform from toysim to scopesim flux values"; provenance unknown.
* Inconsistency: ``CRVAL1D`` is +19.547 mm for detector 1 and -19.547 mm for
  detector 2, the opposite sign to the irdb ``FPA_metis_lms_layout.dat``
  (detector 1 at x = -19.547 mm) and to ``Globals.det_lims`` /
  ``Transform.create_affines`` (detector 1 at negative x).
* Bug: the nearest-transform path (``lms_cfg_code`` not starting with
  ``'eo'``) calls ``Util.find_closest_transforms(wave_ref, ...)`` with
  ``wave_ref`` still None (it is only assigned on the fixed-transform path)
  and then ``PolyFit.wave_to_config(wave_ref.value / 1000., ...)`` although
  ``wave_to_config`` expects um (``wxo_lim`` = 108-115 um x order); the path
  fails or mis-scales by 1000.
* Bug: ``n_exp = int(sim_config['lms_nexp'])`` is immediately overwritten by
  ``n_exp = 1``; only one exposure is ever written.
* Open question: the extended-source convolution uses the slice-13 PSF for
  every slice, and ``psf_dict`` is loaded once for the first order processed
  and reused for all other orders.
* Clean-up: ``waves_mosaic`` and ``tau_ech_mosaic`` are filled but never
  written (the ``debug`` branch writes the image HDUs again under a
  ``_waves`` name); ``w_pnh``, ``mfp_slice`` / ``oob``, ``psf_sig``,
  ``n_rows_written``, ``w_illuminated`` and ``fits_waves_out_path`` are
  unused; the local ``model_configurations`` duplicates
  ``Globals.model_configurations`` with ``coord_out = ('det_x', 'det_y')``
  instead of ``('mfp_x', 'mfp_y')``.
"""
import time
import math
import numpy as np
import scipy.signal
from astropy import units as u
from astropy.io.fits import ImageHDU

from lms_obs_map import ObsMap
from lmsdist_util import Util
from lms_globals import Globals
from lms_filer import Filer
from lms_detector import Detector
from lmssim_model import Model
from lmsdist_polyfit import PolyFit


class Toy:
    """ToySim simulator: EFP source model -> distortion transforms -> detector mosaic.

    Stateless; :meth:`run` is a static method and ``__init__`` does nothing.
    """

    def __init__(self):
        """Create a Toy instance (no state).
        """
        return

    @staticmethod
    def run(sim_configs):
        """Simulate every observation of a test with ToySim and write the mosaics.

        Parameters
        ----------
        sim_configs : dict
            ``{cfg_id: sim_config}`` from ``lms_obs_map.ObsMap.get_configuration``.
            Keys read: ``'lms_msa'`` (opticon), ``'lms_cfg_code'``,
            ``'lms_ech_w_off'`` (wavelength offset simulating an echelle rotation,
            um), ``'lms_pp1'``, ``'wcu_fp2_1'``, ``'cfo_fp2'``, ``'cfo_chop_off_x'`` /
            ``'_y'`` (EFP chop offsets, mm), ``'lms_dit'``, ``'lms_ndit'``,
            ``'lms_nexp'``, plus those used by ``lmssim_model.Model``.

        Returns
        -------
        None

        Notes
        -----
        Algorithm per observation:

        1. Transform selection.  If ``lms_cfg_code`` starts with ``'eo'`` the
           configuration is looked up in the ``svd_fits_index_*_zmx`` pickle and the
           matching ``lms_efp_mfp_*.fits`` is read (fixed transform, giving
           ``wave_ref``, ``pri_ang``, ``ech_ang``, ``ech_ords``); otherwise
           ``Util.find_closest_transforms`` and ``PolyFit.wave_to_config`` are used
           (see Bug below).
        2. Header from ``../config/sim_template.fits``; ``ORIGIN``,
           ``ACHG REF_ECH_ORD`` (``ech_ords[1]`` in extended mode, else
           ``ech_ords[0]``), ``ACHG CFO CHOP X`` / ``Y`` and ``ACHG LASER WAVE`` are
           set.
        3. Focal-plane mask from ``Model.get_fp_mask``; each pinhole position is
           offset by the CFO chop and converted to a field slice number and beta
           phase with ``Util.efp_y_to_slice_no``.
        4. Sources: ``Model.load_source_lists``, ``Model.make_blaze_dictionary``,
           ``Model.make_spectra`` (per-order background and pinhole spectra).
        5. For every slice transform: blaze profile (``Model.make_tau_blaze``);
           slice centre row from ``Util.efp_to_mfp`` + ``Util.mfp_to_dfp``; then,
           for each detector the slice covers, a 250-row strip is built row by row
           with ``Util.dfp_to_efp`` (wavelength and alpha per pixel), filled with
           the interpolated background x blaze where ``|alpha| < efp_x_fov_mm / 2``,
           and convolved (``scipy.signal.convolve2d``, 'same', 'symm') with the
           slice-13 PSF.  Pinholes: for each pinhole within +-4 slices of the
           current slice the point-source spectrum is mapped with
           ``Util.efp_to_dfp``, split bilinearly between the two nearest rows,
           weighted by ``Model.fit_beta_weights`` and convolved with the
           offset-slice PSF.
        6. ``Detector.detect(image, dit, n_dit)`` adds dark, shot and read noise;
           four ``ImageHDU`` extensions are written with ``Filer.write_mosaic`` to
           ``../data/test_toysim/{cfg_id}_000.fits``.

        At debug level 'low' a round-trip test of ``Util.slice_to_efp_y`` /
        ``Util.efp_y_to_slice_no`` is printed first; at 'medium' a per-slice timing
        table.

        * Bug: on the nearest-transform path ``wave_ref`` is None when passed to
          ``Util.find_closest_transforms`` (only the ``'eo'`` path assigns it) and
          ``wave_to_config`` receives ``wave_ref.value / 1000.`` although it
          expects um.
        * Bug: ``n_exp`` is forced to 1 after being read from ``lms_nexp``.
        * Bug: ``beta_phase`` and ``beta_weight`` hold the values of the last
          pinhole processed in step 3 and are applied to every pinhole in step 5,
          so all pinholes are placed at that one beta phase.
        * Bug: the pinhole wavelengths ``efp_ws`` are those of the last illuminated
          row of the extended-background loop, not of the pinhole's own trace.
        * Bug: the extended-background loop covers 251 rows
          (``range(det_row_min, det_row_max + 1)``) for a 250-row strip; an
          illuminated last row raises ``IndexError``.  Column indices are only
          guarded against ``> 2047`` and row indices not at all.
        * Open question: ``psf_dict`` is loaded once (first order encountered) and
          reused for all orders; the slice-13 PSF is used for every slice.
        * Open question: for darks (``ObsMap.is_dark``) the flux calculation is
          skipped but transforms, spectra and blaze dictionary are still built, so
          darks need the same input files as illuminated frames.
        * Inconsistency: the ``'AIT ' + key.upper()`` header keys (``AIT OPTICON``,
          ``AIT PRI_ANG``, ``AIT ECH_ANG``) are taken from ``lms_cfg``, which on the
          fixed-transform path is the last transform's ``lms_configuration``; were
          ``opt_transforms`` empty, ``lms_cfg`` would still be the string left by
          the index loop.
        * Clean-up: ``ID``, ``CRVAL1D``, ``CRVAL2D``, ``X_CEN``, ``Y_CEN`` and
          ``ESO DET3 CHIP GAIN`` are written as formatted strings rather than
          numbers; readers apply ``int()`` / ``float()``.
        * Clean-up: the ``debug`` branch writes the same image HDUs a second time
          as ``{cfg_id}_000.fits_waves`` instead of the ``waves_mosaic`` wavelength
          maps.
        """
        wave_ref, w_pnh = None, None            # Define local variables
        psf_dict = None
        beta_phase, beta_weight = None, None
        analysis_type = 'distortion'
        filer = Filer()

        coord_in = 'efp_x', 'efp_y', 'wavelength'
        coord_out = 'det_x', 'det_y'

        nom_date_stamp = '20240109'
        nom_config = (analysis_type, Globals.nominal, nom_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      coord_in, coord_out)

        ext_date_stamp = '20260112'
        ext_config = (analysis_type, Globals.extended, ext_date_stamp,
                      'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                      coord_in, coord_out)
        model_configurations = {Globals.nominal: nom_config, Globals.extended: ext_config}
        model = Model()
        print(model)

        if Globals.is_debug('low'):        # Test mapping from EFP_Y to slice number, phase
            print('Test of Util.slice_to_efp_y and Util.efp_y_to_slice')
            slice_test_list = {'nom_min': (1, 0.), 'nom_bs': (14, 1.), 'nom_max': (28, 1.),
                               'ext_min': (12, 0.), 'ext_bs': (13, 0.5), 'ext_max': (14, 1.),
                               }
            fmt = "{:>20s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},"
            print(fmt.format('Location', 'slice_no_in', 'phase_in', 'EFP_Y', 'slice_no_out', 'phase_out'))
            fmt = "{:>20s},{:10d},{:10.2f},{:>10.3f},{:10d},{:10.2f},"
            for key in slice_test_list:
                slice_no1_test, phase1_test = slice_test_list[key]
                efp_y_test = Util.slice_to_efp_y(slice_no1_test, phase1_test)
                slice_no2_test, phase2_test = Util.efp_y_to_slice_no(efp_y_test)
                print(fmt.format(key, slice_no1_test, phase1_test, efp_y_test, slice_no2_test, phase2_test))

        # =================================================================================================
        t_start = time.perf_counter()

        # Model PSFs are available up to 4 slices away from the slice the target is centred on.
        zemax_psf_fslice_bounds = [-4, 4]

        for obs_name in sim_configs:
            print('Running simulation - ' + obs_name)

            sim_config = sim_configs[obs_name]
            print(sim_config)
            opticon = sim_config['lms_msa']
            is_ait_data = False      # Use stored SVD transforms
            filer.set_configuration(analysis_type, opticon, is_ait_data)
            lms_cfg_code = sim_config['lms_cfg_code']
            w_ech_off = 0.          # Wavelength shift to simulate echelle rotations.
            is_fixed_transform = lms_cfg_code[0:2] == 'eo'
            if is_fixed_transform:
                ait_tag = '_ait' if is_ait_data else '_zmx'
                fmt = "../output/distortion/{:s}/svd_fits_index_{:s}{:s}"
                svd_dict_path = fmt.format(opticon, opticon[0:3], ait_tag)
                svd_transform_dict = filer.read_pickle(svd_dict_path)
                w_ech_off = float(sim_config['lms_ech_w_off'])
                if Globals.is_debug('low'):         # Print out configuration codes
                    fmt = "{:>10s},{:>14s},{:>14s},{:>14s},{:>24s},"
                    print(fmt.format('Code', 'Ref.Order', 'Ech.Ang', 'Pri.Ang', 'Ref. Wavelength'))
                    fmt = "{:>10s},{:14d},{:14.3f},{:14.3f},{:24.3f},"
                    for code in svd_transform_dict:
                        svd = svd_transform_dict[code]
                        ech_ords = svd['ech_ords']
                        ref_ech_ord = ech_ords[0] if opticon == Globals.nominal else ech_ords[1]
                        print(fmt.format(code, ref_ech_ord, svd['ech_ang'], svd['pri_ang'], svd['wav_ref']))
                if Globals.is_debug('medium'):
                    fmt = "{:2s},{:<10s},{:<10s},{:<10s},{:<10s},{:<10s},{:<50s},"
                    print(fmt.format(' ', 'Config', 'Ref.', 'Prism', 'Ech', 'Ref. ech', 'FITS'))
                    print(fmt.format(' ', 'name', 'wave', 'angle', 'angle', 'order', 'file'))
                    fmt = "{:2s},{:<10s},{:10.3f},{:10.3f},{:>10.2f},{:>10d},{:>50s}"
                for lms_cfg in svd_transform_dict:
                    if lms_cfg != sim_config['lms_cfg_code']:
                        continue
                    svd_entry = svd_transform_dict[lms_cfg]
                    fits_name = svd_entry['fits_name']
                    wave_ref = svd_entry['wav_ref'] * u.micron
                    pri_ang = svd_entry['pri_ang']
                    ech_ang = svd_entry['ech_ang']
                    ech_orders = svd_entry['ech_ords']
                    ref_ech_ord = ech_orders[1] if opticon == Globals.extended else ech_orders[0]
                    svd_entry['ref_ech_ord'] = ref_ech_ord
                    if Globals.is_debug('high'):
                        print(fmt.format('- ', lms_cfg, wave_ref.value, pri_ang, ech_ang, ref_ech_ord, fits_name))
                svd_entry = svd_transform_dict[lms_cfg_code]
                wave_ref = svd_entry['wav_ref'] * u.micron
                fits_name = svd_entry['fits_name']
                if Globals.is_debug('medium'):
                    print(fmt.format('* ', lms_cfg_code, wave_ref.value, svd_entry['pri_ang'], svd_entry['ech_ang'],
                                     svd_entry['ref_ech_ord'], fits_name))
                opt_transforms = filer.read_svd_transforms(inc_tags=[fits_name])
            else:       # Find nearest transform.
                svd_transforms = filer.read_svd_transforms(exc_tags=['fit_parameters', 'mfp_dfp'])
                # Find the list of closest svd transforms for each slice.  This should really be replaced by
                # the interpolated transforms which have their polynomial fit parameters are read in below.
                opt_transforms, ech_orders = Util.find_closest_transforms(wave_ref, opticon, svd_transforms)
                # Calculate the prism and grating angles required to observe the target wavelength.
                # We use the nominal mode prism > wavelength calibration for all cases.
                wpa_fit, wxo_fit, term_fits = filer.read_fit_parameters(opticon)
                lms_cfg = PolyFit.wave_to_config(wave_ref.value / 1000., opticon, wpa_fit, wxo_fit,
                                                 select='min_ech_ang')

            efp_ws = None            # Initialise variables which may be assigned within loops etc.
            lms_pp1 = sim_config['lms_pp1']
            model_config = model_configurations[opticon]
            PolyFit(opticon)                    # Instantiate the polynomial fit tools.
            date_stamp = model_config[2]

            # Read header and data shape (only) in from the template.
            hdu_list = filer.read_zemax_fits('../config/sim_template.fits')
            primary_header = hdu_list[0].header
            primary_header['ORIGIN'] = 'TOYSIM'
            ref_ech_ord = ech_orders[1] if opticon == Globals.extended else ech_orders[0]
            primary_header['HIERARCH ACHG REF_ECH_ORD'] = ref_ech_ord

            # Build mosaic of detector images
            det_shape = hdu_list[1].data.shape
            _, n_det_cols = det_shape
            image_mosaic, waves_mosaic, tau_ech_mosaic = [], [], []
            for det_no in range(1, 5):
                image_mosaic.append(np.zeros(det_shape))
                waves_mosaic.append(np.zeros(det_shape))
                tau_ech_mosaic.append(np.zeros(det_shape))

            # For the pinhole emission, we loop through all pinholes, add CFO chopper position offsets, calculate
            # which slice the central image will fall on and store it in the fp_mask object.
            # there is a PSF which has a trace in this slice, we add it to the spectrum.
            fp_mask = model.get_fp_mask(sim_config['wcu_fp2_1'], sim_config['cfo_fp2'])
            fp_mask['fslice_no_pnh_cen'], fp_mask['efp_xy_pnh_cen'] = [], []
            cfo_chop_x, cfo_chop_y = sim_config['cfo_chop_off_x'], sim_config['cfo_chop_off_y']
            efp_chop_x, efp_chop_y = float(cfo_chop_x), float(cfo_chop_y)

            if fp_mask['efp_xy_bs'] is not None:
                fp_mask['fslice_bounds'] = zemax_psf_fslice_bounds
                efp_xy_list = fp_mask['efp_xy_bs']
                for efp_xy_bs in efp_xy_list:
                    efp_x_bs, efp_y_bs = efp_xy_bs
                    efp_x_pnh = efp_x_bs + efp_chop_x
                    efp_y_pnh = efp_y_bs + efp_chop_y
                    fslice_no_pnh_cen, beta_phase = Util.efp_y_to_slice_no(efp_y_pnh * u.mm)
                    fmt = "PSF will fall in fslice.phase {:2d}.{:5.3f}"
                    print(fmt.format(fslice_no_pnh_cen, beta_phase))
                    fp_mask['fslice_no_pnh_cen'].append(fslice_no_pnh_cen)
                    fp_mask['efp_xy_pnh_cen'].append([efp_x_pnh, efp_y_pnh])
            primary_header['HIERARCH ACHG CFO CHOP X'] = efp_chop_x
            primary_header['HIERARCH ACHG CFO CHOP Y'] = efp_chop_y
            primary_header['HIERARCH ACHG LASER WAVE'] = Model.get_laser_wavelength(sim_config)

            dit = float(sim_config['lms_dit'])          # 1.3  # Integration time in seconds.
            n_dit = int(sim_config['lms_ndit'])         # No. of integrations

            # Find EFP bounds
            efp_xmax = Globals.efp_x_fov_mm
            xh = efp_xmax / 2.

            # Load selected extended background spectrum (units ph/s/m2/as2/um) for wavelength range
            # which overfills mosaic.  This may be the mask structure,
            # f_units_ext_in = 'phot/s/m2/um/arcsec2',
            bgd_src_list, pnh_src_list, lt_w_offset = Model.load_source_lists(sim_config)

            # Prepare to skip flux calculations for darks
            is_dark = ObsMap.is_dark(sim_config)
            blaze = Model.make_blaze_dictionary(opt_transforms)
            spectra = Model.make_spectra(opt_transforms, bgd_src_list, pnh_src_list, lt_w_offset)
            out_folder = '../data/test_toysim/'

            # Set up dictionary of blaze wavelengths from ech_angle=0 transforms
            if Globals.is_debug('medium'):
                fmt = "{:>10s},{:>8s},{:>6s},{:>8s},{:>8s},{:>8s},{:>8s},{:>15s},{:>11s},{:>10s},{:>10s}"
                title_txt = fmt.format('t_elapsed', 'det_no', 'slice',
                                       'pri_ang', 'ech_ang', 'ech_ord',
                                       'w_blaze', 'w_range', 'det_rows', 'f_bgd_max', 'f_psf_max')
                print(title_txt)
            pnh_image = None        # May be no pinholes

            # Loop through each slice/transform.
            for opt_transform in opt_transforms:
                slice_cfg = opt_transform.slice_configuration
                lms_cfg = opt_transform.lms_configuration
                # if is_dark:
                #     continue

                ech_ord = slice_cfg['ech_ord']
                slice_no = slice_cfg['slice_no']
                fslice_no, pslice_no = Util.decode_slice_no(slice_no)
                # spifu_no = slice_cfg['spifu_no']
                ech_ang = lms_cfg['ech_ang']
                pri_ang = lms_cfg['pri_ang']

                # Load PSF dictionary.  The central PSF (calculated for a source centred on a slice) is used
                # for extended source (slice filling) convolution.  The PSFs for slices offset from the centre
                # are used for pinhole image convolution.  Method 'fit_beta_weights' adds weights to the
                # dictionary based on the across slice centring (beta_phase).
                # PSFs are sampled at 4x detector resolution so need to be down-sampled.  We may wish to do this
                # AFTER the convolution.
                if psf_dict is None:
                    psf_dict = Model.load_psf_dict(opticon, ech_ord, downsample=True)
                    _, psf_ext = psf_dict[0]
                if beta_phase is not None:
                    beta_weight = Model.fit_beta_weights(psf_dict, beta_phase)

                waves, f_ext, f_pnh = spectra[ech_ord]
                affines = filer.read_fits_affine_transform(date_stamp)

                n_det_rows_slice = 250  # Approx no. of rows per slice, with comfortable margin...

                w_blaze, tau_blaze = Model.make_tau_blaze(blaze, ech_ord, ech_ang)  # Make echelle blaze profile for this order.
                idx_max = np.argmax(tau_blaze)
                w_blaze_max = w_blaze[idx_max]

                w_min, w_max = slice_cfg['w_min']*u.micron, slice_cfg['w_max']*u.micron
                yc = Util.slice_to_efp_y(fslice_no, 0.5)  # Slice y (beta) coordinate in EFP

                efp_ys = np.array([yc.value]*2)*yc.unit
                efp_xs = np.array([0.]*2)*yc.unit                  # Slice x (alpha) bounds in EFP, map to dfp_y
                efp_w = (np.array([w_min.value, w_max.value]) + w_ech_off)*u.micron
                efp_slice = {'efp_y': efp_ys, 'efp_x': efp_xs, 'efp_w': efp_w}    # Centre of slice for detector
                mfp_slice, oob = Util.efp_to_mfp(opt_transform, efp_slice)
                dfp_slice = Util.mfp_to_dfp(affines, mfp_slice)
                det_row_min = int(dfp_slice['dfp_y'][0] - 100)      # Bracket slices which typically cover 120 rows.
                det_row_max = det_row_min + n_det_rows_slice
                strip_shape = n_det_rows_slice, n_det_cols
                psf_max = 0.0

                for det_no in dfp_slice['det_nos']:
                    det_idx = det_no - 1
                    ext_ill = np.zeros(strip_shape)
                    psf_sig = np.zeros(strip_shape)

                    if is_dark:
                        print('Dark frame')
                        continue

                    dfp_det_nos = np.full(n_det_cols, det_no)
                    dfp_pix_xs = np.arange(n_det_cols)              # Detector column indices

                    # ########################################################################################
                    #  EXTENDED BACKGROUND
                    n_rows_written = 0
                    w_illuminated = []
                    for det_row in range(det_row_min, det_row_max + 1):
                        strip_row = det_row - det_row_min
                        dfp_pix_ys = np.full(n_det_cols, det_row)
                        dfp_row = {'dfp_x': dfp_pix_xs, 'dfp_y': dfp_pix_ys, 'det_nos': dfp_det_nos}
                        efp_row = Util.dfp_to_efp(opt_transform, affines, dfp_row)
                        efp_xs = efp_row['efp_x']
                        idx_illum = np.argwhere(np.abs(efp_xs) < xh.value)
                        n_ib = len(idx_illum)
                        if n_ib == 0:                       # Skip unilluminated rows.
                            continue
                        # Apply extended spectrum (sky or black body) to illuminated rows.  Add a wavelength
                        # offset to simulate echelle rotation.
                        efp_ws = efp_row['efp_w']*u.micron + w_ech_off*u.micron
                        w_obs = efp_ws[idx_illum][:]
                        tau_echelle = np.interp(w_obs, w_blaze, tau_blaze)
                        tau_ech_mosaic[det_idx][det_row, idx_illum] = tau_echelle
                        f_ext_obs = np.interp(w_obs, waves, f_ext)
                        ext_ill[strip_row, idx_illum] = f_ext_obs * tau_echelle
                        waves_mosaic[det_idx][det_row, idx_illum] = w_obs
                        w_illuminated = w_illuminated + list(w_obs[:])
                        n_rows_written += 1
                    # Now convolve background flux map with bright slice psf. (ideally use filled slice psf)
                    # To start with, we just use PSFs for the boresight slice (slice no. 13)
                    image = image_mosaic[det_idx]
                    image[det_row_min:det_row_max, :] += scipy.signal.convolve2d(ext_ill, psf_ext,
                                                                                 mode='same', boundary='symm')

                    # No pinholes selected, so go to the next detector for this slice.
                    is_no_pinholes = fp_mask['id'] in ['open', 'closed'] or lms_pp1 == 'closed'
                    if is_no_pinholes:
                        continue

                    sno_radius = list(psf_dict.keys())[-1]
                    # ##########################################################################################
                    #  PINHOLES !!
                    # Add pinhole illumination to the blank images matched to the PSFs for 'off-centre' slices.
                    psf_illum = {}
                    for slice_no_offset in psf_dict:
                        psf_illum[slice_no_offset] = np.zeros(Globals.det_format)

                    psf_max = 0.0
                    fslice_no_pnh_cens = np.array(fp_mask['fslice_no_pnh_cen'])
                    for fslice_no_pnh_cen in fslice_no_pnh_cens:
                        fslice_no_offset = fslice_no - fslice_no_pnh_cen
                        in_range = math.fabs(fslice_no_offset) <= sno_radius
                        if not in_range:
                            continue
                        _, psf_pnh = psf_dict[fslice_no_offset]
                        efp_xy_list = fp_mask['efp_xy_pnh_cen']
                        psf_ill_image = psf_illum[fslice_no_offset]
                        r1, r2 = 3000, 0        # Min max rows to convolve
                        for efp_xy in efp_xy_list:
                            # Build pinhole point source spectrum in the EFP data cube
                            efp_x = efp_xy[0]
                            efp_y = Util.slice_to_efp_y(fslice_no, beta_phase)
                            n_vals, = efp_ws.shape
                            efp_ys = np.full(n_vals, efp_y)
                            efp_xs = np.full(n_vals, efp_x)

                            efp_pnh = {'efp_x': efp_xs, 'efp_y': efp_ys, 'efp_w': efp_ws}
                            f_ps = np.interp(efp_ws, waves, f_pnh)      # Point source flux

                            dfp_pnh = Util.efp_to_dfp(opt_transform, affines, efp_pnh)

                            # Populate the psf_illum image column by column, with
                            dfp_x_pnh = dfp_pnh['dfp_x']
                            dfp_y_pnh = dfp_pnh['dfp_y']
                            row_min, row_max = int(np.amin(dfp_y_pnh)), int(np.amax(dfp_y_pnh))
                            r1 = r1 if r1 < row_min else row_min
                            r2 = r2 if r2 > row_max else row_max
                            dfp_ys = dfp_pnh['dfp_y']
                            dfp_cols = np.floor(dfp_x_pnh).astype(int)
                            dfp_rows = np.floor(dfp_y_pnh).astype(int)
                            phase = dfp_ys - dfp_rows
                            wt_rows = 1. - np.fabs(phase - 0.5)     # Fraction of flux in row
                            is_above = phase > 0.5
                            f = f_ps.value * beta_weight[fslice_no_offset]
                            for i in range(0, dfp_cols.shape[0]):
                                dfp_col = dfp_cols[i]
                                if dfp_col > 2047:
                                    continue
                                dfp_row = dfp_rows[i]
                                wt = wt_rows[i]
                                psf_ill_image[dfp_row, dfp_col] += wt * f[i]
                                drow = 1 if is_above[i] else -1
                                psf_ill_image[dfp_row+drow, dfp_col] += (1. - wt) * f[i]

                        r1, r2 = r1 - 5, r2 + 5
                        image[r1:r2, :] += scipy.signal.convolve2d(psf_ill_image[r1:r2, :], psf_pnh,
                                                                       mode = 'same', boundary = 'symm')
                        psf_max_slice = np.amax(psf_ill_image[r1:r2, :])
                        psf_max = psf_max if psf_max > psf_max_slice else psf_max_slice

                if Globals.is_debug('medium'):
                    t_now = time.perf_counter()
                    t_el = int(t_now - t_start)
                    det_no_text = '12' if det_no < 3 else '34'
                    fmt = "{:10d},{:>8s},{:6d},{:8.3f},{:8.3f},{:8d},{:8.0f},{:8.0f},{:6.0f},{:5d},{:5d},{:10.1f},{:10.1f}"
                    txt = fmt.format(t_el, det_no_text, slice_no, pri_ang, ech_ang, ech_ord,
                                     int(w_blaze_max.to(u.nm).value), int(w_min.to(u.nm).value), int(w_max.to(u.nm).value),
                                     det_row_min, det_row_max, np.amax(ext_ill), psf_max)
                    print(txt)

            n_exp = int(sim_config['lms_nexp'])
            n_exp = 1
            for obs_idx in range(0, n_exp):

                # Add dark current and read noise to illumination image (Finger, Rauscher)
                hdu_list = []
                for det_no in range(1, 5):
                    det_idx = det_no - 1
                    image = image_mosaic[det_idx]
                    frame = Detector.detect(image, dit, n_dit)
                    hdu = ImageHDU(frame)
                    hdu.name = "DET{:d}.DATA".format(det_no)
                    hdu.header['ID'] = "{:d}".format(det_no)
                    c = 19.547
                    crval1d = {1: +c, 2: -c, 3: -c, 4: +c}
                    crval2d = {1: +c, 2: +c, 3: -c, 4: -c}
                    hdu.header['CRVAL1D'] = "{:8.3f}".format(crval1d[det_no])
                    hdu.header['CRVAL2D'] = "{:8.3f}".format(crval2d[det_no])
                    hdu.header['X_CEN'] = "{:8.3f}".format(crval1d[det_no])
                    hdu.header['Y_CEN'] = "{:8.3f}".format(crval2d[det_no])
                    el_adu = 2.0
                    hdu.header['HIERARCH ESO DET3 CHIP GAIN'] = "{:8.2f}".format(el_adu)
                    hdu.header['HIERARCH AIT PIXEL_PITCH'] = Globals.nom_pix_pitch
                    hdu_list.append(hdu)

                print()
                obs_tag = "_{:03d}.fits".format(obs_idx)
                for key in lms_cfg:
                    primary_header['AIT ' + key.upper()] = lms_cfg[key]

                file_name = obs_name + obs_tag
                print("Writing fits file - {:s}{:s}".format(out_folder, file_name))
                mosaic = file_name, primary_header, hdu_list
                filer.write_mosaic(out_folder, mosaic)

                debug = False
                if debug:
                    fits_waves_out_path = out_folder + '_waves.fits'
                    print("Writing fits file - {:s}".format(fits_waves_out_path))
                    print()
                    mosaic = file_name + '_waves', primary_header, hdu_list
                    filer.write_mosaic(out_folder, mosaic)
        return
