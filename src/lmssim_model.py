#!/usr/bin/env python
"""Source, throughput and PSF models for the ToySim simulator (``lmssim`` program).

:class:`Model` supplies ``lmssim_toy.Toy`` with everything that is not
geometry: the calibration-source spectra (WCU black body at 1000 K, the 300 K
ambient WCU mask, the 70 K CFO mask, the fixed 3.390 um and 5.240 um lasers,
the tunable laser at 4.700 um + offset, and the Leiden sea-level sky), the
focal-plane mask definitions (CFO pinhole, WCU steerable pinhole and grid,
open and closed positions), the echelle blaze profile, the Zemax PSF
dictionary for slice 13 and its neighbours, and the across-slice PSF
weights.

Input files (relative to ``src/``; all outside the repository)
----------------------------------------------------------------
* Zemax PSFs: ``{Filer.psf_folder}lms_{date}_config{ccc}_field{fff}_defoc000um/``
  ``lms_{date}_config{ccc}_field{fff}_defoc000um_spat{ss}_spec{p}_detdesi.fits``
  with ``Filer.psf_folder`` = ``../data/iq/{opticon}/{date_stamp}/``
  (``date_stamp`` from ``Globals.iq_nom_config`` / ``iq_ext_config``),
  ``ccc`` = 41 - echelle order (nominal) or 000 (extended), ``fff`` = 001
  (nominal) or 001-003 (extended), ``ss`` = field slice number, ``p`` = 0
  (nominal) or 1 (extended).  PSF image in the primary HDU, sampled at 4x
  the detector pixel.
* WCU masks: ``./inst_pkgs/METIS/wcu/fp_mask_pinhole_lm.dat`` and
  ``fp_mask_grid_lm.dat`` through ``lms_filer.Filer.read_pinholes``.
* Sky: ``../data/sky/leiden/LBL_A10_w3000_R0120000_ALL_Leiden_LM_R.fits``
  (HDU 1 table, columns ``lam`` in um and ``flux`` in ph/s/m^2/arcsec^2/um).

Nothing is written.  No header keywords are interpreted; the PSF primary
header is passed through unchanged in the PSF dictionary.

Notes
-----
* Clean-up: ``load_psf_dict`` is defined twice in the class body; the first
  definition is dead (shadowed by the second) and would fail anyway because
  it calls ``Filer(model_config)`` with an argument ``Filer.__init__`` does
  not accept, and it hard-codes an extended-mode date stamp ('2024061802')
  that differs from ``Globals.iq_ext_config`` ('2024032401').
* Open question: :meth:`Model.make_spectra` multiplies every spectrum by
  ``k_toy_ss = 0.00177`` "to transform from toysim to scopesim flux values";
  the origin of this factor is not recorded.
* Inconsistency: the ``'tau'`` values in ``Model.bgd_srcs`` are never read;
  :meth:`Model.get_flux` applies ``Model.tau_whs`` to every black body
  (including the 70 K CFO mask and 300 K ambient mask that the table assigns
  ``tau_cfo_lms`` and ``tau_wfp``) and to the lasers, and ``Model.tau_sky``
  to the sky.
* Inconsistency: black-body and laser fluxes use pi * (39 / 2)^2 = 1194.6 m^2
  as the telescope area, the sky uses ``Globals.elt_area`` = 1350 m^2.
* Clean-up: ``synphot.units`` is imported twice (as ``synphot.units`` and as
  ``s_units``) and neither name is used; ``OptimizeWarning`` is unused.
"""
import math
import numpy as np
from astropy.io import fits
from astropy import units as u
from scipy.optimize import curve_fit

from lms_obs_map import ObsMap
from lmsdist_util import Util
from lms_globals import Globals
from lms_filer import Filer
from synphot.models import BlackBody1D
from synphot import SourceSpectrum


class Model:
    """Physical source, throughput, mask and PSF models for ToySim.

    All methods except :meth:`get_fp_mask` and :meth:`__str__` are static; the
    class attributes are shared configuration tables (see the ``#:`` comments).
    """

    #: Placeholder for a kernel blaze profile tau(x), x = wave / blaze_wave - 1.  Never assigned or read.
    tau_blaze_kernel = None # Kernel blaze profile tau(x) where x = (wave / blaze_wave(eo) - 1)
    #: Throughput from the Leiden sky to the CFO pinhole mask (0.7).
    tau_sky_cfo = 0.7       # Sky to CFO pinhole mask
    #: Throughput from the WCU hot source / laser outputs to the CFO pinhole mask (0.1).
    tau_wcu_cfo = 0.1       # WCU hot source and laser outputs to CFO pinhole mask
    #: Throughput from the WCU exit focal plane to the CFO pinhole mask (0.9).
    tau_wfp_cfo = 0.9       # WCU output focal plane to CFO pinhole mask
    #: Throughput from the CFO pinhole mask to the LMS detector, excluding detector QE and echelle blaze (0.2).
    #: Scalar copy of ``Globals.tau_cfo_lms['nominal']``.
    tau_cfo_lms = 0.2       # CFO pinhole mask to LMS detector (excluding detector qe and echelle blaze profile)
    #: Additional transmission of the extended-mode optics (0.8).  Not used.
    tau_lms_ext = 0.8       # Transmission through extended mode optics.

    #: Leiden sky to detector throughput, tau_sky_cfo x tau_wfp_cfo x tau_cfo_lms = 0.126.
    tau_sky = tau_sky_cfo * tau_wfp_cfo * tau_cfo_lms       # Leiden sky to detectors
    #: WCU integrating-sphere input to detector throughput, 0.1 x 0.9 x 0.2 = 0.018.  Applied by
    #: :meth:`get_flux` to every black body and laser.
    tau_whs = tau_wcu_cfo * tau_wfp_cfo * tau_cfo_lms       # WCU integrating sphere input to detectors
    #: WCU exit focal-plane mask to detector throughput, 0.18.  Listed for ``wcu_amb`` but not applied.
    tau_wfp = tau_whs / tau_wcu_cfo                         # WCU exit focal plane mask to detectors

    # Define a list of extended illumination sources.  These images will be convolved with the 'target slice' PSF.
    #: Extended (slice-filling) source table keyed by CSV source name.  Each entry has ``sed`` (``dark``, ``bb``,
    #: ``laser`` or ``sky``) and, as applicable, ``temperature`` (K: wcu_bb 1000, cfo_mask 70, wcu_amb 300),
    #: ``wavelength`` (um: wcu_ls 3.390, wcu_ll 5.240, wcu_lt 4.700 tunable), ``power`` (5e3, used as the laser
    #: line peak), ``wrange`` (unused) and ``tau`` (documentary only; see module Notes).
    bgd_srcs = {'dark': {'sed': 'dark'},
                'wcu_bb': {'sed': 'bb', 'temperature': 1000., 'tau': tau_whs},  # 1000 K black body
                'cfo_mask': {'sed': 'bb', 'temperature': 70., 'tau': tau_cfo_lms},
                'wcu_amb': {'sed': 'bb', 'temperature': 300., 'tau': tau_wfp},
                'wcu_ls': {'sed': 'laser', 'power': 5.E+03, 'wavelength': 3.390, 'wrange':0, 'tau': tau_whs},
                'wcu_ll': {'sed': 'laser', 'power': 5.E+03, 'wavelength': 5.240, 'wrange':0, 'tau': tau_whs},
                'wcu_lt': {'sed': 'laser', 'power': 5.E+03, 'wavelength': 4.700, 'wrange':100, 'tau': tau_whs},
                'sky': {'sed': 'sky', 'tau': tau_sky}  # Model sky emission spectrum
                }

    # Define one or more (point-like) pinhole masks which will spatially filter the extended source.  The model
    # specified PSFs at +-4 slices from the target slice will be convolved with the 'pinhole' images.
    #: Focal-plane mask table keyed by CSV wheel position.  ``id``: ``cfo`` (CFO pinhole), ``wcu`` (steerable WCU
    #: pinhole / grid), ``open`` or ``closed``; ``efp_xy_bs``: pinhole ``[x, y]`` positions in EFP mm at the boresight
    #: chop position (filled from the irdb mask files for ``lm_pinhole`` / ``lm_grid``; None for open / closed);
    #: ``mask_ext``: name of the extended source seen around the pinholes.  PSFs at +-4 slices from the target
    #: slice are convolved with the pinhole images.  Entries are mutated in place by ``lmssim_toy`` (Clean-up).
    fp_masks = {'pnh-1': {'id': 'cfo', 'efp_xy_bs': [[0., 0.]],           # On-axis pinhole on boresight
                    'mask_ext': 'cfo_mask'},
                'lm_pinhole': {'id': 'wcu', 'efp_xy_bs': [[0., 0.]],       # Steerable pinhole in WCU.
                    'mask_ext': 'wcu_mask'},
                'lm_grid': {'id': 'wcu', 'efp_xy_bs': [],                  # Steerable grid of pinholes in WCU.
                    'mask_ext': 'wcu_mask'},
                'align+dark': {'id': 'closed', 'efp_xy_bs': None,             # Blank position in CFO FP wheel
                    'mask_ext': 'cfo_mask'},
                'flatfield': {'id': 'open', 'efp_xy_bs': None,                  # FP-1 open position
                    'mask_ext': 'none'},
                'wcu_closed': {'id': 'closed', 'efp_xy_bs': None,
                    'mask_ext': 'wcu_mask'}
                }

    def __init__(self):
        """Create a Model instance; no instance state is used.
        """
        return

    def __str__(self):
        """Return a two-line summary listing the background source and mask names.
        """
        txt = 'Background sources, '
        for key in self.bgd_srcs:
            txt += "{:s}, ".format(key)
        txt += '\n'
        txt += 'Focal plane masks, '
        for key in self.fp_masks:
            txt += "{:s}, ".format(key)
        return txt

    @staticmethod
    def get_laser_wavelength(sim_config):
        """Return the wavelength of the laser switched on in an observation.

        Parameters
        ----------
        sim_config : dict
            Observation dictionary from ``ObsMap.get_configuration`` with keys
            ``'wcu_laser_sw'``, ``'wcu_laser_lw'``, ``'wcu_laser_tune'`` (the string
            ``'true'`` when on) and ``'wcu_laser_tune_woff'`` (tunable offset, um).

        Returns
        -------
        float
            Laser wavelength in um: 3.390 (short), 5.240 (long) or 4.700 + offset
            (tunable); 0.0 when no laser is on.  Written by ``lmssim_toy`` as
            ``HIERARCH ACHG LASER WAVE``.

        Notes
        -----
        * Open question: with several lasers on the last test wins (tunable over
          long over short); only one wavelength is recorded in the header.
        """
        laser_wave = 0.
        if sim_config['wcu_laser_sw'] == 'true':
            laser_wave = Model.bgd_srcs['wcu_ls']['wavelength']
        if sim_config['wcu_laser_lw'] == 'true':
            laser_wave = Model.bgd_srcs['wcu_ll']['wavelength']
        if sim_config['wcu_laser_tune'] == 'true':
            laser_wave = Model.bgd_srcs['wcu_lt']['wavelength']
            laser_wave += float(sim_config['wcu_laser_tune_woff'])
        return laser_wave

    @staticmethod
    def get_flux(waves, flux, src_list, lt_w_offset):
        """Add the detector-plane flux of a list of extended sources to a spectrum.

        Parameters
        ----------
        waves : astropy.units.Quantity
            Wavelength grid (um) over-filling the order's coverage.
        flux : astropy.units.Quantity
            Spectrum to add to, in ``Globals.u_flux`` (nominally photon / pixel /
            s); modified in place and also returned.
        src_list : list of (str, float)
            ``(Model.bgd_srcs key, scale factor)`` pairs; ``'dark'`` entries are
            skipped.
        lt_w_offset : float
            Tunable-laser wavelength offset (um), passed to
            :meth:`build_laser_emission` for every laser source.

        Returns
        -------
        waves : astropy.units.Quantity
            The input grid, unchanged.
        flux : astropy.units.Quantity
            The accumulated spectrum.

        Notes
        -----
        Per source: black bodies use :meth:`black_body` x resolution-element width
        (``waves[0] / 1e5``) x pi (39 / 2)^2 m^2 x ``alpha_pix * beta_slice`` x
        ``Model.tau_whs``; the sky uses :meth:`load_sky_emission` x
        ``Globals.elt_area`` x per-pixel bandwidth (``waves / 1e5 / 2.5``) x
        ``Model.tau_sky``; lasers use :meth:`build_laser_emission` x ``tau_whs`` x
        area x pixel solid angle x ``delta_w / 2.5``.  Prints the running min/max
        at debug level 'medium'.

        * Inconsistency: the ``'tau'`` entries of ``Model.bgd_srcs`` are ignored;
          ``tau_whs`` is applied to the 70 K and 300 K mask black bodies too.
        * Open question: the black-body branch integrates over a resolution
          element (``delta_w``) rather than a pixel (``pixel_delta_w``), unlike the
          sky branch, so black-body fluxes are 2.5x high relative to the sky.
          ``srp``, ``pixel_delta_w`` and ``pix_delta_w`` duplicate
          ``Globals.pix_spec_res_el`` and each other.
        * Bug: ``lt_w_offset`` shifts every laser line, not only the tunable one.
        * Clean-up: ``df_ext`` is only assigned for ``sed`` in 'bb', 'sky', 'laser';
          any other value raises ``NameError``.
        """
        u_m2 = Globals.u_m2
        u_cm2 = Globals.u_cm2
        for (src, k_scale) in src_list:
            if src == 'dark':
                continue
            source = Model.bgd_srcs[src]
            srp = 100000
            sed = source['sed']
            pixel_delta_w = waves / srp / Globals.pix_spec_res_el
            atel = math.pi * (39. / 2) ** 2 * u_m2          # ELT collecting area
            alpha_pix = Globals.alpha_pix                   # Along slice pixel scale
            beta_slice = Globals.beta_slice                 # Slice width
            omega_pix = alpha_pix * beta_slice
            pix_delta_w = 2.5  # Pixels per spectral resolution element
            delta_w = waves[0] / 100000  # Spectral resolution
            if sed in ['bb']:
                f_bb = Model.black_body(waves, tbb=source['temperature'])   # Units are (ph/sec/Angstrom/cm2/steradian)
                df_ext = f_bb * delta_w.to(u.angstrom) * atel.to(u_cm2) * omega_pix.to(u.sr) * Model.tau_whs     # ph / sec / pix
            if sed == 'sky':
                f_sky = Model.load_sky_emission(waves)      # Units = ph/s/m2/um/arcsec2
                sample_etendue = Globals.elt_area.to(u_m2) * Globals.alpha_pix.to(u.arcsec) * Globals.beta_slice.to(u.arcsec)
                df_ext = sample_etendue * pixel_delta_w.to(u.micron) * Model.tau_sky * f_sky
            if sed == 'laser':
                f_laser = Model.build_laser_emission(source, waves, lt_w_offset)
                f_ext_in = Model.tau_whs * f_laser
                df_ext = f_ext_in * atel.to(u_cm2) * alpha_pix * beta_slice * delta_w / pix_delta_w
            flux += df_ext * k_scale

            f_ext_min, f_ext_max = np.amin(flux), np.amax(flux)
            if Globals.is_debug('medium'):
                fmt = "Adding {:s} flux with min/max signal = {:10.2f} /{:10.2f} el/pix/sec"
                print(fmt.format(sed, f_ext_min.value, f_ext_max.value))
        return waves, flux

    def get_fp_mask(self, wcu_mask, cfo_mask):
        """Select the focal-plane mask definition for an observation.

        Parameters
        ----------
        wcu_mask : str
            WCU FP2-1 wheel position from the CSV (``'flatfield'``, ``'spare'``,
            ``'lm_pinhole'``, ``'lm_grid'``, ...).
        cfo_mask : str
            CFO FP2 wheel position (``'pnh-1'``, ``'align+dark'`` or a field
            position); takes precedence over ``wcu_mask``.

        Returns
        -------
        dict or None
            The ``Model.fp_masks`` entry: ``'id'`` (``'cfo'``, ``'wcu'``, ``'open'``
            or ``'closed'``), ``'efp_xy_bs'`` (list of pinhole ``[x, y]`` in EFP mm
            at the boresight chop position, or None) and ``'mask_ext'``.  For
            ``'lm_pinhole'`` / ``'lm_grid'`` the positions are read from the irdb WCU
            mask files with ``Filer.read_pinholes``.  None (after a printed warning)
            when nothing matches.

        Notes
        -----
        * Bug: ``'spare'`` is accepted but has no ``fp_masks`` entry -> ``KeyError``.
        * Clean-up: the returned dictionary is the shared class-level table entry;
          ``lmssim_toy.Toy.run`` adds ``'fslice_no_pnh_cen'``, ``'efp_xy_pnh_cen'``
          and ``'fslice_bounds'`` to it, so state leaks between observations.
          ``'wcu_closed'`` is unreachable.
        * Clean-up: the warning ``print`` passes two arguments, printing a stray
          space before "not found".
        """
        fp_mask = None
        if cfo_mask in ['pnh-1', 'align+dark']:
            fp_mask = Model.fp_masks[cfo_mask]
            return fp_mask
        if wcu_mask in ['flatfield', 'spare']:
            fp_mask = Model.fp_masks[wcu_mask]
            return fp_mask
        if wcu_mask in ['lm_pinhole']:
            fp_mask = Model.fp_masks[wcu_mask]
            file_name = 'fp_mask_pinhole_lm'
            fp_mask['efp_xy_bs'] = Filer.read_pinholes(file_name)
        if wcu_mask in ['lm_grid']:
            fp_mask = Model.fp_masks[wcu_mask]
            file_name = 'fp_mask_grid_lm'
            fp_mask['efp_xy_bs'] = Filer.read_pinholes(file_name)
        if fp_mask is None:
            print('!! Focal plane mask ' + wcu_mask, ' not found !!')
        return fp_mask

    @staticmethod
    def load_psf_dict(opticon, ech_ord, downsample=False, slice_no_tgt=13):
        """Dead code: superseded by the second ``load_psf_dict`` definition below.

        Python keeps only the last definition of a name in a class body, so this
        method is never reachable as ``Model.load_psf_dict``.  It differs from the
        live version in three ways: it hard-codes the model configuration tuples
        (extended date stamp '2024061802' versus '2024032401' in
        ``Globals.iq_ext_config``), it calls ``Filer(model_config)`` although
        ``Filer.__init__`` takes no arguments (``TypeError``), and it stores
        3-tuples ``(header, psf, zeros(det_format))`` in the dictionary.

        Notes
        -----
        * Clean-up: remove this definition.
        """
        analysis_type = 'iq'

        nominal = Globals.nominal
        nom_iq_date_stamp = '2024073000'
        nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      None, None)

        spifu = Globals.extended
        spifu_date_stamp = '2024061802'
        spifu_config = (analysis_type, spifu, spifu_date_stamp,
                        'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                        None, None)

        model_configurations = {nominal: nom_config, spifu: spifu_config}
        model_config = model_configurations[opticon]
        filer = Filer(model_config)
        defoc_str = '_defoc000um'

        _, _, date_stamp, _, _, _ = model_config
        dataset_folder = '../data/iq/' + opticon + '/' + date_stamp + '/'
        config_no = 41 - ech_ord if opticon == nominal else 0
        config_str = "_config{:03d}".format(config_no)

        psf_sum = 0.
        psf_dict = {}  # Create a new set of psfs

        # Use the boresight field position (field_no = 1) for now...
        (fn_min, fn_max) = (1, 2) if opticon == nominal else (1, 4)
        for field_no in range(fn_min, fn_max):
            field_idx = field_no - 1
            field_str = "_field{:03d}".format(field_no)
            iq_folder = 'lms_' + date_stamp + config_str + field_str + defoc_str
            spec_no = 0
            sn_radius = 4 if opticon == nominal else 1
            sn_min, sn_max = slice_no_tgt - sn_radius, slice_no_tgt + sn_radius + 1

            if opticon == spifu:
                spec_no = 1
                sn_min = slice_no_tgt - 1 + field_idx % 3
                sn_max = sn_min + 1

            for slice_no in range(sn_min, sn_max):
                iq_slice_str = "_spat{:02d}".format(slice_no) + "_spec{:d}_detdesi".format(spec_no)
                iq_filename = iq_folder + iq_slice_str + '.fits'
                iq_path = iq_folder + '/' + iq_filename
                file_path = dataset_folder + iq_path
                hdu_list = filer.read_zemax_fits(file_path)
                hdr, psf = hdu_list[0].header, hdu_list[0].data

                # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
                if downsample:
                    oversampling = 4
                    n_psf_rows, n_psf_ncols = psf.shape
                    n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                    psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)   # down sample
                slice_no_offset = slice_no - slice_no_tgt
                psf_dict[slice_no_offset] = hdr, psf, np.zeros(Globals.det_format)

                psf_sum += np.sum(psf)
            # Normalise the PSFs to have unity total flux in detector space
            for slice_no in range(sn_min, sn_max):
                slice_no_offset = slice_no - slice_no_tgt
                _, psf, _ = psf_dict[slice_no_offset]
                norm_factor = oversampling * oversampling / psf_sum
                psf *= norm_factor
        return psf_dict

    @staticmethod
    def load_source_lists(sim_config):
        """Translate an observation dictionary into background and pinhole source lists.

        Parameters
        ----------
        sim_config : dict
            Observation dictionary from ``ObsMap.get_configuration``.  Keys used:
            ``'wcu_per_arm'`` (``'out'`` = periscope out, looking at the Leiden
            sky), ``'wcu_laser_sw'``, ``'wcu_laser_lw'``, ``'wcu_laser_tune'``,
            ``'wcu_laser_tune_woff'``, ``'wcu_bb_temp'``, ``'wcu_bb_ap_mask'``
            (``'open'``, ``'closed'`` or ``'mask{N}'`` -> N / 15 of the black body
            plus 1 - N / 15 of the 300 K ambient mask), ``'wcu_fp2_1'``,
            ``'cfo_fp2'``; plus those read by ``ObsMap.is_dark``.

        Returns
        -------
        bgd_src_list : list of (str, float)
            ``(Model.bgd_srcs key, scale)`` for the slice-filling background: the
            WCU sources, or ``'wcu_amb'`` when a WCU pinhole mask is in the beam, or
            ``'cfo_mask'`` (70 K) when the CFO mask is in the beam.
        pnh_src_list : list of (str, float)
            Sources seen through the pinholes (a copy of the pre-mask background
            list); empty when no mask is in the beam.
        lt_w_offset : float
            Tunable-laser wavelength offset (um); 0 unless the tunable laser is on.

        Notes
        -----
        Darks return ``[('dark', 1.)]`` for both lists.  Prints the lists at debug
        level 'low'.

        * Open question: the WCU black body is added whenever
          ``wcu_bb_temp != 'off'`` regardless of the temperature value; the model
          temperature is fixed at 1000 K in ``bgd_srcs``.
        """
        lt_w_offset = 0.
        bgd_src_list, pnh_src_list = [], []
        # Trap darks
        is_dark = ObsMap.is_dark(sim_config)
        if is_dark:
            bgd_src_list.append(('dark', 1.))
            pnh_src_list.append(('dark', 1.))
            return bgd_src_list, pnh_src_list, lt_w_offset
        if sim_config['wcu_per_arm'] == 'out':          # IN = Looking at Leiden sky
            bgd_src_list.append(('sky', 1.))
        else:  # Looking at WCU
            if sim_config['wcu_laser_sw'] != 'false':
                bgd_src_list.append(('wcu_ls', 1.))
            if sim_config['wcu_laser_lw'] != 'false':
                bgd_src_list.append(('wcu_ll', 1.))
            if sim_config['wcu_laser_tune'] != 'false':
                bgd_src_list.append(('wcu_lt', 1.))
                lt_w_offset = float(sim_config['wcu_laser_tune_woff'])
            if sim_config['wcu_bb_temp'] != 'off':
                bb_aper = sim_config['wcu_bb_ap_mask']
                k_bb = 1. if bb_aper == 'open' else 0.
                if 'mask' in bb_aper:
                    k_bb = float(bb_aper[4:]) / 15.
                bgd_src_list.append(('wcu_bb', k_bb))
                bgd_src_list.append(('wcu_amb', 1. - k_bb))
            if sim_config['wcu_fp2_1'] == 'lm_pinhole' or sim_config['wcu_fp2_1'] == 'lm_grid':  # Ambient mask in beam
                pnh_src_list = bgd_src_list.copy()
                bgd_src_list = [('wcu_amb', 1.)]
        if 'field' not in sim_config['cfo_fp2']:  # CFO focal plane mask is in beam
            pnh_src_list = bgd_src_list.copy()  # This will be what is seen through the CFO pinhole
            bgd_src_list = [('cfo_mask', 1.)]
        if Globals.is_debug('low'):
            src_list_names = ['Background', 'Pinhole']
            for src_list_name, src_list in zip(src_list_names, [bgd_src_list, pnh_src_list]):
                print(src_list_name)
                print("{:10s},{:14s}".format('source', 'scale_factor'))
                fmt = "{:10s},{:14.2f}"
                for src in src_list:
                    print(fmt.format(src[0], src[1]))
        return bgd_src_list, pnh_src_list, lt_w_offset

    @staticmethod
    def make_spectra(transforms, bgd_src_list, pnh_src_list, lt_w_offset):
        """Build background and pinhole spectra for every echelle order of a transform set.

        Parameters
        ----------
        transforms : list of lms_transform.Transform
            Per-slice transforms; ``slice_configuration`` supplies ``ech_ord``,
            ``w_min`` and ``w_max`` (um).
        bgd_src_list, pnh_src_list : list of (str, float)
            Source lists from :meth:`load_source_lists`.
        lt_w_offset : float
            Tunable-laser offset (um).

        Returns
        -------
        dict
            ``{ech_ord: (waves, f_bgd, f_pnh)}``; ``waves`` spans the order's slice
            wavelength range widened by 10 % on each side, sampled at
            ``w_min / 200000``; the fluxes are :meth:`get_flux` results multiplied by
            ``k_toy_ss``.

        Notes
        -----
        * Open question: ``k_toy_ss = 0.00177`` is described as the factor "to
          transform from toysim to scopesim flux values"; its derivation is not
          recorded anywhere in the package.
        """
        spectra, w_lims = {}, {}
        k_toy_ss = 0.00177  # Scale factor to transform from toysim to scopesim flux values

        for transform in transforms:
            slice_cfg = transform.slice_configuration
            w_min, w_max = slice_cfg['w_min'], slice_cfg['w_max']
            w_slice_min, w_slice_max = w_min * u.micron, w_max * u.micron
            ech_ord = slice_cfg['ech_ord']
            if ech_ord not in w_lims.keys():
                w_lims[ech_ord] = [10.0, 0.0] * u.micron
            w_lims[ech_ord][0] = w_slice_min if w_slice_min < w_lims[ech_ord][0] else w_lims[ech_ord][0]
            w_lims[ech_ord][1] = w_slice_max if w_slice_max > w_lims[ech_ord][1] else w_lims[ech_ord][1]
        for ech_ord in w_lims:
            w_min, w_max = w_lims[ech_ord][0], w_lims[ech_ord][1]
            w_range = w_max - w_min
            w_bounds = [w_min - 0.1 * w_range, w_max + 0.1 * w_range]
            delta_w = w_min / 200000
            waves = np.arange(w_bounds[0].value, w_bounds[1].value, delta_w.value) * w_min.unit
            f_bgd = np.zeros(waves.shape) * Globals.u_flux
            f_pnh = np.zeros(waves.shape) * Globals.u_flux

            waves, f_bgd = Model.get_flux(waves, f_bgd, bgd_src_list, lt_w_offset)
            waves, f_pnh = Model.get_flux(waves, f_pnh, pnh_src_list, lt_w_offset)
            spectra[ech_ord] = waves, f_bgd * k_toy_ss, f_pnh * k_toy_ss
        return spectra

    @staticmethod
    def make_blaze_dictionary(transforms):
        """Collect the blaze (mosaic-centre) wavelength per order and echelle angle.

        Parameters
        ----------
        transforms : list of lms_transform.Transform
            Transform set; only field-slice-13 transforms are used.

        Returns
        -------
        dict
            ``{ech_ord: {ech_ang: wave}}`` where ``wave`` (um, float) is the
            wavelength that slice 13 maps onto the mosaic origin
            ``(mfp_x, mfp_y) = (0, 0)`` through ``Util.mfp_to_efp``.

        Notes
        -----
        * Bug: :meth:`make_tau_blaze` indexes ``blaze[ech_ord][ech_ang]``; an order
          without a slice-13 transform raises ``KeyError`` in ``lmssim_toy``.
        """
        blaze = {}
        for transform in transforms:
            lms_cfg = transform.lms_configuration
            slice_cfg = transform.slice_configuration
            slice_no = slice_cfg['slice_no']
            fslice_no, _ = Util.decode_slice_no(slice_no)
            if fslice_no != 13:
                continue
            ech_ang = lms_cfg['ech_ang']
            mfp_bs = {'mfp_x': [0.], 'mfp_y': [0.]}
            ech_ord = slice_cfg['ech_ord']
            efp_bs = Util.mfp_to_efp(transform, mfp_bs)
            wave = efp_bs['efp_w'][0]
            if ech_ord not in blaze:
                blaze[ech_ord] = {}
            blaze[ech_ord][ech_ang] = wave
        return blaze

    @staticmethod
    def build_laser_emission(laser, waves, wave_offset):
        """Build a laser line as a Gaussian on the wavelength grid.

        Parameters
        ----------
        laser : dict
            ``Model.bgd_srcs`` entry with ``'wavelength'`` (um) and ``'power'`` (peak
            value, 5e3 in the table).
        waves : astropy.units.Quantity
            Wavelength grid (um).
        wave_offset : float
            Added to the laser wavelength (um).

        Returns
        -------
        astropy.units.Quantity
            Spectrum in ``ph / cm^2 / mas^2 / s / um``: zero except for an 11-sample
            Gaussian centred on the grid point just below the laser wavelength, with
            FWHM = lambda / 1e5 (sigma = FWHM / 2.355 in grid samples) and peak
            ``laser['power']``.  All zeros (with a printed message) if the laser lies
            outside the grid.

        Notes
        -----
        * Clean-up: the former docstring (10 mW, 230 arcsec^2 field, "101 pixel
          scale, line centred at pixel 50") does not describe this code; the
          Gaussian occupies 2 x 5 + 1 samples centred at index 5.
        * Open question: ``'power'`` = 5e3 is used directly as the peak in
          ph / cm^2 / mas^2 / s / um with no radiometric derivation; the ``'wrange'``
          entry of the laser table is unused.
        """
        u_phot = u.ph / u.cm / u.cm / u.mas / u.mas / u.s / u.micron
        laser_wave = (laser['wavelength'] + wave_offset) * u.micron
        laser_power = laser['power']
        laser_flux = np.zeros(waves.shape)
        if laser_wave > np.amax(waves) or laser_wave < np.amin(waves):
            print("Laser at {:10.3f} is out of bounds".format(laser_wave))
            return laser_flux * u_phot
        idx_cen = np.argwhere(waves - laser_wave < 0)[:, 0][-1]
        line_width = waves[idx_cen] / 100000
        pix_fwhm = line_width / (waves[idx_cen] - waves[idx_cen-1])
        pix_sigma = pix_fwhm / 2.355
        pix_hw = 5
        n_pix_hw = 2 * pix_hw + 1
        indices = np.arange(n_pix_hw)        # 101 pixel scale, line centred at pixel 50.
        lsf = Globals.gauss(indices, laser_power, 5., pix_sigma.value)
        laser_flux[idx_cen - pix_hw: idx_cen + pix_hw + 1] = lsf
        return laser_flux * u_phot

    @staticmethod
    def black_body(waves, tbb=1000.):
        """Black-body surface brightness from synphot's ``BlackBody1D``.

        Parameters
        ----------
        waves : astropy.units.Quantity
            Wavelengths (converted to um).
        tbb : float, optional
            Temperature in K (default 1000).

        Returns
        -------
        astropy.units.Quantity
            ``BlackBody1D(temperature)`` evaluated at ``waves``, labelled
            ``1 / s / Angstrom / cm^2 / sr`` (photon / s / Angstrom / cm^2 / sr; the
            original comment records a check against Mathcad).

        Notes
        -----
        * Clean-up: ``angstrom_micron`` and ``k`` are computed but unused.
        """
        bb = SourceSpectrum(BlackBody1D, temperature=tbb * u.K)
        angstrom_micron = 1.0E4
        k = angstrom_micron / Globals.mas2_sterad
        flux = bb(waves.to(u.micron)).value
        photlam = 1 / u.s / u.angstrom / Globals.u_cm2 / u.sr
        flux_photlam = flux * photlam
        return flux_photlam

    @staticmethod
    def make_tau_blaze(blaze, ech_ord, ech_ang):
        """Generate the echelle blaze efficiency profile for one order.

        Parameters
        ----------
        blaze : dict
            ``{ech_ord: {ech_ang: w_blaze}}`` from :meth:`make_blaze_dictionary`.
        ech_ord : int
            Echelle order.
        ech_ang : float
            Echelle angle (deg); must be a key of ``blaze[ech_ord]``.

        Returns
        -------
        waves : astropy.units.Quantity
            500 wavelengths (um) spanning ``w_blaze +- 5 * w_wid`` with
            ``w_wid = 0.5 * w_blaze / (ech_ord + 1)``.
        tau : numpy.ndarray
            ``0.7 * sinc(pi * n * (n * w - w_1) / w_1)**2`` with ``n`` the order and
            ``w_1 = n * w_blaze``; peak efficiency 0.7.

        Notes
        -----
        * Open question: ``numpy.sinc(x)`` is already ``sin(pi x) / (pi x)``, so the
          explicit ``math.pi`` in the argument applies pi twice and the profile is
          pi times narrower than the textbook blaze function.  The former docstring's
          "w_width = 0.042 w_blaze from SPIE model" is not what the code does.
        """
        w_n = blaze[ech_ord][ech_ang]
        w_1 = w_n * ech_ord
        n_pts = 500
        w_wid = 0.5 * w_n / (ech_ord + 1)
        w_lo = w_n - 5. * w_wid
        w_hi = w_n + 5. * w_wid
        waves = np.linspace(w_lo, w_hi, n_pts)
        tk = 0.7 * np.power(np.sinc(math.pi * ech_ord * (ech_ord * waves - w_1) / w_1), 2)
        return waves*u.micron, tk

    @staticmethod
    def load_psf_dict(opticon, ech_ord, downsample=False, sno_tgt=13):
        """Load the Zemax PSFs of the target slice and its neighbours for one order.

        Parameters
        ----------
        opticon : str
            ``'nominal'`` or ``'extended'``.
        ech_ord : int
            Echelle order; selects Zemax configuration ``41 - ech_ord`` (nominal)
            or 0 (extended).
        downsample : bool, optional
            If True, block-average the 4x oversampled PSF to detector pixels.
            Default False.
        sno_tgt : int, optional
            Target field slice number (default 13, the boresight slice).

        Returns
        -------
        dict
            ``{slice_offset: (header, psf)}`` for ``slice_offset`` in -4..+4
            (nominal) or -1..+1 (extended); ``psf`` is the (optionally down-sampled)
            image rescaled by ``oversampling**2 / sum(all psfs)``.

        Notes
        -----
        Files: ``{Filer.psf_folder}lms_{date}_config{ccc}_field{fff}_defoc000um/``
        ``..._spat{ss}_spec{p}_detdesi.fits`` (see the module docstring); the
        ``Filer`` is configured with ``set_configuration('iq', opticon, False)``.

        * Bug: ``oversampling`` is only assigned inside ``if downsample:``, so
          ``downsample=False`` raises ``UnboundLocalError`` in the normalisation
          loop.
        * Open question: ``psf_sum`` is accumulated from the already down-sampled
          PSFs, so ``norm_factor = oversampling**2 / psf_sum`` makes the set sum to
          16, not to unity as the comment says.
        * Open question: in extended mode the three field positions
          (``field001``-``field003``) load the same three slice offsets in turn,
          each overwriting the previous entries while ``psf_sum`` keeps
          accumulating, so the final normalisation is inconsistent.
        """
        filer = Filer()
        filer.set_configuration('iq', opticon, False)
        defoc_str = '_defoc000um'
        _, _, data_set, _, _, _ = filer.model_configuration

        config_no = 41 - ech_ord if opticon == Globals.nominal else 0
        config_str = "_config{:03d}".format(config_no)

        psf_sum = 0.
        psf_dict = {}  # Create a new set of psfs
        # Use the boresight field position (field_no = 1) for now...
        (fn_min, fn_max) = (1, 2) if opticon == Globals.nominal else (1, 4)
        for field_no in range(fn_min, fn_max):
            # field_idx = field_no - 1
            field_str = "_field{:03d}".format(field_no)
            iq_folder = 'lms_' + data_set + config_str + field_str + defoc_str

            spec_no = 0
            sno_radius = 4
            if opticon == Globals.extended:
                spec_no = 1
                sno_radius = 1
            sno_min, sno_max = sno_tgt - sno_radius, sno_tgt + sno_radius + 1

            for slice_no in range(sno_min, sno_max):
                iq_slice_str = "_spat{:02d}".format(slice_no) + "_spec{:d}_detdesi".format(spec_no)
                iq_filename = iq_folder + iq_slice_str + '.fits'
                iq_path = iq_folder + '/' + iq_filename
                file_path = filer.psf_folder[:-1] + '_sim/' + iq_path
                hdu_list = filer.read_zemax_fits(file_path)
                hdr, psf = hdu_list[0].header, hdu_list[0].data
                if downsample:
                    oversampling = 4
                    n_psf_rows, n_psf_ncols = psf.shape
                    n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                    psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)  # down sample
                slice_no_offset = slice_no - sno_tgt
                psf_dict[slice_no_offset] = hdr, psf
                psf_sum += np.sum(psf)
            # Normalise the PSFs so that the total flux of all slices sums to unity in detector space
            for slice_no in range(sno_min, sno_max):
                slice_no_offset = slice_no - sno_tgt
                _, psf = psf_dict[slice_no_offset]
                norm_factor = oversampling * oversampling / psf_sum
                psf *= norm_factor
        return psf_dict

    @staticmethod
    def fit_beta_weights(psf_dict, beta_phase):
        """Weight the neighbouring-slice PSFs for an across-slice source position.

        The PSF dictionary is computed for a point source centred on the target
        slice (beta phase 0.5).  The total flux of each slice-offset PSF is fitted
        with a Gaussian in ``slice_offset + 0.5``; the weight for a source at
        ``beta_phase`` is the fitted Gaussian at ``slice_offset + beta_phase``,
        normalised to the fitted peak.

        Parameters
        ----------
        psf_dict : dict
            ``{slice_offset: (header, psf)}`` from :meth:`load_psf_dict`.
        beta_phase : float
            Across-slice position of the source within its slice (0-1).

        Returns
        -------
        dict
            ``{slice_offset: weight}``.

        Notes
        -----
        * Bug: if ``curve_fit`` fails the bare ``except`` swallows the error and
          ``gopt`` is undefined -> ``NameError``; ``is_error`` is never read.
        """
        x_list, y_list = [], []
        for slice_offset in psf_dict:
            _, psf = psf_dict[slice_offset]
            x_list.append(slice_offset + 0.5)
            y_list.append(np.sum(psf))
        # Sum of psf for target centred on slice 0 (beta_phase = 0.5)
        xs = np.array(x_list)
        ys = np.array(y_list)
        p0_guess = [np.amax(ys), 0., 0.5]
        try:
            gopt, gcov = curve_fit(Globals.gauss, xs, ys, p0=p0_guess)
        except:
            is_error = True
        beta_weight = {}
        for slice_offset in psf_dict:
            x = slice_offset + 1. - beta_phase           # eg a target at 0.5 -
            beta_weight[slice_offset] = Globals.gauss(x, *gopt) / gopt[0]
        return beta_weight

    @staticmethod
    def load_sky_emission(waves):
        """Interpolate the Leiden sea-level sky emission onto a wavelength grid.

        Parameters
        ----------
        waves : astropy.units.Quantity
            Target wavelengths (um), ascending.

        Returns
        -------
        astropy.units.Quantity
            Sky surface brightness in ``1 / s / m^2 / arcsec^2 / um``
            (photon / s / m^2 / arcsec^2 / um).

        Notes
        -----
        Reads ``../data/sky/leiden/LBL_A10_w3000_R0120000_ALL_Leiden_LM_R.fits``,
        HDU 1 columns ``lam`` (um because the path contains 'Leiden'; nm otherwise)
        and ``flux``; linear interpolation with a single forward pass through the
        sorted sky table.

        * Bug: (noted in the code) when the requested range overruns the sky
          spectrum the loop breaks and the remaining fluxes stay zero; if that
          happens on the first iteration ``f_units`` is unassigned -> ``NameError``.
        * Bug: ``i`` is incremented after every interpolation, so when the request
          grid is finer than the sky table (R = 200000 here versus 120000) the
          bracket for the next point can lie above it and ``numpy.interp`` clamps to
          the bracket edge instead of interpolating.
        * Clean-up: prints a six-column table header but no rows; ``flux_errs`` is
          unused.
        """
        path = '../data/sky/leiden/LBL_A10_w3000_R0120000_ALL_Leiden_LM_R.fits'
        wave_unit = u.micron if 'Leiden' in path else u.nm
        hdu_list = fits.open(path, mode='readonly')
        data_table = hdu_list[1].data
        waves_all = data_table['lam'] * wave_unit
        flux_all, flux_errs = data_table['flux'], None
        flux = np.zeros(waves.shape)
        i = 0
        fmt = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
        print(fmt.format('New Wave', 'New T', 'W1', 'W2', 'T1', 'T2'))
        n_waves_all, = waves_all.shape
        #  Bug!  Need to implement long wavelength test for when requested wavelength range overruns sky spectrum..
        for j, new_wave in enumerate(waves):
            # print(i, waves_all[i], new_wave)
            while waves_all[i] <= new_wave:
                i += 1
            flux[j] = np.interp(new_wave, waves_all[i - 1:i + 1], flux_all[i - 1:i + 1])
            i += 1
            if i >= n_waves_all - 1:
                break
            f_units = 1 / u.s / u.m / u.m / u.arcsec / u.arcsec / u.micron
        return flux * f_units            # Leiden units - phot/sec/m2/arcsec2/micron * synphot.units.PHOTLAM
