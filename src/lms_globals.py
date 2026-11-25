#!/usr/bin/env python
"""LMS optical, detector and program-control constants shared by all lmsiq programs.

This module defines :class:`Globals`, a namespace class whose class-level
attributes hold the METIS LMS instrument constants, the Zemax model
configuration tuples, the distortion-transform fitting parameters and the
debug-level state used throughout the package.  No instance state is used
(``lmssim.run`` creates one ``Globals()`` and discards it).  It is imported
by every module of the four lmsiq programs (``lmsiq``, ``lmsdist``,
``lmssim`` and ``lmsaiv``).

Input / output files
--------------------
The module reads and writes nothing itself.  It carries path fragments that
other modules use (relative to ``src/`` as the working directory; all data
trees live outside the repository):

* ``as_built_file`` -> ``../output/asbuilt/asbuilt_toysim`` (pickle written
  and read by ``lmsaiv_opt02`` / ``lmsaiv_opt_tests``).
* ``model_configurations`` -> the ``{analysis_type}/{opticon}/{date_stamp}``
  sub-folders that :meth:`lms_filer.Filer.set_configuration` appends to
  ``../data/`` and ``../output/``.
* The plate-scale source quoted in a comment, ``../docs/lb_ps_131124.txt``
  ("LB email 14/11/24"), is not in the repository.

No FITS header keywords are read or written here.  :meth:`Globals.make_lms_config_id`
builds the ``_nom_pa#####_ea[p|n]#####`` file-name stem used for the
``lms_efp_mfp_*.fits`` transform products.

Instrument constants
--------------------
======================= ======================= ======================================================
Attribute               Value                   Source / remark
======================= ======================= ======================================================
``alpha_pix``           8.7 mas / pixel         Along-slice plate scale at the detector.  The irdb
                                                ``METIS_LMS.yaml`` has ``pixel_scale: 0.0082`` arcsec
                                                (8.2 mas) -- see Notes.
``beta_slice``          20.7 mas                Slice width (across-slice sampling); agrees with irdb.
``alpha_fov``           0.897 arcsec            Along-slice field of view, FDR design report
                                                E-REP-ATC-MET-1003.
``beta_fov``            28 x 20.7 mas           = 0.5796 arcsec across-slice field of view.
``efp_arcsec_mm``       0.180723 arcsec / mm    EFP plate scale = 350.06 / 1937 (LB email 14/11/24).
``efp_x_fov_mm``        4.963 mm                alpha_fov / efp_arcsec_mm.  The adjacent code comment
                                                quotes a ray-trace x bound of 5.842 mm -- see Notes.
``efp_y_fov_mm``        3.207 mm                beta_fov / efp_arcsec_mm (ray-trace bound 3.208 mm).
``efp_mm_wcu_as``       4.538 mm / arcsec       1.55 / 0.341550; WCU mask coordinates (arcsec) to EFP
                                                mm.  Differs from 1 / efp_arcsec_mm = 5.533 mm/arcsec.
``pix_spec_res_el``     2.5 pixel               Pixels per spectral resolution element.
``intra_slice_gap``     20 pixel                Nominal gap between adjacent slice images.
``n_fslices``           28 / 3                  Field slices, nominal / extended mode.
``n_pslices``           1 / 6                   Pupil (spectral IFU) slices, nominal / extended.
``fslice_no_range``     1..28 / 12..14          Field slice numbers per mode -- see Notes.
``blaze_angle``         51.23 deg               Echelle blaze angle (unused elsewhere).
``rule_spacing``        18.2 um                 Echelle groove spacing.
``ge_refractive_index`` 4.05                    Germanium immersion index.
``wav_first_order``     109.536 um              21 x 5.216 um; overwrites the first assignment
                                                2 x 18.2 x 4.05 = 147.42 um -- see Notes.
``nom_pix_pitch``       18.0 um                 H2RG pixel pitch.
``det_format``          2048 x 2048             Detector format (pixels).
``mosaic_format``       2 x 2                   Detector mosaic.
``det_size``            36.864 mm               2048 x 18 um.
``det_gap``             3.0 mm                  Gap between active areas in the mosaic.
                                                ``lms_detector.Detector.mosaic_gap`` is 2.23 mm and
                                                the irdb FPA layout (x_cen = +-19.547 mm) implies
                                                2.23 mm -- see Notes.
``pix_margin``          64 pixel                Unilluminated margin at outer detector edges
                                                (``margin`` = 1.152 mm).
``mosaic_size``         74.424 mm               2 x (det_size - margin) + det_gap.
``elt_area``            1350 m^2                ELT collecting area.
``tau_cfo_lms``         0.20 / 0.15             CFO-to-LMS throughput, nominal / extended (unused).
``tau_wcu_isphere``     0.1                     WCU integrating-sphere throughput (unused).
``oversampling``        4                       Zemax PSF sampling relative to detector pixels
                                                (unused; ``lmssim_model`` hard-codes 4).
======================= ======================= ======================================================

Notes
-----
* Inconsistency: ``alpha_pix`` = 8.7 mas here, 8.2 mas in the irdb
  ``METIS_LMS.yaml`` (``pixel_scale: 0.0082``).  Used by ``lmsaiv_opt01``
  (FOV calculation) and ``lmssim_model`` (etendue per pixel).
* Inconsistency: ``det_gap`` = 3.0 mm here versus ``Detector.mosaic_gap`` =
  2.23 mm in ``lms_detector`` and 2.23 mm implied by the irdb
  ``FPA_metis_lms_layout.dat`` (2 x 19.547 - 36.864).  ``det_gap`` feeds
  ``lms_transform.Transform.create_affines`` (MFP <-> detector pixel
  affines) and ``lmsdist_trace.RayTrace._create_mask``; ``Detector`` uses
  its own value, so the two halves of the package disagree.
* Inconsistency: ``fslice_no_range['extended']`` = range(12, 15) (slices
  12-14) but ``lmsaiv_opt01`` ``slice_order`` assigns field slices 11-13 to
  the extended mode.  ``fslice_ext_centre`` = 13.5 is not used anywhere.
* Bug: ``wav_first_order`` is assigned twice; the physically motivated
  value 2 x rule_spacing x ge_refractive_index = 147.42 um is immediately
  overwritten by 21 x 5.216 = 109.536 um (consistent with the
  ``wxo_lim`` = [108, 115] um window in ``lmsdist_polyfit.PolyFit``).  No
  other module reads the attribute; ``lmsdist_util`` recomputes
  2 x d x n itself.
* Bug: ``nom_fov_text`` and ``ext_fov_text`` end with a trailing comma and
  are therefore 1-tuples, not strings.  They are only carried as the fourth
  (ignored) element of the model configuration tuples.
* Bug: ``det_lims['4']`` is ``([+xyf, +xyf], [-xyn, -xyn])``, a zero-width
  box; by symmetry with detector 2 it should be ``([+xyn, +xyf], [-xyf, -xyn])``.
  ``det_lims`` is not referenced by any other module.
* Clean-up: other modules reference attributes that do not exist here and
  would raise ``AttributeError`` when reached: ``Globals.n_slices``
  (``lmsdist_plot``, colour_by='slice'; probably ``n_fslices``) and
  ``Globals.mm_pix`` (``lms_test``).
* Clean-up: ``u.ph`` and ``u.pixel`` are monkey-patched onto the
  ``astropy.units`` namespace as dimensionless at import time, so
  ``u_flux`` is really 1 / s.
* Clean-up: the following attributes are not referenced outside this
  module: ``mas_as``, ``deg_rad``, ``u_plam``, ``tau_wcu_isphere``,
  ``tau_cfo_lms``, ``coord_in_ait`` (only via ``dist_nom_config_ait``),
  ``svd_shape``, ``n_svd_matrices``, ``n_svd_fit_terms``,
  ``zemax_configuration``, ``n_lms_detectors``, ``det_pix_size``,
  ``im_pix_size``, ``optical_configurations``, ``n_fslices``, ``n_pslices``,
  ``fslice_ext_centre``, ``fslice_no_lims``, ``ipc_on_tag``, ``ipc_off_tag``,
  ``blaze_angle``, ``wav_first_order``, ``mosaic_format``, ``det_lims``,
  ``mosaic_size``, ``process_levels``, ``oversampling``, ``debug_levels``
  (internal only).
* Open question: ``efp_mm_wcu_as`` (4.538 mm per WCU arcsec) is not the
  inverse of ``efp_arcsec_mm`` (5.533 mm per EFP arcsec); the WCU mask
  files therefore appear to use a different angular scale from the EFP.
* Open question: ``efp_x_fov_mm`` = 4.963 mm, while the in-line comment
  quotes ray-trace bounds of 5.842 mm (x) and 3.208 mm (y); only y agrees.
* Open question: ``debug_level`` defaults to ``'high'`` at import; every
  program overrides it with :meth:`Globals.set_debug_level`.
"""
import math
from astropy import units as u
import numpy as np


class Globals:
    """Namespace of METIS LMS constants and small static helpers.

    All attributes are class-level; the class is never meant to be
    instantiated for state.  See the module docstring for the table of
    instrument constants and the list of known inconsistencies.  Mutable
    program state lives in ``debug_level`` and ``sim_id``.
    """
    # Constants
    #: Milliarcseconds per arcsecond.  Not referenced outside this module.
    mas_as = 1000.
    #: Degrees per radian.  Not referenced outside this module.
    deg_rad = 180. / math.pi
    #: Milliarcseconds per radian (2.06265e8).
    mas_rad = 2.06265E+8
    #: Square milliarcseconds per steradian (4.2545e16); used by ``lmssim_model.Model.black_body``.
    mas2_sterad = mas_rad * mas_rad
    #: Side effect at import: 'ph' and 'pixel' are added to the astropy.units namespace as dimensionless
    #: aliases, so every unit below that contains them is dimensionless in that factor.
    u.ph = u.dimensionless_unscaled
    u.pixel = u.dimensionless_unscaled
    #: ToySim spectrum flux unit, nominally photon / pixel / s.  With ``u.ph`` and ``u.pixel``
    #: aliased to dimensionless above this is effectively 1 / s.
    u_flux = u.ph / u.pixel / u.s
    #: PHOTLAM-per-steradian surface brightness, photon / s / cm^2 / Angstrom / sr.  Not referenced
    #: outside this module.
    u_plam = u.photon / u.s / u.cm / u.cm / u.angstrom / u.steradian
    #: Square metre unit (astropy Quantity helper).
    u_m2 = u.m * u.m
    #: Square centimetre unit (astropy Quantity helper).
    u_cm2 = u.cm * u.cm

    # Optical parameters
    #: ELT collecting area, 1350 m^2.  ``lmssim_model.Model.get_flux`` uses this for the sky source but
    #: pi * (39 / 2)^2 = 1194.6 m^2 for black-body and laser sources (Inconsistency).
    elt_area = 1350. * u.m * u.m
    #: Detector pixels per spectral resolution element (2.5).
    pix_spec_res_el = 2.5       # Pixels per spectral resolution element
    # Plate scale at the entrance focal plane.  Defined in LB email 14/11/24 (in ../docs/lb_ps_131124.txt)
    # as efp_as_mm = 350.06 / 1937 = 0.180723
    #: Plate scale at the LMS entrance focal plane (EFP), 0.180723 arcsec / mm = 350.06 / 1937.
    #: Source: LB email 14/11/24 (``../docs/lb_ps_131124.txt``, not in the repository).
    efp_arcsec_mm = 0.180723 * u.arcsec / u.mm

    # Plate scale at detector
    #: Along-slice plate scale at the detector, 8.7 mas / pixel.  Inconsistency: the irdb
    #: ``METIS_LMS.yaml`` gives ``pixel_scale: 0.0082`` arcsec = 8.2 mas.
    alpha_pix = 8.7 * u.mas
    #: Slice width (across-slice sampling), 20.7 mas.  Agrees with the irdb.
    beta_slice = 20.7 * u.mas

    #: Nominal gap between adjacent slice images on the detector, pixels (design value 20).
    #: Compared against measured gaps in ``lmsaiv_opt02``.
    intra_slice_gap = 20           # Nominal gap between slices in pixels.

    # Optical throughput of sub-systems (excluding grating)
    #: WCU integrating-sphere throughput (0.1).  Not referenced outside this module;
    #: ``lmssim_model.Model.tau_wcu_cfo`` duplicates the value.
    tau_wcu_isphere = 0.1
    #: CFO-to-LMS-detector optical throughput (excluding grating) per mode: nominal 0.20, extended 0.15.
    #: Not referenced outside this module; ``lmssim_model.Model.tau_cfo_lms`` = 0.2 is a scalar copy.
    tau_cfo_lms = {'nominal': 0.20, 'extended': 0.15}

    # program control
    #: Ordered debug verbosity levels used by :meth:`is_debug`.
    debug_levels = {'off': 0, 'low': 1, 'medium': 2, 'high': 3}
    #: Current debug level (one of the ``debug_levels`` keys).  Defaults to 'high' at import;
    #: set with :meth:`set_debug_level`.
    debug_level = 'high'

    # Simulators
    #: Identifiers for the two simulators driven by ``lmssim``: ``'scopesim'`` and ``'toysim'``.
    scopesim_id, toysim_id = 'scopesim', 'toysim'
    #: Simulator selected by ``lmssim.run`` (``scopesim_id`` or ``toysim_id``); None until set.
    sim_id = None
    # Convert from scope sim detector number (hdu.header['ID'] to mosaic data list index.)
    #: Map from ScopeSim detector number (extension header ``ID``) to index in the mosaic HDU list
    #: (0 = TL, 1 = TR, 2 = BL, 3 = BR).  Applied by ``lms_filer.Filer.read_mosaic`` only when the primary
    #: header has ``ORIGIN == 'SCOPESIM'``.
    mos_idx = {2: 0, 1: 1, 3: 2, 4: 3}

    # Optical configurations and focal planes
    #: Optical configuration tag for the nominal (28 field slice) mode.
    nominal = 'nominal'
    #: Optical configuration tag for the extended (spectral IFU, 3 x 6 slice) mode.
    extended = 'extended'
    #: Input coordinate column names for Zemax distortion ray traces (``lmsdist``).
    coord_in = 'efp_x', 'efp_y', 'wavelength'
    #: Input coordinate column names for AIT-derived (measured) distortion data; only used inside
    #: ``dist_nom_config_ait``.
    coord_in_ait = 'efp_x', 'efp_y', 'efp_w'
    #: Output (mosaic focal plane, mm) coordinate column names for ray traces (``lmsdist``).
    coord_out = 'mfp_x', 'mfp_y'
    #: Description of the nominal mode.  Bug: the trailing comma makes this a 1-tuple, not a string;
    #: it is only carried as the ignored fourth element of the model configuration tuples.
    nom_fov_text = 'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
    #: Description of the extended mode.  Bug: trailing comma makes this a 1-tuple (see ``nom_fov_text``).
    ext_fov_text = 'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
    # Zemax and AIT model configurations with file locations
    #: Model configuration tuple ``(analysis_type, opticon, date_stamp, fov_text, coord_in, coord_out)`` for the
    #: nominal-mode Zemax distortion data set '20240109' (folder ``../data/distortion/nominal/20240109/``).
    dist_nom_config = ('distortion', nominal, '20240109', nom_fov_text, coord_in, coord_out)
    #: Nominal-mode configuration for AIT (measured) distortion data, data set 'ait'.
    dist_nom_config_ait = ('distortion', nominal, 'ait', nom_fov_text, coord_in_ait, coord_out)
    #: Extended-mode Zemax distortion data set '20260112'.
    dist_ext_config = ('distortion', extended, '20260112', ext_fov_text, coord_in, coord_out)
    #: Extended-mode AIT distortion configuration.  Uses ``coord_in`` rather than ``coord_in_ait`` (Inconsistency
    #: with ``dist_nom_config_ait``).
    dist_ext_config_ait = ('distortion', extended, 'ait', ext_fov_text, coord_in, coord_out)
    #: Nominal-mode Zemax image-quality (PSF) data set '2024073000' (folder ``../data/iq/nominal/2024073000/``).
    iq_nom_config = ('iq', nominal, '2024073000', nom_fov_text, coord_in, coord_out)
    #: Extended-mode Zemax image-quality data set '2024032401'.  Note the dead first ``lmssim_model.Model.load_psf_dict``
    #: hard-codes '2024061802' instead.
    iq_ext_config = ('iq', extended, '2024032401', ext_fov_text, coord_in, coord_out)
    #: Lookup ``model_configurations[analysis_type][opticon]`` used by ``lms_filer.Filer.set_configuration``;
    #: ``analysis_type`` is 'distortion', 'distortion_ait' or 'iq'.
    model_configurations = {'distortion': {nominal: dist_nom_config, extended: dist_ext_config},
                            'distortion_ait': {nominal: dist_nom_config_ait, extended: dist_ext_config_ait},
                            'iq': {nominal: iq_nom_config, extended: iq_ext_config}
                            }
    #: Keys of an LMS configuration dictionary (optical mode, prism angle / deg, echelle angle / deg); written
    #: as ``OPTICON``, ``PRI_ANG``, ``ECH_ANG`` in the primary header of ``lms_efp_mfp_*.fits``.
    lms_config_template = {'opticon': None, 'pri_ang': None, 'ech_ang': None}
    #: Keys of a per-slice configuration dictionary (encoded slice number, echelle order, wavelength bounds / um);
    #: written as ``SLICE_NO``, ``ECH_ORD``, ``W_MIN``, ``W_MAX`` in each ``SLICE_n`` extension header.
    slice_config_template = {'slice_no': None, 'ech_ord': None, 'w_min': None, 'w_max': None}   # , 'pslice_no': None

    # Transform parameters
    #: Order of the (upper-triangular) distortion transform matrices A, B, AI, BI (4 x 4, cubic).
    svd_order = 4
    #: Shape of one transform matrix, ``(svd_order, svd_order)``.  Not referenced outside this module.
    svd_shape = svd_order, svd_order
    #: Names of the four transform matrices: A, B (EFP -> MFP) and AI, BI (inverse).
    matrix_names = ['a', 'b', 'ai', 'bi']
    #: Number of transform matrices (4).  Not referenced outside this module.
    n_svd_matrices = len(matrix_names)
    #: Relative singular-value cut-off below which SVD eigenvalues are zeroed in the transform solver.
    svd_cutoff = 1.0e-7                                     # SVD eigenvalues below this value set to zero.
    #: Not referenced outside this module (historical fit-term count).
    n_svd_fit_terms = 6
    # Order of polynomial fit to wave = f(pri_ang, ech_ang) and tform_matrix_term = g(pri_ang, ech_ang)
    #: Order of the bivariate polynomial surface fit of each matrix term (and of wavelength x order) as a
    #: function of prism and echelle angle (3 or 4).
    surface_fit_order = 4  # Matrix order for polynomial surface fit (3 or 4)
    #: Number of non-zero coefficients in the upper-triangular surface-fit matrix of order 4 (10).
    surface_fit_n_coeffs = 10     # Non-zero terms in upper triangular matrix of order 'surface_fit_order'
    #: Polynomial order of the wavelength -> prism angle fit per mode (nominal 6, extended 3), ``lmsdist``.
    wpa_fit_order = {nominal: 6, extended: 3}

    #: Unused placeholder.
    zemax_configuration = None
    #: Number of LMS detectors (4).  Not referenced outside this module.
    n_lms_detectors = 4
    #: Unused placeholders.
    det_pix_size, im_pix_size = None, None

    #: List of the two optical configuration tags.  Not referenced outside this module.
    optical_configurations = [nominal, extended]

    # Field and pupil IFU parameters
    #: Number of field (spatial) slices per mode: nominal 28, extended 3.  Not referenced outside this
    #: module; ``lmsdist_plot`` looks for a non-existent ``Globals.n_slices`` instead (Clean-up).
    n_fslices = {nominal: 28, extended: 3}
    #: Number of pupil (spectral IFU) slices per mode: nominal 1, extended 6.  Not referenced outside
    #: this module.
    n_pslices = {nominal: 1, extended: 6}

    #: Field-slice coordinate of the extended-mode field centre (13.5).  Not referenced outside this module.
    fslice_ext_centre = 13.5
    #: Half-open field-slice number limits per mode.  Not referenced outside this module.
    fslice_no_lims = {nominal: (1, 29), extended: (12, 15)}
    #: Field slice numbers per mode: 1..28 nominal, 12..14 extended (``lmsdist_polyfit``).  Inconsistency:
    #: ``lmsaiv_opt01`` ``slice_order`` uses field slices 11..13 for the extended mode.
    fslice_no_range = {nominal: range(1, 29), extended: range(12, 15)}
    #: Pupil slice numbers per mode: 0 only (nominal), 1..6 (extended).  Combined with the field slice number as
    #: ``slice_no = 100 * pslice_no + fslice_no`` (``lmsdist_util.Util.encode_slice_no``).
    pslice_no_range = {nominal: range(0, 1), extended: range(1, 7)}

    #: File-name tags for IPC/diffusion on and off data sets (``lmsiq`` lineage).  Not referenced outside
    #: this module.
    ipc_on_tag, ipc_off_tag = '_ipc_01_3', '_ipc_00_0'      # IPC/diffusion file tags

    #: Format for a slice identifier ``{opticon}_{ech_ord}_{fslice_no:02d}_{pslice_no}``; evaluated in
    #: ``lmsdist_util.Util.find_closest_transforms`` but the result is unused there.
    slice_id_fmt = "{:s}_{:d}_{:02d}_{:d}"        # Define transforms by opticon, ech_ord, fslice_no, pslice_no

    # The field of view in the optical design is quoted in the FDR design report (E-REP-ATC-MET-1003) is then
    #: Along-slice field of view, 0.897 arcsec (FDR design report E-REP-ATC-MET-1003).
    alpha_fov = 0.897 * u.arcsec
    #: Across-slice field of view, 28 x beta_slice = 0.5796 arcsec (nominal mode).
    beta_fov = beta_slice.to(u.arcsec) * 28
    #: Along-slice EFP field of view, alpha_fov / efp_arcsec_mm = 4.963 mm (Quantity).  Open question: the
    #: in-line comment quotes a ray-trace bound of 5.842 mm.
    efp_x_fov_mm = alpha_fov / efp_arcsec_mm    # EFP field of view (mm) (Note ray trace bounds 5.842063, 3.208105)
    #: Across-slice EFP field of view, 3.207 mm (Quantity); matches the quoted ray-trace bound 3.208 mm.
    efp_y_fov_mm = beta_fov / efp_arcsec_mm

    #: EFP millimetres per WCU-mask arcsecond, 1.55 / 0.34155 = 4.538 (WCU grid limit 0.34155 arcsec at
    #: EFP_Y = 1.55 mm).  Used by ``lms_filer.Filer.read_pinholes``.  Open question: not the inverse of
    #: ``efp_arcsec_mm`` (5.533 mm / arcsec).
    efp_mm_wcu_as = 1.55 / 0.341550             # Based on WCU grid limit of 0.341550 arcsec falling at EFP_Y = 1.55 mm

    # Diffraction grating parameters
    #: Echelle blaze angle, degrees.  Not referenced outside this module.
    blaze_angle = 51.23                                     # Echelle blaze angle (deg)
    #: Echelle groove spacing, um.
    rule_spacing = 18.2			                            # Echelle rule spacing [um]
    #: Refractive index of the germanium immersion grating.
    ge_refractive_index = 4.05                              # Refractive index of germanium
    #: First-order blaze wavelength, um.  Bug: this value (147.42 um) is overwritten on the next line.
    wav_first_order = 2. * rule_spacing * ge_refractive_index
    #: First-order blaze wavelength, 21 x 5.216 = 109.536 um (final value).  Consistent with the
    #: ``wxo_lim`` = [108, 115] um window in ``lmsdist_polyfit.PolyFit.wave_to_config``.  Not referenced
    #: outside this module.
    wav_first_order = 21 * 5.216                            # First order blaze wavelength

    #: Detector (H2RG) pixel pitch, um.
    nom_pix_pitch = 18.0                                    # LMS pixel pitch in microns
    #: Gap between the active areas of the 2 x 2 detector mosaic, mm.  Inconsistency: ``lms_detector.Detector``
    #: uses 2.23 mm and the irdb ``FPA_metis_lms_layout.dat`` (x_cen = +-19.547 mm) implies 2.23 mm.  This value
    #: feeds ``lms_transform.Transform.create_affines`` and ``lmsdist_trace.RayTrace._create_mask``.
    det_gap = 3.0				            # Gap between active regions of detectors in 2 x 2 mosaic (mm)
    #: Unilluminated margin around the outer detector edges, pixels (x, y).
    pix_margin = [64, 64]		            # Unilluminated margin around outer detector edge (pixels)
    #: Outer margin converted to mm (1.152 mm).
    margin = pix_margin[0] * nom_pix_pitch / 1000.          # Convert to mm
    #: Detector format, pixels (rows, columns).
    det_format = 2048, 2048
    #: Mosaic layout, 2 x 2 detectors.  Not referenced outside this module.
    mosaic_format = 2, 2
    #: Detector active-area size, 36.864 mm.
    det_size = det_format[0] * nom_pix_pitch / 1000.                 # Detector size in mm
    #: Distance from the mosaic origin to the nearest light-sensitive pixel, mm (1.5).
    xyn = .5 * det_gap                      # x,y distance from nearest light sensitive pixel to the origin
    #: Distance from the mosaic origin to the farthest illuminated pixel, mm (37.212).
    xyf = xyn + det_size - margin           # x,y distance from farthest light sensitive pixel to the origin
    #: Illuminated MFP bounds ``([x_min, x_max], [y_min, y_max])`` per detector, mm: 1 = top-left, 2 = top-right,
    #: 3 = bottom-left, 4 = bottom-right.  Bug: entry '4' is ``([+xyf, +xyf], [-xyn, -xyn])`` (zero width).
    #: Not referenced outside this module.
    det_lims = {'1': ([-xyf, -xyn], [+xyn, +xyf]),
                '2': ([+xyn, +xyf], [+xyn, +xyf]),
                '3': ([-xyf, -xyn], [-xyf, -xyn]),
                '4': ([+xyf, +xyf], [-xyn, -xyn])
                }

    #: Half-width of the mosaic focal plane bounding box, mm.
    mfp_xy_lim = 40.0
    # Focal plane bounds (mm)
    #: Focal plane bounds ``(x_min, x_max, y_min, y_max)`` in mm for the EFP (Quantities) and MFP (floats);
    #: used by ``lmsdist_util.Util.out_of_bounds``.
    fp_bounds = {'efp': (-efp_x_fov_mm/2., +efp_x_fov_mm/2., -efp_y_fov_mm/2., +efp_y_fov_mm/2.),
                 'mfp': (-mfp_xy_lim, +mfp_xy_lim, -mfp_xy_lim, +mfp_xy_lim)
                 }

    #: Illuminated mosaic extent, 74.424 mm.  Not referenced outside this module.
    mosaic_size = 2. * (det_size - margin) + det_gap
    # Image quality parameters are calculated for three levels of data product
    #: Data-product levels for which image-quality parameters are calculated (``lmsiq`` lineage).
    #: Not referenced outside this module.
    process_levels = ['raw_zemax', 'proc_zemax', 'proc_detector']
    #: Axis labels used by ``lmsiq_summariser``.
    axes = ['spectral', 'spatial']
    # Zemax PSF image oversampling wrt detector pixels.
    #: Zemax PSF sampling relative to detector pixels (4).  Not referenced outside this module;
    #: ``lmssim_model.Model.load_psf_dict`` hard-codes its own 4.
    oversampling = 4

    # The as_built file holds the parameters which define the instrument's performance (eg slice locations, PSF
    # dimensions etc.).  They can be initialised to the as designed values, and then updated with as built values
    # from performance testing.
    #: Path stem of the as-built parameter pickle (slice locations, PSF dimensions, ...) initialised from
    #: design values and updated by performance testing (``lmsaiv_opt02``, ``lmsaiv_opt_tests``).
    as_built_file = '../output/asbuilt/asbuilt_toysim'

    @staticmethod
    def set_debug_level(new_debug_level):
        """Set the package-wide debug verbosity.

        Parameters
        ----------
        new_debug_level : str
            One of the ``Globals.debug_levels`` keys: 'off', 'low', 'medium' or 'high'.

        Returns
        -------
        None

        Notes
        -----
        Prints the new level and the list of valid levels.

        * Clean-up: the argument is not validated; an unknown level is stored and
          makes every later :meth:`is_debug` call raise ``KeyError``.
        """
        Globals.debug_level = new_debug_level
        debug_levels = list(Globals.debug_levels.keys())
        debug_text = "Debug level set to '{:s}'.  ['{:s}'".format(new_debug_level, debug_levels[0])
        for dlt in debug_levels[1:]:
            debug_text += ", '{:s}'".format(dlt)
        print(debug_text + ']')
        return

    @staticmethod
    def is_debug(db_level_query):
        """Test whether output at a given debug level should be produced.

        Parameters
        ----------
        db_level_query : str
            Level of the message being considered ('low', 'medium' or 'high').

        Returns
        -------
        bool
            False if the current level is 'off'; otherwise True when the numeric
            rank of ``db_level_query`` is less than or equal to that of
            ``Globals.debug_level``.
        """
        db_val_set = Globals.debug_levels[Globals.debug_level]
        if db_val_set == 0:
            return False
        db_val_query = Globals.debug_levels[db_level_query]
        is_debug = db_val_query <= db_val_set
        return is_debug

    @staticmethod
    def make_lms_config_id(lms_config):                # Create fits file with primary HDU only
        """Build the file-name stem that encodes an LMS configuration.

        Parameters
        ----------
        lms_config : dict
            Dictionary with keys ``'opticon'`` ('nominal' or 'extended'),
            ``'pri_ang'`` and ``'ech_ang'`` (prism and echelle angles, deg).

        Returns
        -------
        str
            ``'_nom'`` or ``'_ext'``, then ``'_pa#####'`` (abs(prism angle) x 1e4,
            truncated to int) and ``'_ea[p|n]#####'`` (sign character and
            abs(echelle angle) x 1e4), e.g. ``'_nom_pa06500_ean00500'``.  Used by
            ``lms_filer.Filer.write_svd_transforms`` to name
            ``lms_efp_mfp{id}_v{date}.fits``.

        Notes
        -----
        * Open question: the prism-angle tag uses ``abs()`` with no sign
          character, so negative and positive prism angles map to the same
          identifier.
        * Clean-up: the in-line comment "Create fits file with primary HDU
          only" belongs to the caller, not to this function.
        """
        otag = '_nom' if lms_config['opticon'] == 'nominal' else '_ext'
        ptag = "_pa{:05d}".format(abs(int(10000. * lms_config['pri_ang'])))
        ea = lms_config['ech_ang']
        esign = 'p' if ea > 0. else 'n'
        etag = "_ea{:s}{:05d}".format(esign, abs(int(10000. * ea)))
        fmt = "{:s}{:s}{:s}"
        lms_config_id = fmt.format(otag, ptag, etag)
        return lms_config_id

    # Geometry functions
    @staticmethod
    def gauss(x, a, x0, sigma):
        """Evaluate an un-normalised Gaussian ``a * exp(-(x - x0)^2 / (2 sigma^2))``.

        Parameters
        ----------
        x : float or numpy.ndarray
            Abscissa value(s).
        a : float
            Peak amplitude.
        x0 : float
            Centre.
        sigma : float
            Standard deviation.

        Returns
        -------
        float or numpy.ndarray
            Gaussian evaluated at ``x``.  Signature matches
            ``scipy.optimize.curve_fit`` (used by ``lmssim_model.Model.fit_beta_weights``
            and ``lmsaiv_opt02``).
        """
        y = a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        return y

    @staticmethod
    def polynomial(x, *opt, gradient=False):
        """Evaluate a polynomial, or its first derivative, at ``x``.

        Parameters
        ----------
        x : float or array_like
            Abscissa value(s).  A bare float is wrapped in a 1-element array.
        *opt : float
            Polynomial coefficients in ascending order of power (``opt[0]`` is the
            constant term), as returned by ``scipy.optimize.curve_fit``.
        gradient : bool, optional
            If True return ``dy/dx`` instead of ``y``.  Default False.

        Returns
        -------
        numpy.ndarray
            ``y(x)`` or ``dy/dx(x)``, always an array (shape of ``x``; 0-d if ``x``
            was an int).

        Notes
        -----
        * Clean-up: the previous docstring described a ``covariance`` argument
          that does not exist in the signature.
        """
        if isinstance(x, float):
            x = [x]
        x = np.array(x)
        order = len(opt) - 1         # order = highest exponent (last term).
        if gradient:
            dy_dx = np.zeros(x.shape)
            for exp in range(1, order+1):
                dy_dx = np.add(dy_dx, opt[exp] * exp * np.power(x, (exp - 1)))
            return dy_dx
        else:
            y = np.zeros(x.shape)
            for exp in range(0, order+1):
                y = np.add(y, opt[exp] * np.power(x, exp))
        return y
