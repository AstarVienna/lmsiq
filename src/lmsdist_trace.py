#!/usr/bin/env python
"""Ray-trace container: one LMS configuration's Zemax (or AIT) ray coordinates and its fitted transforms.

Purpose
-------
:class:`RayTrace` reads one CSV of ray coordinates for a single LMS
opto-mechanical configuration (prism angle, echelle angle, nominal or
extended mode), and fits the E-TNT-ATC-MET-1058 distortion transforms
(``lms_transform.Transform``, one per slice and echelle order) that map the
entrance focal plane (EFP) onto the detector mosaic focal plane (MFP).  It
also back-projects the detector edges to find each slice's wavelength range
and offers diagnostic plots of the ray coordinates at every intermediate
focal surface.

Program
-------
``lmsdist`` (``lmsdist.run`` creates one ``RayTrace`` per CSV, pickles the list
to ``Filer.trace_file`` and writes each trace's transforms to FITS with
``Filer.write_svd_transforms``).  ``lmssim`` and ``lmsaiv`` read the pickle.

Input data (not in the repository)
----------------------------------
Zemax ray-trace CSV files in ``../data/distortion/<opticon>/<date>/`` relative
to ``src/`` (``Filer.ray_trace_folder``; ``<date>`` is ``20240109`` nominal,
``20260112`` extended, from ``Globals.dist_*_config``).  Extended-mode files
are additionally filtered on ``'in-band'`` in the file name by ``lmsdist.run``.
Layout as parsed by :meth:`RayTrace._read_csv`:

1. Header lines ``<name>: <value>`` until the first line without a colon.
   Recognised names: ``Echelle angle`` -> ``lms_config['ech_ang']``,
   ``Prism angle`` -> ``lms_config['pri_ang']``, ``Spectral order`` -> the
   single echelle order of a nominal-mode file (extended files carry the
   order per ray).
2. One line of comma-separated column names.
3. Data rows until a line with fewer than two tokens.

Column names are mapped to internal series names by ``RayTrace.nom_csv_map``
(nominal: ``sp_slice, order, slice, wavelength, fp2_x, fp2_y, slicer_x,
slicer_y, ifu_x, ifu_y, slit_x, slit_y, det_x, det_y, chief_ray_vignetted?``)
or ``RayTrace.ext_csv_map`` (extended: ``fp1_x, fp1_y`` instead of ``fp2_*``
and ``sp_slicer_x, sp_slicer_y`` instead of ``slit_*``).  Internal names:
``pslice_no, ech_ord, fslice_no, wavelength, efp_x, efp_y, slicer_x/y,
ifu_x/y, slit_x/y | sp_slicer_x/y, mfp_x, mfp_y, cf_vig``.  For AIT data
(``data_id == 'ait'``, folder ``../data/distortion_ait/<opticon>/ait/``) the
CSV is ``LMS_ait_dist_coord.csv`` written by ``lmsaiv_opt02.Opt02._print_dist_coord``
with columns ``slice_no, efp_w, efp_x, efp_y, mfp_x, mfp_y`` and the same
three header lines.

Coordinates: ``efp_x`` mm along slice (alpha), ``efp_y`` mm across slice
(beta), ``wavelength`` microns; ``mfp_x``, ``mfp_y`` mm in the mosaic plane.
The universal slice number ``slice_no = 100 * pslice_no + fslice_no`` is added
to the series by :meth:`RayTrace.load`.

Outputs
-------
- ``self.transforms``: list of ``Transform`` (matrices ``a, b, ai, bi``),
  written by ``Filer.write_svd_transforms`` as
  ``lms_efp_mfp_<tag>_v<date>.fits`` (see ``lms_transform``).
- The ``RayTrace`` objects themselves are pickled by ``lmsdist.run`` to
  ``../output/distortion/<opticon>/<date>/distortion_<opticon>_<date>_trace.pkl``.

Notes
-----
- Inconsistency: :meth:`RayTrace._create_mask` uses ``Globals.det_gap = 3.0``
  mm while ``lms_detector.Detector.mosaic_gap = 2.23`` mm (and the irdb
  layout implies 2.23 mm); the same 3.0 mm feeds
  ``Transform.create_affines``.
- Inconsistency: ``_create_mask`` numbers detectors BL=1, TL=2, BR=3, TR=4
  (enumeration order of ``xy_bounds``) whereas ``Util.mfp_to_dfp`` and the
  affines use 1 = (x>0, y>0), 2 = (x<0, y>0), 3 = (x>0, y<0), 4 = (x<0, y<0).
  The mask value is only used for plotting (zero = missed all detectors).
- Bug: :meth:`RayTrace.create_svd_transforms` accumulates per-slice
  residuals in a local ``a_rms_list`` but never assigns ``self.a_rms``, so the
  ``a_rms`` printed by ``lmsdist.run`` is always 0.
- Bug: the AIT ``csv_map`` in :meth:`RayTrace._read_csv` maps to series
  names ``slice_no`` and ``spifu_no`` which are absent from
  ``RayTrace.series_fmt`` -> ``KeyError``; and :meth:`load` then expects
  ``fslice_no`` / ``pslice_no`` series that the AIT map never fills.  The AIT
  path cannot currently run (``lmsdist.run`` hard-codes ``is_ait_data = False``).
- Bug: :meth:`RayTrace.find_wavelength_bounds` passes the encoded
  ``slice_no`` (not ``fslice_no``) to ``Util.slice_to_efp_y`` and always
  back-projects from detectors 1 and 2, even for slices imaged on 3 and 4.
- Bug: :meth:`RayTrace.plot_scatter` reads ``slice_config['spifu_no']``,
  which is not in ``Globals.slice_config_template`` -> ``KeyError`` whenever
  ``load(..., do_plot=True)``.
- Bug: :meth:`RayTrace.plot_slice_map` references ``self.parameter``,
  ``self.rgb`` and series ``'phase', 'fp2_x', 'det_x'`` that no longer exist.
- Open question: :meth:`RayTrace.get_ifp_boresight` uses ``numpy.interp``
  with ``mfp_x`` as the abscissa; ``interp`` requires increasing ``xp`` and
  ``mfp_x`` decreases with wavelength, so the interpolated boresight
  wavelength may be invalid unless the CSV rows happen to be ordered
  suitably.
- Clean-up: uses ``match`` / ``case`` (Python >= 3.10).
- Clean-up: ``__init__`` accepts ``**kwargs`` (``cfg_id``, ``silent`` are
  passed by ``lmsdist.run``) and ignores them; ``offset_data`` is never set;
  ``cfg_tags`` / ``cfg_id_counter`` are unused class attributes.
"""
import math
import numpy as np
import matplotlib.patches as mpatches

from lms_detector import Detector
from lms_globals import Globals
from lmsdist_util import Util
from lmsdist_plot import Plot
from lms_transform import Transform


class RayTrace:
    """Ray coordinates for one LMS configuration plus the fitted slice transforms.

    Attributes
    ----------
    series : dict of numpy.ndarray
        Per-ray arrays keyed by internal series name (see module docstring),
        plus ``'slice_no'`` (encoded) and ``'mask'`` (detector hit index from
        :meth:`_create_mask`).
    lms_config : dict
        ``{'opticon', 'pri_ang', 'ech_ang'}`` from the CSV header.
    transforms : list of Transform
        One per ``(slice_no, ech_ord)`` with rays, from
        :meth:`create_svd_transforms`.
    unique_ech_ords, unique_slice_nos, unique_waves : numpy.ndarray
        Distinct values in the series.
    wave_reference : float
        Mean of all ray wavelengths (:meth:`_get_wave_reference`).
    n_rays : int
    csv_name, transform_fits_name : str
        Input file name; name of the FITS file written by
        ``Filer.write_svd_transforms`` (set by ``lmsdist.run``).
    a_rms : float
        RMS transform residual (mm; comment says microns).  Never updated
        from its initial 0 (see module notes).
    is_extended : bool
    series_fmt, nom_csv_map, ext_csv_map : dict (class attributes)
        Series dtype table and CSV column maps.
    nominal_focal_planes, spifu_focal_planes : dict (class attributes)
        Focal-plane label -> ``(x_series, y_series)`` for
        :meth:`plot_focal_planes`.  The 'Slicer' plane has swapped axes.
    model_config : tuple (class attribute)
        Last ``Filer.model_configuration`` passed to :meth:`load`.

    Notes
    -----
    - Inconsistency: the in-code comment on ``a_rms`` says microns but the
      residuals computed in :meth:`create_svd_transforms` are in mm
      (``lmsdist.run`` multiplies by 1000 to print microns).
    - Clean-up: ``model_config`` is stored on the class, so traces loaded
      with different configurations in one process share the last one.
    """

    # Configuration dicts.  Note lms_config is instantiated from Globals.
    series_fmt = {'ech_ord': 'int', 'fslice_no': 'int', 'pslice_no': 'int',
                  'wavelength': 'float',
                  'efp_x': 'float', 'efp_y': 'float', 'efp_w': 'float',
                  'slicer_x': 'float', 'slicer_y': 'float',
                  'sp_slicer_x': 'float', 'sp_slicer_y': 'float',
                  'ifu_x': 'float', 'ifu_y': 'float',
                  'slit_x': 'float', 'slit_y': 'float',
                  'mfp_x': 'float', 'mfp_y': 'float',
                  'cf_vig': 'float'}

    nom_csv_map = {'sp_slice': 'pslice_no', 'order': 'ech_ord',
                   'slice': 'fslice_no', 'wavelength': 'wavelength',
                   'fp2_x': 'efp_x',  'fp2_y': 'efp_y',
                   'slicer_x': 'slicer_x', 'slicer_y': 'slicer_y',
                   'ifu_x': 'ifu_x', 'ifu_y': 'ifu_y',
                   'slit_x': 'slit_x', 'slit_y': 'slit_y',
                   'det_x': 'mfp_x', 'det_y': 'mfp_y',
                   'chief_ray_vignetted?': 'cf_vig'
                  }

    ext_csv_map = {'sp_slice': 'pslice_no', 'order': 'ech_ord',
                   'slice': 'fslice_no', 'wavelength': 'wavelength',
                   'fp1_x': 'efp_x', 'fp1_y': 'efp_y',
                   'slicer_x': 'slicer_x', 'slicer_y': 'slicer_y',
                   'ifu_x': 'ifu_x', 'ifu_y': 'ifu_y',
                   'sp_slicer_x': 'sp_slicer_x', 'sp_slicer_y': 'sp_slicer_y',
                   'det_x': 'mfp_x', 'det_y': 'mfp_y',
                   'chief_ray_vignetted?': 'cf_vig'
                   }
    nominal_focal_planes = {'LMS EFP': ('efp_x', 'efp_y'),
                            'Slicer': ('slicer_y', 'slicer_x'),     # This focal plane has swapped axes
                            'IFU': ('ifu_x', 'ifu_y'),
                            'Slit': ('slit_x', 'slit_y'),
                            'Det. mosaic': ('mfp_x', 'mfp_y')}
    spifu_focal_planes = {'LMS EFP': ('efp_x', 'efp_y'),
                          'Slicer': ('slicer_y', 'slicer_x'),       # Swapped axes
                          'IFU': ('ifu_x', 'ifu_y'),
                          'SP slicer': ('sp_slicer_x', 'sp_slicer_y'),
                          'Det. mosaic': ('mfp_x', 'mfp_y')}
    cfg_tags, cfg_id_counter = [], 0

    model_config = None

    def __init__(self, **kwargs):
        """Create an empty ``RayTrace``; call :meth:`load` to populate it.

        Parameters
        ----------
        **kwargs
            Accepted and ignored.  ``lmsdist.run`` passes ``cfg_id`` and
            ``silent``; neither is stored.

        Notes
        -----
        - Clean-up: ``silent`` must be passed to :meth:`load` to take effect,
          not here.
        """
        self.lms_config = None
        self.transform_fits_name, self.csv_name = None, None
        self.transforms = []
        self.offset_data = None
        self.a_rms = 0.                          # RMS transform error for trace (microns)
        self.n_mat_terms = None
        self.is_extended = None
        self.series = None
        self.unique_ech_ords = None
        # self.unique_fslice_nos = None
        # self.unique_pslice_nos = None
        self.unique_slice_nos = None            # Encoded 100 x pslice_no + fslice_no
        self.unique_waves = None
        self.wave_reference = None
        self.n_rays = None
        return

    def load(self, path, model_config, **kwargs):
        """Read a ray-trace CSV, derive the slice bookkeeping and fit the slice transforms.

        Steps: :meth:`_read_csv`; compute ``unique_ech_ords``; add the encoded
        ``series['slice_no'] = 100 * pslice_no + fslice_no`` and
        ``unique_slice_nos``; set ``wave_reference`` and ``unique_waves``;
        build the detector-hit mask (:meth:`_create_mask`); fit the transforms
        (:meth:`create_svd_transforms`).

        Parameters
        ----------
        path : str
            CSV path (forward-slash separated; the file name after the last
            ``'/'`` becomes ``csv_name``).
        model_config : tuple
            ``Filer.model_configuration``: ``(analysis_type, opticon,
            date_stamp, optical_path_label, coord_in, coord_out)``.  Stored on
            the class as ``RayTrace.model_config``.
        silent : bool, optional
            Passed to :meth:`_create_mask` (which does not use it).
        do_plot : bool, optional
            Passed to :meth:`create_svd_transforms`; default True (plots the
            first slice's residuals).  ``lmsdist.run`` passes False.

        Returns
        -------
        None

        Notes
        -----
        - Bug: ``do_plot=True`` (the default) triggers :meth:`plot_scatter`,
          which fails with ``KeyError: 'spifu_no'``.
        - Clean-up: the ``csv_name`` returned by :meth:`_read_csv` is
          discarded in favour of the one derived from ``path`` here.
        - Clean-up: ``silent`` is threaded through but has no effect.
        """
        RayTrace.model_config = model_config
        analysis_type, opticon, date_stamp, optical_path_label, coord_in, coord_out = model_config
        self.csv_name = path.split('/')[-1]
        self.is_extended = opticon == Globals.extended
        self.transform_fits_name = None
        self.transforms = []
        self.offset_data = None
        self.a_rms = 0.                          # RMS transform error for trace (microns)
        self.n_mat_terms = None
        silent = kwargs.get('silent', False)
        csv_name, lms_config, series = self._read_csv(path, model_config)
        self.lms_config = lms_config

        # Count the number of slices and rays
        ech_ords = series['ech_ord']
        self.unique_ech_ords = np.unique(ech_ords)
        fslice_nos = series['fslice_no']
        pslice_nos = series['pslice_no']
        slice_nos = 100 * pslice_nos + fslice_nos
        series['slice_no'] = slice_nos
        self.unique_slice_nos = np.unique(slice_nos)
        waves = series['wavelength']

        self.wave_reference = self._get_wave_reference()
        self.unique_waves = np.unique(waves)
        self.n_rays, = fslice_nos.shape
        self._create_mask(silent)
        do_plot = kwargs.get('do_plot', True)
        self.create_svd_transforms(do_plot=do_plot)
        return

    def _get_wave_reference(self):
        """Return the reference wavelength of the trace: the mean of all ray wavelengths.

        Returns
        -------
        float
            ``numpy.mean(self.series['wavelength'])`` in microns.

        Notes
        -----
        - Open question: for extended-mode files that contain three echelle
          orders the mean straddles the orders and is not a wavelength that
          is imaged in any single one of them; it is used by
          :meth:`find_wavelength_bounds` as the projection wavelength.
        """
        waves = self.series['wavelength']
        wave_ref = np.mean(waves)
        return wave_ref

    def find_wavelength_bounds(self):
        """Set ``w_min`` / ``w_max`` of every slice transform by back-projecting the detector edges.

        For each transform: project the slice-centre point
        ``(efp_x = 0, efp_y = slice centre, efp_w = wave_reference)`` to the
        detector with ``Util.efp_to_dfp`` to obtain its row ``dfp_y``; then
        inverse-project ``(det 1, column 0, dfp_y)`` and ``(det 2, column 2048,
        dfp_y)`` with ``Util.dfp_to_efp`` and store the resulting wavelengths
        as ``transform.slice_configuration['w_min']`` and ``['w_max']``.  With
        the affine convention (column increases towards longer wavelength,
        det 1 at ``mfp_x > 0``) these are the short- and long-wavelength limits
        of the detector pair for that slice and order.

        Returns
        -------
        None
            Results are written into each transform's ``slice_configuration``
            and therefore end up as the ``W_MIN`` / ``W_MAX`` header cards of
            the FITS product.

        Notes
        -----
        - Bug: ``Util.slice_to_efp_y(slice_no, 0.)`` is called with the
          encoded universal slice number (e.g. 312 in extended mode) although
          the function expects the field slice number; the decoded
          ``fslice_no`` is computed on the line above and not used.  Nominal
          mode (pslice 0) is unaffected.
        - Bug: detector numbers ``[1, 2]`` are hard-coded.  Slices 1-14 fall
          on detectors 3 and 4 (module docstring of ``lmsdist``); for them
          ``dfp_y`` is a row on detector 3/4 but is inverted with the detector
          1/2 affine, offsetting ``mfp_y`` by roughly the detector height plus
          gap.  Because wavelength depends mainly on ``mfp_x`` the error in
          ``w_min`` / ``w_max`` is second order, but not zero.
        - Open question: ``Util.mfp_to_dfp`` drops points that fall exactly on
          ``mfp_x == 0`` or ``mfp_y == 0`` and re-orders by detector; the code
          assumes ``dfp_y`` has two entries.
        - Clean-up: ``opticon`` and ``fslice_no`` are computed but unused.
        """
        affines = Transform.affines
        for transform in self.transforms:
            # Start by finding the detector and row for the slice centre.
            slice_no = transform.slice_configuration['slice_no']
            opticon = transform.lms_configuration['opticon']
            fslice_no, _ = Util.decode_slice_no(slice_no)
            efp_y = Util.slice_to_efp_y(slice_no, 0.).value
            efp_ref_points = {'efp_x': np.array([0., 0.]), 'efp_y': np.array([efp_y, efp_y]),
                              'efp_w': np.array([self.wave_reference, self.wave_reference])}

            dfp_slice_bs_points = Util.efp_to_dfp(transform, affines, efp_ref_points)
            dfp_y = dfp_slice_bs_points['dfp_y']
            det_nos = [1, 2]
            dfp_x = [0, 2048]
            dfp_points = {'det_nos': np.array(det_nos), 'dfp_x': np.array(dfp_x), 'dfp_y': np.array(dfp_y)}
            efp_points = Util.dfp_to_efp(transform, affines, dfp_points)
            transform.slice_configuration['w_min'] = efp_points['efp_w'][0]
            transform.slice_configuration['w_max'] = efp_points['efp_w'][1]
        return

    def __str__(self):
        """Return ``"<opticon>, EA=<ech_ang> deg, PA=<pri_ang> deg, slices <min>-<max>"``.

        Notes
        -----
        - Clean-up: ``opticon`` is read from the class attribute
          ``RayTrace.model_config``, not from ``self.lms_config['opticon']``.
        """
        fmt = "{:s}, EA={:6.3f} deg, PA={:6.3f} deg, "
        opticon = RayTrace.model_config[1]
        string = fmt.format(opticon, self.lms_config['ech_ang'], self.lms_config['pri_ang'])
        smin, smax = self.unique_slice_nos[0], self.unique_slice_nos[-1]
        string += "slices {:d}-{:d}".format(int(smin), int(smax))
        return string

    def get_ifp_boresight(self, opticon):
        """Return the boresight wavelength and configuration of this trace.

        The boresight is the wavelength at which rays of field slice 13
        (``pslice_no`` 0 in nominal mode, 3 in extended mode, using the first
        or the middle of the trace's echelle orders respectively) cross
        ``mfp_x = 0``, found with ``numpy.interp(0.0, mfp_x, waves)``.

        Parameters
        ----------
        opticon : str
            ``Globals.nominal`` or ``Globals.extended``.

        Returns
        -------
        tuple
            ``(wave_bs, pri_ang, ech_ang, ech_ord)`` -- microns, deg, deg, int.

        Notes
        -----
        - Open question: ``numpy.interp`` requires a monotonically INCREASING
          abscissa.  ``mfp_x`` is not sorted here and, per the note in
          ``Util.mfp_to_dfp``, decreases with wavelength; if the CSV rows are
          in ascending wavelength order the result is undefined.  The same
          pattern appears in ``Util.get_term_values``.
        - Inconsistency: the boresight slice is 13 here and in
          ``Filer.write_fit_parameters``, but 12 in
          ``PolyFit.create_polynomial_surface_fits``.
        - Inconsistency: ``efp_x`` (alpha) is not filtered, so rays across
          the full slice length contribute to the interpolation.
        """
        pslice_no = 0 if opticon == Globals.nominal else 3
        # In extended mode there are 3 echelle orders in each trace, with pslice_no = 3 using the 'middle' order.
        ech_ord = self.unique_ech_ords[0] if opticon == Globals.nominal else self.unique_ech_ords[1]
        slice_filter = {'fslice_no': 13, 'pslice_no': pslice_no, 'ech_ord': ech_ord}
        mfp_x = self.get_series('mfp_x', slice_filter)
        waves = self.get_series('wavelength', slice_filter)
        wave_bs = np.interp(0.0, mfp_x, waves)              # Find wavelength where 'det_x' (== mfp_x) == 0.
        boresight = wave_bs, self.lms_config['pri_ang'], self.lms_config['ech_ang'], ech_ord
        return boresight

    def create_svd_transforms(self, **kwargs):
        """Fit the A, B, AI, BI distortion matrices for every (slice, echelle order) in the trace.

        For each ``slice_no`` in ``unique_slice_nos`` and ``ech_ord`` in
        ``unique_ech_ords`` with at least one ray:

        1. ``phase = Util.waves_to_phases(waves, ech_ords)`` (``n lambda / 2d``)
           and ``alpha = efp_x``;
        2. ``a, b = Util.solve_svd_distortion(phase, alpha, mfp_x, mfp_y,
           order=Globals.svd_order - 1)`` (forward) and ``ai, bi`` with
           ``inverse=True``, where ``mfp_x/y`` are the series named by
           ``model_config[5]`` (``Globals.coord_out``);
        3. the forward fit is re-applied and the per-ray residual radius
           ``sqrt(dx^2 + dy^2)`` (mm) collected;
        4. a ``Transform(lms_config=self.lms_config, slice_config={'slice_no',
           'ech_ord'})`` with ``wavelength_bounds = (min, max)`` of the rays is
           appended to ``self.transforms``.

        Parameters
        ----------
        do_plot : bool, optional
            If True, :meth:`plot_scatter` is called for the first fitted slice
            only.  Default False.

        Returns
        -------
        None

        Notes
        -----
        - Bug: ``a_rms_list`` is filled but ``self.a_rms`` is never set from
          it, so ``lmsdist.run`` reports ``a_rms = 0.000 microns``.
        - Bug: ``do_plot=True`` -> :meth:`plot_scatter` -> ``KeyError:
          'spifu_no'``.
        - Clean-up: the same ``self.lms_config`` dict object is shared by all
          transforms (see ``Transform.make_hdu_primary`` side effect).
        - Clean-up: the ``n_waves == 0`` check duplicates the ``len(waves) < 1``
          check above it.
        - Open question: no minimum ray count is enforced; a slice/order with
          fewer rays than the 10 free terms per matrix is fitted anyway (the
          SVD cutoff will then zero singular values silently unless debug
          level 'medium' is set).
        """
        do_plot = kwargs.get('do_plot', False)
        n_terms = Globals.svd_order
        slice_order = n_terms - 1
        _, _, _, _, fp_in, fp_out = self.model_config
        is_first = True
        a_rms_list = []
        # for pslice_no in self.unique_pslice_nos:
        #     for fslice_no in self.unique_fslice_nos:
        for slice_no in self.unique_slice_nos:
            for ech_ord in self.unique_ech_ords:
                # fslice_no, pslice_no = Util.decode_slice_no(slice_no)
                slice_filter = {'slice_no': slice_no, 'ech_ord':ech_ord}
                transform = Transform(lms_config=self.lms_config, slice_config=slice_filter)
                waves = self.get_series('wavelength', slice_filter)
                if len(waves) < 1:      # This happens for mismatched spectral slice and echelle order.
                    continue
                transform.wavelength_bounds = np.amin(waves), np.amax(waves)

                n_waves, = waves.shape
                if n_waves == 0:
                    continue
                ech_ords = self.get_series('ech_ord', slice_filter)
                alpha = self.get_series('efp_x', slice_filter)
                mfp_x = self.get_series(fp_out[0], slice_filter)
                mfp_y = self.get_series(fp_out[1], slice_filter)

                phase = Util.waves_to_phases(waves, ech_ords)
                a, b = Util.solve_svd_distortion(phase, alpha, mfp_x, mfp_y, slice_order, inverse=False)
                ai, bi = Util.solve_svd_distortion(phase, alpha, mfp_x, mfp_y, slice_order, inverse=True)

                mfp_x_fit, mfp_y_fit = Util.apply_svd_distortion(phase, alpha, a, b)
                off_mfp_x, off_mfp_y = mfp_x - mfp_x_fit, mfp_y - mfp_y_fit
                off_mfp_a = np.sqrt(np.square(off_mfp_x) + np.square(off_mfp_y))
                a_rms_list.append(off_mfp_a)

                mats = transform.matrices
                mats['a'] = a
                mats['b'] = b
                mats['ai'] = ai
                mats['bi'] = bi
                rays = waves, phase, alpha, mfp_x, mfp_y, mfp_x_fit, mfp_y_fit
                if do_plot and is_first:        # Plot intermediate and full fit to data
                    fmt = "Distortion residuals, A,B, polynomial fit, SVD cutoff = {:5.1e}\n"
                    tlin1 = fmt.format(Globals.svd_cutoff)
                    self.plot_scatter(transform, rays, plot_correction=True, tlin1=tlin1)
                    is_first = False
                self.transforms.append(transform)
        return

    def get_series(self, tag, series_filter):
        """Return one series restricted to rays matching all key/value pairs of a filter.

        A boolean register over all rays is ANDed with ``series[key] ==
        value`` for every key in ``series_filter``; the selected values of
        ``series[tag]`` are returned as a 1-D array in file order.

        Parameters
        ----------
        tag : str
            Series name, e.g. ``'wavelength'``, ``'efp_x'``, ``'mfp_x'``.
        series_filter : dict
            E.g. ``{'slice_no': 13, 'ech_ord': 27}`` or ``{'fslice_no': 13,
            'pslice_no': 0, 'ech_ord': 27}``.  EVERY key must be an existing
            series name.

        Returns
        -------
        numpy.ndarray
            1-D array (may be empty).

        Raises
        ------
        KeyError
            If a filter key is not a series (e.g. ``'w_min'`` when a whole
            ``Transform.slice_configuration`` is passed, as
            ``lmsdist_plot.Plot.series`` does).

        Notes
        -----
        - Clean-up: the ``argwhere`` / fancy-index implementation could be a
          single boolean mask; the commented-out legacy body below the
          ``return`` should be deleted.
        - Clean-up: the ``:rtype:`` annotation in the previous docstring was
          IDE-generated noise.
        """
        series_shape = self.series['slice_no'].shape
        idx_register = np.full(series_shape, True, dtype=bool)
        for key in series_filter:
            array = self.series[key]
            idx = np.argwhere(array != series_filter[key])
            idx_register[idx] = False
            # idx_register[idx] = np.logical_and(idx[0], idx_register[idx])
        idx = np.argwhere(idx_register)
        out = self.series[tag][idx]
        return out[:, 0]

        # slice_no = series_filter['slice_no']
        # # pslice_no = series_filter.get('pslice_no', 0)
        # ech_ord = series_filter.get('ech_ord', None)
        # a = self.series[tag]
        # fslice_nos = self.series['fslice_no']
        # fslice_no_mask = fslice_nos == fslice_no
        # pslice_nos = self.series['pslice_no']
        # pslice_no_mask = pslice_nos == pslice_no
        # mask = np.logical_and(fslice_no_mask, pslice_no_mask)
        # if ech_ord is not None:
        #     ech_ords = self.series['ech_ord']
        #     ech_ord_mask = ech_ords == ech_ord
        #     mask = np.logical_and(mask, ech_ord_mask)
        # a = np.compress(mask, a)
        # return a

    def plot_fit_maps(self, **kwargs):
        """Quiver plot of (Zemax - transform) residual vectors in the mosaic plane, all slices.

        For each transform the slice's rays are projected with
        ``Util.efp_to_mfp`` and arrows from the fitted to the Zemax position
        are drawn (``ax.quiver`` with ``scale=0.001``, i.e. strongly
        exaggerated).

        Parameters
        ----------
        suppress : bool, optional
            Return immediately without plotting.
        plotdiffs : bool, optional
            If False, fix the x-limits to +-40 mm (the mosaic); if True leave
            them automatic.
        subset : bool, optional
            Read but unused (legacy multi-panel layout is commented out).
        field : bool, optional
            Accepted via ``**kwargs`` by ``lmsdist.run`` but not read.

        Returns
        -------
        None

        Notes
        -----
        - Clean-up: ``subset`` / ``field`` are dead parameters; the
          commented-out panel layout should be removed.
        - Clean-up: the ``oob`` flag from ``Util.efp_to_mfp`` is ignored.
        """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return
        plot = Plot()

        plotdiffs = kwargs.get('plotdiffs', False)
        subset = kwargs.get('subset', False)
        name = self.csv_name
        echelle_angle = self.lms_config['ech_ang']
        fig_title = "{:s} ea = {:4.2f}".format(name, echelle_angle)
        # SPIFU 1 column per spatial slice, 1 column
        # n_rows, n_cols = 7, 4
        # unique_slices = self.unique_slice_nos
        # slice_start = unique_slices[0]
        # if self.is_extended:
        #     n_rows = 6
        #     n_cols = 3
        # if subset:
        #     n_rows = 3
        #     n_cols = 1
        xlim = (None if plotdiffs else [-40.0, 40.0])
        fig, ax_list = plot.set_plot_area(fontsize=14.0, sharex=False, sharey=False,
                                          nrows=1, ncols=1, xlim=xlim)
        ax = ax_list[0, 0]
        fig.suptitle(fig_title)
        for transform in self.transforms:
            slice_config = transform.slice_configuration
            slice_no = slice_config['slice_no']
            ech_ord = slice_config['ech_ord']
            fslice_no, pslice_no = Util.decode_slice_no(slice_no)
            slice_filter = {'fslice_no': fslice_no, 'pslice_no': pslice_no, 'ech_ord': ech_ord}
            efp_x = self.get_series('efp_x', slice_filter)
            efp_y = self.get_series('efp_y', slice_filter)
            efp_w = self.get_series('wavelength', slice_filter)
            mfp_x = self.get_series('mfp_x', slice_filter)
            mfp_y = self.get_series('mfp_y', slice_filter)
            efp_points = {'efp_x': np.array(efp_x), 'efp_y': np.array(efp_y), 'efp_w': np.array(efp_w)}
            mfp_fit_points, oob = Util.efp_to_mfp(transform, efp_points)
            det_x_fit, det_y_fit = mfp_fit_points['mfp_x'], mfp_fit_points['mfp_y']
            u, v = mfp_x - det_x_fit, mfp_y - det_y_fit
            _ = ax.quiver(det_x_fit, det_y_fit, u, v,
                          angles='xy', scale_units='xy', scale=.001,
                          width=0.001)
        plot.show()
        return

    def plot_scatter(self, transform, rays, **kwargs):
        """Two-panel scatter of fit residuals (x, y, radial; microns) versus detector x and y.

        Parameters
        ----------
        transform : Transform
            Supplies the title fields (``ech_ord``, ``slice_no``,
            ``spifu_no``).
        rays : tuple
            ``(waves, phase, alpha, mfp_x, mfp_y, mfp_x_fit, mfp_y_fit)`` as
            built in :meth:`create_svd_transforms`.
        slice_list : list of int, optional
            If given, plot only when ``slice_no`` is in the list.
        tlin1 : str, optional
            First title line.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            Always, at ``slice_config['spifu_no']``: the key is not in
            ``Globals.slice_config_template`` and is never added.

        Notes
        -----
        - Bug: ``spifu_no`` KeyError makes this routine (and therefore
          ``load(do_plot=True)``) unusable.
        - Inconsistency: legend colours (salmon / olive) differ from the
          plotted colours (orangered / green).
        - Clean-up: ``lms_config`` is fetched from ``transform`` but the
          angles are read from ``self.lms_config``.
        """
        slice_list = kwargs.get('slice_list', None)
        tlin1_default = "Distortion residuals (Zemax - Zemax Fit): \n"
        tlin1 = kwargs.get('tlin1', tlin1_default)
        slice_config = transform.slice_configuration

        ech_ord = slice_config['ech_ord']
        slice_no = slice_config['slice_no']
        spifu_no = slice_config['spifu_no']
        # Create plot area
        lms_config = transform.lms_configuration
        ech_ang = self.lms_config['ech_ang']
        pri_ang = self.lms_config['pri_ang']
        fmt = "order={:d}, ech. angle={:3.1f}, prism angle={:4.3f}, slice={:d}"
        tlin2 = fmt.format(int(ech_ord), ech_ang, pri_ang, int(slice_no))
        if spifu_no > 0:
            tlin2 += ", spifu={:d}".format(int(spifu_no))
        title = tlin1 + tlin2

        fig, ax_list = Plot.set_plot_area(nrows=1, ncols=2, fontsize=10.0, sharey=True)
        fig.suptitle(title)

        x_list, y_list = [], []
        off_det_x_list, off_det_y_list, off_det_a_list = [], [], []

        if slice_list is None or slice_no in slice_list:
            _, _, _, det_x, det_y, det_x_fit, det_y_fit = rays
            x_list.append(det_x)
            y_list.append(det_y)
            off_det_x, off_det_y = det_x - det_x_fit, det_y - det_y_fit
            off_det_a = np.sqrt(np.square(off_det_x) + np.square(off_det_y))
            off_det_x_list.append(off_det_x)
            off_det_y_list.append(off_det_y)
            off_det_a_list.append(off_det_a)

            for pane in range(0, 2):
                ax = ax_list[0, pane]
                u = det_x if pane == 0 else det_y
                ax.plot(u, 1000. * off_det_x, marker='.', ls='none', ms=1.5, color='orangered')
                ax.plot(u, 1000. * off_det_y, marker='.', ls='none', ms=1.5, color='green')
                ax.plot(u, 1000. * off_det_a, marker='.', ls='none', ms=1.5, color='blue')

        # Add the legend to both plots
        x_labels = ['Det x / mm', 'Det y / mm']
        y_labels = ["Residual / micron.", '']
        gk_text = r'$\sigma$ = ' + '{:4.2f}' + r'$\mu$m'
        fmt = '{:s} residual, ' + gk_text
        for pane in range(0, 2):
            ax = ax_list[0, pane]
            ax.set_xlabel(x_labels[pane])
            ax.set_ylabel(y_labels[pane])
            if pane == 0:       # Only draw the key in pane 0
                off_x_all = np.array(off_det_x_list) * 1000.
                off_y_all = np.array(off_det_y_list) * 1000.
                off_a_all = np.array(off_det_a_list) * 1000.
                off_x_rms = np.std(off_x_all)
                off_y_rms = np.std(off_y_all)
                off_a_rms = np.std(off_a_all)
                label1 = fmt.format('x', off_x_rms)
                label2 = fmt.format('y', off_y_rms)
                label3 = fmt.format('a', off_a_rms)
                patch_x = mpatches.Patch(color='salmon', label=label1)
                patch_y = mpatches.Patch(color='olive', label=label2)
                patch_a = mpatches.Patch(color='blue', label=label3)
                ax.legend(handles=[patch_x, patch_y, patch_a])

        Plot.show()
        return

    @staticmethod
    def _find_residual(det_xy, waves, poly_order):
        """Return residuals of a 1-D polynomial fit ``waves = P(det_xy)``.

        Parameters
        ----------
        det_xy : array_like
            Detector coordinate (mm or pixels).
        waves : array_like
            Wavelengths (microns).
        poly_order : int
            Degree for ``numpy.polyfit``.

        Returns
        -------
        numpy.ndarray
            ``waves - P(det_xy)``.

        Notes
        -----
        - Clean-up: prints the coefficient array to stdout; not called
          anywhere in the repository.
        """
        poly = np.polyfit(det_xy, waves, poly_order)
        text = poly.__str__()
        print(text)
        fit = np.poly1d(poly)
        wfit = fit(det_xy)
        dw = waves - wfit
        return dw

    def get_plot_title(self):
        """Return a LaTeX title string with prism angle, echelle angle and reference wavelength.

        Returns
        -------
        str
            ``"$\\phi_{pri}$=<pa>, $\\psi_{ech}$=<ea>, $\\lambda_{bs}$=<wave_ref> $\\mu$m"``.

        Notes
        -----
        - Inconsistency: labels the reference wavelength as ``lambda_bs``
          (boresight) although it is the mean ray wavelength from
          :meth:`_get_wave_reference`, not the boresight of
          :meth:`get_ifp_boresight`.
        """
        theta_p = r'$\phi_{pri}$' + "={:6.3f}, ".format(self.lms_config['pri_ang'])
        theta_e = r'$\psi_{ech}$' + "={:6.3f}, ".format(self.lms_config['ech_ang'])
        wave_ref = self._get_wave_reference()
        wave_text = r'$\lambda_{bs}$' + "={:6.3f}, ".format(wave_ref) + r'$\mu$m'
        title = theta_p + theta_e + wave_text
        return title

    def plot_slice_map(self, slice_no, **kwargs):
        """Plot (phase, alpha) and (det_x, det_y) for one slice.  Dead code.

        Parameters
        ----------
        slice_no : int
        suppress : bool, optional

        Returns
        -------
        None

        Raises
        ------
        KeyError
            At ``self.get_series('phase', ...)``: there is no ``'phase'``
            series (phase is computed on the fly in
            :meth:`create_svd_transforms`); likewise ``'fp2_x'``, ``'det_x'``,
            ``'det_y'`` are CSV column names, not series names.
        AttributeError
            ``self.parameter`` and ``self.rgb`` do not exist.
        TypeError
            ``Plot.set_plot_area`` does not accept a positional title.

        Notes
        -----
        - Bug: multiple stale references as listed above; the method predates
          the series renaming and cannot run.
        - Clean-up: not called anywhere; remove or rewrite.
        """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return

        plot = Plot()
        titles = ['LMS EFP', 'Detector']

        series_filter = {'slice_no': slice_no}
        phase = self.get_series('phase', series_filter)
        alpha = self.get_series('fp2_x', series_filter)
        det_x = self.get_series('det_x', series_filter)
        det_y = self.get_series('det_y', series_filter)

        xlabels = ['Phase (n lambda / 2 d)', 'X [mm]']
        ylabels = ['alpha [mm]', 'Y [mm]']
        fig_title = "{:s}, slice= {:d}".format(self.parameter['name'], slice_no)
        ax_list = plot.set_plot_area(fig_title, nrows=1, ncols=2)

        xs = [phase, det_x]
        ys = [alpha, det_y]

        for idx in range(0, 2):
            x = xs[idx]
            y = ys[idx]
            xlim = self._find_limits(x, 0.1)
            ylim = self._find_limits(y, 0.1)
            ax = ax_list[0, idx]
            ax.set_title(titles[idx], loc='left')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabels[idx])
            ax.set_ylabel(ylabels[idx])
            plot.plot_points(ax, x, y, fs='full', ms=1.0, mk='o', rgb=self.rgb)

        plot.show()
        return

    def plot_focal_planes(self, **kwargs):
        """Plot ray positions at every intermediate focal surface, one panel per surface.

        Panels follow ``RayTrace.nominal_focal_planes`` (LMS EFP, Slicer, IFU,
        Slit, Det. mosaic) or ``RayTrace.spifu_focal_planes`` (SP slicer
        instead of Slit) depending on ``is_extended``; panels whose series are
        missing are skipped.  Points are blue by default or coloured by
        wavelength (:meth:`_create_wave_colours`).  For the Slit, SP slicer and
        Det. mosaic panels the colours are masked by :meth:`_apply_masks`:
        vignetted rays (``cf_vig != 0``) magenta, rays missing all four
        detectors (``mask == 0``) grey.

        Parameters
        ----------
        suppress : bool, optional
            Return without plotting.
        colour_scheme : str, optional
            ``'wavelength'`` for a wavelength ramp; anything else gives
            uniform blue.  The previous docstring also listed ``'slice_no'``
            and ``'mono'``, which are not implemented.

        Returns
        -------
        None

        Notes
        -----
        - Inconsistency: the documented ``'slice_no'`` / ``'mono'`` colour
          schemes do not exist in :meth:`_create_wave_colours`.
        - Inconsistency: the 'Slicer' panel plots ``(slicer_y, slicer_x)``,
          i.e. swapped axes, as noted in the class-level table.
        - Clean-up: ``fp_plot_mask`` keys must match the focal-plane labels
          exactly; a label missing from it raises ``KeyError``.
        """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return

        fp_plot_mask = {'LMS EFP': False, 'Slicer': False, 'IFU': False,
                        'Slit': True, 'SP slicer': True, 'Det. mosaic': True}
        focal_planes = RayTrace.spifu_focal_planes if self.is_extended else RayTrace.nominal_focal_planes

        n_focal_planes = len(focal_planes)
        colour_scheme = kwargs.get('colour_scheme', 'blue')

        suptitle = self.get_plot_title()

        rgb = self._create_wave_colours(colour_scheme)
        rgb_masked = self._apply_masks(rgb.copy())

        plot = Plot()

        xlabel = 'X [mm]'
        ylabel = 'Y [mm]'
        title = ''
        fig, ax_list = plot.set_plot_area(title=self.__str__(), nrows=1, ncols=n_focal_planes)
        fig.suptitle(suptitle, fontsize=20)
        for pane, title in enumerate(focal_planes):
            if fp_plot_mask[title]:
                rgb = rgb_masked

            ax = ax_list[0, pane]
            fp_x, fp_y = focal_planes[title]
            if fp_x not in self.series.keys() or fp_y not in self.series.keys():
                continue
            x = self.series[fp_x]
            y = self.series[fp_y]
            ax.set_title(title, loc='left')
            ax.set_xlabel(xlabel)
            if pane == 0:
                ax.set_ylabel(ylabel)
            Plot.plot_points(ax, x, y, fs='full', ms=3.0, mk='o', rgb=rgb)
        plot.show()
        return

    def _read_csv(self, path, model_config):
        """Parse a ray-trace CSV into the ``lms_config`` dict and the ``series`` arrays.

        The parser (see module docstring for the file layout):

        1. Chooses the column map: ``nom_csv_map`` (nominal), ``ext_csv_map``
           (extended) or an AIT map (``slice_no, spifu_no, ech_ord, efp_w ->
           wavelength, efp_x, efp_y, mfp_x, mfp_y, cf_vig``) when
           ``model_config[2] == 'ait'``.
        2. Reads ``name: value`` header lines until a line without ``':'``.
           ``Echelle angle`` and ``Prism angle`` go into ``lms_config``;
           ``Spectral order`` is kept as the per-file echelle order for
           nominal files.
        3. Reads the column-name line and builds ``name -> index``.
        4. For each data row (until a line with < 2 tokens) appends every
           mapped column, converting by ``series_fmt`` (``'int'`` /
           ``'float'`` via ``match`` / ``case``).  Nominal rows additionally
           get ``pslice_no = 0``, ``ech_ord = <Spectral order>`` and
           ``cf_vig = 0``.
        5. Converts lists to arrays; ``ech_ord, slice_no, spifu_no`` as int.

        Parameters
        ----------
        path : str
        model_config : tuple
            ``(_, opticon, data_id, _, coord_in, coord_out)``.

        Returns
        -------
        csv_name : str
        lms_config : dict
            ``{'opticon', 'pri_ang', 'ech_ang'}``.
        series : dict of numpy.ndarray
            Also stored as ``self.series``.

        Raises
        ------
        KeyError
            If the CSV contains a column not present in the chosen map, or
            (AIT map) because ``'slice_no'`` / ``'spifu_no'`` are not in
            ``series_fmt``.
        NameError
            For a nominal file without a ``Spectral order:`` header line
            (``nom_ech_order`` is then unbound).
        StopIteration
            If the file ends without a terminating short line.

        Notes
        -----
        - Bug: the AIT branch cannot run (``series_fmt`` lacks ``slice_no`` /
          ``spifu_no``; ``load`` then needs ``fslice_no`` / ``pslice_no`` which
          this map never fills; the AIT CSV from ``lmsaiv_opt02`` has no
          ``spifu_no``, ``ech_ord`` or ``cf_vig`` columns and its
          ``'Spectral order:'`` value is handled only for nominal mode).
        - Inconsistency: ``fslice_no`` / ``pslice_no`` are converted with
          ``np.array(vals)`` (float) although ``series_fmt`` declares them
          ``'int'``; only ``ech_ord, slice_no, spifu_no`` are cast to int.
        - Inconsistency: the ``cf_vig`` column (``chief_ray_vignetted?``) is
          read from the file for extended data but forced to 0 for nominal
          data BEFORE the column loop, so if a nominal file has the column it
          is appended twice per row (list lengths then disagree).
        - Clean-up: uses ``match`` / ``case`` (Python >= 3.10); ``coord_in`` /
          ``coord_out`` are unpacked but unused.
        """
        nom_spifu_no = 0
        _, opticon, data_id, _, coord_in, coord_out = model_config
        csv_map = RayTrace.nom_csv_map if opticon == Globals.nominal else RayTrace.ext_csv_map
        if data_id == 'ait':
            csv_map = {'slice_no': 'slice_no', 'spifu_no': 'spifu_no', 'ech_ord': 'ech_ord',
                       'efp_w': 'wavelength', 'efp_x': 'efp_x', 'efp_y': 'efp_y',
                       'mfp_x': 'mfp_x', 'mfp_y': 'mfp_y',
                       'cf_vig': 'cf_vig'
                       }
        series = {}
        for csv_name in csv_map:
            series_name = csv_map[csv_name]
            series[series_name] = []

        csv_name = path.split('/')[-1]
        with open(path, 'r') as text_file:
            read_data = text_file.read()
        line_list = read_data.split('\n')
        line_iter = iter(line_list)

        lms_config = Globals.lms_config_template.copy()
        lms_config['opticon'] = opticon
        # Read column names and create column indices
        csv_col_list = {}
        while True:
            line = next(line_iter)
            tokens = line.split(':')
            if len(tokens) < 2:
                break

            par_to_lms = {'Echelle angle': 'ech_ang',
                          'Prism angle': 'pri_ang',
                          'Spectral order': 'null'}
            parameter = tokens[0].strip()
            val = float(tokens[1].strip(', '))
            if parameter in par_to_lms.keys():
                if parameter == 'Spectral order':       # Single value per csv file for nominal data (only)
                    nom_ech_order = val                 # The spectral order is written as a series array
                    continue
                lms_name = par_to_lms[parameter]
                lms_config[lms_name] = val

        # Create dictionary of data series, include echelle order, slice, spifu_slice, wavelength, and input and
        # output focal planes.
        line = next(line_iter)
        tokens = line.split(',')
        for i, token in enumerate(tokens):
            if len(token) < 2:  # Empty token (at end of line probably)
                continue
            csv_col_list[token.strip()] = i

        while True:
            line = next(line_iter)
            tokens = line.split(',')
            if len(tokens) < 2:
                break

            if opticon == Globals.nominal:
                series['pslice_no'].append(nom_spifu_no)      # Trap special cases
                series['ech_ord'].append(nom_ech_order)
                series['cf_vig'].append(0.)

            for csv_name in csv_col_list:
                col_idx = csv_col_list[csv_name]
                token = tokens[col_idx].strip()
                series_name = csv_map[csv_name]
                fmt = RayTrace.series_fmt[series_name]
                val = None
                match fmt:
                    case 'float':
                        val = float(token)
                    case 'int':
                        val = int(token)
                    case _:
                        print("Un-supported format {:s} for token {:s}".format(fmt, token))
                series[series_name].append(val)

        # Convert series lists to numpy arrays
        int_arrays = ['ech_ord', 'slice_no', 'spifu_no']
        for name in series:
            vals = series[name]
            series[name] = np.array(vals, dtype=int) if name in int_arrays else np.array(vals)
        self.series = series
        return csv_name, lms_config, series

    def _create_mask(self, silent):
        """Flag which detector (if any) each ray hits and store it as ``series['mask']``.

        Detector active-area bounds in the mosaic plane are built from
        ``edge = Detector.det_size * Detector.det_pix_size / 1000`` (36.864 mm),
        ``xy_n = Globals.det_gap / 2 + Globals.margin`` and
        ``xy_f = Globals.det_gap / 2 + edge - Globals.margin`` (a 64-pixel
        unilluminated margin is excluded), for the four quadrants
        ``BL, TL, BR, TR``.  Rays inside quadrant ``k`` (0-based enumeration
        order) get ``mask = k + 1``; rays hitting no detector get 0.  Also
        stores ``self.xy_bounds``.

        Parameters
        ----------
        silent : bool
            Unused; the hit count is printed at debug level ``'medium'``.

        Returns
        -------
        None

        Notes
        -----
        - Inconsistency: ``Globals.det_gap`` (3.0 mm) is used with
          ``Detector.det_size`` / ``det_pix_size`` although ``Detector`` itself
          defines ``mosaic_gap = 2.23`` mm.
        - Inconsistency: mask numbering BL=1, TL=2, BR=3, TR=4 does not match
          the detector numbering of ``Util.mfp_to_dfp`` / the affines
          (1 = TR quadrant x>0,y>0; 2 = TL; 3 = BR; 4 = BL).  Only the zero /
          non-zero distinction is used (:meth:`_apply_masks`).
        - Clean-up: ``silent`` is a dead parameter.
        """
        edge = Detector.det_size * Detector.det_pix_size / 1000.0
        xy_f = (Globals.det_gap / 2.0) + edge - Globals.margin
        xy_n = (Globals.det_gap / 2.0) + Globals.margin
        xy_bounds = {'BL': [-xy_f, -xy_n, -xy_f, -xy_n], 'TL': [-xy_f, -xy_n,  xy_n,  xy_f],
                     'BR': [xy_n,  xy_f, -xy_f, -xy_n], 'TR': [xy_n,  xy_f,  xy_n,  xy_f]}
        self.xy_bounds = xy_bounds
        x = self.series['mfp_x']
        y = self.series['mfp_y']

        n_pts = len(x)
        mask = np.zeros(n_pts, dtype='int')	    # >0 = Ray hits a detector
        for det_no, det_name in enumerate(xy_bounds):               # Loop through the 4 detectors
            xy_b = xy_bounds[det_name]
            idx_gt_xlo = x > xy_b[0]
            idx_lt_xhi = x < xy_b[1]
            idx_in_x = np.logical_and(idx_gt_xlo, idx_lt_xhi)
            idx_gt_ylo = y > xy_b[2]
            idx_lt_yhi = y < xy_b[3]
            idx_in_y = np.logical_and(idx_gt_ylo, idx_lt_yhi)
            idx_in = np.where(np.logical_and(idx_in_x, idx_in_y))
            mask[idx_in] = det_no + 1
        n_hits = np.count_nonzero(mask)
        if Globals.is_debug('medium'):
            fmt = 'Rays hitting any detector = {:10d} / {:10d}'
            print(fmt.format(n_hits, n_pts))
        self.series['mask'] = mask
        return

    def _apply_masks(self, rgb):
        """Recolour vignetted rays magenta and rays missing all detectors grey.

        Parameters
        ----------
        rgb : numpy.ndarray
            ``(n_rays, 3)`` colour array, modified in place.

        Returns
        -------
        numpy.ndarray
            The same array.  Rows with ``series['cf_vig'] != 0`` become
            ``[1, 0, 1]``; rows with ``series['mask'] == 0`` become
            ``[0.3, 0.3, 0.3]`` (applied second, so it wins).
        """
        palette = {'cf_vig': ([1., 0., 1.], '!='),
                   'mask': ([.3, .3, .3], '==')}

        for pal in palette:
            colour, lop = palette[pal]
            mask = self.series[pal]
            condition = mask == 0 if lop == '==' else mask != 0
            mask_indices = np.argwhere(condition)
            rgb[mask_indices, :] = colour
        return rgb

    def _create_wave_colours(self, colour_scheme):
        """Return an ``(n_rays, 3)`` RGB array: uniform blue, or a wavelength ramp.

        Parameters
        ----------
        colour_scheme : str
            ``'wavelength'`` -> ``Plot.make_rgb_gradient(series['wavelength'])``;
            any other value -> ``[0.3, 0.3, 1.0]`` for every ray.

        Returns
        -------
        numpy.ndarray
        """
        n_pts = len(self.series['wavelength'])
        rgb = np.array([[.3, .3, 1.]]*n_pts)              # Default is blue
        if colour_scheme == 'wavelength':
            waves = self.series['wavelength']
            rgb = Plot.make_rgb_gradient(waves)
        return rgb

    @staticmethod
    def _find_limits(a, margin):
        """Return ``[min - m, max + m]`` with ``m = margin * (max - min)``.

        Parameters
        ----------
        a : array_like
        margin : float
            Fractional padding, e.g. 0.1.

        Returns
        -------
        list of float

        Notes
        -----
        - Clean-up: only used by the dead :meth:`plot_slice_map`.
        """
        amin = min(a)
        amax = max(a)
        arange = amax - amin
        amargin = margin * arange
        limits = [amin - amargin, amax + amargin]
        return limits
