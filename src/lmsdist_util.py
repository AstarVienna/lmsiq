#!/usr/bin/env python
"""Numerical core of ``lmsdist``: distortion polynomials, phase definition and focal-plane mappings.

Purpose
-------
:class:`Util` gathers the static functions that implement the
E-TNT-ATC-MET-1058 LMS distortion model and the bookkeeping around it:

- the grating phase ``phi = n * lambda / (2 d)`` (:meth:`Util.waves_to_phases`,
  :meth:`Util.phases_to_waves`);
- the bivariate polynomial transform and its SVD least-squares solver
  (:meth:`Util.apply_svd_distortion`, :meth:`Util.solve_svd_distortion`);
- the slice <-> EFP across-slice coordinate convention
  (:meth:`Util.slice_to_efp_y`, :meth:`Util.efp_y_to_slice_no`,
  :meth:`Util.encode_slice_no`, :meth:`Util.decode_slice_no`);
- the chained mappings EFP -> MFP -> DFP and back
  (:meth:`Util.efp_to_mfp`, :meth:`Util.mfp_to_dfp`, :meth:`Util.efp_to_dfp`,
  :meth:`Util.dfp_to_mfp`, :meth:`Util.mfp_to_efp`, :meth:`Util.dfp_to_efp`);
- selection helpers over lists of ``lms_transform.Transform``
  (:meth:`Util.filter_transform_list`, :meth:`Util.get_term_values`,
  :meth:`Util.find_closest_transforms`, :meth:`Util.find_matching_transforms`);
- assorted legacy text-format readers/writers and study helpers.

Program
-------
``lmsdist`` (also imported by ``lmssim``, ``lmsaiv`` and ``lmsiq``).

Coordinate systems
------------------
- EFP (entrance focal plane): ``efp_x`` mm along slice (alpha), ``efp_y`` mm
  across slice (beta), ``efp_w`` wavelength in microns.  Plate scale
  ``Globals.efp_arcsec_mm = 0.180723`` arcsec/mm; slice width
  ``Globals.beta_slice = 20.7`` mas.
- MFP (mosaic focal plane): ``mfp_x``, ``mfp_y`` mm, origin at the centre of
  the 2 x 2 detector mosaic.  ``mfp_x`` increases towards SHORTER wavelength;
  slice 1 is at minimum ``mfp_y``.
- DFP (detector focal plane): ``det_nos`` 1..4, ``dfp_x`` column, ``dfp_y``
  row in pixels.  Detector assignment by MFP quadrant: ``x > 0`` -> 1 or 3,
  ``x < 0`` -> 2 or 4, ``y > 0`` -> 1 or 2, ``y < 0`` -> 3 or 4.

Distortion model
----------------
For one slice and echelle order, with ``x = phi`` (phase) and ``y = alpha =
efp_x`` and ``dim = Globals.svd_order = 4``::

    mfp_x = sum_{i,j<4, i+j<=3} A[i, j] * alpha**i * phi**j
    mfp_y = sum_{i,j<4, i+j<=3} B[i, j] * alpha**i * phi**j

(row index = power of the second input, column index = power of the first
input).  The inverse matrices ``AI, BI`` map ``(mfp_x, mfp_y) -> (phi, alpha)``
in the same form.  Elements with ``i + j > 3`` are forced to zero by the
solver (``svd_order = 4`` -> cubic, 10 free terms per matrix).  Across-slice
position is not part of the polynomial: it is degenerate with wavelength on
the detector and is recovered from the slice number only.

Input data
----------
None read directly except by the legacy text readers
(:meth:`Util.read_raytrace_file`, :meth:`Util.read_polyfits_file`,
:meth:`Util.read_text_file`), whose formats predate the FITS products and
have no producer in the repository.  Everything else operates on in-memory
``Transform`` objects and numpy arrays.

Outputs
-------
None on disk except :meth:`Util.write_text_file` / :meth:`Util.print_stats`
(legacy).  The mapping functions return dicts of arrays keyed
``efp_x, efp_y, efp_w`` / ``mfp_x, mfp_y`` / ``det_nos, dfp_x, dfp_y``.

Notes
-----
- Inconsistency: ``Globals.det_gap = 3.0`` mm (used via
  ``Transform.create_affines`` in every MFP <-> DFP call here) versus
  ``lms_detector.Detector.mosaic_gap = 2.23`` mm and the irdb value 2.23 mm.
- Inconsistency: :meth:`Util.mfp_to_dfp` and :meth:`Util.dfp_to_mfp` return
  points GROUPED BY DETECTOR, not in input order, and silently drop points
  with ``mfp_x == 0`` or ``mfp_y == 0``; callers that zip inputs with
  outputs (``RayTrace.find_wavelength_bounds``, ``test_out_and_back``) assume
  order is preserved.
- Bug: :meth:`Util.find_dispersion`, :meth:`Util.read_raytrace_file`,
  :meth:`Util.extract_configs` and :meth:`Util.test_out_and_back` reference
  attributes that no longer exist (``trace.unique_slices``, ``trace.slice``,
  ``trace.waves_to_phase``, dict-style ``transform['configuration']``); they
  are dead code.
- Bug: :meth:`Util.find_closest_transforms` reads
  ``slice_configuration['spifu_no']``, which is not in
  ``Globals.slice_config_template`` (``KeyError``); ``lmssim_toy`` still calls it.
- Inconsistency: ``lmsaiv_opt02`` calls ``Util.slice_to_efp_y(slice_no,
  beta_phase, opticon)`` with three arguments; the function takes two.
- Clean-up: :meth:`Util.filter_transform_list` uses the dict union operator
  ``|`` (Python >= 3.9).
- Clean-up: :meth:`Util.find_fwhm_lin` duplicates
  ``lmsiq_analyse.Analyse._find_fwhm_lin``; :meth:`Util.print_list` is used
  only by ``lmsiq``; ``n_lines_config`` is unused.
- Clean-up: ``least_squares`` and ``curve_fit`` are imported but unused.
- Open question: :meth:`Util.solve_svd_distortion` comments say "converted
  to python from Ronayette's original IDL code by Temim, 21/2/17 ... see note
  on MIRI calibration"; the MIRI note is not referenced by document number.
"""
import math
import numpy as np
from astropy import units as u
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from lms_globals import Globals


class Util:
    """Static helper functions for the LMS distortion model (see module docstring).

    Attributes
    ----------
    n_lines_config : int
        Unused (``-1``).

    Notes
    -----
    - Clean-up: the class is a namespace; ``__init__`` does nothing.
    """

    n_lines_config = -1

    def __init__(self):
        """Create a ``Util`` object.  No state is initialised."""
        return

    @staticmethod
    def print_list(title, val_list):
        """Print ``"<title> - v1, v2, ..."`` for a list of integers.

        Parameters
        ----------
        title : str
        val_list : iterable of int
            Each formatted with ``'{:d}'`` (floats raise ``ValueError``).

        Notes
        -----
        - Clean-up: used only by the ``lmsiq`` program
          (``lmsiq_cuber``, ``lmsiq_phase``), not by ``lmsdist``.
        """
        text = title + ' - '
        for val in val_list:
            text += "{:d}, ".format(val)
        print(text)
        return

    @staticmethod
    def write_text_file(transform_file, text_block):
        """Write ``text_block`` to ``transform_file`` (overwriting) and return the closed file object.

        Parameters
        ----------
        transform_file : str
        text_block : str

        Returns
        -------
        file object
            Already closed; of no practical use to the caller.

        Notes
        -----
        - Clean-up: legacy companion of :meth:`get_transform_text`; not
          called anywhere.
        """
        tf_file = open(transform_file, 'w')
        tf_file.write(text_block)
        tf_file.close()
        return tf_file

    @staticmethod
    def read_text_file(text_file, **kwargs):
        """Return the lines of a text file after skipping ``n_hdr_lines`` header lines.

        Parameters
        ----------
        text_file : str
        n_hdr_lines : int, optional
            Default 0.

        Returns
        -------
        list of str

        Notes
        -----
        - Clean-up: the file handle is never closed explicitly; not called
          anywhere in the repository.
        """
        n_hdr_lines = kwargs.get('n_hdr_lines', 0)
        all_lines = open(text_file, 'r').read().splitlines()
        lines = all_lines[n_hdr_lines:]
        return lines

    @staticmethod
    def waves_to_phases(waves, ech_ords):
        """Convert wavelengths to the dimensionless grating phase used by the transforms.

        .. math::

            \\phi = \\frac{n \\lambda}{2 d}, \\qquad
            d = \\mathrm{rule\\_spacing} \\times n_{Ge} = 18.2\\,\\mu m \\times 4.05

        i.e. ``d = Globals.rule_spacing * Globals.ge_refractive_index`` is the
        rule spacing of the immersed germanium echelle scaled by the
        refractive index, so ``phi`` is ``sin(theta_in) + sin(theta_out)`` up
        to the factor 2 for a Littrow-like configuration.  ``phi`` is the
        first input coordinate of ``A`` / ``B`` in :meth:`apply_svd_distortion`
        and is near 1 for the blaze (e.g. ``n = 27, lambda = 4.65 um ->
        phi ~= 0.85``).

        Parameters
        ----------
        waves : float or numpy.ndarray
            Wavelength(s) in microns.
        ech_ords : int or numpy.ndarray
            Echelle diffraction order(s), broadcastable with ``waves``.

        Returns
        -------
        float or numpy.ndarray
            Phase(s).

        Notes
        -----
        - Open question: the refractive index of germanium is taken as the
          constant 4.05 with no wavelength dependence; whether the Zemax model
          uses dispersion for the immersion medium is not recorded here.
        """
        d = Globals.rule_spacing * Globals.ge_refractive_index
        phases = waves * ech_ords / (2.0 * d)
        return phases      # phases

    @staticmethod
    def phases_to_waves(phases, ech_ords):
        """Inverse of :meth:`waves_to_phases`: ``lambda = 2 d phi / n``.

        Parameters
        ----------
        phases : float or numpy.ndarray
        ech_ords : int or numpy.ndarray
            Echelle order(s); the order must be known independently because
            it cannot be recovered from the phase alone.

        Returns
        -------
        float or numpy.ndarray
            Wavelength(s) in microns.
        """
        d = Globals.rule_spacing * Globals.ge_refractive_index
        waves = phases * 2.0 * d / ech_ords
        return waves

    @staticmethod
    def apply_affine_transform(x, y, aff):
        """Apply a 3 x 3 homogeneous affine matrix to point arrays.

        ``(u, v, 1)^T = aff @ (x, y, 1)^T`` for every point.

        Parameters
        ----------
        x, y : array_like
            Equal-length coordinate arrays.
        aff : numpy.ndarray
            ``(3, 3)`` matrix, e.g. ``Transform.affines['m'][det_idx]``
            (MFP mm -> DFP pixels) or ``['mi'][det_idx]`` (inverse).

        Returns
        -------
        u, v : numpy.ndarray
            Transformed coordinates.
        """
        n_pts = len(x)
        unity = np.full(n_pts, 1.)
        xy = np.array([x, y, unity])
        uv = aff @ xy
        return uv[0], uv[1]

    @staticmethod
    def apply_svd_distortion(x, y, a, b):
        """Apply a distortion matrix pair (A, B or AI, BI) to arrays of points.

        With ``dim = a.shape[0]`` (4) and power vectors
        ``xin = [1, x, x^2, x^3]``, ``yin = [1, y, y^2, y^3]`` per point:

        .. math::

            x_{out} = y_{in}^T A\\, x_{in} = \\sum_{i,j} A_{ij}\\, y^i x^j, \\qquad
            y_{out} = y_{in}^T B\\, x_{in} = \\sum_{i,j} B_{ij}\\, y^i x^j

        so the ROW index is the power of the second input ``y`` and the COLUMN
        index the power of the first input ``x``.  In the forward direction
        ``x = phase`` (:meth:`waves_to_phases`), ``y = alpha = efp_x`` and the
        outputs are ``mfp_x, mfp_y`` (mm); in the inverse direction
        ``x = mfp_x``, ``y = mfp_y`` and the outputs are ``phase, alpha``.
        This is the E-TNT-ATC-MET-1058 transform; the matrices come from
        :meth:`solve_svd_distortion` with the ``i + j > order`` terms zero.

        Parameters
        ----------
        x, y : array_like
            Equal-length input coordinates.
        a, b : numpy.ndarray
            Square ``(dim, dim)`` matrices.

        Returns
        -------
        xout, yout : numpy.ndarray

        Notes
        -----
        - Inconsistency: the previous docstring mentioned an ``affine=True``
          option; no such parameter exists.
        - Clean-up: the per-point Python loop (``for i in range(n_pts)``) could
          be a single ``einsum``; ``RayTrace.create_svd_transforms`` calls this
          for every slice/order.
        - Clean-up: the header comment date 21/2/17 matches the MIRI-derived
          :meth:`solve_svd_distortion` (Temim); the two functions form a pair.
        """
        dim = a.shape[0]
        n_pts = len(x)

        exponent = np.array([range(0, dim), ] * n_pts).transpose()

        xmat = np.array([x, ] * dim)
        xin = np.power(xmat, exponent)
        ymat = np.array([y, ] * dim)
        yin = np.power(ymat, exponent)

        xout = np.zeros(n_pts)
        yout = np.zeros(n_pts)
        for i in range(0, n_pts):
            xout[i] = yin[:, i] @ a @ xin[:, i]
            yout[i] = yin[:, i] @ b @ xin[:, i]
        return xout, yout

    @staticmethod
    def read_raytrace_file(raytrace_file, **kwargs):
        """Read the legacy plain-text ray-trace + transform dump.  Dead code.

        Expected record structure per configuration: a line
        ``<label>,<ea>,<label>,<pa>``; a line ``<label>,<slice_no>,<label>,<spifu_no>``;
        a line ``n_mats=,<n>,n_terms=,<t>,n_layers=,<l>``; then for each
        matrix a title line and ``n_terms`` rows of ``n_terms * n_layers``
        comma-separated values; then ``<label>,<n_pts>``, a column-name line
        and ``n_pts`` data rows.  This is the format written by
        :meth:`get_transform_text` plus ray coordinates, predating the FITS
        products of ``lms_filer.Filer``.

        Parameters
        ----------
        raytrace_file : str
        debug : bool, optional
            Unused.

        Returns
        -------
        configs : numpy.ndarray
            ``(n_cfg, 4)`` rows ``(ea, pa, slice_no, spifu_no)``.
        transforms : numpy.ndarray
            ``(n_cfg, n_mats, n_terms, n_terms, n_layers)``.
        ray_coords : list of dict
            Per-configuration series dicts.

        Notes
        -----
        - Bug: no producer of this format exists in the repository; the
          ``n_layers`` dimension belongs to an earlier per-echelle-angle
          polynomial layout.
        - Clean-up: dead code; not called anywhere.
        """
        debug = kwargs.get('debug', False)

        fmt = "Reading raytrace and transform data from file= {:s}"
        print(fmt.format(raytrace_file))
        lines = open(raytrace_file, 'r').read().splitlines()
        line_iter = iter(lines)
        configs, transforms, ray_coords = [], [], []
        eof = False
        while not eof:
            try:
                line = next(line_iter)
            except StopIteration:
                break
            tokens = line.split(',')
            ea, pa = float(tokens[1]), float(tokens[3])
            line = next(line_iter)
            tokens = line.split(',')
            slice_no, spifu_no = int(tokens[1]), int(tokens[3])
            config = [ea, pa, slice_no, spifu_no]
            configs.append(config)

            line = next(line_iter)
            tokens = line.split(',')
            n_mats, n_terms, n_layers = int(tokens[1]), int(tokens[3]), int(tokens[5])
            matrices = []
            for i in range(0, n_mats):
                next(line_iter)
                matrix = np.zeros((n_terms, n_terms, n_layers))
                for row in range(0, n_terms):
                    line = next(line_iter)
                    tokens = line.split(',')
                    idx = 0
                    for layer in range(0, n_layers):
                        for col in range(0, n_terms):
                            val = float(tokens[idx])
                            idx += 1
                            matrix[row, col, layer] = val
                matrices.append(matrix)
            transforms.append(matrices)
            line = next(line_iter)
            tokens = line.split(',')
            n_pts = int(tokens[1])
            line = next(line_iter)
            tokens = line.split(',')
            series, column = {}, {}

            for j, token in enumerate(tokens[:-1]):
                name = token.strip(' ')
                series[name] = []
                column[j] = name
            for i in range(0, n_pts):
                line = next(line_iter)
                tokens = line.split(',')
                for j, token in enumerate(tokens[:-1]):
                    val = float(tokens[j])
                    col = column[j]
                    series[col].append(val)
            for name in series:
                val_list = series[name]
                series[name] = np.array(val_list)
            ray_coords.append(series)
        return np.array(configs), np.array(transforms), ray_coords

    @staticmethod
    def extract_configs(tf_list, **kwargs):
        """Parse a legacy CSV transform listing into configuration, wavelength-limit and stats arrays.  Dead code.

        Each text row has >= 20 comma-separated tokens: columns 0-5 are
        configuration values (the sixth being a configuration id), 6 a matrix
        id, 8-11 one matrix row, 12-15 four wavelength limits and 16-19 four
        statistics.  Rows are grouped into matrices by column 6 and into
        configurations by column 5.

        Parameters
        ----------
        tf_list : list of str
            Text rows (header already removed).

        Returns
        -------
        configs : list of numpy.ndarray
            Ten arrays: six configuration columns followed by four matrix
            stacks.
        wave_limits : list of numpy.ndarray
            Four arrays (columns 12-15).
        stats : list of numpy.ndarray
            Four arrays (columns 16-19).

        Notes
        -----
        - Bug: no writer for this 20-column format exists in the repository;
          the layout corresponds to the ``configs`` argument of
          ``lms_wcal.Wcal.write_poly``.
        - Clean-up: dead code; not called anywhere.
        """
        config_list = []
        for i in range(0, 10):
            config_list.append([])
        wave_list = []
        for i in range(0, 4):
            wave_list.append([])
        stats_list = []
        for i in range(0, 4):
            stats_list.append([])

        tf_iter = iter(tf_list)
        tf_row = next(tf_iter)
        eof = False
        while not eof:
            tokens = tf_row.split(',')
            for i in range(0, 6):
                val = float(tokens[i])
                config_list[i-0].append(val)
            for i in range(12, 16):
                val = float(tokens[i])
                wave_list[i-12].append(val)
            for i in range(16, 20):
                val = float(tokens[i])
                stats_list[i-16].append(val)

            config_id_old = tokens[5]
            end_of_config = False
            i_mat = 0
            while not end_of_config:
                matrix_id_old = tokens[6]
                matrix = []
                end_of_matrix = False
                while not end_of_matrix:
                    row = []
                    for i in range(8, 12):
                        val = float(tokens[i])
                        row.append(val)
                    matrix.append(row)
                    try:
                        tf_row = next(tf_iter)
                    except StopIteration:
                        eof = True
                        break
                    tokens = tf_row.split(',')
                    matrix_id = tokens[6]
                    end_of_matrix = matrix_id != matrix_id_old
                config_list[6 + i_mat].append(matrix)
                i_mat += 1
                if eof:
                    break
                config_id = tokens[5]
                end_of_config = config_id != config_id_old
        # Convert lists to numpy arrays
        configs = []
        for config in config_list:
            configs.append(np.array(config))
        wave_limits = []
        for waves in wave_list:
            wave_limits.append(np.array(waves))
        stats = []
        for stat in stats_list:
            stats.append(np.array(stat))
        return configs, wave_limits, stats

    @staticmethod
    def find_polyfit(x, coeffs, **kwargs):
        """Fit every element of a stack of square matrices as a 1-D polynomial in ``x``.

        Parameters
        ----------
        x : array_like
            Independent variable (e.g. echelle angle), length ``n``.
        coeffs : numpy.ndarray
            ``(n, ncr, ncr)`` stack of matrices.
        order : int, optional
            Polynomial degree for ``numpy.polyfit``; default 1.
        debug : bool, optional
            Print each ``(x, y)`` pair.

        Returns
        -------
        numpy.ndarray
            ``(ncr, ncr, order + 1)`` coefficients, highest power first
            (``numpy.polyfit`` convention).

        Notes
        -----
        - Clean-up: legacy 1-D precursor of
          ``lmsdist_polyfit.PolyFit.find_slice_fit`` (bivariate); not called
          anywhere.
        """
        debug = kwargs.get('debug', False)
        order = kwargs.get('order', 1)
        _, ncr, _ = coeffs.shape
        poly = np.zeros((ncr, ncr, order+1))
        for i in range(0, ncr):
            for j in range(0, ncr):
                y = coeffs[:, i, j]
                if debug:
                    print(x, y)
                poly[i, j] = np.polyfit(x, y, order)
        return poly

    @staticmethod
    def find_dispersion(traces, configuration):
        """Estimate the dispersion (micron per pixel) at a wavelength from a slice transform.  Dead code.

        Intended algorithm: locate the trace and slice transform for the
        requested order and slice, project ``wave`` and ``wave + 1 nm`` at
        ``alpha = 0`` through matrix A, and convert the resulting ``dx`` (mm)
        to microns per ``Globals.nom_pix_pitch`` pixel.

        Parameters
        ----------
        traces : list of RayTrace
        configuration : tuple
            ``(slice_no, wave, prism, ech_angle, order, im_pix_size)``.

        Returns
        -------
        float
            Dispersion in microns per pixel (would be).

        Raises
        ------
        AttributeError
            ``trace.unique_slices``, ``trace.slice``, ``trace.waves_to_phase``
            and ``trace.apply_svd_distortion`` do not exist on ``RayTrace``.
        UnboundLocalError
            If no trace matches, ``dw_lmspix`` is returned before assignment.

        Notes
        -----
        - Bug: references a pre-2023 ``RayTrace`` API; cannot run.
        - Clean-up: the working equivalent is the finite-difference
          ``'dispersion'`` branch of ``lmsdist_plot.Plot.series``.
        """
        dw = 0.001  # Choose a wavelength interval of 1 nanometre (approx. 4000 / 50 = 80 pixels)

        slice_no, wave, prism, ech_angle, order, im_pix_size = configuration
        for trace in traces:
            is_order = order in trace.unique_ech_ords
            if not is_order:
                continue
            is_slice = slice_no in trace.unique_slices
            if not is_slice:
                continue
            for tf in trace.slice:
                tf_config, tf_matrices, _, _, _ = tf
                tf_ech_ord, tf_slice_no, tf_spifu_no = tf_config
                if order != tf_ech_ord and slice_no != tf_slice_no:
                    continue
                a, b, ai, bi = tf_matrices
                w1 = wave
                w2 = wave + dw
                phases = trace.waves_to_phase(np.array([w1, w2]), np.array([order, order]))
                alpha_middle = 0.0  # FP2 coordinate
                alphas = [alpha_middle, alpha_middle]
                det_x, _ = trace.apply_svd_distortion(np.array(phases), np.array(alphas), a, b)
                dx = det_x[1] - det_x[0]
                dw_dx = dw / dx  # micron per mm
                dw_lmspix = dw_dx * Globals.nom_pix_pitch / 1000.0  # micron per (nominal) pixel
        return dw_lmspix

    @staticmethod
    def print_stats(st_file, data):
        """Print and write a table of transform residual statistics (microns), then close the file.

        Parameters
        ----------
        st_file : file object
            Open for writing (``Filer.stats_file``); closed on return.
        data : tuple
            ``(n_terms, poly_order, offset_xy)`` where ``offset_xy`` has shape
            ``(n, m, 2, k)`` in mm; index 2 selects x / y.

        Returns
        -------
        None

        Notes
        -----
        - Clean-up: legacy; not called anywhere.  Closing a file the caller
          opened is a surprising side effect.
        """
        fmt1 = "{:15s}{:15s}"
        fmt2 = "{:20s}{:20s}{:20s}"
        fmt3 = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
        # Write header
        hdr = fmt1.format("Transform", "Polynomial")
        hdr = hdr + fmt2.format("Mean offset", "Std deviation", "Maximum")
        print(hdr)
        st_file.write(hdr + '\n')
        hdr = fmt1.format("matrix terms", "fit terms")
        hdr = hdr + fmt3.format("X", "Y", "X", "Y", "X", "Y")
        print(hdr)
        st_file.write(hdr + '\n')
        n_terms, poly_order, offset_xy = data
        micron_mm = 1000.0
        for i in range(0, len(offset_xy)):
            x = micron_mm * offset_xy[i, :, 0, :]
            y = micron_mm * offset_xy[i, :, 1, :]
            mx, my = np.mean(x), np.mean(y)
            vx, vy = np.std(x), np.std(y)
            zx, zy = np.max(x), np.max(y)

            fmt4 = "{:10d}{:15d}"
            text = fmt4.format(n_terms, poly_order+1)
            fmt5 = "{:15.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}"
            text += fmt5.format(mx, my, vx, vy, zx, zy)
            print(text)
            st_file.write(text + '\n')
        st_file.close()
        return

    @staticmethod
    def filter_transform_list(transform_list, **kwargs):
        """Return the transforms whose merged configuration matches all keyword filters.

        For each transform the dict
        ``transform.lms_configuration | transform.slice_configuration`` is
        formed and every ``key=value`` in ``kwargs`` must match exactly.

        Parameters
        ----------
        transform_list : list of Transform
        **kwargs
            E.g. ``slice_no=13``, ``ech_ord=27``, ``pri_ang=6.9``.

        Returns
        -------
        list of Transform

        Raises
        ------
        KeyError
            If a filter key is in neither configuration dict.

        Notes
        -----
        - Clean-up: dict union ``|`` requires Python >= 3.9.
        - Clean-up: ``Transform.is_match`` is a two-key special case of this.
        """
        filtered_transform_list = []
        for transform in transform_list:
            cfg = transform.lms_configuration | transform.slice_configuration
            is_match = True
            for key in kwargs:
                is_match = False if cfg[key] != kwargs[key] else is_match
            if is_match:
                filtered_transform_list.append(transform)
        return filtered_transform_list

    @staticmethod
    def parse_slice_locations(slice_locs, **kwargs):
        """Expand an ``lmsiq`` slice-location spec into ``(slice_no, folder, label)`` tuples.

        Parameters
        ----------
        slice_locs : sequence
            ``(target_slice, n_slices, ifu_flag)``: centre slice, number of
            slices centred on it and ``'ifu'`` to append an IFU-image entry.
        boresight : bool, optional
            If True return only the target slice.

        Returns
        -------
        tgt_slice : int
        n_slices : int
        slice_idents : list of tuple
            ``(slice_no, 'slice_<n>/', ', slice <n>')`` entries, plus
            ``(-1, 'ifu/', ', IFU image')`` when requested.

        Notes
        -----
        - Clean-up: belongs to the ``lmsiq`` image-quality folder
          conventions, not to ``lmsdist``; not called from this program.
        """
        boresight = kwargs.get('boresight', False)      # True = Extract boresight slice only
        tgt_slice = int(slice_locs[0])
        slice_idents = []
        if boresight:
            slice_folder = "slice_{:d}/".format(tgt_slice)
            slice_label = ", slice {:d}".format(tgt_slice)
            slice_idents.append((tgt_slice, slice_folder, slice_label))
            return tgt_slice, 1, slice_idents

        n_slices = int(slice_locs[1])
        slice_lo = tgt_slice - int(n_slices / 2)
        for slice_idx in range(n_slices):
            slice_no = slice_lo + slice_idx
            slice_folder = "slice_{:d}/".format(slice_no)
            slice_label = ", slice {:d}".format(slice_no)
            slice_idents.append((slice_no, slice_folder, slice_label))
        if slice_locs[2] == 'ifu':
            slice_folder, slice_label = "ifu/", ", IFU image"
            slice_idents.append((-1, slice_folder, slice_label))
        return tgt_slice, n_slices, slice_idents

    @staticmethod
    def read_polyfits_file(poly_file):
        """Read a legacy CSV of polynomial coefficients into a 2-D float array.

        Skips the first line, then reads comma-separated rows (dropping the
        last, empty, token of each) until a row with fewer than two tokens.

        Parameters
        ----------
        poly_file : str
            E.g. ``Filer.poly_file`` (``*_dist_poly.txt``).

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)``.

        Notes
        -----
        - Inconsistency: ``lms_test.py`` unpacks the result as
          ``poly, ech_bounds = util.read_polyfits_file(poly_file)``, but a
          single array is returned.
        - Clean-up: no current writer of ``*_dist_poly.txt`` exists in the
          repository.
        """
        pf = open(poly_file, 'r')
        lines = pf.read().splitlines()
        pf.close()
        poly = []
        for line in lines[1:]:
            tokens = line.split(',')
            if len(tokens) < 2:
                break
            poly_row = []
            for token in tokens[:-1]:
                val = float(token.strip(' '))
                poly_row.append(val)
            poly.append(poly_row)
        return np.array(poly)

    @staticmethod
    def _decode_tf_list_orders(tf_list):
        """Return the distinct echelle orders found in column 1 of legacy transform text rows.

        Parameters
        ----------
        tf_list : list of str
            Comma-separated rows with the order in token 1.

        Returns
        -------
        list of int
            In order of first appearance.

        Notes
        -----
        - Clean-up: legacy helper for :meth:`extract_configs`; not called.
        """
        ech_ords = []
        for tf in tf_list:
            tokens = tf.split(',')
            eo = int(tokens[1])
            if eo not in ech_ords:
                ech_ords.append(eo)
        return ech_ords

    @staticmethod
    def get_transform_text(transform):
        """Serialise a dict of matrices to the legacy plain-text block read by :meth:`read_raytrace_file`.

        Writes ``n_mats=,<n>,n_terms=,<t>,n_layers=,<l>,`` then, per matrix,
        ``Matrix <name>,`` and ``n_rows`` lines of ``'{:15.6e},'`` values
        (all layers of a row concatenated, separated by three spaces).

        Parameters
        ----------
        transform : dict
            ``{name: ndarray}``; arrays are promoted with ``numpy.atleast_3d``
            so 2-D matrices get ``n_layers = 1``.

        Returns
        -------
        str

        Notes
        -----
        - Clean-up: legacy text format superseded by the FITS product of
          ``Transform.make_hdu_ext``; not called anywhere.
        """
        text = ''
        n_mat = len(transform)
        is_first_matrix = True
        for mat_name in transform:
            matrix = np.atleast_3d(transform[mat_name])
            if is_first_matrix:
                n_rows, n_cols, n_layers = matrix.shape
                fmt2b = "n_mats=,{:4d},n_terms=,{:4d},n_layers=,{:4d},\n"
                text += fmt2b.format(n_mat, n_rows, n_layers)
                is_first_matrix = False
            fmt = "Matrix {:s},\n"
            text += fmt.format(mat_name)
            for row in range(0, n_rows):
                for layer in range(0, n_layers):
                    for col in range(0, n_cols):
                        val = matrix[row, col, layer]
                        text += "{:15.6e},".format(val)
                    text += "   "
                text += '\n'
        return text

    @staticmethod
    def get_term_values(transforms, fslice_no, pslice_no, debug=False):
        """Collect per-configuration angles, boresight wavelength and matrices for one slice.

        For every transform whose ``slice_configuration['slice_no']`` equals
        ``encode_slice_no(fslice_no, pslice_no)``:

        1. the slice-centre EFP point ``(efp_x = 0, efp_y = slice_to_efp_y(
           fslice_no, 0))`` is projected with :meth:`efp_to_mfp` at 100
           wavelengths between the transform's ``w_min`` and ``w_max``;
        2. the boresight wavelength ``w_bs`` at which ``mfp_x == 0`` is
           interpolated with ``numpy.interp(0, mfp_x_list, w_list)``;
        3. ``pri_ang, ech_ang, ech_ord, mfp_y (at the last sample), w_bs`` and
           the matrices dict are appended to the result lists.

        Parameters
        ----------
        transforms : list of Transform
            Must have ``w_min`` / ``w_max`` set (``RayTrace.find_wavelength_bounds``).
        fslice_no : int
            Field slice number.
        pslice_no : int
            Spectral (pupil) slice number; 0 in nominal mode.
        debug : bool, optional
            Print a table of the sampled values.

        Returns
        -------
        dict
            ``{'slice_no': fslice_no, 'spifu_no': pslice_no, 'pri_ang': [...],
            'ech_ang': [...], 'mfp_y': [...], 'w_bs': [...], 'ech_ords': [...],
            'matrices': [...]}`` -- the ``term_values`` consumed by
            ``PolyFit.find_slice_fit`` / ``find_wxo_fit``.

        Notes
        -----
        - Open question: ``numpy.interp`` needs increasing ``xp``.  The
          ``mfp_x_list`` is built in ascending wavelength order and, per the
          note in :meth:`mfp_to_dfp`, ``mfp_x`` DEcreases with wavelength, so
          the abscissa is descending and the interpolated ``w_bs`` is not
          guaranteed to be correct.  Verify against the Zemax boresight.
        - Inconsistency: the returned ``'slice_no'`` is the FIELD slice
          number, not the encoded universal slice number used everywhere else.
        - Inconsistency: the previous docstring referred to "slice 12"; the
          function is generic in ``fslice_no``.
        - Clean-up: ``mfp_y`` is taken from the last of the 100 samples
          (``w_max``), not at the boresight.
        """
        slice_no = Util.encode_slice_no(fslice_no, pslice_no)
        phase = 0.                      # Across slice fractional displacement
        y = Util.slice_to_efp_y(fslice_no, phase)
        efp_y = np.array([y.value])     # Location in EFP.
        efp_x = np.array([0.])
        efp_bs = {'efp_y': efp_y, 'efp_x': efp_x}
        if debug:
            print("Field slice no= {:d}, Pupil slice no= {:d}".format(fslice_no, pslice_no))
            fmt = "{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('pri_ang', 'ech_ang', 'efp_x', 'efp_y', 'mfp_x', 'mfp_y', 'w'))
        term_values = {'slice_no': fslice_no, 'spifu_no': pslice_no,
                       'pri_ang': [], 'ech_ang': [],
                       'mfp_y': [], 'w_bs': [], 'ech_ords': [], 'matrices': []}
        for transform in transforms:
            lms_config = transform.lms_configuration
            slice_cfg = transform.slice_configuration
            if (slice_cfg['slice_no'] != slice_no):
                continue
            # Calculate the wavelength at fps_x = 0.0 (the mosaic column direction mid-line)
            w_min, w_max = slice_cfg['w_min'], slice_cfg['w_max']
            mfp_w_list, mfp_x_list = [], []
            for efp_w in np.linspace(w_min, w_max, 100):
                efp_bs['efp_w'] = np.array([efp_w])
                mfp_bs, _ = Util.efp_to_mfp(transform, efp_bs)
                mfp_x = mfp_bs['mfp_x'][0]
                mfp_x_list.append(mfp_x)
                mfp_w_list.append(efp_w)

            mfp_w = np.interp(0., np.array(mfp_x_list), np.array(mfp_w_list))
            mfp_y = mfp_bs['mfp_y'][0]
            if debug:
                fmt = "{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"
                print(fmt.format(lms_config['pri_ang'], lms_config['ech_ang'],
                                 efp_bs['efp_x'][0], efp_bs['efp_y'][0],
                                 mfp_x, mfp_y, mfp_w))
            term_values['pri_ang'].append(lms_config['pri_ang'])
            term_values['ech_ang'].append(lms_config['ech_ang'])
            term_values['ech_ords'].append(slice_cfg['ech_ord'])
            term_values['mfp_y'].append(mfp_y)
            term_values['w_bs'].append(mfp_w)
            term_values['matrices'].append(transform.matrices)
        return term_values

    @staticmethod
    def find_matching_transforms(a_transforms, b_transforms):
        """Pair transforms from two lists that share the same encoded ``slice_no``.

        Used by ``lmsaiv_opt02`` to pair AIT-measured with Zemax transforms of
        the same slice before comparing matrix terms.

        Parameters
        ----------
        a_transforms, b_transforms : list of Transform

        Returns
        -------
        list of tuple
            ``(a_transform, b_transform)`` for every pair with equal
            ``slice_configuration['slice_no']`` (all cross pairs; the echelle
            order is NOT compared).

        Notes
        -----
        - Open question: if either list holds several orders for one slice
          (extended mode) every order combination is paired, including
          mismatched orders.
        """
        matching_transforms = []
        for a_transform in a_transforms:
            a_slice_no = a_transform.slice_configuration['slice_no']
            for b_transform in b_transforms:
                b_slice_no = b_transform.slice_configuration['slice_no']
                if a_slice_no == b_slice_no:
                    matching_transforms.append((a_transform, b_transform))
        return matching_transforms

    @staticmethod
    def find_closest_transforms(wave, opticon, svd_transforms):
        """Select the configuration (all its slice transforms) that centres a wavelength best.

        Step 1: among transforms with ``slice_no == 13`` and ``spifu_no <= 1``
        pick the one whose band centre ``(w_min + w_max) / 2`` is closest to
        ``wave`` (astropy ``Quantity`` arithmetic in microns / nm).
        Step 2: return every transform whose ``lms_configuration`` equals that
        boresight transform's, plus the list of echelle orders they cover.

        Parameters
        ----------
        wave : astropy.units.Quantity
            Target wavelength (must carry a length unit; a plain float fails
            in the ``Quantity`` subtraction).
        opticon : str
            Only consulted inside a disabled ``spec_order = False`` branch.
        svd_transforms : list of Transform

        Returns
        -------
        opt_transforms : list of Transform
        ech_ords : list of int

        Raises
        ------
        KeyError
            ``slice_config['spifu_no']`` is not a key of
            ``Globals.slice_config_template``.
        AttributeError
            If no slice-13 transform is found (``bs_transform`` stays ``None``).

        Notes
        -----
        - Bug: ``'spifu_no'`` KeyError makes the function unusable with
          transforms produced by the current ``RayTrace`` / ``Transform``;
          ``lmssim_toy`` still calls it.
        - Bug: ``slice_no != 13`` compares the ENCODED slice number, so in
          extended mode (``slice_no = 100 * pslice + 13``) no transform ever
          matches.
        - Inconsistency: the previous docstring mentions "order 24" and
          "spifu = 1" for extended mode; the code contains that logic only in
          the disabled ``spec_order`` branch (``{23: (1, 4), 24: (2, 5), 25:
          (3, 6)}``).
        - Clean-up: ``slice_id`` is computed from ``Globals.slice_id_fmt`` and
          discarded.
        """
        bs_transform = None
        w_off_min = 1000.*u.nm
        for svd_transform in svd_transforms:
            slice_config = svd_transform.slice_configuration
            slice_no = slice_config['slice_no']
            if slice_no != 13:
                continue
            spifu_no = slice_config['spifu_no']
            if spifu_no > 1:        # Catches both nominal and spifu configurations
                continue
            w_min, w_max = slice_config['w_min']*u.micron, slice_config['w_max']*u.micron
            w_off = abs(0.5 * (w_max + w_min) - wave)
            if w_off <= w_off_min:
                bs_transform = svd_transform
                w_off_min = w_off

        # Step 2, select all transforms with same configuration as the boresight. For spifu, use optimum order
        bs_lms_cfg = bs_transform.lms_configuration
        opt_transforms = []
        ech_ords = []             # List of unique echelle orders in optimum transforms
        for svd_transform in svd_transforms:
            lms_cfg = svd_transform.lms_configuration
            is_match = True
            for key in bs_lms_cfg:
                is_match = is_match and bs_lms_cfg[key] == lms_cfg[key]
            if not is_match:
                continue
            slice_cfg = svd_transform.slice_configuration
            ech_ord = slice_cfg['ech_ord']
            # Option to only use specific orders for specific spifu_slices
            spec_order = False
            if spec_order:
                if opticon == Globals.extended:
                    spifu_no = slice_cfg['spifu_no']
                    spifu_slice = {23: (1, 4), 24: (2, 5), 25: (3, 6)}
                    valid_spifu_nos = spifu_slice[ech_ord]
                    if spifu_no not in valid_spifu_nos:
                        continue
            if ech_ord not in ech_ords:
                ech_ords.append(ech_ord)
            slice_no = slice_cfg['slice_no']
            spifu_no = slice_cfg['spifu_no']
            slice_id = Globals.slice_id_fmt.format(opticon, ech_ord, slice_no, spifu_no)
            opt_transforms.append(svd_transform)
        return opt_transforms, ech_ords

    @staticmethod
    def out_of_bounds(fp_id, x, y):             # Check if out of bounds
        """Return True if NO point lies inside the focal-plane bounds (aggregate test).

        Uses ``Globals.fp_bounds[fp_id] = (x1, x2, y1, y2)`` (``'efp'``: the
        EFP field of view; ``'mfp'``: +-40 mm).  ``in_x`` is
        ``numpy.any(x1 < x < x2)`` and ``in_y`` likewise; the result is
        ``not (in_x and in_y)``.

        Parameters
        ----------
        fp_id : {'efp', 'mfp'}
        x, y : numpy.ndarray

        Returns
        -------
        bool
            True only when every point is outside in x, or every point is
            outside in y.  A single in-bounds point makes the whole set
            "in bounds".

        Notes
        -----
        - Inconsistency: the name suggests a per-point or all-points test;
          the implementation is an "any point inside" test and a point set
          with one inlier and many outliers is reported as in bounds.
          Additionally ``in_x`` and ``in_y`` may be satisfied by different
          points.
        """
        bounds = Globals.fp_bounds[fp_id]
        x1, x2, y1, y2 = bounds
        in_x = np.any(np.logical_and(np.less(x1, x), np.greater(x2, x)))
        in_y = np.any(np.logical_and(np.less(y1, y), np.greater(y2, y)))
        in_bounds = in_x and in_y
        return not in_bounds

    @staticmethod
    def efp_y_to_slice_no(efp_y):
        """Convert an EFP across-slice coordinate (mm) to (field slice number, intra-slice phase).

        Inverse of :meth:`slice_to_efp_y`.  With the slice width in the EFP
        ``w = Globals.beta_slice (20.7 mas -> arcsec) / Globals.efp_arcsec_mm
        (0.180723 arcsec/mm) ~= 0.1145 mm``::

            s = 15 - efp_y / w
            slice_no = int(s)          # truncation towards zero
            phase    = s - slice_no    # in [0, 1)

        so ``efp_y = 0`` is the boresight at ``(slice 15, phase 0.0)``
        (equivalently the upper edge of slice 14), positive ``efp_y`` gives
        LOWER slice numbers and slice 28 lies at ``efp_y ~= -1.5`` mm.  The
        centre of the extended-mode field is at ``(12, 0.5)`` per the original
        comment.

        Parameters
        ----------
        efp_y : float
            Across-slice EFP coordinate in mm as a plain float.  (If an
            astropy ``Quantity`` in mm is passed the division is dimensionless
            and ``.value`` still works; ``lmssim_toy`` passes ``* u.mm``.)

        Returns
        -------
        slice_no : int
        phase : float

        Notes
        -----
        - Inconsistency: ``phase`` here is the ACROSS-slice fractional
          position, unrelated to the grating phase of :meth:`waves_to_phases`.
        - Inconsistency: ``test_out_and_back`` calls this with a list
          argument and unpacks two lists; the function is scalar.
        - Open question: negative ``s`` (far above slice 1) truncates towards
          zero rather than flooring, giving a phase in (-1, 0].
        - Clean-up: the local ``phase = 0.`` initialisation is overwritten.
        """
        bs_slice_no, phase = 15, 0.
        efp_slice_width = Globals.beta_slice.to(u.arcsec) / Globals.efp_arcsec_mm
        s_offset = -efp_y / efp_slice_width
        slice_no_float = bs_slice_no + s_offset.value
        slice_no = int(slice_no_float)
        phase = slice_no_float - slice_no
        return slice_no, phase

    @staticmethod
    def decode_slice_no(slice_no):
        """Split an encoded universal slice number into ``(fslice_no, pslice_no)``.

        ``pslice_no = slice_no // 100``, ``fslice_no = slice_no % 100``
        (inverse of :meth:`encode_slice_no`).

        Parameters
        ----------
        slice_no : int
            E.g. 13 (nominal) or 312 (extended: pupil slice 3, field slice 12).

        Returns
        -------
        fslice_no : int
            Field (spatial) slice, 1..28.
        pslice_no : int
            Spectral (pupil / SPIFU) slice, 0 nominal, 1..6 extended.
        """
        pslice_no = slice_no // 100
        fslice_no = slice_no % 100
        return fslice_no, pslice_no

    @staticmethod
    def encode_slice_no(fslice_no, pslice_no):
        """Return the universal slice number ``100 * pslice_no + fslice_no``.

        This encoded value is what ``Transform.slice_configuration['slice_no']``,
        ``RayTrace.series['slice_no']`` and the FITS ``SLICE_NO`` / ``EXTNAME``
        carry.

        Parameters
        ----------
        fslice_no : int
            Field slice 1..28.
        pslice_no : int
            Spectral slice, 0 in nominal mode.

        Returns
        -------
        int
        """
        slice_no = 100 * pslice_no + fslice_no
        return slice_no

    @staticmethod
    def slice_to_efp_y(fslice_no, phase):
        """Return the EFP across-slice coordinate (mm, astropy Quantity) of a slice position.

        ::

            w      = beta_slice (20.7 mas) / efp_arcsec_mm (0.180723 "/mm)   ~ 0.1145 mm
            efp_y  = -(fslice_no + phase - 15) * w

        ``phase = 0`` is the slice edge that coincides with the boresight for
        slice 15 (``efp_y = 0``); ``phase = 0.5`` is the slice centre.  Higher
        slice numbers lie at more negative ``efp_y``.  Inverse:
        :meth:`efp_y_to_slice_no`.

        Parameters
        ----------
        fslice_no : int
            FIELD slice number 1..28 (not the encoded universal number).
        phase : float
            Across-slice fractional position in [0, 1).

        Returns
        -------
        astropy.units.Quantity
            In mm; callers take ``.value`` (``RayTrace.find_wavelength_bounds``,
            ``lmsaiv_opt02``) or use it directly (:meth:`mfp_to_efp`).

        Notes
        -----
        - Inconsistency: the previous docstring said "slice centre" but
          ``phase = 0`` is a slice EDGE; the centre requires ``phase = 0.5``
          (``lmssim_toy`` passes 0.5, ``RayTrace.find_wavelength_bounds`` and
          :meth:`get_term_values` pass 0).
        - Inconsistency: ``lmsaiv_opt02`` calls
          ``slice_to_efp_y(slice_no, beta_phase, opticon)`` with a third
          argument -> ``TypeError``.
        - Bug: (caller) ``RayTrace.find_wavelength_bounds`` passes the ENCODED
          slice number, which is wrong in extended mode.
        - Open question: :meth:`mfp_to_efp` fills an array with the returned
          ``Quantity`` via ``numpy.full``; whether this yields floats or raises
          depends on the astropy/numpy versions and has not been verified.
        """
        bs_slice_no, bs_phase = 15, 0.          # Boresight definition
        efp_slice_width = Globals.beta_slice.to(u.arcsec) / Globals.efp_arcsec_mm
        slice_no_float = fslice_no + phase
        s_offset = slice_no_float - (bs_slice_no + bs_phase)
        efp_y = -s_offset * efp_slice_width
        return efp_y

    @staticmethod
    def efp_to_dfp(transform, affines, efp_points):
        """Map EFP points to detector pixels: :meth:`efp_to_mfp` then :meth:`mfp_to_dfp`.

        Parameters
        ----------
        transform : Transform
            Slice/order transform supplying ``a, b`` and the echelle order.
        affines : dict
            ``Transform.affines`` (``{'m': ..., 'mi': ...}``).
        efp_points : dict
            ``{'efp_x': array, 'efp_y': array, 'efp_w': array}`` (mm, mm, um).

        Returns
        -------
        dict
            ``{'det_nos', 'dfp_x', 'dfp_y'}`` arrays -- GROUPED BY DETECTOR,
            not in input order (see :meth:`mfp_to_dfp`).

        Notes
        -----
        - Clean-up: the out-of-bounds flag from :meth:`efp_to_mfp` is
          discarded.
        """
        mfp_points, _ = Util.efp_to_mfp(transform, efp_points)
        dfp_points = Util.mfp_to_dfp(affines, mfp_points)
        return dfp_points

    @staticmethod
    def mfp_to_dfp(affines, mfp_points):
        """Assign mosaic-plane points to detectors by quadrant and convert to pixel coordinates.

        Detector assignment::

            det 1: mfp_x > 0 and mfp_y > 0      det 2: mfp_x < 0 and mfp_y > 0
            det 3: mfp_x > 0 and mfp_y < 0      det 4: mfp_x < 0 and mfp_y < 0

        then ``(dfp_x, dfp_y) = affines['m'][det - 1] @ (mfp_x, mfp_y, 1)``
        (:meth:`apply_affine_transform`).  Because ``mfp_x`` increases towards
        SHORTER wavelength, detectors 1/3 see shorter wavelengths than 2/4;
        slice 1 is at minimum ``mfp_y`` so detectors 3/4 carry slices 1-14 and
        1/2 carry 15-28 (module docstring of ``lmsdist``).

        Parameters
        ----------
        affines : dict
            ``Transform.affines``.
        mfp_points : dict
            ``{'mfp_x': array, 'mfp_y': array}`` in mm.

        Returns
        -------
        dict
            ``{'det_nos': int array, 'dfp_x': array, 'dfp_y': array}`` in
            pixels.  Points are concatenated detector by detector (1, 2, 3, 4),
            NOT in input order.

        Notes
        -----
        - Inconsistency: output order differs from input order whenever the
          points span more than one detector; callers that index the result
          with the input index (``RayTrace.find_wavelength_bounds``,
          :meth:`test_out_and_back`) are only correct by luck of ordering.
        - Bug: points with ``mfp_x == 0`` or ``mfp_y == 0`` exactly belong to
          no quadrant and are silently dropped, shortening the output.
        - Inconsistency: no check that the pixel coordinates fall inside
          0..2048; points in the inter-detector gap get negative or > 2048
          pixel values on the assigned detector.
        """
        x = mfp_points['mfp_x']
        y = mfp_points['mfp_y']
        idx24 = np.argwhere(x < 0.)
        idx13 = np.argwhere(x > 0.)
        idx34 = np.argwhere(y < 0.)
        idx12 = np.argwhere(y > 0.)
        idx1 = np.intersect1d(idx12, idx13)
        idx2 = np.intersect1d(idx12, idx24)
        idx3 = np.intersect1d(idx13, idx34)
        idx4 = np.intersect1d(idx24, idx34)

        idx_list = [idx1, idx2, idx3, idx4]
        det_nos, dfp_x, dfp_y = [], [], []
        for i, idx in enumerate(idx_list):
            n_idx = len(idx)
            if n_idx == 0:
                continue
            d_nos = [i+1]*n_idx
            affine_fwd = affines['m'][i]
            u, v = Util.apply_affine_transform(x[idx], y[idx], affine_fwd)
            det_nos += d_nos
            dfp_x += list(u)
            dfp_y += list(v)
        dfp_points = {'det_nos': np.array(det_nos), 'dfp_x': np.array(dfp_x), 'dfp_y': np.array(dfp_y)}
        return dfp_points

    @staticmethod
    def dfp_to_mfp(affines, dfp_points):
        """Convert detector pixel coordinates back to mosaic-plane mm using the inverse affines.

        For each detector 1..4 in turn the points with that ``det_nos`` value
        are transformed with ``affines['mi'][det - 1]`` and appended.

        Parameters
        ----------
        affines : dict
            ``Transform.affines``.
        dfp_points : dict
            ``{'det_nos': int array, 'dfp_x': array, 'dfp_y': array}``.

        Returns
        -------
        dict
            ``{'mfp_x': array, 'mfp_y': array}`` in mm, GROUPED BY DETECTOR
            (input order is preserved only if ``det_nos`` is already sorted).

        Notes
        -----
        - Inconsistency: the previous docstring claimed all points are assumed
          to come from the first point's detector; the implementation handles
          mixed detectors but re-orders the output.
        - Clean-up: the ``det_nos`` array is compared against
          ``numpy.full`` arrays instead of a scalar.
        """
        det_nos = dfp_points['det_nos']
        dfp_x = dfp_points['dfp_x']
        dfp_y = dfp_points['dfp_y']
        mfp_x, mfp_y = [], []
        n_pts, = det_nos.shape
        for det_idx in range(4):
            det_no = det_idx + 1
            det_no_array = np.full(n_pts, det_no)
            idx, = np.nonzero(np.equal(det_nos, det_no_array))
            n_idx, = idx.shape
            if n_idx == 0:
                continue
            affine_rev = affines['mi'][det_idx]
            u, v = Util.apply_affine_transform(dfp_x[idx], dfp_y[idx], affine_rev)
            mfp_x += list(u)
            mfp_y += list(v)

        mfp_points = {'mfp_x': np.array(mfp_x), 'mfp_y': np.array(mfp_y)}
        return mfp_points

    @staticmethod
    def efp_to_mfp(transform, efp_points):
        """Forward-project EFP points to the detector mosaic plane with one slice transform.

        ``phase = waves_to_phases(efp_w, ech_ord)`` with the transform's own
        ``slice_configuration['ech_ord']``, ``alpha = efp_x``, then
        ``(mfp_x, mfp_y) = apply_svd_distortion(phase, alpha, a, b)``.  The
        across-slice coordinate ``efp_y`` is NOT used: the transform is
        specific to one slice, so the caller must have chosen the transform
        whose slice contains ``efp_y`` (:meth:`efp_y_to_slice_no`).

        Parameters
        ----------
        transform : Transform
            With ``matrices['a']`` / ``['b']`` populated.
        efp_points : dict
            ``{'efp_x': array (mm), 'efp_y': array (mm, ignored), 'efp_w':
            array (um)}``.

        Returns
        -------
        mfp_points : dict
            ``{'mfp_x': array, 'mfp_y': array}`` in mm.
        oob : bool
            :meth:`out_of_bounds` (``'mfp'``, +-40 mm) aggregate flag.

        Notes
        -----
        - Inconsistency: the previous docstring said the transform "will
          normally have been verified as non-vignetting"; no vignetting check
          exists here (``cf_vig`` is only used for plotting in ``RayTrace``).
        - Clean-up: ``efp_y`` is accepted and ignored; passing it suggests a
          generality the function does not have.
        """
        config, matrices = transform.slice_configuration, transform.matrices
        ech_ord = config['ech_ord']

        alphas, efp_y, efp_w = efp_points['efp_x'], efp_points['efp_y'], efp_points['efp_w']
        ech_ords = np.full(len(efp_w), ech_ord)
        phases = Util.waves_to_phases(efp_w, ech_ords)

        a, b = matrices['a'], matrices['b']
        mfp_x, mfp_y = Util.apply_svd_distortion(phases, alphas, a, b)
        oob = Util.out_of_bounds('mfp', mfp_x, mfp_y)       # Check if out of bounds
        mfp_points = {'mfp_x': mfp_x, 'mfp_y': mfp_y}
        return mfp_points, oob

    @staticmethod
    def dfp_to_efp(transform, affines, dfp_points):
        """Map detector pixels to the EFP: :meth:`dfp_to_mfp` then :meth:`mfp_to_efp`.

        Parameters
        ----------
        transform : Transform
            Slice/order transform (supplies ``ai, bi``, ``ech_ord`` and the
            slice number used to set ``efp_y``).
        affines : dict
            ``Transform.affines``.
        dfp_points : dict
            ``{'det_nos', 'dfp_x', 'dfp_y'}``.

        Returns
        -------
        dict
            ``{'efp_x', 'efp_y', 'efp_w'}``; ``efp_y`` is the slice edge
            (``slice_phase = 0``) for every point, see :meth:`mfp_to_efp`.
        """
        mfp_points = Util.dfp_to_mfp(affines, dfp_points)
        efp_points = Util.mfp_to_efp(transform, mfp_points)
        return efp_points

    @staticmethod
    def mfp_to_efp(transform, mfp_points, slice_phase=0.):
        """Inverse-project mosaic-plane points to the EFP with one slice transform.

        ``(phase, alpha) = apply_svd_distortion(mfp_x, mfp_y, ai, bi)``;
        ``efp_w = phases_to_waves(phase, ech_ord)`` with the transform's
        order; ``efp_x = alpha``.  The across-slice coordinate cannot be
        recovered from the detector position (it is degenerate with
        wavelength), so ``efp_y`` is set for ALL points to
        ``slice_to_efp_y(fslice_no, slice_phase)`` using the transform's
        decoded field slice number.

        Parameters
        ----------
        transform : Transform
            With ``matrices['ai']`` / ``['bi']`` populated.
        mfp_points : dict
            ``{'mfp_x': array, 'mfp_y': array}`` in mm.
        slice_phase : float, optional
            Across-slice fractional position assigned to every point.  Default
            0.0, which is the slice EDGE at the boresight side (the previous
            docstring called this the slice centre; the centre is 0.5).

        Returns
        -------
        dict
            ``{'efp_x': array (mm), 'efp_y': array (mm), 'efp_w': array (um)}``.

        Notes
        -----
        - Inconsistency: default ``slice_phase = 0.`` documented as "centre of
          the slice" but :meth:`slice_to_efp_y` places phase 0 at a slice
          edge.
        - Open question: ``numpy.full(alphas.shape, efp_y_val)`` is filled
          with an astropy ``Quantity`` (mm); whether the result is a float
          array or raises depends on the numpy/astropy versions in use.  Other
          callers of ``slice_to_efp_y`` take ``.value`` first.
        - Clean-up: ``lms_config`` is fetched and unused.
        """
        lms_config = transform.lms_configuration
        ech_ord = transform.slice_configuration['ech_ord']
        slice_no = transform.slice_configuration['slice_no']
        mfp_x, mfp_y = mfp_points['mfp_x'], mfp_points['mfp_y']

        matrices = transform.matrices
        ai, bi = matrices['ai'], matrices['bi']
        phases, alphas = Util.apply_svd_distortion(mfp_x, mfp_y, ai, bi)

        ech_ords = np.full(alphas.shape, ech_ord)
        waves = Util.phases_to_waves(phases, ech_ords)
        fslice_no, _ = Util.decode_slice_no(slice_no)
        efp_y_val = Util.slice_to_efp_y(fslice_no, slice_phase)
        efp_y = np.full(alphas.shape, efp_y_val)

        efp_points = {'efp_x': alphas, 'efp_y': efp_y, 'efp_w': waves}
        return efp_points

    @staticmethod
    def sort(ref, unsort):
        """Sort ``ref`` ascending and apply the same permutation to every array in ``unsort``.

        Parameters
        ----------
        ref : numpy.ndarray
        unsort : sequence of numpy.ndarray
            Arrays of the same length as ``ref``.

        Returns
        -------
        sorted_ref : numpy.ndarray
        sorted_arrays : tuple of numpy.ndarray

        Notes
        -----
        - Clean-up: not called anywhere in the repository.  Would be the
          natural fix for the ``numpy.interp`` ordering issue noted in
          :meth:`get_term_values`.
        """
        indices = np.argsort(ref)
        sort = []
        for array in unsort:
            sort.append(array[indices])
        return ref[indices], tuple(sort)

    @staticmethod
    def lookup_transform_fit(config, prism_angle, transform_fits):
        """Evaluate legacy 1-D (prism-angle) polynomial fits of the matrices for one slice.

        ``transform_fits`` has shape ``(n_slices, n_mats, n_terms, n_terms,
        n_polys)`` and each element polynomial is evaluated with
        ``numpy.poly1d(...)(prism_angle)``.

        Parameters
        ----------
        config : tuple
            ``(ech_ord, slice_no, spifu_no)``; only ``slice_no`` (1-based) is
            used.
        prism_angle : float
            Degrees.
        transform_fits : numpy.ndarray
            5-D coefficient array.

        Returns
        -------
        tuple of numpy.ndarray
            ``n_mats`` matrices of shape ``(n_terms, n_terms)``.

        Notes
        -----
        - Clean-up: superseded by the bivariate
          ``lmsdist_polyfit.PolyFit.make_fit_matrix``; no producer of the 5-D
          array exists; not called anywhere.
        """
        ech_ord, slice_no, spifu_no = config
        n_slices, n_mats, n_terms, _, n_polys = transform_fits.shape
        i_slice = int(slice_no - 1)
        matrix_fits = []
        for i_mat in range(0, n_mats):
            matrix_fit = np.zeros((n_terms, n_terms))
            for row in range(0, n_terms):
                for col in range(0, n_terms):
                    poly = transform_fits[i_slice, i_mat, row, col, :]
                    fit = np.poly1d(poly)
                    term = fit(prism_angle)
                    matrix_fit[row, col] = term
            matrix_fits.append(matrix_fit)
        return tuple(matrix_fits)

    @staticmethod
    def print_poly_transform(transform, **kwargs):
        """Print the upper-left triangle (``col < n_terms - row``) of each matrix in a transform.

        Parameters
        ----------
        transform : sequence of numpy.ndarray
            Matrices in ``Globals.matrix_names`` order (``a, b, ai, bi``).
        label : str, optional
            Printed first; default ``'---'``.

        Returns
        -------
        None

        Notes
        -----
        - Clean-up: expects a tuple of matrices, not a ``Transform`` object
          (``lms_test.py`` passes a legacy tuple); the zero lower-right
          triangle is skipped by construction.
        """
        label = kwargs.get('label', '---')
        print(label)
        for matrix, mat_name in zip(transform, Globals.matrix_names):
            n_terms = matrix.shape[0]
            print("Matrix {:s}".format(mat_name))
            for r in range(0, n_terms):
                line = ""
                for c in range(0, n_terms - r):
                    token = "{:10.3e}".format(matrix[r, c])
                    line = line + token
                print(line)
        return

    @staticmethod
    def find_fwhm_lin(x, y):
        """Find the half-maximum crossings of a 1-D profile by linear interpolation.

        ``yh = max(y) / 2``; the first and last indices where ``y > yh`` are
        found and the crossing abscissae ``xl`` (between ``il - 1`` and ``il``)
        and ``xr`` (between ``ir`` and ``ir + 1``) are linearly interpolated.
        The FWHM is ``xr - xl``.  Referred to as the "linear" FWHM in the
        ``lmsiq`` summary files.

        Parameters
        ----------
        x, y : numpy.ndarray
            Sampled profile, ``x`` increasing.

        Returns
        -------
        xl, xr : float
            Left and right half-maximum positions.
        yh : float
            The half-maximum level.

        Raises
        ------
        IndexError
            If the profile is above half maximum at either end of the array
            (``il == 0`` or ``ir == len(x) - 1``).

        Notes
        -----
        - Open question: assumes a single-peaked profile; secondary peaks
          above half maximum widen the result.
        - Clean-up: duplicate of ``lmsiq_analyse.Analyse._find_fwhm_lin``
          (which additionally returns an error flag); not called from
          ``lmsdist``.
        """
        ymax = np.amax(y)
        yh = ymax / 2.0
        yz = np.subtract(y, yh)       # Search for zero crossing
        iz = np.where(yz > 0)
        il = iz[0][0]
        ir = iz[0][-1]
        xl = x[il-1] - yz[il-1] * (x[il] - x[il-1]) / (yz[il] - yz[il-1])
        xr = x[ir] - yz[ir] * (x[ir+1] - x[ir]) / (yz[ir+1] - yz[ir])
        return xl, xr, yh

    @staticmethod
    def solve_svd_distortion(x_in, y_in, x_out, y_out, order, inverse=False):
        """Least-squares fit of the A, B (or AI, BI) distortion matrices by SVD pseudo-inverse.

        Solves, for ``dim = order + 1`` (4) and every input point ``p``,

        .. math::

            x_{out,p} = \\sum_{r,c} A_{rc}\\, y_{in,p}^{\\,r}\\, x_{in,p}^{\\,c}, \\qquad
            y_{out,p} = \\sum_{r,c} B_{rc}\\, y_{in,p}^{\\,r}\\, x_{in,p}^{\\,c}

        with terms ``r + c > order`` excluded.  The design matrix ``m`` has
        shape ``(dim*dim, n_pts)`` with row ``ind = r*dim + c`` equal to
        ``y_in**r * x_in**c`` masked to zero where ``r + c > order`` (the
        in-code comment says "lower diagonal"; it is the lower-right
        anti-diagonal triangle).  ``numpy.linalg.svd(m)`` gives ``v, w, u``;
        singular values with ``|w| < Globals.svd_cutoff`` (1e-7) are dropped
        (their reciprocal set to 0, a message printed at debug level
        ``'medium'`` if ``w > 1e-16``) and the coefficient vector is
        ``a = x_out @ u.T @ diag(1/w) @ v.T``, reshaped row-major to
        ``(dim, dim)``.  The masked design rows are exactly zero, so their
        singular values are zero and the corresponding matrix elements come
        out as zero.

        This matches :meth:`apply_svd_distortion`, where ``A[r, c]`` multiplies
        ``y**r * x**c``.

        Parameters
        ----------
        x_in, y_in : numpy.ndarray
            Input coordinates (forward: ``phase``, ``alpha``).
        x_out, y_out : numpy.ndarray
            Output coordinates (forward: ``mfp_x``, ``mfp_y`` in mm).
        order : int
            Maximum total polynomial degree (``Globals.svd_order - 1`` = 3).
        inverse : bool, optional
            If True swap the roles of in and out to fit ``AI, BI``
            (``(mfp_x, mfp_y) -> (phase, alpha)``).

        Returns
        -------
        amat, bmat : numpy.ndarray
            ``(dim, dim)`` matrices.

        Notes
        -----
        - Open question: the header comments state the routine was
          "Converted to python from Ronayette's original IDL code by Temim,
          21/2/17 ... see note on MIRI calibration"; the MIRI document is not
          identified.  Given the MIRI heritage the licensing / attribution of
          this routine should be confirmed before it is copied elsewhere.
        - Inconsistency: ``mask`` and ``exponent`` are sized from
          ``x_out.size`` even when ``inverse=True`` (harmless because all four
          arrays have equal length).
        - Clean-up: the in-code comment "Mask lower diagonal" describes the
          ``r + c > order`` triangle; the term is misleading.
        - Clean-up: the fit is unweighted and no residual statistic is
          returned; ``RayTrace.create_svd_transforms`` recomputes residuals by
          re-applying the transform.
        """
        # @author Alistair Glasse + Tea Temim
        # Converted to python from Ronayette's original IDL code by Temim, 21/2/17
        #   1. Changed input parameters to allow use with any data set (Glasse)
        # make a polynomial fit of the imaged grid point
        # see note on MIRI calibration
        # if keyword set inverse: swaps Xout and Xin, Yout and Yin
        # to compute the coefficients for the transform from FPA to entrance plane
        # INPUTS: XYin           Input xo,Y coordinates
        #          XYout
        #          order         Polynomial order for transform
        #          inverse       True = Calculate inverse transform (out -> in)
        #
        # OUTPUTS: amat, bmat, the distortion coefficients
        if inverse:
            xi = x_out.copy()
            yi = y_out.copy()
            xo = x_in.copy()
            yo = y_in.copy()
        else:
            xi = x_in.copy()
            yi = y_in.copy()
            xo = x_out.copy()
            yo = y_out.copy()

        dim = order + 1

        ind = np.arange(0, dim*dim)
        rows = np.floor(ind / dim)
        cols = ind % dim
        mask_ld = rows + cols <= order      # Mask lower diagonal to zero (False)
        mask = np.array([mask_ld.astype(int),] * x_out.size).transpose()
        exponent = np.array([range(0, dim), ] * x_out.size).transpose()

        mx = np.tile(np.power(xi, exponent), (dim, 1))
        my = np.repeat(np.power(yi, exponent), dim, axis=0)
        mxy = np.multiply(mx, my)
        m = np.multiply(mxy, mask)

        v, w, u = np.linalg.svd(m, full_matrices=False)
        n_w = w.size
        wp = np.zeros((n_w, n_w))
        svd_cutoff = Globals.svd_cutoff            # Only use singular values >= svd_cutoff
        for k in range(0, n_w):
            if abs(w[k]) >= svd_cutoff:
                wp[k, k] = 1.0 / w[k]
            else:
                if Globals.is_debug('medium'):
                    if w[k] > 1.0e-16:
                        print("!! Clipping singular value = {:5.3e} !!".format(w[k]))
        a = xo @ u.T @ wp @ v.T
        amat = np.reshape(a, (dim, dim))
        b = yo @ u.T @ wp @ v.T
        bmat = np.reshape(b, (dim, dim))
        return amat, bmat

    @staticmethod
    def test_out_and_back(filer, opticon, date_stamp):
        """Round-trip test EFP -> MFP -> DFP -> MFP -> EFP for one slice.  Dead code.

        Intended to pick the configuration centring 4.65 um
        (:meth:`find_closest_transforms`), sample 30 points across the slice
        length and wavelength range, project them out to detector pixels and
        back, and print a table of the round-trip errors.

        Parameters
        ----------
        filer : Filer
        opticon : str
        date_stamp : str

        Returns
        -------
        None

        Raises
        ------
        TypeError / ValueError
            ``efp_y_to_slice_no([efp_y_cen])`` is called with a list and its
            two scalar results are unpacked as lists.
        TypeError
            ``opt_transform['configuration']`` uses dict-style access on a
            ``Transform`` object (and the key ``'slice'`` never existed in the
            templates); :meth:`find_closest_transforms` also returns a tuple,
            not a list indexable by slice number.
        ValueError
            ``mfp_points = Util.efp_to_mfp(...)`` binds the ``(dict, oob)``
            tuple, which :meth:`mfp_to_dfp` cannot index.

        Notes
        -----
        - Bug: multiple stale API references, listed above; cannot run.
        - Clean-up: superseded by the ``evaluate_transforms`` block of
          ``lmsdist.run`` and the AIT round-trip in ``lmsaiv_opt02``.
        """
        print('lms_distort - Evaluating transforms')

        # Define EFP coordinates and target wavelength for spectrum
        efp_x_cen, efp_y_cen, efp_w_cen = 0., 0., 4.65
        tgt_slice_nos, test_phases = Util.efp_y_to_slice_no([efp_y_cen])
        tgt_slice_no = tgt_slice_nos[0]

        # Generate test spectrum for the wavelength/order which is closest to the mfp_y = 0. column.
        version_tag = "_v{:s}".format(date_stamp)
        affines = filer.read_fits_affine_transform(version_tag)
        svd_transforms = filer.read_svd_transforms()
        opt_transforms = Util.find_closest_transforms(efp_w_cen, opticon, svd_transforms)
        opt_transform = opt_transforms[tgt_slice_no]

        # Check out and back for the wavelength min, max and centre of the spectrum.
        w_min = opt_transform['configuration']['w_min']
        w_max = opt_transform['configuration']['w_max']

        n_pts = 30  # No. of wavelength samples in EFP
        # Create EFP data cube for extended/background emission.
        efp_ws = np.linspace(w_min, w_max, n_pts)
        efp_ys = np.zeros(n_pts)  # EFP across slice coordinate
        efp_fs = np.zeros(n_pts)  # Flux values
        efp_as_mm = Globals.efp_arcsec_mm
        alpha_fov = Globals.alpha_fov
        efp_xmax = alpha_fov / efp_as_mm
        xh = efp_xmax / 2.
        efp_xs = np.linspace(-xh, +xh, n_pts)
        efp_out = {'efp_x': efp_xs, 'efp_y': efp_ys, 'efp_w': efp_ws, 'efp_f': efp_fs}

        test_transforms = []

        config = opt_transform['configuration']
        slice_no = config['slice']
        matrices = opt_transform['matrices']
        mfp_points = Util.efp_to_mfp(opt_transform, efp_out)
        dfp_points = Util.mfp_to_dfp(affines, mfp_points)
        mfp_back = Util.dfp_to_mfp(affines, dfp_points)
        efp_back = Util.mfp_to_efp(opt_transform, mfp_back)

        mfp_x, mfp_y = mfp_points['mfp_x'], mfp_points['mfp_y']
        det_nos, dfp_x, dfp_y = dfp_points['det_nos'], dfp_points['dfp_x'], dfp_points['dfp_y']
        pri_ang = config['pri_ang']
        ech_ang = config['ech_ang']
        ech_ord = config['ech_ord']

        fmt = "{:>7s},{:>7s},{:>7s},{:>8s},{:>9s},{:>9s},{:>12s},{:>9s},{:>12s},{:>9s},{:>9s},{:>8s},{:>8s},{:>15s},{:>12s}"
        print(fmt.format('out', 'out', 'out', 'prism', 'echelle', 'echelle',
                         'mosaic', 'mosaic', 'det', 'det', 'det', 'back', 'back', 'back-out', 'back-out'))
        print(fmt.format('efp_x', 'efp_y', 'efp_w', 'angle', 'angle', 'order',
                         'mfp_x', 'mfp_y', 'no.', 'dfp_x', 'dfp_y', 'efp_x', 'efp_y', 'delta_efp_x', 'delta_efp_y'))
        print(fmt.format('mm', 'mm', 'micron', 'deg.', 'deg.', '-',
                         'mm', 'mm', '-', 'pix', 'pix', 'mm', 'mm', 'mm', 'mm'))
        fmt1 = "{:7.3f},{:7.3f},{:7.3f},{:8.3f},{:9.3f},{:9d},"
        fmt2 = "{:12.3f},{:9.3f},{:12d},{:9.1f},{:9.1f},{:8.3f},{:8.3f},{:15.3f},{:12.3f}"
        fmt = fmt1 + fmt2
        for i in range(0, n_pts):
            efp_out_x, efp_out_y = efp_out['efp_x'][i], efp_out['efp_y'][i]
            efp_back_x, efp_back_y = efp_back['efp_x'][i], efp_back['efp_y'][i]
            delta_efp_x = efp_back_x - efp_out_x
            delta_efp_y = efp_back_y - efp_out_y
            print(fmt.format(efp_out_x, efp_out_y, efp_out['efp_w'][i],
                             pri_ang, ech_ang, ech_ord,
                             mfp_x[i], mfp_y[i],
                             det_nos[i], dfp_x[i], dfp_y[i],
                             efp_back_x, efp_back_y,
                             delta_efp_x, delta_efp_y))
        return
