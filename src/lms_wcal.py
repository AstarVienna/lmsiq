#!/usr/bin/env python
"""Legacy LMS wavelength calibration: linear echelle angle <-> wavelength per order.

Purpose
-------
Holds, per echelle diffraction order, a pair of first-order (linear)
polynomials mapping the boresight wavelength to the echelle rotation angle
(``p_wtoa``) and back (``p_atow``), and provides look-ups on them
(:meth:`Wcal.find_wavelength`, :meth:`Wcal.find_echelle_angle`,
:meth:`Wcal.find_echelle_setting`).

Program
-------
Legacy component of the ``lmsdist`` lineage.  Nothing in the current
``lmsdist.run`` chain calls it; it is imported by ``lmsiq.py`` /
``lmsiq_cuber.py`` (instantiated only), ``lms_efficiency.py`` and the study
script ``lms_test.py``.  Its role (wavelength -> prism/echelle setting) has
been superseded by ``lmsdist_polyfit.PolyFit.wave_to_config``, which uses a
bivariate surface fit in prism and echelle angle instead of a per-order line.

Input data
----------
A CSV written by :meth:`Wcal.write_poly` (``Filer.wcal_file``, i.e.
``../output/distortion/<opticon>/<date>/distortion_<opticon>_<date>_dist_wcal.txt``
relative to ``src/``; not in the repository).  Layout: one header line, then
one row per echelle order::

    Echelle Order, Poly Coeffs., wav->ang, Poly Coeffs., ang->wav,
    <order>, <c1>, <c0>, <ic1>, <ic0>,

Coefficients are written in ``numpy.poly1d`` order (highest power first), so
``angle = c1 * wave + c0`` and ``wave = ic1 * angle + ic0``.  Orders must be
contiguous and ascending because the look-ups index rows by
``order - p[0, 0]``.

Outputs
-------
The same CSV (``write_poly``) and the class attributes ``Wcal.p_wtoa`` and
``Wcal.p_atow`` (``read_poly``), each a ``(n_orders, fit_order + 2)`` array
whose column 0 is the order and columns 1.. are the coefficients.

Notes
-----
- Inconsistency: the irdb ``METIS/TRACE_LMS.fits`` ``WCAL`` extension
  (columns ``Ord, c0, c1, ic0, ic1``, authored by Alistair Glasse) carries the
  same linear model as this class, according to the PIP analysis document
  ``docs/09_lmsiq_extraction_analysis.md`` (section 2.1).  Which of the two
  is the reference has not been established.
- Bug: :meth:`Wcal.find_dispersion` constructs ``Filer(model_configuration)``
  but ``Filer.__init__`` takes no arguments; it also copies
  ``obs_dict['optical_configuration']`` into ``date_stamp`` and returns the
  constant ``1.0`` regardless of input.
- Bug: :meth:`Wcal.get_polyfit_transform` indexes a 6-D array
  ``poly[e_idx, slice_id, m, :, i, j]`` that no current writer produces.
- Clean-up: all state is class-level (``p_atow``, ``p_wtoa``), so two
  calibrations cannot coexist in one process.
- Clean-up: the imports of ``Filer`` and ``Globals`` are needed only by the
  broken ``find_dispersion``.
"""
import numpy as np
from lms_filer import Filer
from lms_globals import Globals


class Wcal:
    """Per-order linear wavelength <-> echelle-angle calibration (legacy).

    Attributes
    ----------
    fit_order : int
        Polynomial order of the per-order fits.  Fixed at 1 (linear).
    p_wtoa : numpy.ndarray or None
        Wavelength-to-angle table, shape ``(n_orders, fit_order + 2)``:
        column 0 is the echelle order, columns 1..fit_order+1 are the
        coefficients in ``numpy.polyval`` order (highest power first).
        Populated by :meth:`read_poly`.
    p_atow : numpy.ndarray or None
        Angle-to-wavelength table, same layout as ``p_wtoa``.

    Notes
    -----
    - Clean-up: state is held on the class, not on instances; ``__init__``
      does nothing.
    """

    fit_order = 1
    p_atow, p_wtoa = None, None

    def __init__(self):
        """Create a ``Wcal`` object.  No state is initialised."""
        return

    @staticmethod
    def find_dispersion(traces, obs_dict):
        """Return the dispersion (micron per pixel) for an observation.  Broken stub.

        The intent (from the original docstring) was to look up the nearest
        distortion transform in ``traces`` for the configuration in
        ``obs_dict`` and derive the dispersion from it.  The body only builds
        a ``Filer`` and reads the trace pickle; the result is discarded and
        the constant ``1.0`` is returned.

        Parameters
        ----------
        traces : list
            Ignored (immediately overwritten by the pickle read).
        obs_dict : dict
            Must contain ``'optical_configuration'``.

        Returns
        -------
        float
            Always ``1.0``.

        Raises
        ------
        TypeError
            Always: ``Filer(model_configuration)`` is called with one positional
            argument but ``lms_filer.Filer.__init__`` accepts none.

        Notes
        -----
        - Bug: ``Filer(model_configuration)`` has the wrong signature; the
          configuration should be applied with ``Filer.set_configuration``.
        - Bug: ``date_stamp = obs_dict['optical_configuration']`` is a copy-
          paste error (the same key is used for the optical configuration).
        - Bug: the return value is the hard-coded constant ``1.0``.
        - Clean-up: dead code; the dispersion is computed instead in
          ``lmsdist_plot.Plot.series('dispersion', ...)`` from ray-trace data.
        """
        dw_lms_pix = 1.0
        optical_configuration = obs_dict['optical_configuration']
        date_stamp = obs_dict['optical_configuration']
        model_configuration = 'distortion', optical_configuration, date_stamp
        dist_filer = Filer(model_configuration)
        traces = dist_filer.read_pickle(dist_filer.trace_file)
        return dw_lms_pix

    @staticmethod
    def write_poly(wcal_file, configs, wave_limits):
        """Fit and write per-order linear wavelength <-> echelle-angle polynomials to CSV.

        For every echelle order between the minimum and maximum present in
        ``configs[0]`` a boresight wavelength ``wbs = (w2 + w3) / 2`` is formed
        from the second and third wavelength-limit arrays and two
        ``numpy.polyfit`` fits of degree ``Wcal.fit_order`` (= 1) are made:
        ``angle = p_wtoa(wbs)`` and ``wbs = p_atow(angle)``.  One CSV row per
        order is written with the coefficients in ``numpy.poly1d`` order
        (highest power first).

        Parameters
        ----------
        wcal_file : str
            Output path (normally ``Filer.wcal_file``).
        configs : sequence of numpy.ndarray
            Per-configuration arrays; at least seven are required because the
            call unpacks ``configs[0:7]`` into ``eos, eas, pas, w1s, w2s, w3s,
            w4s`` (echelle order, echelle angle, prism angle and four
            wavelength limits).  Only ``eos`` and ``eas`` are used.
        wave_limits : sequence of numpy.ndarray
            Four arrays ``(w1s, w2s, w3s, w4s)``; these overwrite the values
            unpacked from ``configs``.

        Returns
        -------
        None

        Notes
        -----
        - Inconsistency: the wavelength limits are unpacked from ``configs``
          and then immediately replaced by ``wave_limits``; one of the two
          sources is redundant.
        - Open question: an order in ``range(eomin, eomax + 1)`` that has no
          rows in ``eos`` yields empty ``w`` / ``a`` arrays and
          ``numpy.polyfit`` will raise; the code assumes contiguous orders.
        - Clean-up: the ``configs`` layout is the legacy list-of-arrays
          format produced by ``lmsdist_util.Util.extract_configs``, which is
          itself dead code.
        """
        eos, eas, pas, w1s, w2s, w3s, w4s = configs[0:7]
        w1s, w2s, w3s, w4s = wave_limits
        eomin = int(np.min(eos))
        eomax = int(np.max(eos))
        n_eos = eomax - eomin + 1
        wbs = (w2s + w3s) / 2.0

        f = open(wcal_file, 'w')
        fmt = "{:>15s},{:>15s},{:>15s},{:>15s},{:>15s},\n"
        f.write(fmt.format("Echelle Order", "Poly Coeffs.", "wav->ang", "Poly Coeffs.", "ang->wav"))
        for i in range(0, n_eos):
            eo = eomin + i
            line = "{:15d},".format(eo)
            idx = np.where(eos == eo)
            w, a = wbs[idx], eas[idx]
            p_wtoa = np.poly1d(np.polyfit(w, a, Wcal.fit_order))        # Wavelength to angle
            for term in p_wtoa:
                token = "{:15.7e},".format(term)
                line = line + token
            p_atow = np.poly1d(np.polyfit(a, w, Wcal.fit_order))        # Angle to wavelength
            for term in p_atow:
                token = "{:15.7e},".format(term)
                line = line + token
            f.write(line + "\n")
        f.close()
        return

    @staticmethod
    def read_poly(wcal_file):
        """Read the per-order polynomial CSV into ``Wcal.p_wtoa`` and ``Wcal.p_atow``.

        Skips one header line, then parses each row as
        ``order, c_1..c_0 (wav->ang), ic_1..ic_0 (ang->wav)``.  Both class
        tables get shape ``(n_orders, fit_order + 2)`` with the order in
        column 0 and the coefficients in columns 1 onwards, preserving the
        highest-power-first ordering expected by ``numpy.polyval``.

        Parameters
        ----------
        wcal_file : str
            Path of the CSV written by :meth:`write_poly`.

        Returns
        -------
        None
            Results are stored on the class.

        Notes
        -----
        - Clean-up: writes to class attributes (hidden global state) rather
          than returning the tables.
        - Open question: a trailing blank line in the file would produce a row
          with a single empty token and ``int('')`` would raise; no guard is
          present.
        """
        fmt = "Reading echelle angle v wavelength polynomials from file= {:s}"
        print(fmt.format(wcal_file))
        lines = open(wcal_file, 'r').read().splitlines()
        n_hdr_lines = 1
        efw_list = lines[n_hdr_lines:]
        n_eos = len(efw_list)
        n_terms = Wcal.fit_order + 1
        p_wtoa, p_atow = np.zeros((n_eos, n_terms + 1)), np.zeros((n_eos, n_terms + 1))

        i = 0
        for rec in efw_list:
            tokens = rec.split(',')
            p_wtoa[i, 0] = int(tokens[0])
            p_atow[i, 0] = int(tokens[0])
            for j in range(0, n_terms):
                p_wtoa[i, 1+j] = float(tokens[1+j])
                p_atow[i, 1+j] = float(tokens[1+j+n_terms])
            i = i + 1
        Wcal.p_wtoa = p_wtoa
        Wcal.p_atow = p_atow
        return

    @staticmethod
    def find_wavelength(angle, order):
        """Return the boresight wavelength for an echelle angle and order.

        Evaluates the linear ``p_atow`` polynomial of the requested order with
        ``numpy.polyval``.  The row is located as ``order - p_atow[0, 0]``,
        i.e. the table is assumed to hold contiguous ascending orders.

        Parameters
        ----------
        angle : float or numpy.ndarray
            Echelle rotation angle in degrees.
        order : int
            Echelle diffraction order.

        Returns
        -------
        float or numpy.ndarray
            Wavelength in microns.

        Raises
        ------
        TypeError
            If :meth:`read_poly` has not been called (``p_atow`` is ``None``).
        IndexError
            If ``order`` lies outside the tabulated range.
        """
        p_atow = Wcal.p_atow
        i = int(order - p_atow[0, 0])
        p = p_atow[i, 1:Wcal.fit_order + 2]
        w = np.polyval(p, angle)
        return w

    @staticmethod
    def get_polyfit_transform(poly, ech_bounds, configuration):
        """Evaluate legacy per-order distortion-matrix polynomials at an echelle angle.

        Legacy accessor for a 6-D coefficient array
        ``poly[order_index, slice, matrix, coeff, row, col]`` in which each
        term of the four ``svd_order x svd_order`` matrices (A, B, AI, BI) is
        a polynomial in the grating (echelle) angle.  Each term is evaluated
        with ``numpy.polyval(poly[e_idx, slice_id, m, :, i, j], grating_angle)``.

        Parameters
        ----------
        poly : numpy.ndarray
            6-D coefficient array as described above.  ``poly.shape[2]`` is
            taken as the number of matrices and ``poly.shape[4]`` as the
            matrix order.
        ech_bounds : sequence
            ``ech_bounds[0]`` is the lowest echelle order in the array.
        configuration : sequence
            Six-element tuple ``(slice_id, _, _, grating_angle, order,
            im_pix_size)``; only ``slice_id``, ``grating_angle`` and ``order``
            are used.

        Returns
        -------
        tuple of numpy.ndarray
            ``(a, b, ai, bi)`` matrices, each ``(n_terms, n_terms)``.

        Notes
        -----
        - Bug: nothing in the repository produces the 6-D ``poly`` layout;
          ``lmsdist_util.Util.read_polyfits_file`` returns a 2-D array and
          ``lmsdist_util.Util.find_polyfit`` a 3-D one.  This method cannot be
          exercised with current outputs.
        - Clean-up: superseded by ``lmsdist_polyfit.PolyFit.make_fit_matrix``
          (bivariate in prism and echelle angle) and the FITS fit-parameter
          product written by ``lms_filer.Filer.write_fit_parameters``.
        """

        slice_id, _, _, grating_angle, order, im_pix_size = configuration
        e_idx = int(order - ech_bounds[0])
        shape = poly.shape
        n_mat = shape[2]
        n_terms = shape[4]
        tr = []

        for m in range(0, n_mat):
            mat = np.zeros((n_terms, n_terms))
            for i in range(0, n_terms):

                for j in range(0, n_terms):
                    poly_coeffs = poly[e_idx, slice_id, m, :, i, j]
                    mat[i, j] = np.polyval(poly_coeffs, grating_angle)
            tr.append(mat)
        return tr[0], tr[1], tr[2], tr[3]

    @staticmethod
    def find_echelle_angle(w, order):
        """Return the echelle angle that places a wavelength at boresight in an order.

        Evaluates the linear ``p_wtoa`` polynomial for the requested order
        with ``numpy.polyval``; the table row is ``order - p_wtoa[0, 0]``.

        Parameters
        ----------
        w : float or numpy.ndarray
            Wavelength in microns.
        order : int
            Echelle diffraction order.

        Returns
        -------
        float or numpy.ndarray
            Echelle rotation angle in degrees.

        Raises
        ------
        TypeError
            If :meth:`read_poly` has not been called (``p_wtoa`` is ``None``).
        IndexError
            If ``order`` lies outside the tabulated range.
        """
        p_wtoa = Wcal.p_wtoa
        i = int(order - p_wtoa[0, 0])
        p = p_wtoa[i, 1:Wcal.fit_order + 2]
        ea = np.polyval(p, w)
        return ea

    @staticmethod
    def find_echelle_setting(w):
        """Return the order and echelle angle with the smallest ``|angle|`` for a wavelength.

        Loops over every tabulated order, evaluates ``p_wtoa`` at ``w`` and
        keeps the order whose echelle angle has the smallest absolute value
        (the setting closest to the blaze).

        Parameters
        ----------
        w : float
            Wavelength in microns.

        Returns
        -------
        eo_opt : int
            Optimum echelle order (``-1`` if no order gave ``|angle|`` < 90 deg).
        ea_min : float
            Corresponding echelle angle in degrees (signed).

        Notes
        -----
        - Clean-up: same selection rule as
          ``lmsdist_polyfit.PolyFit.wave_to_config(select='min_ech_ang')``,
          which supersedes this method.
        """
        p_wtoa = Wcal.p_wtoa
        n_orders = len(p_wtoa)
        eo_opt = -1
        ea_min = 90.0
        for i in range(0, n_orders):
            eo = p_wtoa[i, 0]
            p = p_wtoa[i, 1:Wcal.fit_order+2]
            ea = np.polyval(p, w)
            if abs(ea) < abs(ea_min):
                ea_min = ea
                eo_opt = eo
        return int(eo_opt), ea_min
