#!/usr/bin/env python
"""Polynomial fits of the LMS distortion transforms versus prism and echelle angle.

Purpose
-------
``lmsdist`` fits one set of E-TNT-ATC-MET-1058 distortion matrices
(``A, B, AI, BI``, see ``lms_transform.Transform``) per slice for each Zemax
configuration (prism angle ``pa``, echelle angle ``ea``).  This module
interpolates between those configurations so a transform can be built for an
arbitrary setting:

1. every matrix element ``c_ij`` of every slice is fitted as a bivariate
   polynomial surface in ``(pa, ea)`` (:meth:`PolyFit.find_slice_fit`,
   :meth:`PolyFit.surface_model`, 10 coefficients per element);
2. the product ``n * lambda_bs`` (echelle order times boresight wavelength of
   field slice 12) is fitted with the same surface (:meth:`PolyFit.find_wxo_fit`);
3. the prism angle that centres a wavelength at ``ea = 0`` is fitted as a
   1-D polynomial in wavelength (:meth:`PolyFit.create_polynomial_fit`, called
   from ``lmsdist.run``, order ``Globals.wpa_fit_order[opticon]``);
4. :meth:`PolyFit.wave_to_config` inverts 2 and 3 with ``scipy.optimize.fsolve``
   to give ``(pa, ea, order)`` for a target wavelength, and
   :meth:`PolyFit.make_fit_matrix` rebuilds a 4 x 4 matrix from 1.

Program
-------
``lmsdist`` (``lmsdist.run`` blocks ``fit_transforms``, ``evaluate_transforms``
and ``test_transform_fit``).

Input data
----------
``Transform`` objects read back from the ``lms_efp_mfp_*.fits`` files written
by ``lms_filer.Filer.write_svd_transforms`` (see ``lms_transform`` for the
layout), i.e. derived from the Zemax ray-trace CSVs under
``../data/distortion/<opticon>/<date>/`` relative to ``src/`` (not in the
repository).

Outputs
-------
Python dicts consumed by ``lms_filer.Filer.write_fit_parameters``, which writes
``lms_dist_efp_mfp_{nom|ext}_fit_parameters_v<date>.fits`` in
``../output/distortion/<opticon>/<date>/fits/``:

- ext 1: columns ``WP0..WP{n-1}`` -- wavelength -> prism angle polynomial
  (``wpa_fit['wpa_opt']``), header ``N_COEFFS``;
- ext 2: columns ``SLICE_NO, PA0_EA0, PA1_EA0, ... PA0_EA3`` -- the ``n*lambda``
  surface (``wxo_fit['wxo_opt']``), header ``ORDER``, ``N_COEFFS``;
- ext 3: columns ``SLICE_NO, ROW, COL, A_PA0_EA0 ... BI_PA0_EA3`` -- one row per
  matrix element per slice (16 x n_slices rows), 40 coefficient columns.

Surface model
-------------
With ``order = Globals.surface_fit_order = 4`` the surface has
``Globals.surface_fit_n_coeffs = 10`` terms::

    f(pa, ea) = sum_{j=0}^{3} sum_{i=0}^{3-j} c_ij * pa**i * ea**j

ordered ``(i,j) = (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3)``
(:meth:`PolyFit.make_surface_model_header` names them ``PA{i}_EA{j}``).

Notes
-----
- Inconsistency: the boresight slice used for the ``n*lambda`` surface is
  field slice 12 (``if fslice_no == 12`` in
  :meth:`PolyFit.create_polynomial_surface_fits`), while ``lmsdist.run``'s
  comments, ``Filer.write_fit_parameters`` ('slice 13 wavelength') and
  ``RayTrace.get_ifp_boresight`` (``fslice_no = 13``) use slice 13.
- Inconsistency: :meth:`PolyFit.create_polynomial_fit` treats ``poly_order`` as
  the NUMBER of coefficients (``p0 = [0] * poly_order``), so
  ``Globals.wpa_fit_order = {nominal: 6, extended: 3}`` yields degree 5 and
  degree 2 polynomials, not degree 6 and 3.
- Inconsistency: :meth:`PolyFit.wave_to_config` stores the order under the
  key ``'ech_order'`` whereas every other configuration dict uses ``'ech_ord'``.
- Bug: :meth:`PolyFit.make_fit_transform` and :meth:`PolyFit.make_mfp_projection`
  index ``term_fits[slice_no][spifu_no]`` and use attributes
  (``Transform.configuration``, ``RayTrace.get``) that do not exist; both are
  dead code.
- Clean-up: the nested ``ea_func`` in :meth:`PolyFit.wave_to_config` is
  decorated ``@staticmethod`` inside a function body; this only works because
  ``staticmethod`` objects became callable in Python 3.10.
- Clean-up: ``lmsdist.run`` re-implements :meth:`PolyFit.make_fit_matrix`
  inline in its ``evaluate_transforms`` block.
- Open question: :meth:`PolyFit.surface_model` reads the order from
  ``Globals.surface_fit_order`` rather than from ``len(coeff)``; fit-parameter
  files written with a different ``ORDER`` header would be mis-evaluated.
"""
import math
import numpy as np
from scipy.optimize import curve_fit, fsolve
from lms_globals import Globals
from lmsdist_util import Util
from lmsdist_plot import Plot
from lms_transform import Transform


class PolyFit:
    """Interpolation of wavelength and distortion-matrix terms versus prism/echelle angle.

    All methods are static; the class is a namespace.  See the module
    docstring for the fitting scheme and the surface-model definition.

    Attributes
    ----------
    n_lines_config : int
        Unused (``-1``).

    Notes
    -----
    - Clean-up: ``__init__`` takes ``opticon`` but ignores it (the body that
      used it is commented out); ``n_lines_config`` is never read.
    """
    # wpa_fit_order = None
    n_lines_config = -1

    def __init__(self, opticon):
        """Create a ``PolyFit`` object.

        Parameters
        ----------
        opticon : str
            ``Globals.nominal`` or ``Globals.extended``.  Currently ignored.
        """
        # PolyFit.wpa_fit_order = Globals.wpa_fit_order_dict[opticon]
        return

    @staticmethod
    def wave_to_config(wave, opticon, wpa_fit, wxo_fit, **kwargs):
        """Find the prism angle, echelle angle and order that centre a wavelength on the mosaic.

        The prism angle is evaluated directly from the 1-D fit
        ``pri_ang = poly_model(wave, *wpa_fit['wpa_opt'])``.  Then, for every
        echelle order ``n`` in ``range(21, 42)`` whose product
        ``wave * n`` lies within ``wxo_lim = [108, 115]`` microns (the
        ``n*lambda`` coverage of the echelle), ``scipy.optimize.fsolve`` finds
        the echelle angle ``ea`` satisfying
        ``surface_model((pri_ang, ea), *wxo_fit['wxo_opt']) == wave * n``
        starting from ``ea = 0``.  With ``select='min_ech_ang'`` the order with
        the smallest ``|ea|`` is returned.

        Parameters
        ----------
        wave : float
            Target wavelength in microns.
        opticon : str
            ``Globals.nominal`` or ``Globals.extended``; copied into the result.
        wpa_fit : dict
            ``{'wpa_opt': coefficients, ...}`` from
            :meth:`create_polynomial_fit` / ``Filer.read_fit_parameters``.
        wxo_fit : dict
            ``{'wxo_opt': 10 surface coefficients, ...}`` from
            :meth:`find_wxo_fit` / ``Filer.read_fit_parameters``.
        debug : bool, optional
            Passed to ``ea_func`` but not used there (it consults
            ``Globals.is_debug('medium')`` instead).  Default ``False``.
        select : {'min_ech_ang', 'all'}, optional
            ``'min_ech_ang'`` (default) returns the single best configuration;
            ``'all'`` returns a list with one entry per admissible order.
        print_header : bool, optional
            Read but never used.

        Returns
        -------
        dict or list of dict
            For ``'min_ech_ang'``: a copy of ``Globals.lms_config_template``
            with keys ``opticon, pri_ang, ech_ang, wave, ech_order``.  For
            ``'all'``: a list of dicts.

        Notes
        -----
        - Bug: with ``select='all'`` each appended dict contains only
          ``wave`` and ``opticon``; ``pri_ang``, ``ech_ang`` and the order are
          never stored because those assignments sit inside the
          ``'min_ech_ang'`` branch.
        - Inconsistency: the order is stored as ``'ech_order'`` while the rest
          of lmsiq uses ``'ech_ord'``.
        - Inconsistency: if no order falls inside ``wxo_lim`` the template
          copy is returned with ``pri_ang`` / ``ech_ang`` still ``None`` and
          no warning.
        - Open question: the ``wxo_lim = [108, 115]`` micron window and the
          order range 21..41 are hard-coded; their source (E-TNT-ATC-MET-1058?)
          is not stated.
        - Clean-up: ``fsolve``'s ``ier`` / ``msg`` convergence flags are
          discarded.
        - Clean-up: ``debug`` and ``print_header`` are unused parameters.
        """
        @staticmethod
        def ea_func(ea, *fargs):
            """Residual ``surface_model((pa, ea)) - n*lambda`` for ``fsolve``.

            Parameters
            ----------
            ea : numpy.ndarray
                One-element array holding the trial echelle angle (deg).
            *fargs : tuple
                ``(pa, wxo, fcoeff, debug)``: prism angle, target ``n*lambda``,
                10 surface coefficients, unused debug flag.

            Returns
            -------
            float
                ``f(pa, ea) - wxo`` in microns.

            Notes
            -----
            - Clean-up: decorated ``@staticmethod`` inside a function body;
              relies on ``staticmethod`` objects being callable (Python >= 3.10).
            """
            pa, wxo, fcoeff, debug = fargs
            xy = pa, ea[0]
            f = PolyFit.surface_model(xy, *fcoeff)
            if Globals.is_debug('medium'):
                print("{:>52.3f},{:>20.3f} -> dw = {:10.3f}".format(pa, ea[0], f - wxo))
            return f - wxo

        debug = kwargs.get('debug', False)
        select = kwargs.get('select', 'min_ech_ang')
        print_header = kwargs.get('print_header', True)

        lms_configs = []

        wpa_coeffs = wpa_fit['wpa_opt']
        pri_ang = PolyFit.poly_model(wave, *wpa_coeffs)

        # print("Finding configuration for wavelength = {:10.3f}".format(wave))
        wxo_coeffs = wxo_fit['wxo_opt']
        wxo_lim = np.array([108., 115.])      # Wavelength bounds of w x order coverage (in microns)
        ech_ang_min = 100.
        lms_config = Globals.lms_config_template.copy()
        for ech_ord in range(21, 42):
            lms_config['wave'] = wave
            lms_config['opticon'] = opticon

            wxo_tgt = wave * ech_ord
            if wxo_tgt < wxo_lim[0] or wxo_tgt > wxo_lim[1]:   # The accessible wavelength coverage in each order is wxo_lim / n
                continue
            ech_ang_guess = np.zeros(1)
            args = pri_ang, wxo_tgt, wxo_coeffs, debug
            ech_ang_sol, _, ier, msg = fsolve(ea_func, ech_ang_guess, args=args, full_output=True)
            ech_ang = ech_ang_sol[0]

            if select == 'all':
                lms_configs.append(lms_config)
                lms_config = Globals.lms_config_template.copy()
            if select == 'min_ech_ang':
                abs_ech_ang = math.fabs(ech_ang)
                if abs_ech_ang < ech_ang_min:
                    ech_ang_min = abs_ech_ang
                    lms_config['pri_ang'] = pri_ang
                    lms_config['ech_ang'] = ech_ang
                    lms_config['ech_order'] = ech_ord
        if select == 'all':
            return lms_configs
        return lms_config

    @staticmethod
    def create_polynomial_surface_fits(opticon, svd_transforms, plot_terms=False, plot_wxo=True):
        """Fit the ``n*lambda`` surface and every matrix term of every slice versus (pa, ea).

        Loops over ``Globals.fslice_no_range[opticon]`` x
        ``Globals.pslice_no_range[opticon]`` (28 slices nominal; 3 x 6 = 18
        extended), and for each encoded slice number:

        1. selects that slice's transforms from ``svd_transforms`` with
           ``Util.filter_transform_list``;
        2. gathers ``(pri_ang, ech_ang, ech_ord, boresight wavelength,
           matrices)`` per configuration with ``Util.get_term_values``;
        3. fits the 16 x 4 matrix terms with :meth:`find_slice_fit`;
        4. for field slice 12 only, fits ``n*lambda_bs`` with :meth:`find_wxo_fit`.

        Parameters
        ----------
        opticon : str
            ``Globals.nominal`` or ``Globals.extended``.
        svd_transforms : list of Transform
            All slice transforms of all configurations (from
            ``Filer.read_svd_transforms``).
        plot_terms : bool, optional
            Unused.
        plot_wxo : bool, optional
            Unused; plotting is controlled by ``Globals.is_debug('medium')``.

        Returns
        -------
        wxo_fit : dict
            From :meth:`find_wxo_fit` for the last pslice of fslice 12.
        wxo_header : list of str
            Column names ``PA{i}_EA{j}`` from :meth:`make_surface_model_header`.
        term_fits : list
            ``[[slice_no, slice_fit], ...]`` where ``slice_fit`` is the dict
            returned by :meth:`find_slice_fit`.

        Raises
        ------
        TypeError
            If no transform has ``fslice_no == 12`` (``wxo_fit`` stays ``None``
            and ``wxo_fit['order']`` fails).

        Notes
        -----
        - Inconsistency: the boresight slice is 12 here but 13 elsewhere (see
          module notes).
        - Inconsistency: in extended mode ``wxo_fit`` is overwritten for each
          pslice 1..6 of fslice 12 and only the last survives.
        - Clean-up: ``plot_terms`` / ``plot_wxo`` are dead parameters;
          ``baseline_debug_level`` is saved and restored without being changed.
        - Clean-up: a local ``slice_fits`` dict is built and discarded.
        """
        slice_fits = {}
        term_fits = []
        wxo_fit, wxo_fit_order = None, None
        surface_model = PolyFit.surface_model
        for fslice_no in Globals.fslice_no_range[opticon]:
            for pslice_no in Globals.pslice_no_range[opticon]:
                slice_no = Util.encode_slice_no(fslice_no, pslice_no)
                slice_transforms = Util.filter_transform_list(svd_transforms, slice_no=slice_no)
                term_values = Util.get_term_values(slice_transforms, fslice_no, pslice_no)
                term_fit = PolyFit.find_slice_fit(term_values)
                baseline_debug_level = Globals.debug_level
                if Globals.is_debug('medium'):
                    Plot.transform_fit(term_fit, term_values, surface_model, do_plots=True)
                    Plot.transform_fit(term_fit, term_values, surface_model, do_plots=True, plot_residuals=True)
                    Globals.set_debug_level(baseline_debug_level)
                term_fits.append([slice_no, term_fit])
                slice_fits[slice_no] = term_fit
                if fslice_no == 12:
                    wxo_fit = PolyFit.find_wxo_fit(term_values, Globals.surface_fit_order)
                    if Globals.is_debug('medium'):
                        Plot.wxo_fit(wxo_fit, term_values, surface_model, plot_residuals=False)
                        Plot.wxo_fit(wxo_fit, term_values, surface_model, plot_residuals=True)
                        Globals.set_debug_level(baseline_debug_level)
        wxo_header = PolyFit.make_surface_model_header(wxo_fit['order'])
        return wxo_fit, wxo_header, term_fits

    @staticmethod
    def make_surface_model_header(order):
        """Return the coefficient names ``PA{i}_EA{j}`` in :meth:`surface_model` order.

        Parameters
        ----------
        order : int
            Surface fit order (``Globals.surface_fit_order``, 4).

        Returns
        -------
        list of str
            ``order * (order + 1) / 2`` names, e.g. for order 4:
            ``['PA0_EA0', 'PA1_EA0', 'PA2_EA0', 'PA3_EA0', 'PA0_EA1', 'PA1_EA1',
            'PA2_EA1', 'PA0_EA2', 'PA1_EA2', 'PA0_EA3']``.

        Notes
        -----
        - Clean-up: the ``ij`` counter is incremented but never used.
        """
        ij = 0
        header = []
        for j in range(0, order):
            for i in range(0, order - j):
                text = "PA{:d}_EA{:d}".format(i, j)
                header.append(text)
                ij += 1
        return header

    @staticmethod
    def surface_model(xy, *coeff):
        """Bivariate polynomial surface used for all (prism angle, echelle angle) fits.

        With ``order = Globals.surface_fit_order`` (4) the model is the cubic
        surface with all terms of total degree <= 3:

        .. math::

            f(x, y) = \\sum_{j=0}^{order-1} \\sum_{i=0}^{order-1-j} c_{ij}\\, x^i y^j

        i.e. ``c00 + c10 x + c20 x^2 + c30 x^3 + c01 y + c11 xy + c21 x^2 y
        + c02 y^2 + c12 x y^2 + c03 y^3`` (10 coefficients), where ``x`` is the
        prism angle and ``y`` the echelle angle in degrees.  Coefficients are
        consumed in the order ``(i, j) = (0,0), (1,0), (2,0), (3,0), (0,1),
        (1,1), (2,1), (0,2), (1,2), (0,3)``, matching
        :meth:`make_surface_model_header`.  The signature is that required by
        ``scipy.optimize.curve_fit``.

        Parameters
        ----------
        xy : tuple
            ``(x, y)`` scalars or arrays of equal shape: prism angle, echelle
            angle.
        *coeff : float
            At least ``Globals.surface_fit_n_coeffs`` (10) coefficients.

        Returns
        -------
        float or numpy.ndarray
            Model value(s).

        Raises
        ------
        IndexError
            If fewer than 10 coefficients are supplied.

        Notes
        -----
        - Inconsistency: the previous docstring called this a "quadratic"
          surface and contained typos (``a03 x^2``, duplicated ``a20``); the
          implemented model is cubic as written above.
        - Open question: the order is taken from ``Globals.surface_fit_order``
          and not from ``len(coeff)``; extra coefficients are silently ignored.
        """
        xp, ye = xy
        ij = 0
        f = 0.
        order = Globals.surface_fit_order
        for j in range(0, order):
            y_term = np.power(ye, j)
            for i in range(0, order - j):
                x_term = np.power(xp, i)
                f += coeff[ij] * x_term * y_term
                ij += 1
        return f

    @staticmethod
    def poly_model(x, *coeff):
        """Evaluate ``y = c0 + c1 x + c2 x^2 + ...`` (lowest power first).

        Parameters
        ----------
        x : float or numpy.ndarray
        *coeff : float
            Coefficients in ascending power order (``len(coeff) - 1`` is the
            degree).  Note this is the REVERSE of ``numpy.polyval`` /
            ``lms_wcal.Wcal`` ordering.

        Returns
        -------
        float or numpy.ndarray

        Notes
        -----
        - Clean-up: duplicates ``lms_globals.Globals.polynomial`` (which also
          offers the gradient).
        """
        f = 0.
        order = len(coeff)
        for i in range(0, order):
            f += coeff[i] * np.power(x, i)
        return f

    @staticmethod
    def create_polynomial_fit(waves, pri_angs, poly_order=3):
        """Least-squares 1-D polynomial fit ``pri_angs = poly_model(waves)``.

        Uses ``scipy.optimize.curve_fit`` on :meth:`poly_model` with
        ``poly_order`` zero-initialised coefficients.  Despite the parameter
        names, ``lmsdist.run`` also uses it for echelle angle -> wavelength and
        wavelength -> prism angle fits per order.

        Parameters
        ----------
        waves : array_like
            Independent variable (e.g. wavelength in microns).
        pri_angs : array_like
            Dependent variable (e.g. prism angle in degrees).
        poly_order : int, optional
            NUMBER of coefficients, i.e. polynomial degree ``poly_order - 1``.
            Default 3 (quadratic).

        Returns
        -------
        dict
            ``{'wpa_opt': coefficients (ascending power), 'wpa_cov':
            covariance, 'n_coeffs': poly_order}``.

        Notes
        -----
        - Inconsistency: ``poly_order`` is the coefficient count, not the
          degree, so ``Globals.wpa_fit_order[nominal] = 6`` gives a quintic
          and ``[extended] = 3`` a quadratic through the three extended-mode
          points (an exact fit with zero residual).
        - Open question: with ``curve_fit`` and ``p0 = 0`` for a linear model
          the result equals ordinary least squares; ``numpy.polyfit`` would be
          simpler and return the same coefficients (reversed).
        """
        p0 = [0] * poly_order
        wpa_opt, wpa_cov = curve_fit(PolyFit.poly_model, xdata=waves, ydata=pri_angs, p0=p0)
        wpa_fit = {'wpa_opt': wpa_opt, 'wpa_cov': wpa_cov, 'n_coeffs': poly_order}
        return wpa_fit

    @staticmethod
    def find_slice_fit(term_vals):
        """Fit every element of A, B, AI, BI for one slice as a surface in (pa, ea).

        For each matrix tag in ``['a', 'b', 'ai', 'bi']`` and each of the 16
        ``(row, col)`` positions, the element values across all configurations
        in ``term_vals['matrices']`` are fitted with
        ``curve_fit(surface_model, (pri_ang, ech_ang), values, p0=[0]*10)``.

        Parameters
        ----------
        term_vals : dict
            Output of ``lmsdist_util.Util.get_term_values``: lists
            ``'pri_ang'``, ``'ech_ang'`` (deg) and ``'matrices'`` (one
            ``{'a','b','ai','bi'}`` dict per configuration), all of equal
            length.

        Returns
        -------
        dict
            ``{'a': ndarray (4, 4, 10), 'b': ..., 'ai': ..., 'bi': ...}``;
            ``[row, col, :]`` are the 10 surface coefficients of that element.

        Notes
        -----
        - Open question: elements in the masked lower-right triangle of each
          matrix (``row + col > 3``) are identically zero in every
          configuration; ``curve_fit`` on all-zero data returns zero
          coefficients but may emit ``OptimizeWarning`` about the covariance.
        - Open question: with only three extended-mode configurations
          (one per order) the 10-coefficient surface is heavily
          under-determined; ``curve_fit`` returns a minimum-norm-like solution
          whose extrapolation is unconstrained.
        - Clean-up: the matrix order 4 is hard-coded rather than taken from
          ``Globals.svd_order``.
        """
        n_coefficients = Globals.surface_fit_n_coeffs     # No. of fit coefficients (upper triangular elements)

        x = np.array(term_vals['pri_ang'])
        y = np.array(term_vals['ech_ang'])
        matrices = term_vals['matrices']
        slice_fit = {}
        mat_tags = ['a', 'b', 'ai', 'bi']
        for i, mat_tag in enumerate(mat_tags):
            term_array = np.zeros((4, 4, n_coefficients))
            slice_fit[mat_tag] = term_array
            for row in range(0, 4):
                for col in range(0, 4):
                    term_list = []
                    for mat_set in matrices:
                        mat = mat_set[mat_tag]
                        term = mat[row, col]
                        term_list.append(term)
                    terms = np.array(term_list)
                    term_opt, term_cov = curve_fit(PolyFit.surface_model,
                                                   xdata=(x, y), ydata=terms, p0=[0]*n_coefficients)
                    term_array[row, col, :] = term_opt
        return slice_fit

    @staticmethod
    def find_wxo_fit(term_values, order):
        """Fit ``n * lambda_bs = surface_model(pa, ea)`` for the boresight slice.

        ``lambda_bs`` is the wavelength that falls at ``mfp_x = 0`` for the
        slice centre (``term_values['w_bs']``, computed by
        ``Util.get_term_values``) and ``n`` the echelle order
        (``term_values['ech_ords']``).  Their product is a smooth function of
        the two angles across orders, which is why it, rather than the
        wavelength itself, is fitted.

        Parameters
        ----------
        term_values : dict
            Output of ``lmsdist_util.Util.get_term_values`` (keys
            ``slice_no, spifu_no, pri_ang, ech_ang, w_bs, ech_ords``).
        order : int
            Surface order recorded in the result (the caller passes
            ``Globals.surface_fit_order``); the number of coefficients is
            nevertheless taken from ``Globals.surface_fit_n_coeffs``.

        Returns
        -------
        dict
            ``{'slice_no', 'spifu_no', 'wxo_opt' (10 coefficients), 'wxo_cov',
            'order', 'n_coefficients'}``.

        Notes
        -----
        - Inconsistency: ``order`` is stored but not used to size the fit.
        - Inconsistency: ``term_values['slice_no']`` is the FIELD slice number
          (``fslice_no``) here, whereas ``'slice_no'`` elsewhere denotes the
          encoded universal slice number.
        """
        n_coefficients = Globals.surface_fit_n_coeffs
        slice_no = term_values['slice_no']
        spifu_no = term_values['spifu_no']
        x = np.array(term_values['pri_ang'])
        y = np.array(term_values['ech_ang'])
        wxo = np.array(term_values['w_bs']) * np.array(term_values['ech_ords'])
        wxo_opt, wxo_cov = curve_fit(PolyFit.surface_model,
                                     xdata=(x, y), ydata=list(wxo), p0=[0]*n_coefficients)
        wxo_fit = {'slice_no': slice_no, 'spifu_no': spifu_no,
                   'wxo_opt': wxo_opt, 'wxo_cov': wxo_cov,
                   'order': order, 'n_coefficients': n_coefficients}
        return wxo_fit

    @staticmethod
    def make_fit_matrix(poly_matrix, pa, ea):
        """Evaluate one fitted 4 x 4 distortion matrix at an arbitrary (pa, ea).

        ``matrix[i, j] = surface_model((pa, ea), *poly_matrix[i, j])`` for all
        ``i, j < Globals.svd_order``.

        Parameters
        ----------
        poly_matrix : numpy.ndarray
            Shape ``(4, 4, 10)`` surface coefficients for one matrix, e.g.
            ``find_slice_fit(...)['a']`` or ``term_fits[slice_no]['a']`` from
            ``Filer.read_fit_parameters``.
        pa : float
            Prism angle in degrees.
        ea : float
            Echelle angle in degrees.

        Returns
        -------
        numpy.ndarray
            ``(svd_order, svd_order)`` matrix usable in
            ``Util.apply_svd_distortion``.

        Notes
        -----
        - Inconsistency: the previous docstring said "2nd order 2D polynomial";
          the surface is cubic (``Globals.surface_fit_order = 4``).
        - Clean-up: ``lmsdist.run`` (block ``evaluate_transforms``) duplicates
          this loop inline instead of calling it.
        """
        svd_order = Globals.svd_order
        svd_shape = svd_order, svd_order
        matrix = np.zeros(svd_shape)
        for i in range(0, svd_order):
            for j in range(0, svd_order):
                fit_terms = list(poly_matrix[i, j])
                matrix[i, j] = PolyFit.surface_model((pa, ea), *fit_terms)
        return matrix

    @staticmethod
    def make_fit_transform(configs, fit_data):
        """Build a ``Transform`` from the surface fits for a given configuration.  Dead code.

        Parameters
        ----------
        configs : tuple
            ``(lms_config, slice_config)`` dicts with ``pri_ang, ech_ang`` and
            ``slice_no, spifu_no`` respectively.
        fit_data : tuple
            ``(wpa_fit, wxo_fit, term_fits)`` as returned by
            ``Filer.read_fit_parameters``.

        Returns
        -------
        Transform
            With matrices from :meth:`make_fit_matrix`.

        Notes
        -----
        - Bug: indexes ``term_fits[slice_no][spifu_no][mat_name]`` but
          ``Filer.read_fit_parameters`` returns ``term_fits[slice_no][mat_name]``
          (no spifu level) -> ``KeyError``.
        - Bug: ``slice_config['spifu_no']`` is not a key of
          ``Globals.slice_config_template``.
        - Clean-up: only called from :meth:`make_mfp_projection`, itself dead;
          ``lmsdist.run`` builds the equivalent inline.
        """
        wpa_fit, wxo_fit, term_fits = fit_data
        lms_config, slice_config = configs
        pa, ea = lms_config['pri_ang'], lms_config['ech_ang']
        slice_no, spifu_no = slice_config['slice_no'], slice_config['spifu_no']

        matrices = {}
        for mat_name in Globals.matrix_names:
            poly_matrix = term_fits[slice_no][spifu_no][mat_name]
            matrix = PolyFit.make_fit_matrix(poly_matrix, pa, ea)
            matrices[mat_name] = matrix
        transform = Transform(lms_config=lms_config, slice_config=slice_config, matrices=matrices)
        return transform

    @staticmethod
    def make_mfp_projection(slice_no, spifu_no, trace, svd_transform, wxo_fit, term_fits):
        """Project one slice's rays to the MFP via Zemax, SVD and fitted transforms.  Dead code.

        Intended to return ``{'zemax': ..., 'svd': ..., 'fit': ...}`` MFP point
        dicts for comparison plots.

        Parameters
        ----------
        slice_no, spifu_no : int
            Slice identifiers (``slice_no`` is immediately overwritten from
            ``svd_transform``).
        trace : RayTrace
        svd_transform : Transform
        wxo_fit : dict
            Unused.
        term_fits : dict

        Returns
        -------
        dict

        Notes
        -----
        - Bug: ``svd_transform.configuration`` does not exist
          (``Transform.lms_configuration`` / ``slice_configuration`` do).
        - Bug: ``trace.get(...)`` does not exist (``RayTrace.get_series``
          does) and the series names ``'det_x'`` / ``'det_y'`` were renamed
          ``'mfp_x'`` / ``'mfp_y'``.
        - Bug: :meth:`make_fit_transform` is called with ``(svd_cfg,
          term_fit)`` instead of ``(configs, fit_data)`` tuples.
        - Clean-up: not called anywhere; superseded by the
          ``evaluate_transforms`` block of ``lmsdist.run``.
        """

        svd_cfg = svd_transform.configuration
        svd_slice_config = svd_transform.slice_configuration
        slice_no = svd_slice_config['slice_no']

        term_fit = term_fits[slice_no][spifu_no]
        fit_transform = PolyFit.make_fit_transform(svd_cfg, term_fit)

        slice_id = {'slice_no': slice_no, 'spifu_no': spifu_no}
        efp_x = trace.get('efp_x', **slice_id)
        efp_y = trace.get('efp_y', **slice_id)
        efp_w = trace.get('wavelength', **slice_id)
        efp_points = {'efp_y': efp_y, 'efp_x': efp_x, 'efp_w': efp_w}  # Centre of slice for detector
        mfp_points_zemax = {'mfp_x': trace.get('det_x', **slice_id),
                            'mfp_y': trace.get('det_y', **slice_id)}
        mfp_points_svd = Util.efp_to_mfp(svd_transform, efp_points)
        mfp_points_fit = Util.efp_to_mfp(fit_transform, efp_points)
        projection = {'zemax': mfp_points_zemax, 'svd': mfp_points_svd[0], 'fit': mfp_points_fit[0]}
        return projection
