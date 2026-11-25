"""
Image-quality metrics for stacks of LMS PSF images (LSF, FWHM, enslitted energy, Strehl).

Purpose
-------
Pure-numpy / photutils / scipy routines that turn a list of PSF images -- or the
pseudo-cubes assembled by :mod:`lmsiq_cuber` -- into image-quality numbers: Gaussian and
linear-interpolated FWHM of a collapsed profile, the normalised line spread function
(LSF) sampled with fractional-pixel rectangular apertures, enslitted / encircled energy
(EE) curves, a relative 'Strehl' against a perfect-model image, the local dispersion from
an ``lmsdist`` trace, and a number of legacy helpers for the pixel-phase study.

Belongs to the ``lmsiq`` program (called from :mod:`lmsiq_cuber`, :mod:`lmsiq_phase`
and :mod:`lms_baffle`).  Everything is static; the class is instantiated once only to
set ``Analyse.fwhm_sigma``.

Image-stack convention
----------------------
Throughout, a list or stack of images is ordered so that index 0 is the *perfect*
optical model, index 1 the *as-designed* model and indices 2.. the Monte-Carlo
as-built instances (``MC_<n>``).  Monte-Carlo statistics are always formed over
``[:, 2:]``.

Input data
----------
Images in memory.  In ``lmsiq`` these are either the 128 x 128 pixel Zemax PSFs at 4.5
um sampling (4x oversampled 18 um H2RG pixels, from
``../data/iq/<opticon>/<date_stamp>/``), the same after :meth:`lms_ipc.Ipc.apply`, or
their 32 x 32 down-sampled versions; the ``oversample`` keyword tells the routines how
to rescale their x axes to detector pixels.  ``get_dispersion`` needs the list of
:class:`lmsdist_trace.Trace` objects from the ``lmsdist`` trace pickle.

Outputs
-------
None written to disk; all methods return arrays or dictionaries.

Notes
-----
* Bug: ``Analyse.fwhm_sigma`` is ``None`` until ``Analyse()`` has been called, and
  :meth:`Analyse._gauss` divides by it; calling any fitting method on a fresh import
  raises ``TypeError``.
* Bug: :meth:`Analyse.get_dispersion` passes the whole ``ds_dict`` to
  ``Trace.get_series(tag, series_filter)``, which indexes ``trace.series[key]`` for
  every key of the filter; ``ds_dict`` contains keys such as ``'folder'`` and
  ``'file_names'`` that are not trace series, so a ``KeyError`` is expected.
* Bug: :meth:`Analyse.find_strehls` normalises to the image in slot 0 (the perfect
  model), so the returned quantity is a peak-to-total ratio relative to the perfect
  model, not a Strehl ratio, and is undefined when no perfect model is present.
* Bug: :meth:`Analyse._find_fwhm_lin` resets ``yh`` to 0.0 before returning, so the
  reported half-maximum level is always zero.
* Bug: :meth:`Analyse._find_line_widths` stores the perfect-model width under the
  ``'design'`` key.
* Inconsistency: :meth:`Analyse.eed` accepts but ignores a ``normalise`` keyword that
  :mod:`lmsiq_cuber` passes as ``'to_average'``; the curves are always normalised by
  the image total.
* Clean-up: ``photutils`` is imported as a module but only the two explicit imports
  are used.  :meth:`Analyse.ycompress` is O(n^4) in Python loops and unused.
"""
import numpy as np
import math
import photutils
from photutils.centroids import centroid_com
from photutils.aperture import RectangularAperture, CircularAperture
from scipy.optimize import curve_fit


class Analyse:
    """
    Static image-quality metric functions for LMS PSF image stacks.

    Attributes
    ----------
    fwhm_sigma : float or None
        Conversion factor ``FWHM / sigma = 2 sqrt(2 ln 2) = 2.3548`` for a Gaussian.
        ``None`` until the class has been instantiated once.

    Notes
    -----
    * Bug: the class attribute is set in ``__init__`` rather than in the class body,
      so static use without instantiation fails in :meth:`_gauss`.
    """

    fwhm_sigma = None

    def __init__(self):
        """
        Set ``Analyse.fwhm_sigma = 2 sqrt(2 ln 2)``.

        Notes
        -----
        * Clean-up: the only reason to instantiate the class; ``lmsiq.main`` and
          ``lms_baffle.main`` do so for this side effect.
        """
        Analyse.fwhm_sigma = 2 * math.sqrt(2.0 * math.log(2.0))
        return

    @staticmethod
    def extract_cube_strips(image_list, oversampling):
        """
        Collapse each image over a central column band to give one along-slice strip.

        For the first image the column range is set from its width ``nzc``:
        ``zc_cen = nzc // 2``, ``zc_hwid = int(0.5 * 2.5 * oversampling)`` and columns
        ``[zc_cen - zc_hwid, zc_cen + zc_hwid)`` are summed along axis 1.  Every image
        in the list is collapsed over the same columns.  In :meth:`lmsiq_cuber.Cuber.build`
        the resulting strips become one slice column of the pseudo-cube.

        Parameters
        ----------
        image_list : list of ndarray, each shape (n_rows, n_cols)
            Images for one slice, ordered perfect, design, MC.. (see module docstring).
        oversampling : int
            Nominal image pixels per detector pixel used to size the band.

        Returns
        -------
        strips : ndarray, shape (n_images, n_rows)
            Column-summed profiles, one row per image.

        Notes
        -----
        * Bug: ``Cuber.build`` calls this on images that have already been
          down-sampled to detector pixels but still passes ``oversampling = 4``, so
          ``zc_hwid = 5`` and 10 detector columns are summed, whereas the comment
          "spectral width per slice = 2.5 pix" implies a 2-3 detector pixel band.
        * Inconsistency: the band is centred on the geometric image centre, not on the
          PSF centroid.
        """
        first_obs = True
        n_obs = len(image_list)
        c1, c2 = 0, 0
        strips = None
        for i, image in enumerate(image_list):
            if first_obs:
                nzr, nzc = image.shape
                zc_cen = int(nzc / 2)
                spec_width_pix = 2.5  # Spectral width per slice for reconstructed cubes.
                zc_hwid = int(0.5 * spec_width_pix * oversampling)
                c1, c2 = zc_cen - zc_hwid, zc_cen + zc_hwid
                strips = np.zeros((n_obs, nzr))
                first_obs = False
            strip = np.sum(image[:, c1:c2], axis=1)
            strips[i, :] = strip
        return strips

    @staticmethod
    def _gauss(x, amp, fwhm, xpk):
        """
        Three-parameter Gaussian parameterised by FWHM, with no baseline term.

        ``y = amp * exp(-0.5 * ((x - xpk) / sigma)^2)`` with
        ``sigma = fwhm / Analyse.fwhm_sigma``.

        Parameters
        ----------
        x : ndarray
            Abscissa.
        amp : float
            Peak value.
        fwhm : float
            Full width at half maximum, same units as ``x``.
        xpk : float
            Peak position.

        Returns
        -------
        y : ndarray
            Gaussian evaluated at ``x``.

        Notes
        -----
        * Bug: raises ``TypeError`` if ``Analyse.fwhm_sigma`` is still ``None``
          (class never instantiated).
        * Inconsistency: no constant/baseline term, so a profile with a non-zero
          background biases the fitted FWHM upward.  The pipeline's
          ``drl.lines.fit_gaussian`` includes a baseline.
        """
        sigma = fwhm / Analyse.fwhm_sigma
        k = (x - xpk) / sigma
        y = amp * np.exp((-k ** 2) / 2.0)
        return y

    @staticmethod
    def get_dispersion(ds_dict, ray_traces):
        """
        Estimate the linear dispersion dw/dx (um per mm) at a data set's wavelength.

        Algorithm: pick the trace whose ``wave_reference`` is closest to
        ``ds_dict['wavelength']``; pull its ``'wavelength'`` and ``'mfp_x'`` ray series
        (filtered by ``ds_dict``); form first differences ``dw / dx`` between
        consecutive rays; sort by wavelength and linearly interpolate to the target
        wavelength.  :meth:`lmsiq_cuber.Cuber.build` multiplies the result by the 18 um
        pixel pitch to obtain nm per detector pixel (um/mm x um = nm/pixel).

        Parameters
        ----------
        ds_dict : dict
            Data-set dictionary from :meth:`lmsiq_image_manager.ImageManager.load_dataset`;
            ``'wavelength'`` (um) is used, and the whole dictionary is passed to
            ``Trace.get_series`` as the ray filter.
        ray_traces : list of lmsdist_trace.Trace
            Traces from the ``lmsdist`` trace pickle (``dist_filer.trace_file``).

        Returns
        -------
        dw_dx : float
            Dispersion in um per mm at the mosaic focal plane.

        Notes
        -----
        * Bug: ``Trace.get_series(tag, series_filter)`` iterates over every key of
          ``series_filter`` and indexes ``self.series[key]``; ``ds_dict`` contains keys
          (``'folder'``, ``'config_no'``, ``'file_names'``, ``'debug'``, ...) that are not
          ray-series columns, so a ``KeyError`` is expected.  Whether this ever ran with
          an earlier ``Trace`` API is an open question.
        * Inconsistency: consecutive rays in the series are not guaranteed to be
          adjacent in x, so individual ``dw/dx`` differences can be noisy or of either
          sign; only the interpolated value at ``wave`` is used.
        * Inconsistency: the original docstring spoke of interpolating ``dw_dx_mean``
          values over all traces; the code uses a single nearest trace.
        """
        wave = ds_dict['wavelength']

        # wave_list, dw_dx_mean_list, dw_dx_std_list = [], [], []
        dw_min, trace_dw_min = 999., None
        for trace in ray_traces:
            wave_ref = trace.wave_reference
            dw = abs(wave_ref - wave)
            if dw < dw_min:
                trace_dw_min = trace
                dw_min = dw
        waves = trace_dw_min.get_series('wavelength', {})
        mfp_x = trace_dw_min.get_series('mfp_x', {})
        dw_dx = (waves[1:] - waves[:-1]) / (mfp_x[1:] - mfp_x[:-1])
        ws = waves[1:]
        sort_indices = np.argsort(ws)
        w_sort = ws[sort_indices]
        dw_dx_sort = dw_dx[sort_indices]
        dw_dx = np.interp(wave, w_sort, dw_dx_sort)
        return dw_dx

    @staticmethod
    def find_fwhm_multi(image_list, **kwargs):
        """
        Average the Gaussian and linear FWHM results of :meth:`find_fwhm` over images.

        Images whose Gaussian fit fails are skipped entirely; images whose linear
        half-max search fails contribute to the Gaussian statistics only.

        Parameters
        ----------
        image_list : list of ndarray
            Images to measure (normally the Monte-Carlo subset ``images[2:]``).
        **kwargs
            Passed to :meth:`find_fwhm` (``oversample``, ``axis``).

        Returns
        -------
        gauss : tuple
            ``(False, mean_fit, std_fit)`` where ``mean_fit`` / ``std_fit`` are the mean
            and standard deviation over images of the ``[amp, fwhm, xpk]`` fit vectors.
        linear : tuple
            ``(False, fwhm_lin_mean, fwhm_lin_std, xl_mean, xr_mean, yh_mean)``.

        Notes
        -----
        * Inconsistency: the error flag in both tuples is hard-wired ``False`` even if
          every image failed (in which case ``np.mean`` of an empty array returns
          ``nan`` with a warning).
        * Bug: ``yh_mean`` is always 0.0 because :meth:`_find_fwhm_lin` returns
          ``yh = 0``.
        * Clean-up: ``debug`` is accepted via ``**kwargs`` by callers but not used here.
        """

        fits, fit_errs, fwhms_lin, xls, xrs, yhs = [], [], [], [], [], []
        for image in image_list:
            gauss, linear = Analyse.find_fwhm(image, **kwargs)
            is_error_gau, fit, fit_err = gauss
            if is_error_gau:
                continue
            fits.append(fit)
            fit_errs.append(fit_err)

            is_error_lin, fwhm_lin, _, xl, xr, yh = linear
            if is_error_lin:
                continue
            xls.append(xl)
            xrs.append(xr)
            yhs.append(yh)
            fwhms_lin.append(fwhm_lin)

        fits = np.array(fits)
        mean_fit = np.mean(fits, axis=0)
        mean_fit_err = np.std(fits, axis=0)
        gauss = False, mean_fit, mean_fit_err

        xl_mean = np.mean(np.array(xls))
        xr_mean = np.mean(np.array(xrs))
        yh_mean = np.mean(np.array(yhs))
        fwhm_lin_mean = np.mean(np.array(fwhms_lin))
        fwhm_lin_err = np.std(np.array(fwhms_lin))
        linear = False, fwhm_lin_mean, fwhm_lin_err, xl_mean, xr_mean, yh_mean

        return gauss, linear

    @staticmethod
    def find_fwhm(image, **kwargs):
        """
        Measure the FWHM of an image profile collapsed around the geometric centre.

        Algorithm:

        1. ``half_aperture = 2 * oversample``; rows (or columns) ``rc_lo:rc_hi`` with
           ``rc_lo = n_cols // 2 - half_aperture`` and width ``2 * half_aperture + 1``
           (i.e. 4 detector pixels + 1 image pixel) are selected around the *geometric*
           image centre.
        2. For ``axis = 0`` the selected rows are averaged to give a profile against
           column index; for ``axis = 1`` the selected columns are averaged to give a
           profile against row index.
        3. ``x = arange(n_pts) / oversample`` converts the abscissa to detector pixels.
        4. :meth:`_fit_gaussian` and :meth:`_find_fwhm_lin` are applied to ``(x, y)``.

        Parameters
        ----------
        image : ndarray, shape (n_rows, n_cols)
            Input image.
        **kwargs
            ``oversample`` : int, default 1 -- image pixels per detector pixel.
            ``axis`` : int, default 0 -- 0 = profile along columns (spectral /
            across-slice direction), 1 = profile along rows (spatial / along-slice).
            ``debug`` is accepted by callers but ignored here.

        Returns
        -------
        gauss : tuple
            ``(is_error, fit, fit_err)`` with ``fit = [amp, fwhm, xpk]`` in detector
            pixels and ``fit_err`` the square root of the covariance diagonal.
        linear : tuple
            ``(is_error, fwhm_lin, 0.0, xl, xr, yh)`` from :meth:`_find_fwhm_lin`;
            ``fwhm_lin = xr - xl``.

        Notes
        -----
        * Inconsistency: ``rc_lo`` is computed from ``n_cols`` for both axes, so the
          method assumes a square image; the band is centred on the image centre, not
          the PSF centroid.
        * Inconsistency: :meth:`lmsiq_phase.Phase.process` calls this with
          ``oversample = 4`` on an already down-sampled image, which scales the FWHM
          by 1/4 and widens the band to 9 detector pixels.
        * Bug: when the Gaussian fit fails ``covar`` is a zero array of shape (4, 3)
          from :meth:`_fit_gaussian`; ``np.diag`` still yields three zeros so no
          exception is raised, but the errors are meaningless.
        """
        oversample = kwargs.get('oversample', 1)
        axis = kwargs.get('axis', 0)

        nr5, nc5 = image.shape
        half_aperture = 2 * oversample
        rc_lo = int(nc5 / 2) - half_aperture
        rc_hi = rc_lo + 2 * half_aperture + 1

        n_rows, n_cols = image.shape
        n_pts = n_cols if axis == 0 else n_rows
        sub_image = image[rc_lo:rc_hi] if axis == 0 else image[:, rc_lo:rc_hi]
        y = np.mean(sub_image, axis=axis)
        x = np.arange(0.0, float(n_pts)) / oversample        # Find FWHM in detector units
        is_error_gau, fit, covar = Analyse._fit_gaussian(x, y)
        cv1 = np.array(covar)
        cv2 = np.diag(cv1)
        fit_err = np.sqrt(cv2)
        gauss = is_error_gau, fit, fit_err

        is_error_lin, xl, xr, yh = Analyse._find_fwhm_lin(x, y)
        fwhm_lin = xr - xl
        fwhm_lin_err = 0.
        linear = is_error_lin, fwhm_lin, fwhm_lin_err, xl, xr, yh
        return gauss, linear

    @staticmethod
    def find_profile_fwhm(x, y):
        """
        Gaussian and linear FWHM of an already extracted 1-D profile.

        Parameters
        ----------
        x : ndarray
            Abscissa (detector pixels in :meth:`lsf`).
        y : ndarray
            Profile values.

        Returns
        -------
        gauss : tuple
            ``(is_error, [amp, fwhm, xpk], fit_err)``.
        linear : tuple
            ``(is_error, xr - xl, 0.0, xl, xr, yh)``.

        Notes
        -----
        * Clean-up: identical to the tail of :meth:`find_fwhm`; the two could share
          code.
        """
        is_error_gau, fit, covar = Analyse._fit_gaussian(x, y)
        cv1 = np.array(covar)
        cv2 = np.diag(cv1)
        fit_err = np.sqrt(cv2)
        gauss = is_error_gau, fit, fit_err

        is_error_lin, xl, xr, yh = Analyse._find_fwhm_lin(x, y)
        fwhm_lin = xr - xl
        fwhm_lin_err = 0.
        linear = is_error_lin, fwhm_lin, fwhm_lin_err, xl, xr, yh
        return gauss, linear

    @staticmethod
    def _fit_gaussian(x, y, **kwargs):
        """
        Least-squares fit of the baseline-free Gaussian :meth:`_gauss` to a profile.

        Algorithm: initial guess ``p0 = [y[argmax(y)], 2.0, x[argmax(y)]]`` (peak
        value, FWHM of 2.0 abscissa units, peak position); ``scipy.optimize.curve_fit``
        with ``method='trf'`` and ``xtol=1e-8``, no bounds, no weights.  On
        ``RuntimeError`` (no convergence) a message is printed and zero arrays are
        returned with ``is_error = True``.

        Parameters
        ----------
        x : ndarray
            Abscissa.
        y : ndarray
            Profile values.
        **kwargs
            ``debug`` : bool -- print guess and fit.

        Returns
        -------
        is_error : bool
            True if ``curve_fit`` raised ``RuntimeError``.
        fit : ndarray, shape (3,)
            ``[amp, fwhm, xpk]``; zeros on error.
        covar : ndarray
            Covariance matrix, shape (3, 3) on success; shape (4, 3) of zeros on error.

        Notes
        -----
        * Inconsistency: the error-path ``covar`` shape (4, 3) does not match the
          success-path (3, 3).
        * Inconsistency: the FWHM guess of 2.0 is in whatever units ``x`` carries
          (detector pixels after rescaling, image pixels otherwise).
        * Clean-up: only ``RuntimeError`` is caught; ``ValueError`` from NaN input or
          ``OptimizeWarning`` for a singular covariance propagate.
        """
        debug = kwargs.get('debug', False)
        imax = np.argmax(y)
        guess_amp, guess_fwhm, guess_xpk = y[imax], 2.0, x[imax]
        guess = [guess_amp, guess_fwhm, guess_xpk]
        is_error = False
        try:
            fit, covar = curve_fit(Analyse._gauss, x, y,
                                   p0=guess, method='trf', xtol=1E-8)
        except RuntimeError:
            print('lmsiq_analyse._fit_gaussian !! Runtime error !!')
            is_error = True
            order = len(guess) + 1
            fit, covar = np.zeros(order-1), np.zeros((order, order-1))
        if debug:
            fmt = "{:<8s} - amp={:8.6f} fwhm={:5.2f} xpk={:9.6f}"
            print(fmt.format('Guess', guess_amp, guess_fwhm, guess_xpk))
            print(fmt.format('Fit  ', fit[0], fit[1], fit[2]))
            print()
        return is_error, fit, covar

    @staticmethod
    def _find_fwhm_lin(x, y):
        """
        FWHM from the outermost half-maximum crossings, by linear interpolation.

        Algorithm: ``yh = max(y) / 2``; ``yz = y - yh``; ``il`` and ``ir`` are the
        first and last indices with ``yz > 0``; the left crossing is interpolated
        linearly between samples ``il - 1`` and ``il``, the right crossing between
        ``ir`` and ``ir + 1``.  Referred to as the 'linear' FWHM in the summary
        outputs, as opposed to the Gaussian-fit FWHM.

        Parameters
        ----------
        x : ndarray
            Abscissa, assumed monotonically increasing.
        y : ndarray
            Profile values.

        Returns
        -------
        is_error : bool
            True if ``ir + 1`` ran off the end of the array (``IndexError``).
        xl : float
            Left half-maximum crossing (0.0 on error).
        xr : float
            Right half-maximum crossing (0.0 on error).
        yh : float
            Always 0.0 (see Notes).

        Notes
        -----
        * Bug: ``yh`` is reset to 0.0 after being used, so the returned half-maximum
          level is always zero.
        * Bug: if ``il == 0`` the expression ``x[il - 1]`` wraps to the last sample
          without raising, giving a nonsensical ``xl`` with ``is_error = False``.
        * Inconsistency: using the first and last samples above half maximum makes the
          width sensitive to noise spikes or secondary maxima (any wing sample above
          ``yh`` extends the FWHM); the method is intended for smooth model PSFs.
        """
        ymax = np.amax(y)
        yh = ymax / 2.0
        yz = np.subtract(y, yh)       # Search for zero crossing
        iz = np.where(yz > 0)
        il = iz[0][0]
        ir = iz[0][-1]
        xl, xr, yh = 0., 0., 0.
        is_error = False
        try:
            xl = x[il-1] - yz[il-1] * (x[il] - x[il-1]) / (yz[il] - yz[il-1])
            xr = x[ir] - yz[ir] * (x[ir+1] - x[ir]) / (yz[ir+1] - yz[ir])
        except IndexError:
            is_error = True
        return is_error, xl, xr, yh

    @staticmethod
    def find_phases(data_id, centroids):
        """
        Legacy: convert centroid positions to pixel phases relative to the perfect model.

        ``phase_ref = ndet_cols / 2``; for each model column ``col >= 1``,
        ``phases[:, col] = centroids[:, col] - centroids[:, 0] - phase_ref``;
        ``phase_diffs`` holds the row-to-row differences of those phases.  Column 0 of
        both outputs is a copy of ``centroids[:, 0]`` (the perfect-model positions).

        Parameters
        ----------
        data_id : tuple
            Six-element legacy identifier; only the last element (``ndet_cols``) is used.
        centroids : ndarray, shape (n_shifts, n_models)
            Centroid positions; column 0 = perfect model.

        Returns
        -------
        phases : ndarray, shape (n_shifts, n_models)
        phase_abs_diffs : ndarray, shape (n_shifts, n_models)
            ``abs(phase_diffs)``; last row of columns >= 1 is zero.

        Notes
        -----
        * Clean-up: unused at HEAD (legacy pixel-phase study).
        """
        _, _, _, _, _, ndet_cols = data_id
        phase_ref = ndet_cols / 2.0
        _, n_mcruns = centroids.shape

        phases, phase_diffs = np.zeros(centroids.shape), np.zeros(centroids.shape)
        for col in range(1, n_mcruns):
            phases[:, col] = centroids[:, col] - centroids[:, 0] - phase_ref
        phase_diffs[:-1, 1:] = phases[1:, 1:] - phases[:-1, 1:]
        phase_diffs[:, 0], phases[:, 0] = centroids[:, 0], centroids[:, 0]
        return phases, np.abs(phase_diffs)

    @staticmethod
    def find_phot(image, **kwargs):
        """
        Simple photometry: full-image sum or fractional-pixel rectangular aperture.

        Parameters
        ----------
        image : ndarray
            Input image.
        **kwargs
            ``method`` : {'aperture', 'full_image'}, default ``'fixed_aperture'``.
            ``method_parameters`` : tuple ``(ap_pos, width, height)`` for
            ``'aperture'``; ``ap_pos`` is ``(x, y)`` in pixel index units.

        Returns
        -------
        phot : float
            Aperture or total signal; 0.0 if the method is unrecognised or the aperture
            parameters are missing (a message is printed in the latter case).

        Notes
        -----
        * Inconsistency: the default ``method`` value ``'fixed_aperture'`` matches
          neither branch, so calling without ``method`` silently returns 0.0.
        * Clean-up: :meth:`lmsiq_phase.Phase.process` always uses ``'full_image'``.
        """
        method = kwargs.get('method', 'fixed_aperture')
        method_parameters = kwargs.get('method_parameters', None)
        phot = 0.
        if method == 'aperture':
            if method_parameters is not None:
                ap_pos, width, height = method_parameters
                rect_aper = RectangularAperture(ap_pos, w=width, h=height)
                phot = Analyse.exact_rectangular(image, rect_aper)
            else:
                print("!! Analyse.find_phot aperture parameter not set !!")
        if method == 'full_image':
            phot = np.sum(image)
        return phot

    @staticmethod
    def subtract_phases(values):
        """
        Legacy: subtract column 0 (perfect model) from every other column, in place.

        Parameters
        ----------
        values : ndarray, shape (n_rows, n_runs)
            Modified in place.

        Returns
        -------
        values : ndarray
            The same array.

        Notes
        -----
        * Clean-up: unused at HEAD.
        """
        _, n_runs = values.shape
        for col in range(1, n_runs):
            values[:, col] = values[:, col] - values[:, 0]
        return values

    @staticmethod
    def find_stats(data_list):
        """
        Legacy: mean and rms of centroid shift differences per configuration.

        For each entry the sub-block ``centroid_abs_diffs[1:12, 1:]`` (rows 1-11, all
        non-perfect models) is averaged.

        Parameters
        ----------
        data_list : list of tuple
            Each ``(data_id, zemax_configuration, centroids, photometry,
            centroid_abs_diffs)``; ``zemax_configuration[1]`` is the wavelength.

        Returns
        -------
        stats : ndarray, shape (n_configs, 3)
            Columns: wavelength, mean absolute difference, standard deviation.

        Notes
        -----
        * Inconsistency: the row range ``1:12`` is hard-coded and tied to the legacy
          21-step shift grid.  Unused at HEAD.
        """
        n_configs = len(data_list)
        stats = np.zeros((n_configs, 3))
        for i, data in enumerate(data_list):
            data_id, zemax_configuration, centroids, photometry, centroid_abs_diffs = data
            wave = zemax_configuration[1]
            abs_diffs = centroid_abs_diffs[1:12, 1:]
            mean_abs_diff = np.mean(abs_diffs)
            mean_abs_diff_err = np.std(abs_diffs)
            stats[i, :] = [wave, mean_abs_diff, mean_abs_diff_err]
        return stats

    @staticmethod
    def fix_offset(centroids):
        """
        Legacy: remove a step between two hard-coded row blocks of a centroid table.

        For each model column ``run >= 1`` the difference between the means of rows
        ``3:20`` and ``23:40`` is subtracted from rows ``3:20``, in place.

        Parameters
        ----------
        centroids : ndarray, shape (>= 40, n_runs)
            Modified in place.

        Returns
        -------
        centroids : ndarray
            The same array.

        Notes
        -----
        * Inconsistency: the row blocks are hard-coded for a legacy 40+ step shift
          scan; rows 0-2 and 20-22 are untouched.  Unused at HEAD.
        """
        _, n_cols = centroids.shape
        for run in range(1, n_cols):
            mean_offset = np.mean(centroids[3:20, run]) - np.mean(centroids[23:40, run])
            centroids[3:20, run] -= mean_offset
        return centroids

    @staticmethod
    def find_centroid(image):
        """
        Centre-of-mass centroid of an image (``photutils.centroids.centroid_com``).

        Parameters
        ----------
        image : ndarray

        Returns
        -------
        centroid : ndarray, shape (2,)
            ``(x, y)`` in pixel index units (x = column).

        Notes
        -----
        * Clean-up: thin wrapper, unused at HEAD.
        """
        centroid = centroid_com(image)
        return centroid

    @staticmethod
    def _find_mean_centroid(images):
        """
        Centroid of the mean image of a stack.

        The images are stacked, averaged along axis 0 and the centre of mass of the
        average is returned, so all models in a stack share one reference position.

        Parameters
        ----------
        images : list of ndarray, all the same shape

        Returns
        -------
        centroid : ndarray, shape (2,)
            ``(x, y)`` of the mean image.

        Notes
        -----
        * Inconsistency: a common centroid means the per-model LSFs in :meth:`lsf`
          include any model-to-model centroid shift as an apparent broadening or offset.
        """
        cube = np.array(images)
        image_ave = np.average(cube, axis=0)
        centroid = centroid_com(image_ave)
        return centroid

    @staticmethod
    def find_phase_shift_rates(centroids):
        """
        Legacy: row-to-row derivative of each model's centroid against the perfect model.

        ``psr[i, j] = (c[i, j] - c[i-1, j]) / (c[i, 0] - c[i-1, 0])`` for ``i, j >= 1``;
        column 0 is copied and row 0 is zero.

        Parameters
        ----------
        centroids : ndarray, shape (n_rows, n_cols)

        Returns
        -------
        phase_shift_rates : ndarray, shape (n_rows, n_cols)

        Notes
        -----
        * Clean-up: pure-Python double loop; unused at HEAD.
        """
        nrows, ncols = centroids.shape
        phase_shift_rates = np.zeros(centroids.shape)
        phase_shift_rates[:, 0] = centroids[:, 0]
        for i in range(1, nrows):
            for j in range(1, ncols):
                psr = (centroids[i, j] - centroids[i-1, j]) / (centroids[i, 0] - centroids[i-1, 0])
                phase_shift_rates[i, j] = psr
        return phase_shift_rates

    @staticmethod
    def find_strehls(cube_stack):
        """
        Relative peak-to-total ratio of each model image, normalised to the perfect model.

        Algorithm, for each image ``run_idx`` of the stack:

        1. ``isum = sum(image)``.
        2. Find the brightest pixel ``(rmax, cmax)`` and take the column ``image[:,
           cmax]`` through it as a 1-D profile against row index.
        3. Fit the baseline-free Gaussian (:meth:`_fit_gaussian`) and take its
           amplitude ``imax``.
        4. For the first image (index 0, the perfect model) store
           ``p_factor = isum / imax``.
        5. ``strehl = (imax / isum) * p_factor``, so the perfect model is exactly 1.0 by
           construction and every other model is its peak/total ratio relative to the
           perfect model's.

        Parameters
        ----------
        cube_stack : ndarray, shape (n_models, n_rows, n_cols)
            Pseudo-cube from :meth:`lmsiq_cuber.Cuber.build`: index 0 = perfect model,
            1 = as-designed, 2.. = Monte-Carlo instances.

        Returns
        -------
        strehls : ndarray, shape (n_models,)
            Relative 'Strehl' per model; element 0 is 1.0.

        Notes
        -----
        * Bug: this is not a Strehl ratio.  It is a 1-D Gaussian amplitude divided by
          the image total, normalised to the same quantity for image 0; it is
          meaningless if image 0 is not a diffraction-limited perfect model, and the
          ``p_factor`` is taken from whatever image happens to be first.
        * Inconsistency: the profile is a single column through the brightest pixel of
          a 32-row x (2*slice_radius + 7)-column pseudo-cube plane whose columns are
          slices, so the 'peak' is the along-slice profile of the brightest slice.
        * Inconsistency: ``np.argmax(image, keepdims=True)`` returns a (1, 1) array so
          ``rmax``, ``cmax`` are arrays; indexing still works after ``np.squeeze``.
        * Clean-up: the fit error flag from :meth:`_fit_gaussian` is discarded; a
          failed fit gives ``imax = 0`` and ``strehl = 0`` (or a division by zero for
          image 0).
        """
        n_runs, n_rows, n_cols = cube_stack.shape
        strehls = np.zeros(n_runs)
        p_factor = None
        rows = np.arange(0, n_rows)
        first_time = True
        for run_idx in range(0, n_runs):
            image = cube_stack[run_idx]
            isum = np.sum(image)
            rmax, cmax = np.unravel_index(np.argmax(image, keepdims=True), image.shape)
            x = rows
            y = np.squeeze(image[:, cmax])
            _, gauss_fit, _ = Analyse._fit_gaussian(x, y)
            imax = gauss_fit[0]
            if first_time:
                psum = isum
                pmax = imax
                p_factor = psum / pmax
                first_time = False
            strehl = (imax / isum) * p_factor
            # print("{:03d}, {:8.5f}, {:8.5f}, {:6.3f}".format(run_idx, imax, isum, strehl))
            strehls[run_idx] = strehl
        return strehls

    @staticmethod
    def eed(image_list, cube_dict, axis_name, **kwargs):
        """
        Enslitted (or encircled) energy versus aperture half-size for every image.

        Algorithm:

        1. ``r_max = min(nx, ny) / 2 - 1`` (aperture kept inside the image);
           ``radii = arange(0.1, r_max, 0.1)``.  If ``log10sampling`` (default True)
           the same number of points is remapped to be uniform in ``log10(r)`` between
           0.1 and ``r_max``.
        2. For each image ``j``: ``centroid = centroid_com(image)``; for each radius
           ``r`` build the aperture centred on the centroid:

           * ``axis_name == 'radial'``: ``CircularAperture(centroid, r)`` measured with
             ``photutils`` ``do_photometry`` (encircled energy);
           * ``'spectral'`` or ``'across-slice'``: ``RectangularAperture(w=2r,
             h=2*r_max)`` -- a slit of half-width ``r`` in x, full height in y;
           * ``'spatial'`` or ``'along-slice'``: ``RectangularAperture(w=2*r_max,
             h=2r)`` -- full width in x, half-height ``r`` in y;

           rectangular apertures are summed with :meth:`exact_rectangular`.
        3. Each image's curve is divided by ``np.sum(image)`` (the image total, *not*
           the largest aperture), so ``EE -> 1`` only if all flux lies within
           ``r_max`` of the centroid.
        4. If ``cube_dict['mc_bounds']`` is not None, the Monte-Carlo mean and rms
           curves are formed over columns ``[:, 2:]`` (indices 0 and 1 being the
           perfect and as-designed models).
        5. ``xvals = radii / oversample`` converts to detector pixels.

        Parameters
        ----------
        image_list : ndarray or list of ndarray, shape (n_models, ny, nx)
            Image stack; index 0 = perfect, 1 = design, 2.. = MC.  In ``Cuber.build``
            this is the pseudo-cube (axis 1 along-slice rows, axis 2 slices).
        cube_dict : dict
            Needs ``'mc_bounds'`` (None or ``(lo, hi)``) and, for debug output,
            ``'name'``.
        axis_name : {'radial', 'spectral', 'across-slice', 'spatial', 'along-slice'}
            Aperture growth direction (see step 2).
        **kwargs
            ``debug`` : bool, default False.
            ``log10sampling`` : bool, default True.
            ``oversample`` : float, default 1.0 -- image pixels per detector pixel.
            ``normalise`` : accepted by callers but ignored (see Notes).

        Returns
        -------
        ees_data : dict
            ``'xvals'`` : ndarray (n_points,) aperture half-size in detector pixels;
            ``'yvals'`` : ndarray (n_points, n_models) normalised EE curves;
            ``'ees_mc_mean'``, ``'ees_mc_rms'`` : ndarray (n_points,) or 0.0 if no MC.

        Notes
        -----
        * Inconsistency: the ``normalise='to_average'`` keyword passed by
          :meth:`lmsiq_cuber.Cuber.build` is never read; curves are always normalised
          by the image total.
        * Inconsistency: for the pseudo-cube, x is the slice index and y the
          along-slice detector row, so ``'across-slice'`` EE is measured in units of
          slices (20.7 mas) and ``'along-slice'`` in detector pixels (8.7 mas); the
          rectangular aperture's fixed dimension ``2 * r_max`` is the same number in
          both cases even though the cube is far from square (32 rows x ~15 columns).
        * Open question: an unknown ``axis_name`` leaves ``aperture = None`` and
          :meth:`exact_rectangular` raises ``AttributeError``.
        * Clean-up: ``im_sums`` and ``im_peaks`` are computed but not returned; the
          out-of-bounds warning is only printed in debug mode.
        """
        debug = kwargs.get('debug', False)
        is_log = kwargs.get('log10sampling', True)  # Use sampling equispaced in log10
        oversample = kwargs.get('oversample', 1.)

        ny, nx = image_list[0].shape
        nxy_min = nx if nx < ny else ny

        n_obs = len(image_list)
        r_sample = 0.1
        r_start = r_sample
        # Set maximum radial size of aperture (a bit less than the full image size to avoid OOB errors)
        r_max = (nxy_min / 2.) - 1.

        radii = np.arange(r_start, r_max, r_sample)
        n_points = radii.shape[0]
        if is_log:      # Remap the same number of points onto a log uniform scale.
            k = math.log10(r_max / r_start) / (n_points - 1)
            lr = math.log10(r_start)
            for i in range(0, n_points):
                radii[i] = math.pow(10., lr)
                lr += k

        ees_all = np.zeros((n_points, n_obs))
        im_sums = np.zeros(n_obs)
        im_peaks = np.zeros(n_obs)
        if debug:
            print()

        for j, image in enumerate(image_list):
            # file_name = obs_dict['file_names'][j]
            img_min, img_max = np.amin(image), np.amax(image)
            centroid = centroid_com(image)
            umin, vmin = centroid[0] - r_max, centroid[1] - r_max
            umax, vmax = centroid[0] + r_max, centroid[1] + r_max
            is_x_oob = umin < 0 or umax > nx - 1
            is_y_oob = vmin < 0 or vmax > ny - 1
            if debug:
                fmt = "\r- Cube {:s}, processing image {:d} {:10.1e} {:10.1e}"
                print(fmt.format(cube_dict['name'], j, img_min, img_max))
                if is_x_oob or is_y_oob:
                    txt = "!! U/V out of bounds (signal truncated) !! - "
                    fmt = "u={:5.1f} -{:5.1f}, v={:5.1f} -{:5.1f}"
                    txt += fmt.format(umin, umax, vmin, vmax)
                    print(txt)

            for i in range(0, n_points):        # One radial point per row
                r = radii[i]      # Increase aperture width to measure spectral cog profile
                aperture = None
                if axis_name == 'radial':
                    aperture = CircularAperture(centroid, r)
                    signal_list = aperture.do_photometry(image)
                    signal = signal_list[0][0]
                else:
                    if axis_name in ['spectral', 'across-slice']:
                        aperture = RectangularAperture(centroid, w=2.*r, h=2.*r_max)     # Spectral
                    if axis_name in ['spatial', 'along-slice']:
                        aperture = RectangularAperture(centroid, w=2.*r_max, h=2.*r)     # Spatial
                    signal = Analyse.exact_rectangular(image, aperture)
                ees_all[i, j] = signal

            enorm = np.sum(image)
            ees_all[:, j] = np.divide(ees_all[:, j], enorm)
            im_sums[j] = ees_all[n_points-1, j]
            im_peaks[j] = img_max

        # Rescale x axis from image scale to LMS pixels
        # ees_perfect, ees_design = ees_all[:, 0], ees_all[:, 1]
        # mc_bounds = obs_dict['mc_bounds']
        ees_mc_mean, ees_mc_rms = 0., 0.
        if cube_dict['mc_bounds'] is not None:
            ees_mcs = ees_all[:, 2:]
            ees_mc_mean = np.mean(ees_mcs, axis=1)
            ees_mc_rms = np.std(ees_mcs, axis=1)
        xdet = np.divide(radii, oversample)         # Convert x scale to LMS detector pixels
        ees_data = {'xvals': xdet, 'yvals': ees_all,
                    'ees_mc_mean': ees_mc_mean, 'ees_mc_rms': ees_mc_rms}
        return ees_data

    @staticmethod
    def find_ee_axis_references(wav, ee_data):
        """
        Legacy: interpolate the perfect, design and MC-mean EE curves at a reference size.

        The reference aperture half-size is ``x_ref = 2.0 * max(1, wav / 3.7)`` detector
        pixels, i.e. 2 pixels up to 3.7 um and growing with the diffraction limit
        beyond.  Each curve is linearly interpolated at ``x_ref`` using the first sample
        with ``x > x_ref`` and its predecessor (or the last two samples if none).

        Parameters
        ----------
        wav : float
            Wavelength in um.
        ee_data : tuple
            Legacy 7-tuple ``(axis, x, ee_per, ee_des, ee_mc, ee_mc_err, ee_mc_all)`` --
            *not* the dictionary returned by :meth:`eed`.

        Returns
        -------
        x_ref : float
            Reference half-size in detector pixels.
        ee_refs : dict
            ``'ee_<axis4>_ref'`` : ``x_ref``; ``'ee_<axis4>_per'``, ``'_des'``,
            ``'_mean'`` : interpolated EE values, where ``<axis4>`` is the first four
            letters of the axis name.

        Notes
        -----
        * Inconsistency: the input tuple layout matches the legacy
          :meth:`lmsiq_summariser.Summariser.read_profiles` output, not :meth:`eed`;
          unused at HEAD.
        """
        axis, x, ee_per, ee_des, ee_mc, ee_mc_err, ee_mc_all = ee_data

        axis_tag = '_' + axis[0:4]
        ee_tag = 'ee' + axis_tag + '_'

        x_ref_min = 2.
        broadening = 1.0 if wav < 3.7 else wav / 3.7
        x_ref = x_ref_min * broadening

        ees = {ee_tag + 'per': ee_per,
               ee_tag + 'des': ee_des,
               ee_tag + 'mean': ee_mc}

        ee_refs = {ee_tag + 'ref': x_ref}
        iz = np.where(x > x_ref)
        is_index = len(iz[0])
        i = iz[0][0] if is_index else len(x) - 1
        for key in ees:
            ee = ees[key]
            ee_val = ee[i - 1] + (x_ref - x[i - 1]) * (ee[i] - ee[i - 1]) / (x[i] - x[i - 1])
            ee_refs[key] = ee_val
        return x_ref, ee_refs

    @staticmethod
    def ycompress(cube_in, factor):
        """
        Legacy: rebin a 4-D cube along its row axis by a (possibly non-integer) factor.

        For every layer, observation, column and output row a 1 x ``factor`` pixel
        ``RectangularAperture`` centred at ``(col + 0.5, factor * (row_out + 0.5))`` is
        summed with :meth:`exact_rectangular`.

        Parameters
        ----------
        cube_in : ndarray, shape (n_layers, n_obs, n_rows_in, n_cols)
        factor : float
            Compression factor along rows.

        Returns
        -------
        cube_out : ndarray, shape (n_layers, n_obs, int(n_rows_in / factor) + 1, n_cols)

        Notes
        -----
        * Clean-up: four nested Python loops creating a photutils aperture per output
          pixel (O(n^4)); unused at HEAD.
        * Inconsistency: the ``+ 0.5`` column centre offset, combined with the
          ``pco = 0.5`` convention in :meth:`exact_rectangular`, places the aperture on
          pixel boundaries rather than pixel centres.
        """
        n_layers, n_obs, n_rows_in, n_cols = cube_in.shape
        n_rows_out = int(n_rows_in / factor) + 1
        shape_out = n_layers, n_obs, n_rows_out, n_cols
        cube_out = np.zeros(shape_out)
        ap_width = 1.
        for i in range(0, n_layers):
            for j in range(0, n_obs):
                image = cube_in[i, j, :, :]
                for col in range(0, n_cols):
                    ap_centre_col = col + .5
                    for row_out in range(0, n_rows_out):
                        ap_centre_row = factor * (row_out + 0.5)
                        ap_centre = ap_centre_col, ap_centre_row
                        ap_height = factor
                        aperture = RectangularAperture(ap_centre, w=ap_width, h=ap_height)
                        cube_out[i, j, row_out, col] = Analyse.exact_rectangular(image, aperture)
        return cube_out

    @staticmethod
    def _boxcar(series_in, box_width):
        """
        Running-mean smooth of each column of a 2-D array along axis 0.

        Sample ``i`` is replaced by the mean over ``[i - hw, i + hw)`` with
        ``hw = int(box_width / 2)``, clipped to the array bounds.

        Parameters
        ----------
        series_in : ndarray, shape (n_pts, n_series)
        box_width : float
            Box width in samples (the ``oversample`` factor in :meth:`lsf`).

        Returns
        -------
        series_out : ndarray, shape (n_pts, n_series)

        Notes
        -----
        * Inconsistency: the window ``[i - hw, i + hw)`` is asymmetric (one more
          sample on the low side) and its width is ``2 * hw`` rather than
          ``box_width``.
        """
        n_pts, _ = series_in.shape
        series_out = np.zeros(series_in.shape)
        box_hw = int(box_width / 2.)
        for i in range(0, n_pts):
            i1, i2 = i - box_hw, i + box_hw
            i1 = i1 if i1 >= 0 else 0
            i2 = i2 if i2 < n_pts else n_pts
            series_out[i, :] = np.mean(series_in[i1:i2, :], axis=0)
        return series_out

    @staticmethod
    def lsf(image_list, cube_dict, axis, **kwargs):
        """
        Normalised line spread function of every image in a stack, plus FWHM estimates.

        Algorithm:

        1. Geometry.  ``n_rows, n_cols`` from the first image.  For ``axis = 0``
           (spectral / across-slice, profile along x = columns): ``vcoadd = n_rows - 1``
           if ``v_coadd == 'all'`` else ``v_coadd``; ``uradius = (n_cols - 1) / 2`` if
           ``u_radius == 'all'`` else ``u_radius``.  For ``axis = 1`` (spatial /
           along-slice, profile along y = rows) the roles of rows and columns swap.
           Profile positions ``uvals = arange(-uradius, uradius + 1, 1.0)`` -- one
           sample per image pixel, in image pixels, relative to the centroid.
        2. Reference position.  If ``centroid_relative`` (default True) the common
           centroid of the *mean* image (:meth:`_find_mean_centroid`) is used for all
           models; otherwise (0, 0).
        3. Sampling.  For each image and each ``u``, a ``RectangularAperture`` of width
           1 image pixel along the profile axis and ``vcoadd`` pixels across it is
           placed at ``centroid + u`` and summed with :meth:`exact_rectangular`.
        4. Normalisation.  Each profile is divided by its own maximum
           (``lsf_all / amax(lsf_all, axis=0)``), so every model peaks at 1.0 and the
           LSFs carry no throughput information.
        5. ``xvals = uvals / oversample`` converts to detector pixels; if ``boxcar``
           the normalised profiles are smoothed with :meth:`_boxcar` of width
           ``oversample``.
        6. For every profile ``j`` the Gaussian and linear FWHM are measured on
           ``(xvals, lsf_norm[:, j])`` with :meth:`find_profile_fwhm`; the Gaussian
           half-maximum positions are ``xcen +- fwhm / 2``.

        Parameters
        ----------
        image_list : list of ndarray or ndarray, shape (n_models, n_rows, n_cols)
            Image stack; index 0 = perfect, 1 = design, 2.. = MC.  ``Cuber.build`` calls
            this once on the IPC-processed 128 x 128 Zemax images of the target slice
            (``axis=0, oversample=4, v_coadd=12.0``) and once per axis on the
            down-sampled pseudo-cube (``oversample=1, v_coadd=3.0``).
        cube_dict : dict
            Needs ``'mc_bounds'``.
        axis : {0, 1}
            0 = profile along columns (spectral / across-slice); 1 = along rows.
        **kwargs
            ``debug`` : bool (unused).
            ``centroid_relative`` : bool, default True.
            ``oversample`` : float, default 1.0 -- image pixels per detector pixel.
            ``boxcar`` : bool, default False.
            ``v_coadd`` : float or 'all', default 'all' -- aperture extent across the
            profile, in image pixels.
            ``u_radius`` : float or 'all', default 'all' -- half-extent of the profile,
            in image pixels.

        Returns
        -------
        lsf_data : dict
            ``'xvals'`` : ndarray (n_points,) position relative to the centroid, in
            detector pixels;
            ``'yvals'`` : ndarray (n_points, n_models) peak-normalised LSFs;
            ``'lin_fwhm'``, ``'lin_xl'``, ``'lin_xr'`` : lists (n_models) from
            :meth:`_find_fwhm_lin`, detector pixels;
            ``'gau_fwhm'``, ``'gau_xl'``, ``'gau_xr'``, ``'gau_amp'`` : lists
            (n_models) from the Gaussian fit.

        Notes
        -----
        * Bug: the model-tag loop uses the stale variable ``i`` left over from the
          sampling loop (``mc_bounds[0] + i``, ``pd_tags[i]``) instead of ``j``; the
          resulting ``model_tags`` list is wrong but is never used or returned.
        * Inconsistency: with ``u_radius='all'`` the aperture centres run from
          ``centroid - (n-1)/2`` to ``centroid + (n-1)/2`` and so extend past the image
          edge whenever the centroid is off-centre; :meth:`exact_rectangular` clips them.
        * Inconsistency: for the pseudo-cube, ``axis = 0`` is *across* slices (x =
          slice index) while for the raw Zemax image ``axis = 0`` is the spectral
          direction; ``Cuber.build`` stores the former under ``'acr_'`` and the latter
          under ``'lsf_'`` keys.
        * Clean-up: the ``debug`` keyword is read but unused; the original docstring
          was truncated mid-sentence and described a return tuple rather than the
          dictionary.
        """
        debug = kwargs.get('debug', False)
        centroid_relative = kwargs.get('centroid_relative', True)
        oversample = kwargs.get('oversample', 1.)   # Pixel oversampling with respect to detector
        boxcar = kwargs.get('boxcar', False)        # t=Apply boxcar filter of width = oversample
        v_coadd = kwargs.get('v_coadd', 'all')      # Number of image pixels to coadd orthogonal to profile
        u_radius = kwargs.get('u_radius', 'all')    # Maximum radial size of aperture (a bit less than 1/2 image size)
        usample = 1.0                               # Sample psf once per pixel to avoid steps
        ustart = 0.0                                # Offset from centroid

        n_rows, n_cols = image_list[0].shape

        vcoadd = n_cols - 1 if v_coadd == 'all' else v_coadd
        uradius = (n_rows - 1) / 2.0 if u_radius == 'all' else u_radius
        if axis == 0:
            vcoadd = n_rows - 1 if v_coadd == 'all' else v_coadd
            uradius = (n_cols - 1) / 2.0 if u_radius == 'all' else u_radius

        uvals = np.arange(ustart - uradius, ustart + uradius + 1., usample)
        n_points, = uvals.shape

        n_files = len(image_list)
        lsf_all = np.zeros((n_points, n_files))

        centroid = np.zeros((2,))
        if centroid_relative:
            centroid = Analyse._find_mean_centroid(image_list)

        for j, image in enumerate(image_list):
            ap_pos = np.array(centroid)
            ucen = centroid[0] if axis == 0 else centroid[1]
            us = np.add(uvals, ucen)
            for i in range(0, n_points):  # One radial point per row
                u = us[i]
                if axis == 0:       # spectral or across-slice
                    ap_pos[0] = u
                    aperture = RectangularAperture(ap_pos, w=usample, h=vcoadd)
                    lsf_all[i, j] = Analyse.exact_rectangular(image, aperture)
                if axis == 1:       # Spatial or along-slice
                    ap_pos[1] = u
                    aperture = RectangularAperture(ap_pos, w=vcoadd, h=usample)  # Spatial
                    lsf_all[i, j] = Analyse.exact_rectangular(image, aperture)
        lsf_norm = lsf_all / np.amax(lsf_all, axis=0)
        xvals = np.divide(uvals, oversample)         # Convert x scale to LMS detector pixels
        if boxcar:
            lsf_norm = Analyse._boxcar(lsf_norm, oversample)

        lsf_data = {'xvals': xvals, 'yvals': lsf_norm}
        mc_bounds = cube_dict['mc_bounds']
        pd_tags, model_tags = ['perfect', 'design'], []
        lsf_data['lin_fwhm'], lsf_data['lin_xl'], lsf_data['lin_xr'] = [], [], []
        lsf_data['gau_fwhm'], lsf_data['gau_xl'], lsf_data['gau_xr'], lsf_data['gau_amp'] = [], [], [], []

        n_pts, n_profiles = lsf_all.shape
        for j in range(0, n_profiles):
            mc_tag = '' if mc_bounds is None else "MC_{:03d}".format(mc_bounds[0] + i)
            key = mc_tag if i > 1 else pd_tags[i]
            model_tags.append(key)

            gauss, linear = Analyse.find_profile_fwhm(xvals, lsf_norm[:, j])
            # print('Analyse.lsf ', gauss)
            is_error_lin, fwhm_per_lin, _, xl_lin, xr_lin, _ = linear
            is_error_gau, (amp_gau, fwhm_per_gau, xcen_gau), covar = gauss
            xl_gau = xcen_gau - 0.5 * fwhm_per_gau
            xr_gau = xcen_gau + 0.5 * fwhm_per_gau
            # Scale to detector pixels and write to data dictionary
            lsf_data['lin_fwhm'].append(fwhm_per_lin)
            lsf_data['lin_xl'].append(xl_lin)
            lsf_data['lin_xr'].append(xr_lin)
            lsf_data['gau_fwhm'].append(fwhm_per_gau)
            lsf_data['gau_xl'].append(xl_gau)
            lsf_data['gau_xr'].append(xr_gau)
            lsf_data['gau_amp'].append(amp_gau)

        return lsf_data

    @staticmethod
    def _find_line_widths(image_list, axis):
        """
        Legacy: linear FWHM of the perfect, design and mean Monte-Carlo images.

        Parameters
        ----------
        image_list : list of ndarray
            Index 0 = perfect, 1 = design, 2.. = MC.
        axis : {0, 1}
            Profile axis, passed to :meth:`find_fwhm`.

        Returns
        -------
        line_widths : dict
            ``'perfect'``, ``'design'``, ``'mc_mean'`` : ``[fwhm_lin, xl, xr]``.

        Notes
        -----
        * Bug: ``line_widths['design']`` is assigned ``fwhm_per_lin`` (the perfect
          model's width) instead of ``fwhm_des_lin``; the design FWHM is measured and
          discarded.
        * Clean-up: the Gaussian results are computed and dropped; unused at HEAD.
        """
        line_widths = {}
        per_gauss, per_linear = Analyse.find_fwhm(image_list[0], axis=axis, debug=True)
        _, fwhm_per_lin, _, xl, xr, yh = per_linear
        line_widths['perfect'] = [fwhm_per_lin, xl, xr]

        des_gauss, des_linear = Analyse.find_fwhm(image_list[1], axis=axis, debug=True)
        _, fwhm_des_lin, _, xl, xr, yh = des_linear
        line_widths['design'] = [fwhm_per_lin, xl, xr]

        # Find line widths of Monte-Carlo data
        mc_gauss, mc_linear = Analyse.find_fwhm_multi(image_list[2:], axis=axis, debug=True)
        _, mc_fit, mc_fit_err = mc_gauss
        # fwhm_mc_gau, fwhm_mc_gau_err = mc_fit[1], mc_fit_err[1]
        _, fwhm_lin_mc, fwhm_lin_mc_err, xl, xr, yh = mc_linear
        line_widths['mc_mean'] = [fwhm_lin_mc, xl, xr]
        return line_widths

    @staticmethod
    def exact_rectangular(image, aperture):
        """
        Sum the image inside a rectangular aperture with fractional-pixel edge weights.

        Pixel convention: pixel ``i`` is centred on index ``i`` and spans
        ``[i - 0.5, i + 0.5]`` (``pco = 0.5``).  Algorithm:

        1. Aperture edges ``x1 = cx - w/2 + pco``, ``x2 = x1 + w``, ``y1 = cy - h/2 +
           pco``, ``y2 = y1 + h`` (``cx, cy`` from ``aperture.positions``, ``w, h`` from
           ``aperture.w``, ``aperture.h``).  In these shifted coordinates ``int(x1)``
           is the index of the first pixel touched and ``int(x2)`` the last.
        2. ``c1, c2, r1, r2 = int(x1), int(x2), int(y1), int(y2)``, clipped to
           ``[0, n-1]``; if ``x1 > n_cols`` return 0.0.
        3. Build a weight array of ones over the ``(r2-r1+1) x (c2-c1+1)`` sub-image;
           multiply the first column by ``fc1 = 1 - (x1 - c1)`` (fraction of the first
           pixel inside), the last column by ``fc2 = x2 - c2``; if only one column,
           by ``fc1 + fc2 - 1``.  Same for rows with ``fr1``, ``fr2``.
        4. Return ``sum(sub_image * weights)``.

        Parameters
        ----------
        image : ndarray, shape (n_rows, n_cols)
        aperture : photutils.aperture.RectangularAperture
            Only ``positions`` (``(x, y)``), ``w`` and ``h`` are read; ``theta`` is
            ignored (no rotation).

        Returns
        -------
        f : float
            Weighted sum.

        Notes
        -----
        * Bug: apertures extending below index 0 are mishandled: ``c1`` (or ``r1``)
          is clipped to 0 but ``fc1 = 1 - (x1 - 0)`` then exceeds 1 for ``x1 < 0``,
          over-weighting the first column/row; and ``int()`` truncates toward zero so
          ``int(-0.3) == 0`` rather than -1.  Only ``x1 > n_cols`` is trapped on the
          high side, and no ``y`` bounds are trapped at all.
        * Inconsistency: after clipping, ``fc2 = x2 - c2`` can exceed 1 when the
          aperture extends past the last column.
        * Clean-up: functionally equivalent to
          ``aperture.to_mask(method='exact').multiply(image).sum()`` for unrotated
          apertures fully inside the image.
        """
        pco = 0.5       # Pixel centre offset,

        cen = aperture.positions
        w = aperture.w
        h = aperture.h
        x1 = cen[0] - (w / 2.0) + pco
        x2 = x1 + w
        y1 = cen[1] - (h / 2.0) + pco
        y2 = y1 + h
        c1, c2, r1, r2 = int(x1), int(x2), int(y1), int(y2)
        c1, r1 = max(0, c1),max(0, r1)
        rmax, cmax = image.shape
        c2, r2 = min(cmax-1, c2), min(rmax-1, r2)
        # Catch off image apertures
        if x1 > cmax:
            return 0.

        # Number of rows and columns in subarray, 1 pixel extra to allow sub-pixel fragments.
        nr = r2 - r1 + 1
        nc = c2 - c1 + 1
        wts = np.ones((nr, nc))
        im = image[r1:r1+nr, c1:c1+nc]

        fc1 = 1. - (x1 - c1)
        fc2 = x2 - c2
        if nc == 1:
            fc = fc1 + fc2 - 1.
            wts[:, 0] *= fc
        else:
            wts[:, 0] *= fc1
            wts[:, nc-1] *= fc2

        fr1 = 1. - (y1 - r1)
        fr2 = y2 - r2
        if nr == 1:
            fr = fr1 + fr2 - 1.
            wts[0, :] *= fr
        else:
            wts[0, :] *= fr1
            wts[nr-1, :] *= fr2

        wtim = np.multiply(im, wts)
        f = np.sum(wtim)
        return f
