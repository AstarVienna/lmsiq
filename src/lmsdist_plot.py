#!/usr/bin/env python
"""Matplotlib diagnostic plots for the ``lmsdist`` distortion-fitting program.

Purpose
-------
Static plotting helpers used while fitting the E-TNT-ATC-MET-1058 distortion
transforms: wavelength coverage / dispersion per configuration
(:meth:`Plot.series`), the prism-angle-versus-wavelength fit
(:meth:`Plot.wave_v_prism_angle`), 3-D views of the ``n*lambda`` and
matrix-term surface fits (:meth:`Plot.wxo_fit`, :meth:`Plot.transform_fit`),
and ray-trace versus transform comparisons in the mosaic focal plane
(:meth:`Plot.plot_mfp_points`).  ``lmsdist_trace.RayTrace`` and
``lmsdist_polyfit.PolyFit`` call into this module; ``lmsaiv`` has its own
``lmsaiv_plot`` module.

Program
-------
``lmsdist`` (all plots are gated by ``Globals.is_debug(...)`` in
``lmsdist.run`` and ``PolyFit``).

Input data
----------
In-memory objects only: ``RayTrace`` traces (``trace.series``,
``trace.transforms``, ``trace.lms_config``), fit dictionaries from
``PolyFit`` and ``term_values`` dicts from ``lmsdist_util.Util.get_term_values``.
No files are read.

Outputs
-------
None on disk.  Every public routine ends in ``matplotlib.pyplot.show()``
(via :meth:`Plot.show`), which blocks in an interactive backend, and returns
``None`` (``transform_fit`` also returns its residual dict).

Notes
-----
- Bug: :meth:`Plot.efficiency_v_wavelength` calls ``ax.plot_focal_planes``,
  which is not a ``matplotlib.axes.Axes`` method -> ``AttributeError``.
- Bug: :meth:`Plot.series` with ``colour_by='slice'`` reads
  ``Globals.n_slices``, which does not exist (``Globals.n_fslices`` does) ->
  ``AttributeError``; and then indexes ``rgb[slice_no - 1]`` with ``rgb`` still
  ``None``.
- Bug: :meth:`Plot.series` passes ``transform.slice_configuration`` (keys
  ``slice_no, ech_ord, w_min, w_max``) as the filter to
  ``RayTrace.get_series``, which looks every key up in ``trace.series``;
  ``w_min`` is not a series -> ``KeyError`` once
  ``RayTrace.find_wavelength_bounds`` has run (always, in ``lmsdist.run``).
- Bug: :meth:`Plot.plot_mfp_points` reads ``mfp_plot['spifu_no']`` but the
  dict built in ``lmsdist.run`` (``mfp_plot_points``) has no such key ->
  ``KeyError`` at debug level ``'low'``.
- Bug: :meth:`Plot.round_trip` calls ``set_plot_area(title, ...)`` with a
  positional argument (the function accepts keywords only) and expects a bare
  ``ax_list`` return where a ``(fig, ax_list)`` tuple is returned.
- Clean-up: every function calls ``plt.show()`` inside the compute path; for
  reuse as report generators they would need to return ``Figure`` objects.
- Clean-up: :meth:`Plot.show` re-imports ``matplotlib.pyplot`` locally
  although it is already imported at module level.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from lms_globals import Globals
from lmsdist_util import Util


class Plot:
    """Namespace of static plotting routines for ``lmsdist`` (see module docstring).

    Notes
    -----
    - Clean-up: instances carry no state; ``efficiency_v_wavelength`` is the
      only non-static method and only uses ``self`` to reach static methods.
    """

    def __init__(self):
        """Create a ``Plot`` object.  No state is initialised."""
        return

    @staticmethod
    def set_plot_area(**kwargs):
        """Create a figure with an ``nrows x ncols`` grid of axes and common decorations.

        Wraps ``plt.subplots(..., squeeze=False)`` so ``ax_list`` is always
        2-D, sets the global font size, applies common ``xlim`` / ``ylim`` /
        aspect to every axis, labels only the bottom-left axis, and optionally
        removes axes listed in ``remplots``.

        Parameters
        ----------
        figsize : list of float, optional
            Inches, default ``[9, 6]``.
        xlim, ylim : sequence of float, optional
            Common axis limits (default none).
        xlabel, ylabel : str, optional
            Applied to axis ``[nrows-1, 0]`` only.
        ncols, nrows : int, optional
            Grid size, default 1 x 1.
        remplots : array_like, optional
            ``(k, 2)`` list of ``[row, col]`` axes to remove.
        aspect : str or float, optional
            Passed to ``Axes.set_aspect``; default ``'auto'``.
        fontsize : float, optional
            Sets ``plt.rcParams['font.size']`` globally; default 16.
        sharex, sharey : bool, optional
            Passed to ``plt.subplots``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax_list : numpy.ndarray
            2-D array of ``Axes`` of shape ``(nrows, ncols)``.

        Notes
        -----
        - Inconsistency: ``RayTrace.plot_slice_map`` and :meth:`round_trip`
          call this with a positional title and expect only ``ax_list`` back;
          both would fail (``TypeError``).
        - Clean-up: modifies the global ``rcParams`` font size as a side
          effect.
        """
        figsize = kwargs.get('figsize', [9, 6])
        xlim = kwargs.get('xlim', None)            # Common limits for all plots
        ylim = kwargs.get('ylim', None)            # Common limits for all plots
        xlabel = kwargs.get('xlabel', '')          # Common axis labels
        ylabel = kwargs.get('ylabel', '')
        ncols = kwargs.get('ncols', 1)             # Number of plot columns
        nrows = kwargs.get('nrows', 1)
        remplots = kwargs.get('remplots', None)
        aspect = kwargs.get('aspect', 'auto')      # 'equal' for aspect = 1.0
        fontsize = kwargs.get('fontsize', 16)

        plt.rcParams.update({'font.size': fontsize})

        sharex = kwargs.get('sharex', False)
        sharey = kwargs.get('sharey', False)
        fig, ax_list = plt.subplots(nrows, ncols, figsize=figsize,
                                    sharex=sharex, sharey=sharey,
                                    squeeze=False)
        fig.patch.set_facecolor('white')

        for i in range(0, nrows):
            for j in range(0, ncols):
                ax = ax_list[i,j]
                ax.set_aspect(aspect)       # Set equal axes
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if (i == nrows-1 and j == 0):
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
        if remplots is not None:
            rps = np.atleast_2d(remplots)
            for i in range(0, len(rps)):
                ax_list[rps[i,0], rps[i, 1]].remove()
        return fig, ax_list

    def efficiency_v_wavelength(self, weoas, **kwargs):
        """Plot echelle blaze efficiency versus wavelength, one curve per order.  Broken.

        Parameters
        ----------
        weoas : tuple
            ``(waves, effs, orders, angles)`` as returned by
            ``lms_efficiency.Efficiency.test``; ``waves`` and ``effs`` have
            shape ``(n_angles, n_orders)``.
        xlim : sequence of float, optional
            Wavelength limits in microns (default data range).
        ylim : sequence of float, optional
            Default ``[0, 1]``.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            Always: ``ax.plot_focal_planes`` is not a ``matplotlib`` ``Axes``
            method (``ax.plot`` was evidently intended).

        Notes
        -----
        - Bug: ``ax.plot_focal_planes(...)`` -> ``AttributeError``; looks like
          an accidental rename of ``ax.plot``.
        - Clean-up: only called from the study script ``lms_test.py``.
        """

        waves, effs, orders, angles = weoas
        n_angs, n_orders = waves.shape
        xlim_default = [np.min(waves), np.max(waves)]
        ylim_default = [0.0, 1.0]
        xlim = kwargs.get('xlim', xlim_default)
        ylim = kwargs.get('ylim', ylim_default)
        xcoverage = xlim[1] - xlim[0]
        xtick_spacing = 0.05 if xcoverage < 1.0 else 0.2
        fig, ax_list = self.set_plot_area(xlim=xlim, xlabel='Wavelength [um]',
                                          ylim=ylim, ylabel='Efficiency')
        fig.suptitle('Echelle eficiency')
        ax = ax_list[0, 0]
        xtick_vals = np.arange(xlim[0], xlim[1], xtick_spacing)
        xtick_labels = []
        for v in xtick_vals:
            xtl = "{:10.2f}".format(v)
            xtick_labels.append(xtl)
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels)
        colours = Plot._make_colours(n_orders)
        for i in range(0, n_orders):
            order = orders[i]
            x = waves[:, i]
            y = effs[:, i]
            col = colours[i]
            ax.plot_focal_planes(x, y, clip_on=True, ls='-', lw=5.0, color=col)
            jmid = int(n_angs / 2)
            xt, yt = x[jmid], y[jmid]
            ax.text(xt, yt, "{:3d}".format(order), color=col, ha='left', va='bottom')
        self.show()
        return

    @staticmethod
    def _make_colours(n_colours):
        """Return ``n_colours`` RGB triples by stepping (r, g, b) += (0.9, 0.3, 0.5) mod 1.

        Parameters
        ----------
        n_colours : int

        Returns
        -------
        list of list of float
            RGB values in [0, 1); the first colour is ``[0.9, 0.3, 0.5]``, not
            black (black is only the starting point before the first step).

        Notes
        -----
        - Inconsistency: the comment "Always start with black" is not what the
          loop does; the increment is applied before the first colour is
          appended.
        """
        colours = []
        r, g, b = 0.0, 0.0, 0.0         # Always start with black
        for i in range(0, n_colours):
            r += 0.9
            g += 0.3
            b += 0.5
            r = r if r < 1.0 else r - 1.0
            g = g if g < 1.0 else g - 1.0
            b = b if b < 1.0 else b - 1.0
            colours.append([r, g, b])
        return colours

    @staticmethod
    def make_rgb_gradient(vals):
        """Map values linearly onto a blue -> green -> red colour ramp.

        With ``f = (vals - min) / (max - min)``: ``R = f``, ``G = sin(pi f)``,
        ``B = 1 - f``.

        Parameters
        ----------
        vals : array_like
            Values to colour-code (e.g. wavelengths or slice indices).

        Returns
        -------
        numpy.ndarray
            Shape ``(len(vals), 3)`` RGB array.

        Raises
        ------
        ZeroDivisionError / RuntimeWarning
            If all values are equal (``max == min``); with numpy arrays this
            yields NaN colours rather than an exception.
        """
        n_pts = len(vals)
        rgb = np.zeros((n_pts, 3))

        w_min = np.amin(vals)
        w_max = np.amax(vals)
        r_min, r_max = 0., 1.
        b_min, b_max = 1., 0.

        f = (vals - w_min) / (w_max - w_min)
        rgb[:, 0] = r_min + f * (r_max - r_min)
        rgb[:, 1] = np.sin(f * math.pi)
        rgb[:, 2] = b_min + f * (b_max - b_min)
        return rgb

    @staticmethod
    def wave_v_prism_angle(wpa_fit, wpa_model, ea_zero_waves, ea_zero_pas, all_boresights,
                           differential=False, echelle_angles=False):
        """Plot the prism angle = f(wavelength) fit and its residuals (two stacked panels).

        Upper panel: the ``(ea_zero_waves, ea_zero_pas)`` fit points as open
        circles and the polynomial ``wpa_model(x, *wpa_fit['wpa_opt'])`` as a
        line, with the fitted coefficients written into the title.  Lower
        panel: ``data - fit`` residuals.

        Parameters
        ----------
        wpa_fit : dict
            ``{'wpa_opt': coefficients (ascending power), ...}`` from
            ``PolyFit.create_polynomial_fit``.
        wpa_model : callable
            ``PolyFit.poly_model`` or compatible ``f(x, *coeff)``.
        ea_zero_waves, ea_zero_pas : array_like
            Per-order wavelength (microns) and prism angle (deg) at echelle
            angle zero, the data that were fitted.
        all_boresights : numpy.ndarray
            ``(n_configs, 4)`` rows ``(wave, pri_ang, ech_ang, ech_ord)``; only
            used when ``echelle_angles`` is True.
        differential : bool, optional
            Only changes the y-axis label text; no derivative is computed.
        echelle_angles : bool, optional
            If True, over-plot ``pri_ang + ech_ang`` versus wavelength for each
            order from ``all_boresights``.

        Returns
        -------
        None

        Notes
        -----
        - Inconsistency: ``differential=True`` relabels the axis as a
          sensitivity (micron/deg) but the plotted quantity is unchanged.
        - Clean-up: the coefficient string in the title labels the linear term
          with a bare ``lambda`` and higher terms with ``lambda^i``, but the
          constant term gets no label and the ordering logic sets ``lam_text``
          one iteration late.
        """
        x, y = np.array(ea_zero_waves), np.array(ea_zero_pas)

        fig, [ax1, ax2] = plt.subplots(figsize=(12, 8), ncols=1, nrows=2, sharex=True)
        ylabel = 'Prism rotation sensitivity micron / deg' if differential else 'Prism rotation angle / deg'
        xlabel = "Wavelength / " + r'$\mu$m'

        coeffs = wpa_fit['wpa_opt']
        y_fit = wpa_model(x, *coeffs)
        form_text = r', $\phi_{prism} = $'
        lam_text = ''
        for i, coeff in enumerate(coeffs):
            if i > 1:
                lam_text = r'$\lambda^{' + str(i) + r'}$'
            form_text += "{:+6.3f}{:s}".format(coeff, lam_text) # + plus_text
            lam_text = r'$\lambda$'

        title1 = 'Prism wavelength best fit' + form_text
        ax1.set_title(title1, loc='left')
        ax1.set_ylabel(ylabel)
        ax1.plot(x, y, color='blue', clip_on=True,
                 fillstyle='none', marker='o', mew=2., ms=10., linestyle='None')
        ax1.plot(x, y_fit, color='blue', clip_on=True, lw=2., linestyle='solid')
        if echelle_angles:
            ech_orders = all_boresights[:, 3]
            for ech_order in np.unique(all_boresights[:, 3]):
                idx = ech_orders == ech_order
                x_bs = all_boresights[idx, 0]
                y_bs = all_boresights[idx, 1] + all_boresights[idx, 2]
                ax1.plot(x_bs, y_bs, marker='o', fillstyle='full', color='green', ms=2., linestyle='solid')

        ax1.grid(True)

        ax2.set_title('Residual (data - fit)', loc='left')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        dy = y - y_fit
        ax2.plot(x, dy, color = 'blue', clip_on = True,
                 fillstyle = 'none', marker = 'o', mew = 2., ms = 10., linestyle = 'None')
        ax2.grid(True)
        plt.show()
        return

    @staticmethod
    def wxo_fit(wxo_fit, term_values, surface_model, plot_residuals=False):
        """3-D scatter of ``n*lambda_bs`` versus (prism angle, echelle angle) with the fitted surface.

        Data points are ``term_values['w_bs'] * term_values['ech_ords']`` at
        ``(pri_ang, ech_ang)``; the surface is ``surface_model`` evaluated on a
        50 x 50 grid over prism angle 5..8 deg and echelle angle -6..+6 deg.
        The residual standard deviation is printed in nm.

        Parameters
        ----------
        wxo_fit : dict
            ``{'wxo_opt': 10 coefficients, ...}`` from ``PolyFit.find_wxo_fit``.
        term_values : dict
            From ``Util.get_term_values`` (``pri_ang, ech_ang, w_bs, ech_ords``).
        surface_model : callable
            ``PolyFit.surface_model``.
        plot_residuals : bool, optional
            If True plot ``(data - model) * 1000`` (nm) instead of the data and
            omit the surface.

        Returns
        -------
        None

        Notes
        -----
        - Open question: the grid ranges 5..8 deg (prism) and -6..6 deg
          (echelle) are hard-coded; whether they span the actual LMS
          mechanism ranges is not stated.
        - Clean-up: the local name ``wxo_fit`` shadows the method name.
        - Clean-up: prints to stdout instead of returning the residual
          statistic.
        """
        x = np.array(term_values['pri_ang'])
        y = np.array(term_values['ech_ang'])
        wxo = np.array(term_values['w_bs']) * np.array(term_values['ech_ords'])
        wxo_opt = wxo_fit['wxo_opt']       # Fit parameters
        wxo_model = surface_model((x, y), *wxo_opt)
        f_resid = (wxo - wxo_model) * 1000  # Fractional residual (for fit points)
        f_res_std = np.std(f_resid)
        print("Residual stdev = {:7.3f} nm".format(f_res_std))
        z = wxo
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8),
                               ncols=1, nrows=1,
                               sharex=True, sharey=True)

        zlabel = r'n $\lambda$ / $\mu$m'
        suptitle = r'$n \lambda = f(\phi_{pri}, \psi_{ech})$'
        if plot_residuals:
            suptitle = 'Residuals ' + suptitle
            z = f_resid
            zlabel = r'n $\lambda$ residual / nano-m'
        fig.suptitle(suptitle)
        ax.scatter(x, y, z, color='blue')
        ax.set_xlabel(r'prism angle, $\phi_{pri}$ / deg')
        ax.set_ylabel(r'echelle angle, $\psi_{ech}$ / deg')
        ax.set_zlabel(zlabel)

        if not plot_residuals:
            x_range = np.linspace(5, 8., 50)
            y_range = np.linspace(-6., 6., 50)
            xy_grid = np.meshgrid(x_range, y_range)
            z_grid = surface_model(xy_grid, *wxo_opt)
            ax.plot_surface(xy_grid[0], xy_grid[1], z_grid, color='red', alpha=0.5)
        plt.show()
        return

    @staticmethod
    def transform_fit(term_fit, term_values, surface_model, plot_residuals=False, do_plots=True):
        """Grid of 3-D plots of the fitted surface for each element of matrix A of one slice.

        Creates an ``svd_order x svd_order`` grid of 3-D axes, removes those
        with ``row + col > svd_order - 1`` (the masked lower-right triangle),
        and for each remaining element ``c[row, col]`` of matrix ``'a'`` shows
        the per-configuration values (red) and the fitted surface (grey) or,
        with ``plot_residuals``, the fractional residual
        ``100 * (data - model) / model`` in percent (black).

        Parameters
        ----------
        term_fit : dict
            ``{'a': ndarray (4, 4, 10), ...}`` from ``PolyFit.find_slice_fit``.
        term_values : dict
            From ``Util.get_term_values`` (``pri_ang, ech_ang, matrices,
            slice_no``).
        surface_model : callable
            ``PolyFit.surface_model``.
        plot_residuals : bool, optional
        do_plots : bool, optional
            Unused; the figure is always drawn.

        Returns
        -------
        dict
            ``{'slice_no': n, 'a': {row: {col: residual_array}}}``.

        Notes
        -----
        - Inconsistency: only matrix ``'a'`` is plotted (``for mat_tag in
          ['a']``); B, AI, BI are not inspected.
        - Open question: the fractional residual divides by ``z_model`` and
          will blow up for near-zero coefficients.
        - Clean-up: ``do_plots`` is a dead parameter; the grid ranges 5..8 and
          -6..6 deg are duplicated from :meth:`wxo_fit`.
        """
        pri_ang = np.array(term_values['pri_ang'])
        ech_ang = np.array(term_values['ech_ang'])
        matrices = term_values['matrices']
        slice_no = term_values['slice_no']
        x, y = pri_ang, ech_ang

        svd_order = Globals.svd_order
        fig, ax_list = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 10),
                                    ncols=svd_order, nrows=svd_order,
                                    sharex=True, sharey=True)
        for row in range(0, svd_order):
            for col in range(0, svd_order):
                if row + col > svd_order - 1:
                    ax_list[row, col].remove()

        st1 = r'$c_{i,j} = g_{i,j}(\phi_{pri}, \psi_{ech})$'
        st2 = '\nTransform fit coefficients $c_{row, col}$'
        st3 = " for slice no.= {:d}".format(slice_no)
        zlabel = 'coefficient value'
        if plot_residuals:
            st2 = '\nTransform fit coefficient fractional residuals '
            zlabel = 'residual / %'
        suptitle = st1 + st2 + st3
        fig.suptitle(suptitle)

        f_residuals = {'slice_no': slice_no}
        for mat_tag in ['a']:
            f_residuals[mat_tag] = {}
            for row in range(0, svd_order):
                f_residuals[mat_tag][row] = {}
                for col in range(0, svd_order):
                    if row + col > svd_order - 1:
                        continue
                    z_term = []
                    for matrix in matrices:
                        mat = matrix[mat_tag]
                        z_term.append(mat[row, col])
                    z = np.array(z_term)
                    mat = term_fit[mat_tag]
                    term_opt = mat[row, col]
                    z_model = surface_model((x, y), *term_opt)
                    f_resid = 100. * (z - z_model) / z_model       # Fractional residual (for fit points) / %
                    f_residuals[mat_tag][row][col] = f_resid
                    ax = ax_list[row, col]
                    title = "c[{:d}, {:d}]".format(row, col)
                    ax.set_title(title)
                    ax.xaxis.set_major_formatter("{x:1.2f}")
                    if row == 2 and col == 1:
                        ax.set_xlabel(r'$\phi_{pri}$ / deg')
                        ax.set_ylabel(r'$\psi_{ech}$ / deg')
                        ax.set_zlabel(zlabel)
                    if plot_residuals:
                        ax.scatter(x, y, f_resid, color='black', alpha=0.5)
                        continue
                    x_range = np.linspace(5, 8., 50)
                    y_range = np.linspace(-6., 6., 50)
                    xy_grid = np.meshgrid(x_range, y_range)
                    z_grid = surface_model(xy_grid, *term_opt)
                    ax.scatter(x, y, z, color='red', alpha=0.5)
                    ax.plot_surface(xy_grid[0], xy_grid[1], z_grid, color='grey', alpha=0.5)
        Plot.show()
        return f_residuals

    @staticmethod
    def plot_mfp_points(mfp_plot, decorations, **kwargs):
        """Overlay ray-trace, slice-transform and fit-transform positions in the mosaic plane.

        For each of the three point sets ``'ray'``, ``'sli'``, ``'fit'`` in
        ``mfp_plot`` (lists of ``{'mfp_x', 'mfp_y'}`` dicts, one per
        slice/order) the points are drawn with the marker/colour from the
        matching entry of ``decorations``.  With ``ref_xy`` the ``'ray'``
        coordinates are subtracted so the plot shows offsets.  With ``grid`` a
        pixel grid of pitch ``xy_pix = 0.018`` mm is drawn.  Each ``'fit'``
        set is labelled ``"{slice}_{spifu}_{order}"``.

        Parameters
        ----------
        mfp_plot : dict
            Keys ``'slice_no', 'spifu_no', 'ech_ord'`` (lists) and
            ``'ray', 'sli', 'fit'`` (lists of point dicts).
        decorations : list of tuple
            Three ``(label, colour, marker, markersize)`` tuples.
        ref_xy : bool, optional
            Plot differences relative to the ``'ray'`` set.
        grid : bool, optional
            Draw an 18 micron pixel grid.
        title : str, optional

        Returns
        -------
        None

        Notes
        -----
        - Bug: ``mfp_plot['spifu_no']`` is required but ``lmsdist.run`` builds
          ``mfp_plot_points`` with keys ``slice_no, ech_ord, ray, sli, fit``
          only -> ``KeyError`` when the plot is enabled (debug ``'low'``).
        - Inconsistency: the grid pitch ``0.018`` mm is hard-coded instead of
          ``Globals.nom_pix_pitch / 1000``.
        - Clean-up: ``x -= x_ref`` modifies arrays created from the input
          lists via ``np.array`` so the caller's data are safe, but only by
          accident of the copy.
        """
        xy_ref = kwargs.get('ref_xy', False)
        do_grid = kwargs.get('grid', False)
        title = kwargs.get('title', '')

        fig, ax_list = Plot.set_plot_area(xlabel="x / mm", ylabel="y / mm")     # , aspect='equal')
        ax = ax_list[0, 0]
        ax.set_title(title)
        handles = []
        slice_nos = mfp_plot['slice_no']
        spifu_nos = mfp_plot['spifu_no']
        ech_ords = mfp_plot['ech_ord']

        for i, mfp_name in enumerate(['ray', 'sli', 'fit']):
            mfp_list = mfp_plot[mfp_name]
            for j in range(len(mfp_list)):
                mfp = mfp_list[j]
                x, y = np.array(mfp['mfp_x']), np.array(mfp['mfp_y'])
                if xy_ref:
                    x_ref = np.array(mfp_plot['ray'][j]['mfp_x'])
                    y_ref = np.array(mfp_plot['ray'][j]['mfp_y'])
                    x -= x_ref
                    y -= y_ref
                if mfp_name == 'fit':
                    x_lab, y_lab = np.amax(x), np.amax(y)
                    s_label = "{:02d}_{:02d}_{:02d}".format(slice_nos[j], spifu_nos[j], ech_ords[j])
                    ax.text(x_lab, y_lab, s_label, ha='right', va='bottom', size=12, color='blue')

                label, colour, marker, msize = decorations[i]
                handle, = ax.plot(x, y,
                                  color=colour, marker=marker, markersize=msize,
                                  linestyle='none', label=label)
                if j == 0:
                    handles.append(handle)
        ax.legend(handles=handles)
        xy_pix = 0.018
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if do_grid:
            xg = xmin
            while xg < xmax:
                ax.plot([xg, xg], [ymin, ymax], ls='solid', lw=0.5, color='grey')
                xg += xy_pix
            yg = ymin
            while yg < ymax:
                ax.plot([xmin, xmax], [yg, yg], ls='solid', lw=0.5, color='grey')
                yg += xy_pix
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.show()
        return

    @staticmethod
    def round_trip(y_arr, y_rtn_arr, **kwargs):
        """Plot original versus round-tripped sample values.  Broken.

        Parameters
        ----------
        y_arr : numpy.ndarray
            Original values (flattened for plotting).
        y_rtn_arr : numpy.ndarray
            Values after a forward-and-inverse transform.
        title : str, optional

        Returns
        -------
        None

        Raises
        ------
        TypeError
            Always: :meth:`set_plot_area` accepts keyword arguments only but is
            called with ``title`` positionally.

        Notes
        -----
        - Bug: positional ``title`` argument to ``set_plot_area``; and the
          return value is treated as ``ax_list`` although a ``(fig, ax_list)``
          tuple is returned.
        - Clean-up: not called anywhere in the repository.
        """
        title = kwargs.get('title', 'title')
        ax_list = Plot.set_plot_area(title,
                                     xlabel="sample",
                                     ylabel="val")
        ax = ax_list[0, 0]
        y = y_arr.flatten()
        n_samples, = y.shape
        y_rtn = y_rtn_arr.flatten()
        x = np.arange(0, n_samples, 1)
        ax.plot(x, y,
                color='black', clip_on=True,
                fillstyle='none', marker='o', mew=1.0, ms=3.0, linestyle='None')
        ax.plot(x, y_rtn,
                color='red', clip_on=True,
                fillstyle='none', marker='x', mew=1.0, ms=3.0, linestyle='None')
        Plot.show()
        return

    @staticmethod
    def _cycle_ijk(ijk, dir):
        """Rotate the three RGB channel indices in ``ijk`` in place by ``dir`` (mod 3).

        Parameters
        ----------
        ijk : list of int
            Three channel indices, modified in place.
        dir : int
            ``+1`` or ``-1``.

        Returns
        -------
        None

        Notes
        -----
        - Clean-up: ``dir`` shadows the Python built-in; only used by
          :meth:`_make_colour_list`.
        """
        for v in range(0, 3):
            ijk[v] += dir
            if ijk[v] < 0:
                ijk[v] = 2
            if ijk[v] > 2:
                ijk[v] = 0
        return

    @staticmethod
    def _make_colour_list(n_shades_ramp=1, sat=1.):
        """Return a tuple of RGB colours obtained by walking around the colour hexagon.

        Builds ``6 * (3 * n_shades_ramp - 1)`` colours by holding one channel at
        ``sat``, ramping a second up then down, and cycling the channel roles
        with :meth:`_cycle_ijk`.

        Parameters
        ----------
        n_shades_ramp : int, optional
            Shades per triangle edge; default 1.
        sat : float, optional
            Saturation (maximum channel value); default 1.0.

        Returns
        -------
        tuple of numpy.ndarray
            Each element an RGB triple.

        Notes
        -----
        - Open question: ``up`` / ``down`` from ``numpy.arange`` have
          ``dcol`` elements only when ``sat / dcol`` divides exactly; for other
          ``n_shades_ramp`` values the slice assignment
          ``rgbs[c:c+dcol, j] = up`` may raise a shape mismatch.
        - Clean-up: not called anywhere in the repository.
        """
        n_triangle_edges = 3
        n_colours_ramp = n_shades_ramp * n_triangle_edges
        n_ramps = 2 * n_triangle_edges
        dcol = n_colours_ramp - 1
        drgb = sat / dcol                       # Delta colour
        up = np.arange(0., sat, drgb)
        down = np.arange(sat, 0., -drgb)
        n_colours = n_ramps * dcol
        rgbs = np.zeros((n_colours, 3))         # Array to hold rgb values
        ijk = [0, 1, 2]
        c = 0
        while c < n_colours:
            [i, j, k] = ijk
            rgbs[c:c+dcol, i] = sat
            rgbs[c:c+dcol, j] = up
            rgbs[c:c+dcol, k] = 0.
            c += dcol
            rgbs[c:c+dcol, i] = down
            rgbs[c:c+dcol, j] = sat
            rgbs[c:c+dcol, k] = 0.
            c += dcol
            Plot._cycle_ijk(ijk, 1)

        colours = tuple(rgbs)
        return colours

    @staticmethod
    def series(plot_type, traces, model_config, plot_perimeters=True,
               colour_by='config', xlimits=None, ylimits=None, labels=False):
        """Plot wavelength coverage, dispersion or mosaic bandwidth for a list of ray traces.

        For every ``RayTrace`` in ``traces`` and every slice transform in it,
        the per-slice ray series are extracted with ``trace.get_series`` and
        plotted against wavelength according to ``plot_type``:

        - ``'coverage'``: ``y = pri_ang + 0.1 * ech_ang + 0.004 * fslice_no
          + 0.001 * mfp_y`` -- a synthetic ordinate that separates
          configurations and slices so the wavelength extent of each can be
          read off; the outline of the lowest and highest slice is drawn in
          black.
        - ``'dispersion'``: ``-1000 * (nom_pix_pitch / 1000) * dlambda / dmfp_x``
          in nm per pixel, from finite differences of consecutive rays
          (``waves[1:] - waves[:-1]`` over ``mfp_x[1:] - mfp_x[:-1]``); the
          sign flip reflects ``mfp_x`` increasing towards shorter wavelength.
        - ``'nm_det'``: the slice bandwidth ``(w_max - w_min) * 1000`` nm from
          ``slice_configuration`` as a constant per slice.

        Parameters
        ----------
        plot_type : {'coverage', 'dispersion', 'nm_det'}
        traces : list of RayTrace
        model_config : tuple
            ``Filer.model_configuration``; only ``opticon`` (element 1) is
            used, and only when ``colour_by == 'slice'``.
        colour_by : {'config', 'slice', 'slice_wavelength'}, optional
            ``'config'`` colours each trace; ``'slice_wavelength'`` colours
            each ray by wavelength; ``'slice'`` is broken (see Notes).
        xlimits, ylimits : sequence of float, optional
        labels : bool, optional
            Annotate each trace with ``"PA{pri_ang}, EA{ech_ang}"``.

        Returns
        -------
        None

        Notes
        -----
        - Bug: ``trace.get_series('wavelength', slice_config)`` uses the full
          ``slice_configuration`` dict (including ``w_min`` / ``w_max``) as a
          filter; ``RayTrace.get_series`` indexes ``trace.series[key]`` for each
          key, and there is no ``'w_min'`` series -> ``KeyError`` for any trace
          on which ``find_wavelength_bounds`` has run.
        - Bug: ``colour_by='slice'`` references the non-existent
          ``Globals.n_slices`` (``AttributeError``) and later indexes
          ``rgb[slice_no - 1]`` while ``rgb`` is ``None``.
        - Open question: the ``'dispersion'`` finite difference assumes
          consecutive rays in the series are neighbours in wavelength along the
          same alpha; rays at the same wavelength but different alpha give a
          zero denominator and infinite dispersion.
        - Inconsistency: the ``'coverage'`` ordinate weights (0.1, 0.004,
          0.001) are arbitrary display constants with no physical meaning.
        - Clean-up: the loop variable ``i`` is reused for the inner
          per-point loop, clobbering the trace index used for colouring.
        """
        _, opticon, _, _, _, _ = model_config
        titles = {'coverage': ('Wavelength coverage', r'$\theta_{prism}$ + 0.1 $\theta_{echelle}$ + det(y) / metre'),
                  'dispersion': ('Dispersion [nm / column]', 'Dispersion [nm / pixel]'),
                  'nm_det': ('Instantaneous mosaic bandwidth', 'Mosaic bandwidth [nm]')}
        title, ylabel = titles[plot_type]
        fig, ax_list = Plot.set_plot_area(xlabel=r'Wavelength / $\mu$m',
                                          ylabel=ylabel)
        fig.suptitle(title)
        ax = ax_list[0, 0]
        if xlimits is not None:
            ax.set_xlim(xlimits)
        if ylimits is not None:
            ax.set_ylim(ylimits)

        if colour_by == 'slice':
            n_fslices = Globals.n_slices[opticon]
            slice_rgb = Plot.make_rgb_gradient(np.arange(n_fslices))
        n_traces = len(traces)
        if colour_by == 'config':
            slice_rgb = Plot.make_rgb_gradient(np.arange(n_traces))
        for i, trace in enumerate(traces):
            ech_angle, prism_angle = trace.lms_config['ech_ang'], trace.lms_config['pri_ang']
            colour, rgb = None, None
            if colour_by == 'config':
                idx = n_traces - i - 1
                colour = slice_rgb[idx]
            xperi, yperi = None, None
            for transform in trace.transforms:
                slice_config = transform.slice_configuration
                slice_no = slice_config['slice_no']
                fslice_no, pslice_no = Util.decode_slice_no(slice_no)
                waves = trace.get_series('wavelength', slice_config)
                mfp_x = trace.get_series('mfp_x', slice_config)
                mfp_y = trace.get_series('mfp_y', slice_config)
                if colour_by == 'slice_wavelength':
                    rgb = Plot.make_rgb_gradient(waves)
                if colour_by == 'slice':
                    colour = rgb[slice_no - 1]

                x, y = None, None
                if plot_type == 'coverage':
                    x = waves
                    y = prism_angle + 0.1 * ech_angle + 0.004 * fslice_no + 0.001 * mfp_y
                nm_micron = 1000.0
                if plot_type == 'nm_det':
                    x = waves
                    w_min, w_max = slice_config['w_min'], slice_config['w_max']
                    dw = (w_max - w_min) * nm_micron
                    y = np.full(waves.shape, dw)
                if plot_type == 'dispersion':
                    mm_pix = Globals.nom_pix_pitch / 1000.
                    dw_dlmspix = -nm_micron * mm_pix * (waves[1:] - waves[:-1]) / (mfp_x[1:] - mfp_x[:-1])
                    x = waves[1:]
                    y = dw_dlmspix

                n_pts = len(x)

                if colour_by == 'slice_wavelength':
                    for i in range(0, n_pts):
                        ax.plot(x[i], y[i], color=rgb[i, :], clip_on=True,
                                fillstyle='none', marker='.', mew=1., ms=1, ls='None')
                else:
                    ax.plot(x, y, color=colour, clip_on=True,
                            fillstyle='none', marker='.', mew=1., ms=1, ls='None')

                # Plot perimeter of dot pattern
                if plot_perimeters:
                    unique_waves = np.unique(waves)
                    n_waves = len(unique_waves)
                    if xperi is None:
                        n_peri = 2 * n_waves + 1  # Number of points on perimeter, including return to first
                        xperi, yperi = np.zeros(n_peri), np.zeros(n_peri)
                    is_slice_lower = slice_no == trace.unique_slice_nos[0]
                    if is_slice_lower:
                        for i, w in enumerate(unique_waves):
                            idx = np.argwhere(w == x)
                            ys = y[idx]
                            ymin = np.min(ys)
                            xperi[i], yperi[i] = w, ymin
                    is_slice_upper = slice_no == trace.unique_slice_nos[-1]
                    if is_slice_upper:
                        for i, w in enumerate(unique_waves):
                            idx = np.argwhere(w == x)
                            ys = y[idx]
                            ymax = np.max(ys)
                            xperi[n_peri-i-2], yperi[n_peri-i-2] = w, ymax
                    xperi[n_peri-1], yperi[n_peri-1] = xperi[0], yperi[0]
                    ax.plot(xperi, yperi, color='black', linestyle='solid', lw=1.0)

            if labels:
                all_waves = trace.series['wavelength']
                label_idx = np.argmax(all_waves)
                x_label = all_waves[label_idx]
                y_label = prism_angle + 0.1 * ech_angle + 0.004 * fslice_no + 0.01
                lms_config = trace.lms_config
                label = "PA{:6.3f}, EA{:6.3f}".format(lms_config['pri_ang'], lms_config['ech_ang'])
                ax.text(x_label, y_label, label)
            ax.grid(True)
        Plot.show()
        return

    @staticmethod
    def plot_points(ax, x, y, **kwargs):
        """Scatter-plot points on an existing axis, optionally with a per-point RGB colour.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        x, y : array_like
        fs : str, optional
            Marker fill style (default ``'none'``).
        mk : str, optional
            Marker (default ``'o'``).
        mew : float, optional
            Marker edge width (default 1.0).
        ms : float, optional
            Marker size (default 3).
        color : str, optional
            Single colour (default ``'black'``); ignored if ``rgb`` given.
        rgb : numpy.ndarray, optional
            ``(n_pts, 3)`` per-point colours; forces one ``ax.plot`` call per
            point.
        label : str, optional
            Legend label.  With ``rgb`` the label is attached to every point.

        Returns
        -------
        None

        Notes
        -----
        - Clean-up: the per-point loop in the ``rgb`` branch is O(n) plot
          calls; ``ax.scatter(x, y, c=rgb)`` would do the same in one call.
        """
        n_pts = len(x)
        fs = kwargs.get('fs', 'none')
        mk = kwargs.get('mk', 'o')
        mew = kwargs.get('mew', 1.0)
        ms = kwargs.get('ms', 3)
        color = kwargs.get('color', 'black')
        rgb = kwargs.get('rgb', None)
        label = kwargs.get('label', None)
        if rgb is None:
            ax.plot(x, y, color=color, clip_on=True,
                    fillstyle=fs, marker=mk, mew=mew, ms=ms, linestyle='None', label=label)
        else:
            for i in range(0, n_pts):
                ax.plot(x[i], y[i], color=rgb[i, :], clip_on=True,
                        fillstyle=fs, marker=mk, mew=mew, ms=ms, label=label)
        return

    @staticmethod
    def show():
        """Call ``matplotlib.pyplot.show()`` (blocking in interactive backends).

        Notes
        -----
        - Clean-up: re-imports ``matplotlib.pyplot`` locally although the
          module already imports it as ``plt``.
        """
        import matplotlib.pyplot as plt
        plt.show()
