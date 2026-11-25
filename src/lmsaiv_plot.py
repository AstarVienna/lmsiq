"""
Matplotlib figures for the LMS AIT analyses (lmsaiv program).

Plotting helpers used by ``lmsaiv_opt01`` (LMS-OPT-01: ``mosaic``,
``histograms``, ``profiles``, ``slice_gaps``) and ``lmsaiv_opt02``
(LMS-OPT-02: ``mosaic`` with trace overlays, ``det_traces``,
``gap_data``).  All functions operate on ``lms_mosaic.Mosaic`` objects or
on the trace / gap dictionaries built by the analyses; nothing is read
from disk.

**Output**: every function creates a figure and ends with ``plt.show()``,
which blocks until the window is closed, and returns None.  No figure or
file is returned or saved.

Notes
-----
- Clean-up: return the ``Figure`` objects and drop ``plt.show()`` so the
  plots can be embedded in notebooks / reports and the analyses can run
  unattended.
- Bug: in ``mosaic`` the ``overlay`` default is ``{}`` but the
  ``ImageGrid`` / colour-bar branch tests ``overlay is None``, so it is
  dead code and no colour bar is ever drawn.
- Bug: ``mosaic`` with the ``sb`` (slice bounds) keyword and no overlay
  fails with TypeError because ``det_idx`` is only set in the overlay
  branch.
- Inconsistency: overlay dictionaries spell the trace type 'iso_alpha' /
  'iso_lambda' while the trace dicts use 'iso-alpha' / 'iso-lambda'.
- Inconsistency: ``Opt02.dist`` passes a ``bounds`` keyword to
  ``mosaic`` which is ignored (``det_traces`` does accept ``bounds``).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from lmsdist_util import Util
from lms_globals import Globals


class Plot:
    """Namespace of static plotting functions for the LMS AIT analyses."""

    def __init__(self):
        """Do nothing; all methods are static."""
        return

    @staticmethod
    def gap_data(gap_data, det_thetas):
        """Plot the inter-detector column gap versus row for detector pairs 1/2 and 3/4.

        Two vertically stacked panes; each shows ``col_gap`` against ``u_mean``
        (row) for the slices whose 'det_no' belongs to the pair, titled with the
        fitted rotation ``theta +- theta_err`` in degrees.

        Parameters
        ----------
        gap_data : dict
            ``{slice_no: {'u_mean', 'col_gap', 'det_no', ...}}`` from
            ``Opt02._find_detector_gaps``.
        det_thetas : dict
            ``{'12': (theta, theta_err) or None, '34': ...}``; a pane whose entry
            is None is left empty.

        Returns
        -------
        None
            Shows the figure (blocking).

        Notes
        -----
        - Bug: (cosmetic) the x-label condition ``pane == 2 or pane == 3`` is
          never true for a two-pane figure, so no 'Row' label appears.
        - Slices with 'col_gap' None would break ``ax.plot``; the caller only
          passes fitted gaps in practice.
        """
        figsize = [8, 8]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize,
                                 sharex='all', squeeze=True)

        for pane, det_nos in enumerate(['12', '34']):
            ax = axes[pane]
            x, y = [], []
            for slice_no in gap_data:
                gap = gap_data[slice_no]
                if str(gap['det_no']) in det_nos:
                    x.append(gap['u_mean'])
                    y.append(gap['col_gap'])
            if det_thetas[det_nos] is None:
                continue
            theta, theta_err = det_thetas[det_nos]
            title = r'$\theta_{:s}$ = {:8.3f}$\pm${:5.3f} deg.'.format(det_nos, theta, theta_err)

            ax.set_title(title)
            if pane == 2 or pane == 3:
                ax.set_xlabel('Row')
            if pane == 0 or pane == 2:
                ax.set_ylabel("Intra mosaic gap {:s}".format(det_nos))
            ax.plot(x, y, linestyle='none', marker='x', color='blue')
        plt.show()
        return

    @staticmethod
    def mosaic(mosaic, **kwargs):
        """Show the four detector images of a mosaic, optionally with slice bounds or trace overlays.

        Without ``overlay`` a 2 x 2 grid is drawn with detector ``det_no`` (from
        the per-HDU 'ID' keyword) in pane ``det_no - 1``, full 0..2048 extent,
        common ``vmin`` / ``vmax`` (nanmin / nanmax over all HDUs unless given).
        With ``overlay = {'type': 'det_traces', 'trace_type': 'iso_alpha' |
        'iso_lambda', 'data': det_traces}`` one pane is created per (detector,
        pupil slice) and zoomed to the traces: full width and ``v_fid +- 20``
        rows for iso-alpha; ``v_fid +- 10`` columns by ``u_mean +- 80`` rows
        (union over traces) for iso-lambda.  Sampled centroids are drawn as cyan
        points and the polynomial fit ``Globals.polynomial(x, *popt)`` as a blue
        line.

        Parameters
        ----------
        mosaic : Mosaic
            Image to display; ``mosaic.name`` goes into the super-title.
        **kwargs
            cmap : str, default 'hot'.
            title : str, default '-'.
            vmin, vmax : float, override the common colour scale.
            mask : tuple (value, colour); pixels equal to ``value`` are drawn in
            ``colour`` (used for slice maps with ``(0.0, 'black')``).
            sb : dict with 'det_no', 'det_col', 'det_row_min', 'det_row_max'
            arrays; slice-bound markers (red lower / green upper).
            overlay : dict; trace overlay as described above.

        Returns
        -------
        None
            Shows the figure (blocking).

        Notes
        -----
        - Bug: ``overlay`` defaults to ``{}`` so ``if overlay is None`` is never
          true; the ``ImageGrid`` branch with a shared colour bar is dead and the
          ``plt.subplots`` branch is always used.
        - Bug: ``sb`` without an overlay uses ``det_idx`` (None) -> TypeError.
        - Bug: unused overlay panes are removed with ``axes[ax_idx]`` on the 2-D
          ``axes`` array returned by ``plt.subplots`` (IndexError when
          ``n_rows > 1`` and ``n_axes < n_panes``), and the pane-count formula
          ``n_rows = n_axes // 3 + 1`` with two columns appears to allocate too
          few panes for some trace counts (e.g. five).
        - Inconsistency: 'iso_alpha' (overlay) versus 'iso-alpha' (trace 'type').
        - Open question: with ``origin='lower'`` and an ``extent`` starting at
          ``y_min - 1.5`` the image appears offset by one row relative to the
          overlaid coordinates; not verified.
        - Clean-up: the ``bounds`` keyword passed by ``Opt02.dist`` is ignored;
          ``ax_bounds`` is initialised twice and ``det_idx`` is only meaningful in
          the overlay branch.
        """
        det_idx = None

        cmap_name = kwargs.get('cmap', 'hot')
        cmap = mpl.colormaps[cmap_name]
        sb = kwargs.get('sb', {})
        title = kwargs.get('title', '-')
        suptitle = mosaic.name + '\n' + title
        fig_size = 8, 7
        n_rows, n_cols = 2, 2
        ax_datas, ax_bounds = {}, {}
        overlay = kwargs.get('overlay', {})
        shareall = True                                     # Share axis scaling for all plots
        ax_bounds = {}                                      # Plot bounds for each axis
        axes_pad = 0.15, 0.15
        n_axes = 4
        if not overlay:                                     # An empty dictionary returns False
            for hdu in mosaic.hdu_list:
                det_no = int(hdu.header['ID'])
                ax_tag = "det_no={:d}".format(det_no)
                ax_bounds[ax_tag] = 0, 2048, 0, 2048
                ax_data = {'ax_idx': det_no - 1, 'image': hdu.data}
                ax_datas[ax_tag] = [ax_data]
        else:
            if overlay['type'] == 'det_traces':         # Make individual plots, 4 per row.
                # Each detector/pupil number is plotted in the same axis.
                is_iso_alpha = overlay['trace_type'] == 'iso_alpha'
                shareall = False
                axes_pad = 0.5, 0.5
                n_panes_row = 2
                det_traces = overlay['data']
                ax_idx_next = 0
                for det_trace in det_traces:
                    det_no = det_trace['det_no']
                    slice_no = det_trace['slice_no']
                    _, pslice_no = Util.decode_slice_no(slice_no)
                    ax_tag = "det_no={:d}, pslice={:d}".format(det_no, pslice_no)
                    det_idx = det_no - 1
                    v_fid = int(det_trace['v_fid'])
                    u_mean = int(det_trace['u_mean'])
                    ax_data = {'ax_idx': ax_idx_next, 'image': mosaic.hdu_list[det_idx].data, 'det_trace': det_trace}
                    if ax_tag in ax_datas:
                        ax_idx_copy = ax_datas[ax_tag][0]['ax_idx']
                        ax_data['ax_idx'] = ax_idx_copy
                        ax_datas[ax_tag].append(ax_data)
                        if is_iso_alpha:
                            x_min, x_max, y_min, y_max = 0, 2048, v_fid - 20, v_fid + 20
                        else:
                            old_bounds = ax_bounds[ax_tag]
                            x_min = min(old_bounds[0], v_fid - 10)
                            x_max = max(old_bounds[1], v_fid + 10)
                            y_min = min(old_bounds[2], u_mean - 80)
                            y_max = max(old_bounds[3], u_mean + 80)
                        ax_bounds[ax_tag] = x_min, x_max, y_min, y_max
                    else:
                        ax_datas[ax_tag] = [ax_data]
                        if is_iso_alpha:
                            ax_bounds[ax_tag] = 0, 2048, v_fid - 20, v_fid + 20
                        else:
                            ax_bounds[ax_tag] = v_fid - 10, v_fid + 10, u_mean - 80, u_mean + 80
                        ax_idx_next += 1
                n_axes = ax_idx_next
                n_rows, n_cols = (n_axes // (n_panes_row +1)) + 1, n_panes_row

        nrows_ncols = n_rows, n_cols
        if overlay is None:
            fig = plt.figure(figsize=fig_size)  # Set up figure and image grid
            grid = ImageGrid(fig, 111,
                             nrows_ncols=nrows_ncols, axes_pad=axes_pad, cbar_location="right",  share_all=shareall,
                             cbar_mode="single", cbar_size="7%", cbar_pad=0.15,
                             )
        else:
            n_rows, n_cols = nrows_ncols
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size,
                                     sharex='none', squeeze=True)
            n_panes = n_rows * n_cols
            for ax_idx in range(n_axes, n_panes):
                axes[ax_idx].remove()
            grid = axes.flatten()
        fig.suptitle(suptitle)

        # Find common signal bounds
        vmin, vmax = 1.E6, -1.E6
        for hdu in mosaic.hdu_list:
            vmin_hdu, vmax_hdu = np.nanmin(hdu.data), np.nanmax(hdu.data)
            vmin = min(vmin, vmin_hdu)
            vmax = max(vmax ,vmax_hdu)
        if 'vmin' in kwargs:
            vmin = kwargs.get('vmin', vmin)
        if 'vmax' in kwargs:
            vmax = kwargs.get('vmax', vmax)
        ax, im = None, None
        for ax_tag in ax_datas:
            ax_data_list = ax_datas[ax_tag]             # Get list of data sets to plot in this axis
            ax_idx = ax_data_list[0]['ax_idx']          # All data in list (should) share the same axis index.
            ax = grid[ax_idx]
            x_min, x_max, y_min, y_max = ax_bounds[ax_tag]
            ax.set_xlim(x_min - 1, x_max + 1)
            ax.set_ylim(y_min - 1, y_max + 1)
            aspect_ratio = (x_max - x_min) / (y_max - y_min)
            ax.set_aspect(aspect_ratio)

            for ax_data in ax_data_list:
                x1, x2, y1, y2 = int(x_min), int(x_max), int(y_min), int(y_max)
                image = ax_data['image']
                mask = kwargs.get('mask', None)
                if mask is not None:
                    mask_value, mask_colour = mask
                    image = np.ma.masked_where(image == mask_value, image)
                    cmap.set_bad(color=mask_colour)
                im = ax.imshow(image[y1:y2, x1:x2], extent=(x_min-.5, x_max+.5, y_min-1.5, y_max-.5),
                               interpolation='nearest', aspect=aspect_ratio,
                               cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                if sb:
                    det_no = sb['det_no']
                    idx = np.argwhere(det_no == det_idx + 1)
                    x = sb['det_col'][idx]
                    yrmin = sb['det_row_min'][idx]
                    ax.plot(x, yrmin, marker='o', ms=2.0, color='red', linestyle='none')
                    yrmax = sb['det_row_max'][idx]
                    ax.plot(x, yrmax, marker='o', ms=2.0, color='green', linestyle='none')
                # overlay = kwargs.get('overlay', None)
                if overlay:
                    if overlay['type'] == 'det_traces':
                        det_trace = ax_data['det_trace']
                        if det_trace is None:
                            continue
                        is_alpha = det_trace['type'] == 'iso-alpha'
                        pt_u_coords = det_trace['pt_u_coords']
                        pt_v_coords = det_trace['pt_v_coords']
                        xs = pt_u_coords if is_alpha else pt_v_coords
                        ys = pt_v_coords if is_alpha else pt_u_coords
                        # title = "{:d}, {:d}".format(det_trace['det_no'], det_trace['slice_no'])
                        ax.set_title(ax_tag)
                        ax.plot(xs, ys, marker='o', ms=4.0, color='cyan', linestyle='none')
                        popt = det_trace['popt']
                        xhw = 1024 if is_alpha else 40
                        x1 = det_trace['u_mean'] - xhw
                        x2 = x1 + 2 * xhw
                        x = np.arange(x1, x2, 10)
                        y = Globals.polynomial(x, *popt)
                        if is_alpha:
                            ax.plot(x, y, color='blue', linestyle='solid')
                        else:
                            ax.plot(y, x, color='blue', linestyle='solid')
        if overlay is None:
            ax.cax.colorbar(im)
        plt.show()
        return

    @staticmethod
    def det_traces(alpha_traces, lambda_traces, config_id, **kwargs):
        """Draw the fitted iso-alpha and iso-lambda trace polynomials on a 2 x 2 detector grid.

        Iso-alpha fits are drawn in blue (``row = poly(col)`` over ``u_mean +-
        1024``), iso-lambda fits in red (``col = poly(row)`` over ``u_mean +-
        40``), each in the pane of its detector.  Optional intersection points
        are drawn as black dots.  No image is shown.

        Parameters
        ----------
        alpha_traces : list of dict
            Iso-alpha traces (keys 'det_no', 'type', 'popt', 'u_mean').
        lambda_traces : list of dict
            Iso-lambda traces.
        config_id : str
            LMS configuration id used in the super-title only.
        **kwargs
            bounds : tuple (xmin, xmax, ymin, ymax); default the full detector
            ``Globals.det_format``.
            coords : dict with 'det_no', 'row', 'col' lists (one entry of the
            ``Opt02._find_trace_intersections`` output).

        Returns
        -------
        None
            Shows the figure (blocking).

        Notes
        -----
        - Open question: the traces are not filtered by ``config_id``; all
          supplied traces are drawn whatever their 'lms_config_id'.
        - Clean-up: the axis limits are re-applied for every trace.
        """
        fig = plt.figure(figsize=(8, 7))
        fig.suptitle('Configuration ' + config_id)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2), axes_pad=(0.15, 0.15), share_all=True
                         )
        # Set plot limits
        xmin, xmax = 0, Globals.det_format[1]
        ymin, ymax = 0, Globals.det_format[0]
        bounds = kwargs.get('bounds', (xmin, xmax, ymin, ymax))
        xmin, xmax, ymin, ymax = bounds

        for det_traces in [alpha_traces, lambda_traces]:
            for det_trace in det_traces:
                det_no = det_trace['det_no']
                ax = grid[det_no - 1]
                ax.set_xlim(xmin - 1, xmax + 1)
                ax.set_ylim(ymin - 1, ymax + 1)
                aspect_ratio = (xmax - xmin) / (ymax - ymin)
                ax.set_aspect(aspect_ratio)

                is_alpha = det_trace['type'] == 'iso-alpha'
                popt = det_trace['popt']
                xhw = 1024 if is_alpha else 40
                x1 = det_trace['u_mean'] - xhw
                x2 = x1 + 2 * xhw
                x = np.arange(x1, x2, 10)
                y = Globals.polynomial(x, *popt)
                colour = 'blue' if is_alpha else 'red'
                if is_alpha:
                    ax.plot(x, y, color=colour, linestyle='solid', lw=.5)
                else:
                    ax.plot(y, x, color=colour, linestyle='solid', lw=.5)
        coords = kwargs.get('coords', {})
        if coords:
            det_nos = np.array(coords['det_no'])
            rows = np.array(coords['row'])
            cols = np.array(coords['col'])
            for i in range(0, 4):
                idx = np.argwhere(det_nos == i+1)[:]
                x, y = cols[idx], rows[idx]
                ax = grid[i]
                ax.set_xlim(xmin - 1, xmax + 1)
                ax.set_ylim(ymin - 1, ymax + 1)
                aspect_ratio = (xmax - xmin) / (ymax - ymin)
                ax.set_aspect(aspect_ratio)
                ax.plot(x, y, marker='o', ms=2.0, color='black', linestyle='none')

        plt.show()
        return

    @staticmethod
    def histograms(mosaic, n_bins=200, range=None, make_labels=True):
        """Plot a histogram of the pixel values of each detector HDU in a 2 x 2 grid.

        Parameters
        ----------
        mosaic : Mosaic
            Four HDUs, drawn in list order (not by 'ID').
        n_bins : int, optional
            Number of histogram bins.  Default 200.
        range : tuple of float, optional
            ``(lower, upper)`` histogram range passed to ``Axes.hist``; None
            (default) uses the data range.  ``Rad04.pix_flat`` uses
            (0.99, 1.01).
        make_labels : bool, optional
            Accepted but not used.

        Returns
        -------
        tuple
            ``(n_bins, n_pts)`` where ``n_pts`` is the number of values counted
            in the histogram of the last detector drawn.  The 8 x 8 inch figure
            is shown (blocking) before returning.

        Notes
        -----
        - Inconsistency: panes follow ``hdu_list`` order (index ``i``) whereas
          ``mosaic`` uses the 'ID' keyword; the two agree for ToySim data only
          (see ``Filer.read_mosaic``).
        - Clean-up: ``n_pts`` is overwritten per detector, so only the last
          detector's count is returned; ``hist_pars``, ``det_no`` and
          ``n_vals`` (the NaN count) are computed but unused; ``make_labels``
          is ignored; the parameter name ``range`` shadows the builtin.
        """

        fig_size = [8, 8]
        fig, axs = plt.subplots(2, 2, figsize=fig_size,
                                sharex=True, sharey=True, tight_layout=True)
        fig.suptitle(mosaic.name)
        hist_pars = {}

        for i, hdu in enumerate(mosaic.hdu_list):
            det_no = i + 1
            ax_row, ax_col = int(i / 2), i % 2
            vals = hdu.data.flatten()
            n_vals = np.count_nonzero(np.isnan(vals))
            bin_pts, _, _ = axs[ax_row, ax_col].hist(vals, bins=n_bins, range=range)
            n_pts = np.sum(np.array(bin_pts))
        plt.show()
        return n_bins, n_pts

    @staticmethod
    def slice_gaps(gap_list):
        """Plot the measured inter-slice gap against slice number, one pane per detector.

        Parameters
        ----------
        gap_list : list of tuple
            ``(det_no, slice_no_a, slice_no_b, col_min, col_max, gap)`` from
            ``Opt01._find_gaps``; ``gap`` (pixels) is plotted against
            ``slice_no_a``.

        Returns
        -------
        None
            Shows the figure (blocking).

        Notes
        -----
        - Clean-up: the design gap ``Globals.intra_slice_gap`` (20 pixels) is
          not drawn for comparison.
        """
        gaps = np.array(gap_list)
        figsize = [8, 8]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                                 sharex='all', sharey='all', squeeze=True)
        fig.suptitle('Intra-slice gaps')
        ax_list = np.array(axes).flatten()
        uni_det_nos = np.unique(gaps[:,0])
        for det_no in uni_det_nos:
            pane = int(det_no) - 1
            ax = ax_list[pane]
            ax.set_title('Det {:d}'.format(pane + 1))
            if pane == 2 or pane == 3:
                ax.set_xlabel('Slice no.')
            if pane == 0 or pane == 2:
                ax.set_ylabel('Gap / pixels')
            idx = np.argwhere(gaps[:, 0] == det_no)
            x, y = gaps[idx, 1], gaps[idx, 5]
            ax.plot(x, y, linestyle='none', marker='x', color='black')
        plt.show()
        return

    @staticmethod
    def profiles(profiles, nax_rows=2, nax_cols=2):
        """Plot along-column signal profiles with their detected slice edges, one pane per detector.

        Parameters
        ----------
        profiles : list of tuple
            ``(label, colour, det_no, profile_column, vals, pts)`` as built by
            ``Opt01._find_slices``: ``vals`` is the raw 9-column mean profile
            versus row, ``pts`` a list of ``(row, cut_level)`` edge points drawn
            as 'x' markers in the profile colour.
        nax_rows, nax_cols : int, optional
            Grid shape; default 2 x 2, pane index ``det_no - 1``.

        Returns
        -------
        None
            Shows the figure (blocking).

        Notes
        -----
        - Inconsistency: the legacy ``Opt01.flood_stats`` builds profile tuples
          with a different layout ``(title, x_indices, y_signal, [on_points,
          off_points])`` which this function cannot unpack.
        - Clean-up: ``label`` is unpacked but not used (no legend).
        """
        figsize = [8, 8]
        n_axes = nax_rows * nax_cols
        fig, axes = plt.subplots(nrows=nax_rows, ncols=nax_cols, figsize=figsize,
                                 sharex='all', sharey='all', squeeze=True)
        ax_list = [axes] if n_axes < 2 else axes
        ax_list = np.array(ax_list).flatten()
        for profile in profiles:
            label, colour, det_no, profile_column, vals, pts = profile
            iax = det_no - 1
            x = np.arange(len(vals))
            title = "Det {:d}".format(det_no)
            ax = ax_list[iax]
            ax.plot(x, vals, color=colour, linestyle='solid', lw=.5)
            ax.set_title(title)
            for pt in pts:
                x_pts, y_pts = pt
                ax.plot(x_pts, y_pts, linestyle='none', marker='x', color=colour)
        plt.show()
