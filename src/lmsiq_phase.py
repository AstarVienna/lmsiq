"""
Pixel-phase study: sensitivity of centroid, FWHM and photometry to sub-pixel PSF shifts.

Purpose
-------
:meth:`Phase.process` takes the model PSF images of one slice, shifts each by a grid of
sub-pixel offsets along the spectral (column) axis, applies the detector diffusion / IPC
model and down-samples to detector pixels, then measures the fitted x and y centroid,
the x FWHM and the total signal at every shift.  The variation of these quantities with
pixel phase is plotted per (configuration, field).  It quantifies how much the
detector sampling alone modulates line centroids and photometry.

Belongs to the ``lmsiq`` program; run from :func:`lmsiq.main` when
``process_phase_data`` is True (default False).

Input data
----------
Zemax PSF stacks via :class:`lmsiq_image_manager.ImageManager` from
``../data/iq/<opticon>/<date_stamp>/`` (128 x 128 pixels at 4.5 um = 4x oversampled 18
um H2RG pixels; file naming as described in :mod:`lmsiq_image_manager`).

Outputs
-------
PNG plots in ``../output/iq/<opticon>/<date_stamp>/phase/`` named
``<xcentroids|ycentroids|photometry|xfwhm>_<ipc>_<cc>_field<ff>_tgt_slice<ss>_spifu_<pp>.png``
via :meth:`lmsiq_plot.Plot.phase_shift`.  No numeric results are written.

Notes
-----
* Bug: :meth:`Phase.sub_pixel_shift` discards the return value of ``np.moveaxis``, so
  the intended transpose for ``axis == 'spectral'`` never happens and the shift is
  always applied along columns (axis 1).
* Inconsistency: ``tgt_slice_no`` is read from ``data_identifier['field_tgt_slice']``
  (a dict) and then overwritten with the constant 13; ``slice_nos`` and ``spifu_nos``
  are likewise forced to ``[13]`` and ``[1]``.
* Inconsistency: only the first four models (``image_list[0:4]`` = perfect, design,
  MC_lo, MC_lo+1) are processed, but the result arrays are sized for all models, so
  columns 4.. stay zero and :meth:`lmsiq_plot.Plot.phase_shift` divides by a zero mean
  for them.
* Inconsistency: the y-centroid fit uses ``Analyse.find_fwhm(im_det, oversample=4)`` on
  an already down-sampled image, so its FWHM/centroid are in units of 1/4 detector pixel
  and the collapse band is 4x wider than intended.
* Clean-up: ``xcen_obs``, ``xcen_obs_rms``, ``ycen_obs``, ``xfwhm_obs``, ``phot_obs``,
  ``phot_obs_rms``, ``waves`` and ``dataset_indices`` are filled but never used or
  returned; the ``plot_level`` kwarg passed by ``lmsiq.main`` is ignored.
"""
import numpy as np
import time
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot
from lmsiq_analyse import Analyse
from lmsdist_util import Util


class Phase:
    """
    Namespace for the sub-pixel phase-shift analysis.

    Notes
    -----
    * Clean-up: the class holds no state.
    """

    def __init__(self):
        """No-op constructor."""
        return

    @staticmethod
    def process(data_identifier, process_control, iq_filer, image_manager, **kwargs):
        """
        Run the phase-shift scan and plot centroid, FWHM and photometry variations.

        Flow:

        1. Shift grid ``det_shifts = arange(-1.0, 1.1, 0.1)`` detector pixels (21
           steps); target slice fixed to 13, spifu slice to 1.
        2. For each ``inter_pixel`` in ``process_control[1]`` (``Ipc.set_inter_pixel``),
           configuration and field: load the model stack with
           :meth:`lmsiq_image_manager.ImageManager.load_dataset`.
        3. For each of the first four models and each shift:
           ``im_shift = det_shift * oversampling`` image pixels;
           :meth:`sub_pixel_shift` (``resolution=50``); :meth:`lms_ipc.Ipc.apply` if
           ``inter_pixel``; :meth:`lms_detector.Detector.down_sample`.  For the first
           data set and model index 2 only, a 3-pane illustration (Zemax image, with
           diffusion, sampled) is shown.
        4. Measure ``Analyse.find_fwhm(im_ipc, oversample=4, axis=0)`` and
           ``find_fwhm(im_det, oversample=1, axis=0)`` for the x centroid / FWHM in
           detector pixels, ``find_fwhm(im_det, oversample=4, axis=1)`` for the y
           centroid, and ``Analyse.find_phot(im_det, method='full_image')`` for the
           total signal.  Store ``xcen_det - det_shift`` (residual after removing the
           applied shift), ``ycen``, ``xfwhm_det`` and ``phot``.
        5. Plot the four ``(n_shifts, n_models)`` arrays with
           :meth:`lmsiq_plot.Plot.phase_shift` to ``<output_folder>/phase/``.

        Parameters
        ----------
        data_identifier : dict
            ``'field_tgt_slice'`` and ``'slice_radius'`` are read (then overridden).
        process_control : tuple
            ``(mc_bounds, inter_pixels)``.
        iq_filer : lms_filer.Filer
            ``model_configuration`` (for the date stamp), ``output_folder``,
            ``get_folder``.
        image_manager : lmsiq_image_manager.ImageManager
        **kwargs
            ``config_nos`` : list of int or None -- default all configurations.
            ``field_nos`` : list of int or None -- default all fields.
            Other keywords (``plot_level``) are ignored.

        Returns
        -------
        None

        Notes
        -----
        * Bug: the ``det_shift += det_shift_increment`` at the end of the shift loop
          rebinds the loop variable and has no effect.
        * Bug: ``ds_dict['mc_bounds']`` is unpacked as ``(mc_start, mc_end)``; it is
          None when the data set has no MC instances.
        * Inconsistency: ``lmsiq.main`` passes ``config_nos=[0]``; whether a
          configuration 0 exists depends on the folder naming (configurations decoded
          from ``config<nnn>`` codes are normally >= 1), in which case
          ``load_dataset`` returns None and the following ``ds_dict['im_pix_size']``
          raises ``TypeError``.
        * Inconsistency: the ``Plot.images`` illustration has its ``png_path`` commented
          out, so it is only shown interactively.
        * Clean-up: ``ycen_obs``, ``xcen_obs`` etc. are normalised by their means but
          never used; the ``phase_plots`` arrays passed to the plotter are the raw
          (un-normalised) shift arrays, which ``Plot.phase_shift`` normalises again.
        """
        field_nos = kwargs.get('field_nos', None)
        config_nos = kwargs.get('config_nos', None)

        t_start = time.perf_counter()

        mc_bounds, inter_pixels = process_control
        _, _, date_stamp, _, _, _ = iq_filer.model_configuration
        tgt_slice_no, slice_radius = data_identifier['field_tgt_slice'], data_identifier['slice_radius']
        tgt_slice_no = 13
        slice_tag = "slice_{:d}".format(tgt_slice_no)

        # Set up spectral shifts in detector pixels
        det_shift_start, det_shift_end, det_shift_increment = -1.0, +1.1, 0.1
        det_shifts = np.arange(det_shift_start, det_shift_end, det_shift_increment)
        n_shifts = len(det_shifts)

        png_folder = iq_filer.output_folder + '/phase'
        png_folder = iq_filer.get_folder(png_folder)

        # Dictionary of photometry rms values and centroid phase locations for both ipc settings
        for inter_pixel in inter_pixels:
            Ipc.set_inter_pixel(inter_pixel)
            ipc_tag = Ipc.tag
            # Initialise wavelength dependent value lists
            waves = []
            dataset_indices = []
            dataset_idx = 0

            uni_par = image_manager.unique_parameters
            if config_nos is None:
                config_nos = uni_par['config_nos']
            if field_nos is None:
                field_nos = uni_par['field_nos']
            slice_nos = uni_par['slice_nos']
            spifu_nos = uni_par['spifu_nos']
            print('Nominal data set ')
            print("Detector diffusion = {:s}".format(str(inter_pixel)))
            Util.print_list('configurations', config_nos)
            Util.print_list('field numbers ', field_nos)
            Util.print_list('slice numbers ', slice_nos)
            Util.print_list('spifu numbers ', spifu_nos)
            print()
            print('Changing to..')
            slice_nos = [13]
            spifu_nos = [1]
            Util.print_list('configurations', config_nos)
            Util.print_list('field numbers ', field_nos)
            Util.print_list('slice numbers ', slice_nos)
            Util.print_list('spifu numbers ', spifu_nos)

            is_first_dataset = True         # Only plot pipeline illustration for first dataset
            for config_no in config_nos:
                config_tag = "config_{:d}".format(config_no)
                for field_no in field_nos:
                    field_tag = "field_{:d}".format(field_no)
                    for tgt_spifu_no in spifu_nos:
                        spifu_tag = "spifu_{:d}".format(tgt_spifu_no)
                        dataset_indices.append(dataset_idx)
                        dataset_idx += 1
                        selection = {'config_no': config_no,
                                     'field_no': field_no,
                                     'slice_no': tgt_slice_no,
                                     'spifu_no': tgt_spifu_no,
                                     'mc_bounds': mc_bounds}
                        image_list, ds_dict = image_manager.load_dataset(iq_filer,
                                                                         selection,
                                                                         # xy_shift=(10, -5, -5),
                                                                         debug=False)
                        im_pix_size = ds_dict['im_pix_size']
                        n_images = len(image_list)

                        phase_labels = ['perfect', 'design']
                        mc_start, mc_end = ds_dict['mc_bounds']
                        for mc_no in range(mc_start, mc_end+1):
                            phase_labels.append("MC_{:04d}".format(mc_no))

                        wave = ds_dict['wavelength']
                        waves.append(wave)
                        oversampling = int(Detector.det_pix_size / im_pix_size)
                        xcen_obs = {'phase shift': det_shifts}
                        xcen_obs_rms = {}
                        ycen_obs = {'phase shift': det_shifts}
                        xfwhm_obs = {'phase shift': det_shifts}
                        phot_obs = {'phase shift': det_shifts}
                        shape = n_shifts, n_images
                        xcen_shift, ycen_shift = np.zeros(shape), np.zeros(shape)
                        xfwhm_shift, phot_shift = np.zeros(shape), np.zeros(shape)
                        phot_obs_rms = {}
                        for img_idx, img_in in enumerate(image_list[0:4]):
                            t_now = time.perf_counter()
                            t_min = (t_now - t_start) / 60.

                            fmt = "\r- configuration {:d}, field {:d} of {:d}, " +\
                                  "slice {:d}, spifu {:d}, model {:03d} of {:03d} at t= {:8.3f} min"
                            print(fmt.format(config_no, field_no, len(field_nos),
                                  tgt_slice_no, tgt_spifu_no, img_idx + 1, n_images,
                                  t_min),
                                  end="", flush=True)
                            obs_key = phase_labels[img_idx]
                            for shift_idx, det_shift in enumerate(det_shifts):
                                im_shift = det_shift * oversampling
                                im_shifted = Phase.sub_pixel_shift(img_in, 'spectral', im_shift,
                                                                   resolution=50, debug=False)
                                im_ipc = Ipc.apply(im_shifted, oversampling) if inter_pixel else im_shifted
                                im_det = Detector.down_sample(im_ipc, im_pix_size)
                                if is_first_dataset and img_idx == 2:
                                    img_tag = "img_{:d}".format(img_idx)
                                    fmt = "processed_data_{:s}_{:s}_{:s}_{:s}_{:s}_{:s}"
                                    png_file = fmt.format(ipc_tag, field_tag, spifu_tag,
                                                          config_tag, slice_tag, img_tag)
                                    png_path = png_folder + png_file

                                    title = png_file
                                    pane_titles = ['Zemax image', 'with detector\ndiffusion',
                                                   'sampled at\ndetector']
                                    Plot.images([im_shifted, im_ipc, im_det],
                                                title=title, pane_titles=pane_titles,
                                                nrowcol=(1, 3))   # png_path=png_path

                                    is_first_dataset = False

                                # Find the FWHM and <signal> in the detector plane
                                xgauss_ipc, _ = Analyse.find_fwhm(im_ipc, oversample=oversampling,
                                                                  debug=False, axis=0)
                                xgauss_det, _ = Analyse.find_fwhm(im_det, oversample=1,
                                                                  debug=False, axis=0)
                                _, xfwhm_ipc, xcen_ipc = xgauss_ipc[1]
                                _, xfwhm_det, xcen_det = xgauss_det[1]

                                ygauss, _ = Analyse.find_fwhm(im_det, oversample=oversampling,
                                                              debug=False, axis=1)
                                is_error, yfit, yfit_err = ygauss
                                if is_error:
                                    continue
                                _, _, ycen = yfit
                                method = 'full_image'          # Valid methods are 'aperture' or 'full_image'
                                ap_pos, ap_width, ap_height = (xcen_det, ycen), 16., 16.
                                phot = Analyse.find_phot(im_det,
                                                         method=method,
                                                         method_parameters=(ap_pos, ap_width, ap_height))

                                xcen_shift[shift_idx, img_idx] = xcen_det - det_shift
                                ycen_shift[shift_idx, img_idx] = ycen
                                xfwhm_shift[shift_idx, img_idx] = xfwhm_det
                                phot_shift[shift_idx, img_idx] = phot
                                det_shift += det_shift_increment

                            xcen_shift_mean = np.mean(xcen_shift)
                            xcen_shift_rms = np.std(xcen_shift)
                            ycen_shift_mean = np.mean(ycen_shift)
                            phot_shift_mean = np.mean(phot_shift)
                            phot_shift_rms = np.std(phot_shift)
                            phot_obs[obs_key] = phot_shift / phot_shift_mean
                            xcen_obs_rms[obs_key] = xcen_shift_rms
                            phot_obs_rms[obs_key] = phot_shift_rms
                            xcen_obs[obs_key] = xcen_shift / xcen_shift_mean
                            ycen_obs[obs_key] = ycen_shift / ycen_shift_mean
                            xfwhm_obs[obs_key] = xfwhm_shift

                        fmt = "{:s}_{:s}_{:02d}_field{:02d}_tgt_slice{:02d}_spifu_{:02d}"
                        phase_plots = {'xcentroids': xcen_shift, 'ycentroids': ycen_shift,
                                       'photometry': phot_shift, 'xfwhm': xfwhm_shift}
                        for key in phase_plots:
                            png_name = fmt.format(key, ipc_tag, config_no, field_no, tgt_slice_no, tgt_spifu_no)
                            data_tuple = det_shifts, phase_plots[key]
                            png_path = png_folder + '' + png_name
                            title = "{:s}, ".format(date_stamp) + png_name
                            Plot.phase_shift(key, data_tuple, ds_dict, ipc_tag, title=title, png_path=png_path)
            print()
        print()
        return

    @staticmethod
    def sub_pixel_shift(image, axis, sp_shift, **kwargs):
        """
        Shift an image along its columns by a fractional pixel amount via super-sampling.

        Algorithm: copy the input; build a super-sampled image with ``resolution``
        sub-columns per column; write each input column into sub-columns
        ``[col * res + shift, col * res + shift + res)`` where
        ``shift = round(sp_shift * res)``, clipping at the array edges (data shifted
        out are lost, vacated sub-columns stay zero); then average every ``res``
        sub-columns back to one output column.

        Parameters
        ----------
        image : ndarray, shape (n_rows, n_cols)
            Not modified.
        axis : str
            ``'spectral'`` is intended to select the row axis by transposing; see
            Notes -- in practice the shift is always along columns.
        sp_shift : float
            Shift in image pixels (positive -> towards higher column index).
        **kwargs
            ``debug`` : bool.
            ``resolution`` : int, default 10 -- sub-pixels per pixel (50 in
            :meth:`process`).

        Returns
        -------
        image_out : ndarray, shape (n_rows, n_cols)

        Notes
        -----
        * Bug: ``np.moveaxis(image_in, 0, 1)`` returns a view that is discarded
          (twice), so the 'spectral' transpose has no effect.  Because the spectral
          direction of the Zemax images is the column axis (see
          :meth:`lmsiq_analyse.Analyse.lsf`, ``axis=0``), the shift nevertheless lands
          on the intended axis.
        * Inconsistency: the upper clip ``c2 = ncols_ss - 1`` drops the last
          sub-column when the shifted block reaches the edge.
        * Clean-up: two Python loops over columns per call; ``np.roll`` on the
          super-sampled array (as in ``ImageManager.load_dataset``'s ``xy_shift``) or
          ``scipy.ndimage.shift`` would be equivalent apart from the edge handling.
        """
        debug = kwargs.get('debug', False)
        resolution = kwargs.get('resolution', 10)       # Number of sub-pixels per image pixel

        image_in = np.array(image)                      # Clone input to avoid corrupting observation
        if axis == 'spectral':                          # Rotate input image
            np.moveaxis(image_in, 0, 1)
        nrows, ncols = image_in.shape
        ncols_ss = resolution * ncols
        image_ss = np.zeros((nrows, ncols_ss))          # Create super-sampled image
        col_shift = round(sp_shift * resolution)
        if debug:
            fmt = "Shifting {:s} axis image, ({:d}x) oversampled by {:d} columns"
            print(fmt.format(axis, resolution, col_shift))
        for col in range(0, ncols):     # Map shifted image data into super-sampled image
            c1 = col * resolution + col_shift
            c2 = c1 + resolution
            c1 = c1 if c1 > 0 else 0
            c2 = c2 if c2 < ncols_ss else ncols_ss - 1
            for c_ss in range(c1, c2):
                image_ss[:, c_ss] = image_in[:, col]
        image_out = np.zeros(image_in.shape)
        for col in range(0, ncols):     # Resample super-sampled image onto output image
            col_ss = col * resolution
            strip = np.mean(image_ss[:, col_ss:col_ss + resolution], axis=1)
            image_out[:, col] = strip
        if axis == 'spectral':                      # Rotate input image back to original orentation
            np.moveaxis(image_out, 0, 1)
        return image_out        # , params
