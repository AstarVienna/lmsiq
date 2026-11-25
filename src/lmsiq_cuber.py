"""
Cube assembly and image-quality series analysis for the ``lmsiq`` program.

Purpose
-------
:class:`Cuber` is the workhorse of ``lmsiq``.  :meth:`Cuber.build` walks every
(inter-pixel setting, field, spectral-IFU slice, configuration, focus) combination
in an :class:`lmsiq_image_manager.ImageManager` dictionary, loads the per-slice stack
of Zemax PSF images (perfect, as-designed, Monte-Carlo instances), applies the detector
diffusion / IPC model and down-samples to detector pixels, measures the spectral LSF on
the target slice, stacks the slices into a pseudo (model, along-slice, across-slice)
cube, measures enslitted energy, along/across-slice LSF and a relative Strehl on the
cube, and writes the cube (FITS) plus a ``field_series`` metric dictionary (pickle).
:meth:`Cuber.plot` reloads the pickles and plots each metric against wavelength (or
spectral-IFU slice) per field with Monte-Carlo percentile bands and the MET-2745 /
MET-3739 requirement lines.

Input data
----------
Zemax PSF FITS images under ``../data/iq/<opticon>/<date_stamp>/<config folder>/``,
named ``<folder>_spat<ss>_spec<p>_det<tag>.fits`` (``tag`` = ``perf``, ``desi`` or a
4-digit MC number), 128 x 128 pixels at 4.5 um (4x oversampled 18 um H2RG pixels), with
one ``.txt`` parameter file per folder (keys ``wavelength``, ``prism angle``, ``grating
angle``, ``grating order``, ``Pixel size (micron)``).  The ``lmsdist`` trace pickle
(``dist_filer.trace_file``) supplies the dispersion.

Outputs
-------
Under ``../output/iq/<opticon>/<date_stamp>/``:

* ``cube/fits/cube_<ipc>_field_<f>_spifu_<p>_cfg_<c>_focus_<d>_slices_<a>_to_<b>.fits``
  (:meth:`lmsiq_fitsio.FitsIo.write_cube`, no header);
* ``cube/pkl/<same stem>.pkl`` holding ``(cube_ident, cube, field_series)``;
* ``profiles/ee/ee_<axis>_<cube>.png`` and ``profiles/lsf/lsf_<axis>_<cube>.png``;
* ``cube/series/png/<opticon>_<metric>_v_<abscissa>_<ipc>.png`` and ``key.png`` from
  :meth:`Cuber.plot`;
* ``cube/series/csv/series.csv`` from :meth:`Cuber.write_csv` (not called at HEAD).

Notes
-----
* Bug: :meth:`Cuber.build` calls ``dist_filer.set_configuration('distortion',
  opticon)`` with two arguments; :meth:`lms_filer.Filer.set_configuration` requires
  three, so the method raises ``TypeError`` at its first statement after setup.
* Bug: :meth:`lmsiq_analyse.Analyse.extract_cube_strips` is called on down-sampled
  (detector-pixel) images with ``oversampling = 4``, so each cube column sums 10
  detector columns rather than the 2.5 pixel slice width the comment intends.
* Bug: if the target slice of a field is not present in the data set,
  ``field_series`` stays ``None`` and the per-axis assignments
  ``field_series[axis_tag + ...]`` raise ``TypeError``.
* Inconsistency: ``write_pickle = False`` is hard-coded twice so the LSF and EE
  profile pickles are never written, although the PNGs are.
* Clean-up: :class:`lms_wcal.Wcal` is instantiated in the constructor for no effect;
  :class:`lmsiq_fitsio.FitsIo`, ``Util``, ``Plot`` are used, ``Globals`` only for debug
  checks.
"""
import numpy as np
import time

from lms_globals import Globals
from lmsiq_fitsio import FitsIo
from lms_wcal import Wcal
from lmsdist_util import Util
from lms_filer import Filer
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot
from lmsiq_analyse import Analyse


class Cuber:
    """
    Static methods to build, filter, tabulate and plot lmsiq image-quality cubes.

    Cube package layout (as pickled by :meth:`build` and read by :meth:`read_pkl_cubes`)::

        cube_ident   = (cube_name, ipc_tag, field_no, spifu_no, config_no)
        cube         = ndarray (n_models, 32, n_cube_cols)
        field_series = dict with keys
            'ipc_on', 'field_no', 'spifu_no', 'config_no', 'dw_dlmspix' (nm/pixel),
            'lsf_gau_fwhms', 'lsf_lin_fwhms'   (spectral LSF of target slice, det. pix),
            'acr_gau_fwhms', 'acr_lin_fwhms'   (across-slice LSF of cube, slices),
            'alo_gau_fwhms', 'alo_lin_fwhms'   (along-slice LSF of cube, det. pix),
            'strehls', plus every key of the data-set dictionary ds_dict
            ('wavelength', 'prism_angle', 'grating_angle', 'grating_order',
            'focus_shift', 'folder', 'file_names', 'mc_bounds', ...).

    All per-model lists follow the convention index 0 = perfect, 1 = as-designed,
    2.. = Monte-Carlo instances.
    """

    def __init__(self):
        """
        Instantiate :class:`lms_wcal.Wcal` (no effect) and return.

        Notes
        -----
        * Clean-up: ``Wcal()`` has a no-op constructor; the call can be removed.
        """
        Wcal()
        return

    @staticmethod
    def build(data_identifier, process_control, image_manager, iq_filer, dist_filer, **kwargs):
        """
        Build pseudo-cubes and image-quality metrics for every data set in the model.

        Flow:

        1. ``dist_filer.set_configuration('distortion', opticon)`` (raises at HEAD, see
           Notes) and load the ``lmsdist`` traces from ``dist_filer.trace_file``.
        2. ``oversampling = int(Detector.det_pix_size / im_pix_size)`` = 4;
           ``n_runs = 2 if mc_bounds is None else mc_bounds[1] - mc_bounds[0] + 3``
           (perfect + design + MC instances); ``n_cube_rows = 128 / oversampling`` = 32.
        3. Loop over ``inter_pixels`` (calls :meth:`lms_ipc.Ipc.set_inter_pixel`), then
           field, spifu slice, configuration and focus shift.  For each field the slice
           range is ``slice_tgt +- slice_radius`` with
           ``slice_tgt = data_identifier['field_tgt_slice'][field_no]``; the cube has
           ``n_cube_cols = 2 * slice_radius + 1 + 2 * 3`` columns (3 empty margin
           columns each side).
        4. For each slice, ``image_manager.load_dataset`` returns the model stack and
           ``ds_dict``.  Each image is passed through :meth:`lms_ipc.Ipc.apply` (if
           ``inter_pixel``) and :meth:`lms_detector.Detector.down_sample`.
        5. Target slice only: :meth:`lmsiq_analyse.Analyse.lsf` on the IPC-processed
           (still oversampled) images with ``axis=0, oversample=4, v_coadd=12.0``
           gives the spectral LSF; :meth:`lmsiq_analyse.Analyse.get_dispersion`
           gives ``dw_dx`` (um/mm) and ``dw_lmspix = dw_dx * 18`` (nm/pixel);
           ``field_series`` is created with these and all ``ds_dict`` entries.
        6. Every slice: :meth:`lmsiq_analyse.Analyse.extract_cube_strips` collapses
           the down-sampled images to along-slice strips that fill cube column
           ``slice_no - slice_start + 3``.
        7. On the finished cube, for ``axis_name`` in ``('across-slice', 'along-slice')``:
           :meth:`lmsiq_analyse.Analyse.eed` (``oversample=1``, log sampling) and
           :meth:`lmsiq_analyse.Analyse.lsf` (``oversample=1, v_coadd=3.0``), each
           plotted with :meth:`lmsiq_plot.Plot.plot_cube_profile`; FWHM lists stored
           under ``'acr_'`` / ``'alo_'`` keys.  Then
           :meth:`lmsiq_analyse.Analyse.find_strehls`.
        8. Write the cube with :meth:`lmsiq_fitsio.FitsIo.write_cube` and pickle
           ``(cube_ident, cube, field_series)`` to ``cube/pkl/<cube_name>.pkl``.

        Parameters
        ----------
        data_identifier : dict
            From ``lmsiq.main``: ``'optical_path'``, ``'field_tgt_slice'`` (field ->
            target slice map), ``'slice_radius'`` and others.
        process_control : tuple
            ``(mc_bounds, inter_pixels)``; ``mc_bounds`` is None or ``(lo, hi)``,
            ``inter_pixels`` a list of bools.
        image_manager : lmsiq_image_manager.ImageManager
            Must have had ``make_dictionary`` run; supplies ``model_dict`` and
            ``unique_parameters``.
        iq_filer : lms_filer.Filer
            Configured for ``'iq'``; provides ``psf_folder``, ``output_folder``,
            ``cube_folder``.
        dist_filer : lms_filer.Filer
            Filer for the ``'distortion'`` analysis (re-configured here).
        **kwargs
            ``plot_to_png`` : bool, default False -- write slicer-image PNGs in debug
            mode.  ``debug`` is accepted by the caller but not read.

        Returns
        -------
        None

        Notes
        -----
        * Bug: two-argument ``set_configuration`` call (see module Notes).
        * Bug: ``extract_cube_strips`` receives down-sampled images but
          ``oversampling = 4`` (module Notes).
        * Bug: ``field_series`` is None if ``slice_tgt`` is not in the slice range
          actually loaded (module Notes).
        * Inconsistency: ``images is None`` triggers the message "exiting cuber.build"
          but the code continues (``continue``) to the next slice.
        * Inconsistency: the ``eed`` call passes ``normalise='to_average'`` which
          ``Analyse.eed`` ignores.
        * Inconsistency: the spectral LSF (step 5) is measured on IPC-processed but
          *not* down-sampled images and rescaled by ``oversample=4``, whereas the
          across/along-slice LSFs are measured on the down-sampled cube; the
          along-slice FWHM therefore includes pixel-integration broadening the spectral
          one does not.
        * Inconsistency: ``cube_dict['wavelength']`` is taken from the last-loaded
          ``ds_dict``, assumed identical across slices of one configuration.
        * Clean-up: ``plot_images`` and both ``write_pickle`` flags are hard-coded
          False; ``lsf_data`` for the target slice is overwritten by the cube LSFs
          before the loop ends.
        * Clean-up: the wall-clock progress line is printed with ``\\r`` and no final
          newline.
        """
        plot_to_png = kwargs.get('plot_to_png', False)
        t_start = time.perf_counter()
        ds_dict, lsf_data = None, None
        axes = ['across-slice', 'along-slice']
        opticon = data_identifier['optical_path']
        dist_filer.set_configuration('distortion', opticon)
        traces = dist_filer.read_pickle(dist_filer.trace_file)

        profile_folder = iq_filer.get_folder(iq_filer.output_folder + 'profiles')

        mc_bounds, inter_pixels = process_control
        im_pix_size = image_manager.model_dict['im_pix_size']
        oversampling = int(Detector.det_pix_size / im_pix_size)

        uni_par = image_manager.unique_parameters
        config_nos = uni_par['config_nos']
        field_nos = uni_par['field_nos']
        spifu_nos = uni_par['spifu_nos']
        slice_nos = uni_par['slice_nos']
        focus_shifts = uni_par['focus_shifts']
        n_runs = 2 if mc_bounds is None else mc_bounds[1] - mc_bounds[0] + 3
        n_cube_rows = int(128 / oversampling)
        for ipc_idx, inter_pixel in enumerate(inter_pixels):
            Ipc.set_inter_pixel(inter_pixel)
            ipc_tag = Ipc.tag

            print('\nData set comprises,')
            Util.print_list('field numbers ', field_nos)
            Util.print_list('configurations', config_nos)
            Util.print_list('slice numbers', slice_nos)
            Util.print_list('spifu numbers ', spifu_nos)
            Util.print_list('focus shifts', focus_shifts)
            print("no. of optical models (2 + Monte-Carlo instances) = {:d}".format(n_runs))
            slice_radius = data_identifier['slice_radius']

            for field_no in field_nos:
                fts = data_identifier['field_tgt_slice']
                slice_tgt = fts[field_no]
                field_tag = "field_{:d}".format(field_no)
                slice_start, slice_end = slice_tgt - slice_radius, slice_tgt + slice_radius
                slices_tag = "slices_{:d}_to_{:d}".format(slice_start, slice_end)
                slice_margin = 3
                n_cube_cols = slice_end - slice_start + 1 + 2 * slice_margin
                fmt = "\nBuilding cubes for field {:d}, using slices {:d} to {:d}"
                print(fmt.format(field_no, slice_start, slice_end))
                print("- target centred on slice = {:d}".format(slice_tgt))

                for spifu_idx, spifu_no in enumerate(spifu_nos):
                    spifu_tag = "spifu_{:d}".format(spifu_no)

                    for config_idx, config_no in enumerate(config_nos):
                        config_tag = "cfg_{:d}".format(config_no)

                        for focus_idx, focus in enumerate(focus_shifts):
                            focus_tag = "focus_{:d}".format(focus)
                            fmt = "cube_{:s}_{:s}_{:s}_{:s}_{:s}_{:s}"
                            cube_name = fmt.format(ipc_tag, field_tag, spifu_tag, config_tag, focus_tag, slices_tag)
                            cube_shape = n_runs, n_cube_rows, n_cube_cols
                            cube, cube_dict = None, None
                            field_series = None

                            for slice_no in range(slice_start, slice_end+1):
                                t_now = time.perf_counter()
                                t_min = (t_now - t_start) / 60.
                                if Globals.is_debug('low'):
                                    # Don't include IFU image in analysis, just plot it.
                                    selection = {'config_no': config_no,
                                                 'slice_no': slice_no,
                                                 'spifu_no': spifu_no,
                                                 'field_no': field_no,
                                                 'focus_shift': focus,
                                                 'focal_plane': 'slicer',
                                                 'mc_bounds': mc_bounds,
                                                 }
                                    slicer_obs_list = image_manager.load_dataset(iq_filer, selection)
                                    slicer_folder = '/slicer_image'
                                    plot_title = "slicer_config{:02d}_{:s}".format(config_no, ipc_tag)
                                    png_path = None
                                    if plot_to_png:
                                        png_folder = iq_filer.get_folder(iq_filer.output_folder + slicer_folder)
                                        png_path = png_folder + plot_title
                                    if Globals.is_debug('low'):
                                        Plot.images(slicer_obs_list[0][:4],
                                                    figsize=[8, 10], nrowcol=(2, 2), shrink=1.0, colourbar=True,
                                                    title=plot_title, do_log=True, png_path=png_path)

                                selection = {'config_no': config_no,
                                             'field_no': field_no,
                                             'focus_shift': focus,
                                             'focal_plane': 'det',
                                             'slice_no': slice_no,
                                             'spifu_no': spifu_no,
                                             'mc_bounds': mc_bounds,
                                             }

                                images, ds_dict = image_manager.load_dataset(iq_filer,
                                                                             selection,
                                                                             # xy_shift=(10, -5, -5),
                                                                             debug=False
                                                                             )
                                if images is None:      # No images found
                                    print('No images found, exiting cuber.build')
                                    continue

                                fmt = "\r- diffusion {:s}, field {:02d}, spifu_no {:01d}, configuration {:03d}," + \
                                      " focus {:03d}," + " extracting slice_no {:02d} at t= {:7.2f} min"
                                print(fmt.format(str(inter_pixel), field_no, spifu_no, config_no,
                                                 focus, slice_no, t_min),
                                      end="", flush=True)

                                # Collect fully processed images for image quality calculation
                                det_images, ipc_images = [], []
                                for zem_img in images:
                                    ipc_img = Ipc.apply(zem_img, oversampling) if inter_pixel else zem_img
                                    ipc_images.append(ipc_img)
                                    det_img = Detector.down_sample(ipc_img, im_pix_size)
                                    det_images.append(det_img)
                                # Find the line widths (perfect, design, mc_mean etc.) for the 'target' slice
                                if slice_no == slice_tgt:

                                    # Calculate line spread function parameters
                                    axis_idx, axis_name = 0, 'spectral'
                                    lsf_data = Analyse.lsf(ipc_images, ds_dict, axis_idx,
                                                           oversample=oversampling,
                                                           debug=False,
                                                           v_coadd=12.0,
                                                           u_radius='all')

                                    lsf_folder = Filer.get_folder(profile_folder + 'lsf')
                                    lsf_name = 'lsf_' + axis_name + '_' + cube_name
                                    lsf_path = lsf_folder + lsf_name
                                    write_pickle = False
                                    if write_pickle:
                                        iq_filer.write_pickle(lsf_path, (lsf_data, ds_dict))
                                    if Globals.is_debug('low'):
                                        Plot.plot_cube_profile('lsf', lsf_data, lsf_name,
                                                               ds_dict, axis_name,
                                                               xlim=[-6., 6.],
                                                               png_path=lsf_path)
                                    dw_dx = Analyse.get_dispersion(ds_dict, traces)  # micron / mm
                                    dw_lmspix = dw_dx * Detector.det_pix_size         # nm / pixel
                                    # Add FWHM values to summary table (params v wavelength/field etc.)
                                    field_series = {'ipc_on': inter_pixel, 'field_no': field_no,
                                                    'spifu_no': spifu_no, 'config_no': config_no,
                                                    'dw_dlmspix': dw_lmspix,
                                                    'lsf_gau_fwhms': lsf_data['gau_fwhm'],
                                                    'lsf_lin_fwhms': lsf_data['lin_fwhm']
                                                    }
                                    # field_series['wavelength'] = ds_dict['wavelength']
                                    for ds_key in ds_dict:
                                        field_series[ds_key] = ds_dict[ds_key]

                                # Add slice to data cube
                                col_idx = slice_no - slice_start + slice_margin
                                det_strips = Analyse.extract_cube_strips(det_images, oversampling)
                                if cube is None:
                                    cube = np.zeros(cube_shape)
                                    cube_dict = {'name': cube_name, 'field_no': field_no, 'mc_bounds': mc_bounds,
                                                 'wavelength': ds_dict['wavelength']}
                                cube[:, :, col_idx] = det_strips

                            if cube is None:
                                continue
                            plot_images = False
                            if plot_images:
                                pane_titles = ['perfect', 'design', 'MC-001', 'MC-002']
                                png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/png')
                                png_path = png_folder + cube_name
                                plot_image_list = list(cube[0:4])
                                Plot.images(plot_image_list,
                                            aspect='auto',
                                            title=cube_name, png_path=png_path,
                                            pane_titles=pane_titles, nrowcol=(2, 2))

                            for axis, axis_name in enumerate(axes):
                                ee_data = Analyse.eed(cube, cube_dict, axis_name,
                                                      oversample=1,
                                                      debug=False,
                                                      log10sampling=True,
                                                      normalise='to_average')

                                ee_folder = Filer.get_folder(profile_folder + 'ee')
                                ee_name = 'ee_' + axis_name + '_' + cube_name
                                ee_path = ee_folder + ee_name
                                write_pickle = False
                                if write_pickle:
                                    iq_filer.write_pickle(ee_path, (ee_data, cube_dict))
                                Plot.plot_cube_profile('ee', ee_data, ee_name, cube_dict, axis_name,
                                                       xlog=True, png_path=ee_path)

                                lsf_folder = Filer.get_folder(profile_folder + 'lsf')
                                lsf_name = 'lsf_' + axis_name + '_' + cube_name
                                lsf_path = lsf_folder + lsf_name
                                lsf_data = Analyse.lsf(cube, cube_dict, axis,
                                                       oversample=1,        # Cube is in detector pixels...
                                                       debug=False,
                                                       v_coadd=3.0,
                                                       u_radius='all')
                                axis_tag = axis_name[0:3]

                                field_series[axis_tag + '_gau_fwhms'] = lsf_data['gau_fwhm']
                                field_series[axis_tag + '_lin_fwhms'] = lsf_data['lin_fwhm']

                                Plot.plot_cube_profile('lsf', lsf_data, lsf_name,
                                                       ds_dict, axis_name,
                                                       png_path=lsf_path)

                            strehls = Analyse.find_strehls(cube)  # Find Strehls for all runs
                            field_series['spifu_no'] = spifu_no
                            field_series['strehls'] = strehls

                            FitsIo.write_cube(cube, cube_name, iq_filer)
                            cube_ident = cube_name, ipc_tag, field_no, spifu_no, config_no
                            cube_package = cube_ident, cube, field_series

                            cube_pkl_folder = iq_filer.get_folder(iq_filer.cube_folder + 'pkl')
                            cube_series_path = cube_pkl_folder + cube_name
                            iq_filer.write_pickle(cube_series_path, cube_package)
        return

    @staticmethod
    def read_pkl_cubes(iq_filer):
        """
        Load every cube package pickle from ``<cube_folder>/pkl/``.

        Files whose name contains ``'slicer'`` are excluded.

        Parameters
        ----------
        iq_filer : lms_filer.Filer
            Configured ``'iq'`` Filer; ``cube_folder`` is used.

        Returns
        -------
        cube_packages : list of tuple
            ``(cube_ident, cube, field_series)`` per file, in directory listing order.

        Notes
        -----
        * Open question: no filter on the IPC tag, so packages from ``ipc_on`` and
          ``ipc_off`` runs would be mixed if both had been built; :meth:`plot` only
          iterates over ``ipc_on``.
        """
        inc_tags, excl_tags = [], []
        cube_pkl_folder = iq_filer.get_folder(iq_filer.cube_folder + 'pkl')
        file_list = iq_filer.get_file_list(cube_pkl_folder,
                                           inc_tags=inc_tags,
                                           exc_tags=['slicer'])
        cube_packages = []
        for file in file_list:
            file_path = cube_pkl_folder + file
            cube_package = iq_filer.read_pickle(file_path)
            cube_packages.append(cube_package)
        return cube_packages

    @staticmethod
    def remove_configs(cube_packages, remove_config_list):
        """
        Drop cube packages whose ``field_series['config_no']`` is in a list.

        Parameters
        ----------
        cube_packages : list of tuple
            As returned by :meth:`read_pkl_cubes`.
        remove_config_list : list of int
            Configuration numbers to exclude (``[21]`` in ``lmsiq.main`` for the
            nominal path).

        Returns
        -------
        cube_packages_out : list of tuple

        Notes
        -----
        * Open question: ``lmsiq.main`` removes nominal configuration 21 as a "weird
          nominal configuration"; what is wrong with it is not recorded.
        """
        cube_packages_out = []
        for cube_package in cube_packages:
            cube_ident, cube, field_series = cube_package
            config_no = field_series['config_no']
            if config_no not in remove_config_list:
                cube_packages_out.append(cube_package)
        return cube_packages_out

    @staticmethod
    def write_csv(image_manager, cube_packages, iq_filer):
        """
        Write Monte-Carlo mean Strehl and LSF FWHM per (configuration, field) to a CSV.

        For each of ``'strehls'``, ``'lsf_gau_fwhms'``, ``'lsf_lin_fwhms'`` a table is
        built with one row per configuration (or spectral-IFU slice for ``'spifu'``),
        columns ``prism_angle, grating_angle, grating_order, wavelength (or spifu_no),
        field_1 .. field_n``, where the field value is ``mean(values[2:])`` (Monte-Carlo
        models only).  Rows are sorted by the fourth column.  The text block is printed
        and written to ``<output_folder>/cube/series/csv/series.csv``.

        Parameters
        ----------
        image_manager : lmsiq_image_manager.ImageManager
            Supplies ``model_dict['optical_path']`` and ``unique_parameters``.
        cube_packages : list of tuple
        iq_filer : lms_filer.Filer

        Returns
        -------
        None

        Notes
        -----
        * Inconsistency: ``is_spifu = optical_path == 'spifu'`` but the optical paths
          are ``'nominal'`` / ``'extended'`` (``Globals``), so the spifu branch is never
          taken.
        * Inconsistency: ``values`` is sorted in place inside the metric loop and then
          fully rewritten by row index on the next metric; this only works because
          every ``(row, field)`` cell is re-assigned each pass -- a missing package
          would leave a stale, mis-sorted value behind.
        * Clean-up: not called anywhere at HEAD.
        """
        model_dict = image_manager.model_dict
        optical_path = model_dict['optical_path']
        is_spifu = optical_path == 'spifu'
        unique_parameters = image_manager.unique_parameters
        n_fields = len(unique_parameters['field_nos'])
        n_wavelengths = len(unique_parameters['config_nos'])

        field_col_idx = 4
        n_rows = n_wavelengths
        n_cols = n_fields + field_col_idx
        values = np.zeros((n_rows, n_cols))
        csv_array_keys = {'strehls': (['<Strehl>', '-'], '10.3f'),
                          'lsf_gau_fwhms': (['<Gauss FWHM>', 'pix.'], '15.3f'),
                          'lsf_lin_fwhms': (['<Linear FWHM>', 'pix.'], '15.3f')}

        csv_text_block = ''
        for akey in csv_array_keys:
            for cube_package in cube_packages:
                _, _, field_series = cube_package
                field_no = field_series['field_no']
                col = field_no + field_col_idx - 1

                config_no = field_series['config_no']
                spifu_no = field_series['spifu_no']
                row = spifu_no - 1 if is_spifu else config_no - 1  # Row in data block
                values[row, 0] = field_series['prism_angle']
                values[row, 1] = field_series['grating_angle']
                values[row, 2] = field_series['grating_order']
                values[row, 3] = spifu_no if is_spifu else field_series['wavelength']

                vals = field_series[akey]
                val = np.mean(vals[2:])
                values[row, col] = val

            indices = np.argsort(values[:, 3])
            values = values[indices, :]
            common_fmts = ['{:8.4f},', '{:6.1f},', '{:6.0f},', '{:8.3f},']
            val_fmt = '{:8.3f},'
            csv_text_block += "\n"
            csv_text_block += "\n{:s}".format(akey)
            for row in range(0, n_rows):
                csv_text_block += "\n"
                for col in range(0, 4):
                    csv_text_block += common_fmts[col].format(values[row, col])
                for col in range(4, n_cols):
                    csv_text_block += val_fmt.format(values[row, col])
        print(csv_text_block)

        csv_folder = Filer.get_folder(iq_filer.output_folder + 'cube/series/csv')
        csv_path = csv_folder + 'series.csv'
        print("Filer.write_profiles to {:s}".format(csv_path))
        with open(csv_path, 'w', newline='') as csv_file:
            print(csv_text_block, file=csv_file)
        return

    @staticmethod
    def plot(optical_path, cube_packages, iq_filer, **kwargs):
        """
        Plot each image-quality metric against wavelength (or spifu slice), one panel per field.

        For every cube package the ``field_series`` metrics are appended to a per-field
        ``plot_data`` entry: ``lsf_gau_fwhms``, ``acr_gau_fwhms``, ``alo_gau_fwhms``,
        ``strehls`` (each an array over models) and the spectral resolving power
        ``srps = 1000 * wavelength / (lsf_gau_fwhms * dw_dlmspix)`` (wavelength in um,
        FWHM in detector pixels, dispersion in nm/pixel).  The abscissa is the
        wavelength for the nominal path or ``spifu_no`` for the spectral IFU.  Each
        metric is then drawn with :meth:`lmsiq_plot.Plot.series` using a 3 x 3 panel
        grid (nominal), 3 x 1 (spifu) or 1 x 3 (defocus), fixed y limits per metric and
        optical path, Monte-Carlo 10th / 90th percentile lines, and requirement
        overlays: MET-2745 for ``srps`` (nominal only) and MET-3739 for
        ``lsf_gau_fwhms``.  A separate legend-only ``key.png`` is written once.

        Parameters
        ----------
        optical_path : str
            ``'nominal'`` or ``'spifu'``; used to select layouts and limits.
        cube_packages : list of tuple
            From :meth:`read_pkl_cubes` (optionally filtered by :meth:`remove_configs`).
        iq_filer : lms_filer.Filer
        **kwargs
            ``is_defocus`` : bool, default False -- use the defocus layout / abscissa.

        Returns
        -------
        None

        Notes
        -----
        * Bug: ``fig_layouts[optical_path]`` and the limit dictionaries are keyed
          ``'spifu'`` / ``'nominal'``, but ``lmsiq.main`` passes ``opticon`` which is
          ``'nominal'`` or ``'extended'`` (``Globals``); the extended path raises
          ``KeyError``.
        * Inconsistency: the date stamp shown in titles is parsed from
          ``field_series['file_names'][0][4:14]``, i.e. from the Zemax file name, not
          from the Filer configuration.
        * Inconsistency: the 'defocus' abscissa entry is labelled "Wavelength" and
          keyed ``'defocus'``, but :meth:`lmsiq_plot.Plot.series` still plots
          ``x_values`` (wavelength) and distinguishes defocus values by marker.
        * Clean-up: ``ipc_on`` loops over ``[True]`` only; ``n_plots`` is unused.
        """
        print()
        is_defocus = kwargs.get('is_defocus', False)

        png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/series/png')

        is_spifu = optical_path == 'spifu'
        n_plots = len(cube_packages)
        fig_layouts = {'spifu': (3, 1, (6, 8)), 'nominal': (3, 3, (12, 8))}
        fig_layout = fig_layouts[optical_path]
        fig_layout = (1, 3, (12, 8)) if is_defocus else fig_layout

        key_only = True
        for ipc_on in [True]:
            plot_data = {}
            unique_field_nos = []
            datestamp = None
            for cube_package in cube_packages:
                _, _, field_series = cube_package
                if datestamp is None:
                    datestamp = field_series['file_names'][0][4:14]
                field_no = field_series['field_no']
                mc_bounds = field_series['mc_bounds']
                field_id = "field_{:d}".format(field_no)
                if field_no not in unique_field_nos:
                    unique_field_nos.append(field_no)
                    field_data = {'field_no': field_no, 'ipc_on': ipc_on, 'is_spifu': is_spifu,
                                  'mc_bounds': mc_bounds, 'focus_shifts': [],
                                  'x_values': [],
                                  'lsf_gau_fwhms': [], 'acr_gau_fwhms': [], 'alo_gau_fwhms': [],
                                  'strehls': [], 'srps': []}
                    plot_data[field_id] = field_data

                # Write parameters to the field_data sub-dictionary of plot_data
                field_data = plot_data[field_id]
                focus_shift = field_series['focus_shift']
                field_data['focus_shifts'].append(focus_shift)
                wave = field_series['wavelength']
                lsf_gau_fwhms = np.array(field_series['lsf_gau_fwhms'])
                field_data['lsf_gau_fwhms'].append(lsf_gau_fwhms)
                acr_gau_fwhms = np.array(field_series['acr_gau_fwhms'])
                field_data['acr_gau_fwhms'].append(acr_gau_fwhms)
                alo_gau_fwhms = np.array(field_series['alo_gau_fwhms'])
                field_data['alo_gau_fwhms'].append(alo_gau_fwhms)
                strehls = np.array(field_series['strehls'])
                field_data['strehls'].append(strehls)
                dw_dlmspix = field_series['dw_dlmspix']
                srps = 1000. * wave / (lsf_gau_fwhms * dw_dlmspix)
                field_data['srps'].append(srps)
                x_val = field_series['spifu_no'] if is_spifu else wave
                field_data['x_values'].append(x_val)

            abs_id = optical_path
            abscissae = {'spifu': ('spifu_no', 'Spectral IFU slice no.', [0.5, 6.5]),
                         'nominal': ('wavelength', r'Wavelength $\mu$m', [2.67, 6.72]),
                         'defocus': ('defocus', r'Wavelength $\mu$m', [2.67, 6.72])
                         }
            abs_id = 'defocus' if is_defocus else abs_id
            abscissa = abscissae[abs_id]

            lsf_fwhm_lims = {'spifu': ([2.5, 2.9]), 'nominal': ([1.5, 3.5])}
            acr_fwhm_lims = {'spifu': ([0.5, 2.0]), 'nominal': ([0.3, 2.0])}
            alo_fwhm_lims = {'spifu': ([2.5, 5.0]), 'nominal': ([1.5, 8.0])}
            srp_lims = {'spifu': [90000, 120000], 'nominal': [70000, 160000]}
            ordinates = {'lsf_gau_fwhms': ('lsf_gau_fwhms', 'FWHM / pix.', lsf_fwhm_lims[optical_path]),
                         'acr_gau_fwhms': ('acr_gau_fwhms', 'FWHM / slice', acr_fwhm_lims[optical_path]),
                         'alo_gau_fwhms': ('alo_gau_fwhms', 'FWHM / pix.', alo_fwhm_lims[optical_path]),
                         'srps': ('srps', r'SRP ($\lambda / \Delta\lambda$)', srp_lims[optical_path]),
                         'strehls': ('strehls', 'Strehl', [0.4, 1.1])}
            mc_percentiles = [(10, 'dashed', 1.0), (90, 'dashed', 1.0)]

            for ykey in ordinates:
                do_fwhm_rqt = ykey == 'lsf_gau_fwhms'
                do_srp_rqt = ykey == 'srps' and not is_spifu

                ova_tag = "{:s}_v_{:s}".format(ykey, abscissa[0])
                fmt = "{:s}_{:s}_{:s}"
                ipc_tag = 'ipc_on' if ipc_on else 'ipc_off'
                png_file = fmt.format(optical_path, ova_tag, ipc_tag)
                png_path = png_folder + png_file
                # png_path = None
                ordinate = ordinates[ykey]
                fmt = "\r- Plotting diffusion {:s}, spectral IFU= {:s}, to file {:s}"
                print(fmt.format(str(ipc_on), str(is_spifu), png_file))
                op_text = 'extended spectral coverage' if is_spifu else 'nominal spectral coverage'
                title = "{:s}, {:s}\n{:s}, {:s}".format(datestamp, ova_tag, op_text, ipc_tag)
                plot_parameters = {'fig_layout': fig_layout, 'title': title,
                                   'key_only': False, 'plot_key': False,
                                   'abscissa': abscissa, 'ordinate': ordinate,
                                   'xscale': 'auto',
                                   'mc_percentiles': mc_percentiles, 'png_path': png_path,
                                   'do_srp_rqt': do_srp_rqt, 'do_fwhm_rqt': do_fwhm_rqt
                                   }
                Plot.series(plot_data, plot_parameters)

                if key_only:
                    # Explicitly plot a key that can be pasted into the report
                    plot_parameters['png_path'] = png_folder + 'key'
                    plot_parameters['do_srp_rqt'] = False
                    plot_parameters['do_fwhm_rqt'] = False
                    plot_parameters['key_only'] = key_only
                    plot_parameters['plot_key'] = True
                    Plot.series(plot_data, plot_parameters)
                    key_only = False
        return
