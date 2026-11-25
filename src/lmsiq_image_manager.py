"""
Catalogue and loader for Zemax PSF image stacks used by the ``lmsiq`` program.

Purpose
-------
:class:`ImageManager` scans the Zemax data tree once (:meth:`ImageManager.make_dictionary`),
decoding configuration, field and defocus numbers from folder names, wavelength and
grating parameters from the per-folder ``.txt`` file, and slice / spectral-IFU / Monte-Carlo
numbers from the FITS file names.  :meth:`ImageManager.load_dataset` then returns, for one
(configuration, field, focus, slice, spifu) selection, the ordered list of model images
[perfect, as-designed, MC_lo .. MC_hi] together with a flat ``ds_dict`` of the selection's
parameters.  All results are held in class attributes.

Input data
----------
``../data/iq/<opticon>/<date_stamp>/`` (``iq_filer.psf_folder``), containing one
sub-folder per Zemax configuration.  Folder names carry ``config<nnn>``, ``field<nnn>``
and optionally ``defoc<nnn>`` (three digits after each code; ``rfind`` is used, so the
code may appear anywhere).  Each folder holds:

* one ``.txt`` parameter file, colon-separated ``key: value`` lines with keys
  ``wavelength`` (um), ``prism angle`` (deg), ``grating angle`` (deg), ``grating order``
  and ``Pixel size (micron)``; spectral-IFU data sets omit the grating keys and receive
  defaults ``prism_angle = 7.0``, ``grating_angle = 0.0``, ``grating_order = -99``;
* FITS files ``<folder>_spat<ss>_spec<p>_det<tag>.fits`` (image in the primary HDU,
  128 x 128 pixels at 4.5 um = 4x oversampled 18 um H2RG pixels) where ``spat`` is
  followed by a 2-digit spatial slice number, ``spec`` by a 1-digit spectral-IFU slice
  number and ``det`` by ``perf`` (perfect model), ``desi`` (as-designed) or a 4-digit
  Monte-Carlo instance number; files containing ``sli`` (slicer-plane images) are
  ignored.

No FITS header keywords are read; all metadata come from names and the ``.txt`` file.

Outputs
-------
None written to disk.

Notes
-----
* Inconsistency: ``im_pix_size`` is hard-coded to 4.5 um before the parameter file is
  read; the file's ``Pixel size (micron)`` overrides it only if present.
* Inconsistency: the folder-code loop appends to ``config_nos``, ``field_nos`` and
  ``focus_shifts`` independently, so if any folder lacks a code (e.g. no ``defoc``)
  the lists have different lengths and :meth:`load_dataset` indexes them with a common
  ``dataset_idx``; ``field_no = config_dict['field_nos'][-1]`` also assumes every folder
  has a ``field`` code.
* Bug: :meth:`load_dataset` computes ``n_mcs = mc_bounds[1] - mc_bounds[0] + 3``
  unconditionally, so ``mc_bounds = None`` (perfect + design only) raises ``TypeError``
  despite the ``if mc_bounds is not None`` guard just above.
* Clean-up: about 130 lines of the previous ``make_dictionary_old`` are kept as a
  comment block; ``_find_new_zemax_parameters`` and ``_make_inc_tags`` are unused
  legacy; the ``'field position'`` branch of ``_read_param_file`` is unreachable.
* Open question: a comment notes "the slice number information is wrong in the new
  zemax format (since start of 2024)"; which quantity is wrong and how it is
  compensated is not recorded.
"""
import numpy as np
from astropy.io import fits


class ImageManager:
    """
    Static catalogue of a Zemax PSF data set and loader for individual model stacks.

    Attributes
    ----------
    model_dict : dict or None
        Data-set-wide values: ``'optical_path'``, ``'im_pix_size'`` (um),
        ``'fits_trailer'`` (always None), ``'mc_bounds'`` (``(lo, hi)`` or None).
    config_dict : dict or None
        Per-folder lists: ``'folders'``, ``'config_nos'``, ``'field_nos'``,
        ``'focus_shifts'``, ``'slice_tgts'``, ``'slice_nos'`` (list of lists),
        ``'spifu_nos'`` (list of lists), ``'wavelengths'``, ``'prism_angles'``,
        ``'grating_angles'``, ``'grating_orders'``.
    unique_parameters : dict or None
        For each ``config_dict`` key, the list of distinct values (flattened for the
        list-of-lists keys).
    is_new_zemax : None
        Legacy flag, never set.

    Notes
    -----
    * Clean-up: state is class-level, so two data sets cannot be open at once.
    """
    model_dict, config_dict = None, None
    unique_parameters = None
    is_new_zemax = None

    def __init__(self):
        """No-op constructor; call :meth:`make_dictionary` to populate the class."""
        return

    @staticmethod
    def make_dictionary(data_identifier, iq_filer):
        """
        Scan the PSF folder tree and populate ``model_dict``, ``config_dict`` and ``unique_parameters``.

        For each sub-folder of ``iq_filer.psf_folder`` (excluding names containing
        ``.csv``):

        1. Decode ``config``, ``field`` and ``defoc`` codes (3 digits after the code)
           into ``config_nos``, ``field_nos``, ``focus_shifts``.
        2. Look up the target slice for the field in
           ``data_identifier['field_tgt_slice']``; set ``optical_path`` and the default
           ``im_pix_size = 4.5``.
        3. Read the first ``.txt`` file with :meth:`_read_param_file`; keys present in
           ``model_dict`` overwrite it, keys whose plural is in ``config_dict`` are
           appended there, others are reported.
        4. List the ``.fits`` files (excluding ``.txt`` and ``sli``); collect distinct
           ``spat`` (2-digit) and ``spec`` (1-digit) numbers, and for files that are
           neither ``desi`` nor ``perf`` the 4-digit number after ``det`` as an MC
           instance.  ``mc_bounds`` becomes ``(min, max)`` of all MC numbers seen.
        5. :meth:`_find_unique_parameters` builds ``unique_parameters``.

        Parameters
        ----------
        data_identifier : dict
            Needs ``'field_tgt_slice'`` (field number -> target slice) and
            ``'optical_path'``.
        iq_filer : lms_filer.Filer
            Configured ``'iq'`` Filer; ``psf_folder`` and ``get_file_list`` are used.

        Returns
        -------
        None

        Raises
        ------
        IndexError
            If a folder contains no ``.txt`` file (a message is printed first) or lacks
            a ``field`` code.
        KeyError
            If a decoded field number is missing from ``field_tgt_slice``.

        Notes
        -----
        * Inconsistency: ``mc_bounds`` is the overall min/max across *all* folders; a
          folder with fewer MC instances will make :meth:`load_dataset` look for
          missing files.
        * Inconsistency: ``focus_shifts`` receives entries only from folders with a
          ``defoc`` code (see module Notes on list alignment).
        """
        model_dict = {'optical_path': None, 'im_pix_size': None,
                      'fits_trailer': None, 'mc_bounds': None,
                      }
        config_dict = {'folders': [],
                       'config_nos': [],
                       'field_nos': [],
                       'focus_shifts': [],
                       'slice_tgts': [],
                       'slice_nos': [], 'spifu_nos': [],
                       'wavelengths': [],
                       'prism_angles': [], 'grating_angles': [], 'grating_orders': [],
                       }
        mc_code, mc_width = 'det', 4
        mc_codelen = len(mc_code)

        psf_folder = iq_filer.psf_folder
        folder_list = iq_filer.get_file_list(psf_folder, exc_tags=['.csv'])
        folder_codes = {'config': 'config_nos',
                        'field': 'field_nos',
                        'defoc': 'focus_shifts'}

        for folder in folder_list:
            config_dict['folders'].append(folder)
            for code in folder_codes:       # Add folder-wide values
                if code in folder:
                    codelen = len(code)
                    c1 = folder.rfind(code) + codelen
                    cfg_key = folder_codes[code]
                    cfg_tag = folder[c1: c1+3]
                    config_dict[cfg_key].append(int(cfg_tag))

            field_no = config_dict['field_nos'][-1]
            fts = data_identifier['field_tgt_slice']
            slice_tgt = fts[field_no]
            optical_path = data_identifier['optical_path']
            model_dict['optical_path'] = optical_path
            model_dict['im_pix_size'] = 4.5                 # Default image pixel size in microns
            config_dict['slice_tgts'].append(slice_tgt)

            fits_folder = psf_folder + folder + '/'
            par_files = iq_filer.get_file_list(fits_folder, inc_tags=['.txt'])
            if len(par_files) == 0:
                print("Text file not found in {:s}".format(fits_folder))
            par_path = fits_folder + par_files[0]
            # Note that the slice number information is wrong in the new zemax format (since start of 2024)
            parameters = ImageManager._read_param_file(par_path)

            for key in parameters:
                p = parameters[key]
                if key in model_dict:
                    model_dict[key] = p
                    continue
                cfg_key = key + 's'
                if cfg_key in config_dict:
                    config_dict[cfg_key].append(p)
                    continue
                print("Keyword {:s} not found in model_dict or config_dict".format(key))

            inc_tags = ['.fits']
            exc_tags = ['.txt', 'sli']
            fits_files = iq_filer.get_file_list(fits_folder,
                                                inc_tags=inc_tags,
                                                exc_tags=exc_tags)

            # Read in and decode Monte-Carlo file names and slice/spifu numbers.
            mc_no_list = []
            slice_no_list, spifu_no_list = [], []
            for fits_file in fits_files:
                file_codes = {'spat': ('slice_no', 2, slice_no_list),
                              'spec': ('spifu_no', 1, spifu_no_list)
                              }
                for code in file_codes:
                    if code in fits_file:
                        codelen = len(code)
                        c1 = fits_file.rfind(code) + codelen
                        cfg_key, width, val_list = file_codes[code]
                        val = int(fits_file[c1: c1 + width])
                        if val not in val_list:
                            val_list.append(val)

                if 'sli' in fits_file:
                    continue
                if ('desi' in fits_file) or ('perf' in fits_file):
                    continue
                if mc_code in fits_file:
                    c1 = fits_file.rfind(mc_code) + mc_codelen
                    mc_tag = fits_file[c1: c1 + mc_width]
                    # print(folder, mc_tag)
                    mc_no = int(mc_tag)
                    mc_no_list.append(mc_no)
            if len(mc_no_list) > 0:             # If 'MC' data is included in dataset
                mc_array = np.array(mc_no_list)
                mc_start, mc_end = np.amin(mc_array), np.amax(mc_array)
                model_dict['mc_bounds'] = mc_start, mc_end
            config_dict['slice_nos'].append(slice_no_list)
            config_dict['spifu_nos'].append(spifu_no_list)

        ImageManager._find_unique_parameters(config_dict)
        ImageManager.model_dict = model_dict
        ImageManager.config_dict = config_dict
        return

    # @staticmethod
    # def make_dictionary_old(data_identifier, iq_filer):
    #     """ Make the image table dictionary which describes all data for this model as a dictionary object
    #     (config_dict) which has a single entry for each folder/configuration in the model data.  Within a
    #     folder there are multiple files ('perfect', 'design' and M-C instances) and may be multiple field
    #     positions, spatial and spectral slices, so these are stored as lists.  A 'dataset' is then defined
    #     as the unique set of files for a single value of all these parameters,
    #     eg, folder/config_a, field_b, spat_slice_c, spec_slice_d.
    #     Method 'load_dataset' then returns the images plus a dictionary (obs_dict) containing the single
    #     values corresponding to the selected data.
    #     """
    #     # is_new_zemax = data_identifier['zemax_format'] == 'new_zemax'
    #     # ImageManager.is_new_zemax = is_new_zemax
    #     model_dict = {'optical_path': None,
    #                   'im_pix_size': None,
    #                   'fits_trailer': None,
    #                   'mc_bounds': None,
    #                   }
    #     config_dict = {'folders': [],
    #                    'config_nos': [],
    #                    'field_nos': [],
    #                    'focus_shifts': [],
    #                    'slice_nos': [], 'slice_tags': [], 'slice_tgts': [],
    #                    'spifu_nos': [],
    #                    'wavelengths': [],
    #                    'prism_angles': [], 'grating_angles': [], 'grating_orders': [],
    #                    }
    #     dataset_folder = iq_filer.data_folder
    #     folder_list = iq_filer.get_file_list(dataset_folder, exc_tags=['.csv'])
    #     config_no_list, field_no_list = [], []
    #     for folder in folder_list:
    #         is_valid_field = False
    #         slice_fields = data_identifier['slice_fields']
    #         for key in slice_fields:
    #             field_nos = slice_fields[key]
    #             for field_no in field_nos:
    #                 field_text = '_field{:d}'.format(field_no)
    #                 is_valid_field = True if field_text in folder else is_valid_field
    #         if not is_valid_field:
    #             continue
    #         print('Folder {:s} included in analysis'.format(folder))
    #
    #         optical_path = data_identifier['optical_path']
    #         model_dict['optical_path'] = optical_path
    #         config_dict['folders'].append(folder)
    #         slice_tgt = 13
    #         config_dict['slice_tgts'].append(slice_tgt)
    #
    #         fits_folder = dataset_folder + folder + '/'
    #         par_files = iq_filer.get_file_list(fits_folder, inc_tags=['.txt'])
    #         par_path = fits_folder + par_files[0]
    #         # Note that the slice number information is wrong in the new zemax format (since start of 2024)
    #         parameters = ImageManager._read_param_file(par_path)
    #         for key in parameters:
    #             p = parameters[key]
    #             if key in model_dict:
    #                 model_dict[key] = p
    #                 continue
    #             cfg_key = key + 's'
    #             if cfg_key in config_dict:
    #                 config_dict[cfg_key].append(p)
    #                 continue
    #             print("Keyword {:s} not found in model_dict or config_dict".format(key))
    #
    #         if data_identifier['optical_path'] == 'spifu':
    #             config_dict['prism_angles'].append(6.99716)
    #             config_dict['grating_angles'].append(0.00000)
    #             config_dict['grating_orders'].append(-23)
    #
    #         # Get list of file names for a single run (e.g. identical except for MC run no., 'perfect' or 'design' tags
    #         if optical_path == 'nominal':
    #             exc_tags = ['specslice', 'preslice']
    #             fits_trailer = '_MC_spatslice.fits'
    #         else:
    #             exc_tags = ['spatslice', 'preslice']
    #             fits_trailer = '_MC_specslice.fits'
    #         model_dict['fits_trailer'] = fits_trailer
    #         inc_tags = ['.fits']
    #         fits_files = iq_filer.get_file_list(fits_folder,
    #                                             inc_tags=inc_tags,
    #                                             exc_tags=exc_tags)
    #
    #         tokens = folder.split('_')
    #         field_no = int(tokens[3][-2:])
    #         field_no_list.append(field_no)
    #         config_no = int(tokens[4][-2:])
    #         config_no_list.append(config_no)
    #         if len(tokens) > 4:     # Defocus data present
    #             focus_shift = float(tokens[5][1:4])
    #             config_dict['focus_shifts'].append(int(focus_shift))
    #
    #         # Read in and decode Monte-Carlo file names and slice/spifu numbers.
    #         mc_no_list = []
    #         slice_no_list, slice_tag_list, spifu_no_list = [], [], []
    #         for fits_file in fits_files:
    #
    #             tokens = fits_file.split('_')
    #             slice_idx = 3
    #             spifu_idx = slice_idx + 1
    #             if 'slicer' in fits_file:
    #                 continue
    #             slice_tag = tokens[slice_idx]
    #             slice_no = int(slice_tag)
    #             spifu_no = int(tokens[spifu_idx])
    #             is_new_slice = slice_tag not in slice_tag_list
    #             is_new_spifu = spifu_no not in spifu_no_list
    #             if is_new_slice or is_new_spifu:
    #                 slice_tag_list.append(slice_tag)
    #                 slice_no_list.append(slice_no)
    #                 spifu_no_list.append(spifu_no)
    #
    #             if ('design' in fits_file) or ('perfect' in fits_file):
    #                 continue
    #             if '_MC_' in fits_file:
    #                 mc_idx = spifu_idx + 1
    #                 mc_tag = tokens[mc_idx]
    #
    #                 if mc_tag.isdigit():
    #                     mc_no = int(mc_tag)
    #                     mc_no_list.append(mc_no)
    #         if len(mc_no_list) > 0:             # If 'MC' data is included in dataset
    #             mc_array = np.array(mc_no_list)
    #             mc_start, mc_end = np.amin(mc_array), np.amax(mc_array)
    #             model_dict['mc_bounds'] = mc_start, mc_end
    #         config_dict['slice_tags'].append(slice_tag_list)
    #         config_dict['slice_nos'].append(slice_no_list)
    #         spifu_no_list = [0] if len(spifu_no_list) == 0 else spifu_no_list
    #         config_dict['spifu_nos'].append(spifu_no_list)
    #         config_dict['config_nos'].append(config_no)
    #         config_dict['field_nos'].append(field_no)
    #
    #     ImageManager._find_unique_parameters(config_dict)
    #     ImageManager.model_dict = model_dict
    #     ImageManager.config_dict = config_dict
    #     return

    @staticmethod
    def _find_new_zemax_parameters(config_no, optical_configuration, date_stamp):
        """
        Legacy: infer spatial / spectral slice numbers from the configuration number.

        Valid only for date stamps ``20240209``, ``20240229``, ``20240305``: nominal
        ``spat = 12 + (config_no - 1) % 3``; spifu ``spat = 12 + (config_no - 1) % 6``,
        ``spec = (config_no - 1) % 3``.

        Parameters
        ----------
        config_no : int
        optical_configuration : {'nominal', 'spifu'}
        date_stamp : str

        Returns
        -------
        spat_slice_no : int or None
        spec_slice_no : int or None
            -1 where not applicable; ``(None, None)`` for an unsupported date stamp.

        Notes
        -----
        * Clean-up: unused at HEAD; slice numbers now come from file names.
        """
        if date_stamp not in ['20240209', '20240229', '20240305']:
            print("ImageManager._find_new_zemax_parameters - NOT VALID FOR {:s}".format(date_stamp))
            return None, None
        spat_slice_no, spec_slice_no = -1, -1
        config_index = config_no - 1
        if optical_configuration == 'nominal':
            spat_slice_no = 12 + config_index % 3
        if optical_configuration == 'spifu':
            spat_slice_no = 12 + config_index % 6
            spec_slice_no = config_index % 3
        return spat_slice_no, spec_slice_no

    @staticmethod
    def _find_unique_parameters(image_table):
        """
        Build ``ImageManager.unique_parameters``: distinct values per ``config_dict`` key.

        ``'slice_nos'`` and ``'spifu_nos'`` are lists of lists and are flattened; order
        of first appearance is kept (no sorting).

        Parameters
        ----------
        image_table : dict
            The ``config_dict``.

        Returns
        -------
        None
        """
        uni_par = {}
        for key in image_table:
            uni_par[key] = []
            values = image_table[key]
            for value in values:
                if key in ['slice_nos', 'spifu_nos']:      # Trap special case of list of lists
                    for val_item in value:
                        if val_item in uni_par[key]:
                            continue
                        uni_par[key].append(val_item)
                    continue
                if value in uni_par[key]:
                    continue
                uni_par[key].append(value)
        ImageManager.unique_parameters = uni_par
        return

    @staticmethod
    def _read_param_file(path):
        """
        Parse a Zemax ``.txt`` parameter file into a dictionary.

        Lines of the form ``key: value`` are matched against
        ``{'wavelength': 'wavelength' (float), 'prism angle': 'prism_angle' (float),
        'grating angle': 'grating_angle' (float), 'grating order': 'grating_order'
        (int), 'Pixel size (micron)': 'im_pix_size' (float)}``; other keys are ignored.
        Defaults ``prism_angle = 7.0``, ``grating_angle = 0.0``, ``grating_order = -99``
        are set first so that spectral-IFU files, which lack the grating keys, still
        yield complete dictionaries.

        Parameters
        ----------
        path : str
            Path to the ``.txt`` file.

        Returns
        -------
        params : dict
            Keys as above (``wavelength`` in um, angles in degrees).

        Notes
        -----
        * Clean-up: the ``'field position'`` branch (``[x, y]`` vs integer formats) is
          dead code because ``'field position'`` is not in ``param_translator`` and
          is skipped by the ``continue`` above it; the ``'2fl_2fl'`` and ``'str_str'``
          formats are therefore never used.
        * Open question: a value that cannot be parsed (e.g. an empty string) raises
          ``ValueError``; there is no error handling.
        """
        params = {}
        pf = open(path, 'r')
        lines = pf.read().splitlines()
        pf.close()
        # Note, field position format changed from [x, y] to single integer in 20240209 dataset
        param_translator = {'wavelength': ('wavelength', 'fl_fl'),
                            'prism angle': ('prism_angle', 'fl_fl'),
                            'grating angle': ('grating_angle', 'fl_fl'),
                            'grating order': ('grating_order', 'fl_int'),
                            'Pixel size (micron)': ('im_pix_size', 'fl_fl')
                            }
        # Set default parameters for SPIFU mode, they don't appear in spifu parameter files.
        params['prism_angle'] = 7.0
        params['grating_angle'] = 0.0
        params['grating_order'] = -99
        for line in lines:
            if ':' in line:
                tokens = line.split(':')
                if tokens[0] not in param_translator:
                    continue
                key_in = tokens[0]
                value = tokens[1]
                key, fmt = param_translator[key_in]

                if key_in == 'field position':
                    if '[' in value:     # Trap change in field position/number format
                        key = 'field_xy'
                        fmt = '2fl_2fl'
                        params['field_no'] = 0
                    else:
                        key = 'field_no'
                        fmt = 'fl_int'
                        params['field_xy'] = [0., 0.]
                if fmt == 'fl_fl':
                    params[key] = float(value)
                    continue
                if fmt == 'fl_int':
                    params[key] = int(value)
                    continue
                if fmt == '2fl_2fl':
                    v1, v2 = value.split(',')
                    v1 = float(v1.replace('[', ''))
                    v2 = float(v2.replace(']', ''))
                    params[key] = float(v1), float(v2)
                if fmt == 'str_str':
                    params[key] = value.strip()
        return params

    @staticmethod
    def read_zemax_image(path):
        """
        Read the primary-HDU image of a Zemax FITS file as a new array.

        Parameters
        ----------
        path : str

        Returns
        -------
        image : ndarray
            Copy of the primary HDU data (128 x 128 for the standard PSFs).

        Notes
        -----
        * Clean-up: the HDU list is not closed; :meth:`load_dataset` duplicates this
          code inline rather than calling it.
        """
        hdu_list = fits.open(path, mode='readonly')
        hdu = hdu_list[0]
        image_in = hdu.data
        image = np.array(image_in)  # Make a copy of the raw image (maybe shift it).
        return image

    @staticmethod
    def load_dataset(iq_filer, selection, **kwargs):
        """
        Load the [perfect, design, MC..] image stack for one selection.

        Algorithm:

        1. For each key of ``selection`` find the ``config_dict`` keys that *contain*
           it as a substring (``'slice_no'`` matches ``'slice_nos'`` but also
           ``'field_no'`` matches ``'field_nos'`` etc.) and collect the folder indices
           whose value equals (or, for list-valued entries, contains) the selection
           value.  Intersect the index lists over all keys; exactly one folder index
           must remain, otherwise ``(None, None)`` is returned.
        2. Build ``ds_dict`` from that folder: every ``config_dict`` key with its final
           ``'s'`` stripped, the requested ``slice_no`` / ``spifu_no`` (looked up in the
           folder's lists), all ``model_dict`` entries, and all ``kwargs``.
        3. ``obs_tag = '<folder>_spat<ss>_spec<p>_det'``; model tags are ``['perf',
           'desi']`` plus ``'%04d' % n`` for ``n`` in ``mc_bounds``.
        4. Check the folder file count against ``n_slices * n_spifus * (n_mc + 2) + 1``
           (one ``.txt``); print a warning if different.
        5. For each tag, list files containing ``[tag, obs_tag, '.fits']`` and not
           ``'sli'``, warn if not exactly one, and read the primary HDU of the first.
           If ``xy_shift = (sampling, dr, dc)`` is given, the image is super-sampled by
           ``sampling`` (block replication), rolled by ``(dr, dc)`` super-pixels with
           ``np.roll`` (wrap-around) and block-averaged back.

        Parameters
        ----------
        iq_filer : lms_filer.Filer
            ``psf_folder`` and ``get_file_list`` are used.
        selection : dict
            Must contain ``'slice_no'``, ``'spifu_no'``, ``'mc_bounds'`` and the folder
            selectors (``'config_no'``, ``'field_no'``, ``'focus_shift'``; ``Cuber``
            also passes ``'focal_plane'``, which matches no ``config_dict`` key and is
            ignored).
        **kwargs
            ``debug`` : bool, default False.
            ``xy_shift`` : tuple ``(sampling, dr, dc)`` or None.
            Every kwarg is also copied into ``ds_dict``.

        Returns
        -------
        images : list of ndarray or None
            ``[perfect, design, MC_lo, ..., MC_hi]`` images; None if the selection is
            not unique.
        ds_dict : dict or None
            Selection parameters plus ``'file_names'`` (list, same order as
            ``images``).

        Raises
        ------
        TypeError
            If ``selection['mc_bounds']`` is None (``n_mcs`` arithmetic).
        IndexError
            If no file matches a tag (a message is printed, then ``file_list[0]``
            fails).

        Notes
        -----
        * Bug: ``n_mcs = mc_bounds[1] - mc_bounds[0] + 3`` is evaluated even when
          ``mc_bounds`` is None (module Notes).
        * Inconsistency: substring matching in step 1 means a selection key such as
          ``'no'`` would match several ``config_dict`` keys; the current keys happen to
          be unambiguous.
        * Inconsistency: ``mc_bounds`` in ``ds_dict`` comes from ``model_dict`` (the
          folder scan) rather than from ``selection``; the two can differ when
          ``lmsiq.main`` overrides the bounds.
        * Clean-up: the ``xy_shift`` path is two pure-Python double loops over
          ``sampling^2 * n_pix`` elements; ``np.repeat`` / ``reshape().mean()`` would
          be equivalent.  It is commented out in every caller.
        * Clean-up: ``kwargs`` (``debug``, ``xy_shift``) leak into ``ds_dict`` and from
          there into the pickled ``field_series``.
        """
        model_dict = ImageManager.model_dict
        config_dict = ImageManager.config_dict

        debug = kwargs.get('debug', False)
        xy_shift = kwargs.get('xy_shift', None)
        slice_no = selection['slice_no']
        spifu_no = selection['spifu_no']

        # Find the single data set which matches the selection parameters.
        all_indices = {}
        for kw_key in selection:
            indices = []
            for config_key in config_dict:
                if kw_key in config_key:
                    kw_val = selection[kw_key]
                    cfg_vals = config_dict[config_key]
                    for idx, cfg_val in enumerate(cfg_vals):
                        is_list = isinstance(cfg_val, list)
                        if is_list:
                            if kw_val in cfg_val:
                                indices.append(idx)
                        else:
                            if kw_val == cfg_val:
                                indices.append(idx)
            if len(indices) > 0:
                all_indices[kw_key] = indices

        # Find the index that is common to all 'indices' lists
        common_indices = None
        for idx_key in all_indices:
            indices = all_indices[idx_key]
            if common_indices is None:
                common_indices = indices
                continue
            new_cis = []
            for cidx in common_indices:
                if cidx in indices:
                    new_cis.append(cidx)
            common_indices = new_cis

        n_cis = len(common_indices)
        if n_cis != 1:
            if debug:
                print('!! ImageManager.load_dataset - unique data set not found !!')
                for item in selection:
                    print("{:12s}, {:s}".format(item, str(selection[item])))
            return None, None
        dataset_idx = common_indices[0]

        # Generate the dictionary for this dataset ('ds') combining config, model and keyword data
        ds_dict = {}
        for config_key in config_dict:
            ds_key = config_key[:-1]          # Strip final 's' off keyword (slice_nos -> slice_no etc.)
            ds_value = config_dict[config_key][dataset_idx]
            if ds_key == 'slice_no':
                slice_nos = np.array(config_dict['slice_nos'][dataset_idx])
                list_index = np.argwhere(slice_nos == slice_no)[0][0]
                ds_dict[ds_key] = ds_value[list_index]
                continue
            if ds_key == 'spifu_no':
                spifu_nos = np.array(config_dict['spifu_nos'][dataset_idx])
                list_index = np.argwhere(spifu_nos == spifu_no)[0][0]
                ds_dict[ds_key] = ds_value[list_index]
                continue
            ds_dict[ds_key] = ds_value
        for model_key in model_dict:
            ds_dict[model_key] = model_dict[model_key]
        for kw_key in kwargs:
            ds_dict[kw_key] = kwargs[kw_key]

        folder = ds_dict['folder']
        fits_folder = iq_filer.psf_folder + folder + '/'

        slice_no = ds_dict['slice_no']
        spifu_no = ds_dict['spifu_no']
        fmt = "{:s}_spat{:02d}_spec{:01d}_det"
        obs_tag = fmt.format(folder, slice_no, spifu_no)

        # Load perfect and design images first, then MC images if requested (mc_bounds is not None)
        mc_tags = ['perf', 'desi']           # Initialise mc tag list for old format
        mc_bounds = selection['mc_bounds']
        if mc_bounds is not None:
            mc_start, mc_end = mc_bounds
            for mc_no in range(mc_start, mc_end + 1):
                mc_tag = "{:04d}".format(mc_no)
                mc_tags.append(mc_tag)

        # Check no. of files in folder against expectation
        n_slices = len(slice_nos)
        n_spifus = len(spifu_nos)
        n_mcs = mc_bounds[1] - mc_bounds[0] + 3
        n_text = 1
        n_expected = n_slices * n_spifus * n_mcs + n_text
        all_files = iq_filer.get_file_list(fits_folder)
        n_files = len(all_files)
        if n_files != n_expected:
            print("Found {:d} files, {:d} expected in folder {:s}".format(n_files, n_expected, folder))

        images, file_names = [], []
        for mc_tag in mc_tags:
            inc_tags = [mc_tag, obs_tag, '.fits']
            file_list = iq_filer.get_file_list(fits_folder,
                                               inc_tags=inc_tags,
                                               exc_tags=['sli'])
            n_files = len(file_list)
            if n_files > 1:
                print("!! multiple files named {:s} in {:s}".format(file_list[0], folder))
            if n_files < 1:
                print("\n!! No files found with {:s}{:s}".format(obs_tag, mc_tag))

            file = file_list[0]
            if debug:
                fmt = "Reading {:s}, mc_tag={:s}"
                print(fmt.format(file, mc_tag))
            path = fits_folder + file
            hdu_list = fits.open(path, mode='readonly')
            hdu = hdu_list[0]
            image_in = hdu.data
            image = np.array(image_in)          # Make a copy of the raw image (maybe shift it).
            if xy_shift is not None:            # Oversample, shift, resample
                sampling, dr, dc = xy_shift
                n_rows, n_cols = image_in.shape
                # Create super-sampled copy of image
                osim = np.zeros((sampling * n_rows, sampling * n_cols))
                for row in range(0, n_rows):
                    for col in range(0, n_cols):
                        r1, c1 = row * sampling, col * sampling
                        r2, c2 = r1 + sampling, c1 + sampling
                        osim[r1:r2, c1:c2] = image[row, col]
                # Apply the integer pixel shift
                osim = np.roll(osim, (dr, dc), axis=(0, 1))
                # Rebin into image
                for row in range(0, n_rows):
                    for col in range(0, n_cols):
                        r1, c1 = row * sampling, col * sampling
                        r2, c2 = r1 + sampling, c1 + sampling
                        image[row, col] = np.mean(osim[r1:r2, c1:c2])

            images.append(image)
            file_names.append(file)

        ds_dict['file_names'] = file_names
        return images, ds_dict

    @staticmethod
    def _make_inc_tags(config_no, slice_no, mc_tag):
        """
        Legacy: build file-name inclusion tags for the old Zemax naming scheme.

        Parameters
        ----------
        config_no : int
        slice_no : int
        mc_tag : str

        Returns
        -------
        inc_tags : list of str
            ``['.fits', '<cc>_<ss><mc_tag>']`` or, if ``is_new_zemax``,
            ``['.fits', '_<cc+1>']``.

        Notes
        -----
        * Clean-up: unused at HEAD; ``is_new_zemax`` is never set.
        """
        inc_tags = ['.fits']
        fmt = "{:02d}_{:02d}{:s}"
        id_tag = fmt.format(config_no, slice_no, mc_tag)
        if ImageManager.is_new_zemax:
            id_tag = "_{:02d}".format(config_no+1)
        inc_tags.append(id_tag)
        return inc_tags
