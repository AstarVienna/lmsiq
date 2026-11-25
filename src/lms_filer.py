"""File and folder management for the lmsiq package.

:class:`Filer` centralises the path conventions and most of the FITS, pickle
and text I/O of the four lmsiq programs (``lmsiq``, ``lmsdist``, ``lmssim``,
``lmsaiv``): the ``../data`` / ``../output`` folder layout for a Zemax or AIT
model configuration, reading and writing four-detector LMS mosaics, the
``lms_efp_mfp_*.fits`` distortion-transform products with their polynomial
fit-parameter and affine-transform companions, WCU pinhole-mask files, and
pickles.

All paths are relative to ``src/`` as the working directory; the data trees
live outside the repository.

Folders and files
-----------------
* :meth:`Filer.set_configuration` derives, from the
  ``(analysis_type[_ait], opticon, date_stamp)`` entry of
  ``Globals.model_configurations``:
  ``psf_folder`` = ``ray_trace_folder`` =
  ``../data/{analysis_type}/{opticon}/{date_stamp}/``
  (e.g. ``../data/distortion/nominal/20240109/`` or
  ``../data/distortion_ait/nominal/ait/``), ``output_folder`` =
  ``../output/{analysis_type}/{opticon}/{date_stamp}/``, ``tf_dir`` =
  ``{output_folder}fits/``, ``cube_folder`` = ``{output_folder}/cube/`` and
  the stems ``{output_folder}{analysis_type}_{opticon}_{date_stamp}_trace``
  (ray-trace pickle), ``..._dist_poly.txt``, ``..._dist_wcal.txt``,
  ``..._dist_stats.txt``, ``..._dist_tf_fit``.
* :meth:`Filer.set_test_data_folder` -> class attribute
  ``Filer.test_data_folder`` = ``../data/{simulator_name}`` (e.g.
  ``../data/test_toysim``), where the ``lmsaiv`` tests read their mosaics.
* :meth:`Filer.read_pinholes` -> ``./inst_pkgs/METIS/wcu/{name}.dat`` (irdb
  WCU mask files ``fp_mask_pinhole_lm.dat`` and ``fp_mask_grid_lm.dat`` as
  downloaded by ScopeSim into ``src/inst_pkgs``).
* Transform products in ``tf_dir``:
  ``lms_efp_mfp{_nom|_ext}_pa#####_ea[p|n]#####_v{date}.fits``
  (:meth:`Filer.write_svd_transforms` / :meth:`Filer.read_svd_transforms`),
  ``lms_dist_efp_mfp_{nom|ext}_fit_parameters_v{date}.fits``
  (:meth:`Filer.write_fit_parameters` / :meth:`Filer.read_fit_parameters`),
  ``lms_dist_mfp_dfp_{tag}.fits`` (:meth:`Filer.write_affine_transform` /
  :meth:`Filer.read_fits_affine_transform`).
* Pickles ``{path}.pkl`` via :meth:`Filer.read_pickle` /
  :meth:`Filer.write_pickle` (ray traces, ``svd_fits_index_*``, as-built
  parameters).

Header keywords
---------------
* Read by :meth:`Filer.read_mosaic`: primary ``ORIGIN``; per detector
  extension ``ID`` (detector number), ``CRVAL1D``, ``CRVAL2D`` (debug print
  only).
* Written by :meth:`Filer.write_fit_parameters`: ``OPTICON`` (primary);
  ``DESCR``, ``N_COEFFS``, ``ORDER`` (table HDUs); read back by
  :meth:`Filer.read_fit_parameters` together with column ``SLICE_NO``.
* Written by :meth:`Filer.write_affine_transform` and read by
  :meth:`Filer.read_fits_affine_transform`: ``N_MATS``, ``MAT_ORD``.
* Written through ``lms_transform.Transform.make_hdu_primary`` /
  ``make_hdu_ext`` in :meth:`Filer.write_svd_transforms`: ``OPTICON``,
  ``PRI_ANG``, ``ECH_ANG``, ``N_MATS``, ``MAT_ORD`` (primary) and
  ``SLICE_NO``, ``ECH_ORD``, ``W_MIN``, ``W_MAX`` (each ``SLICE_n``
  extension); read back by :meth:`Filer.read_svd_transforms`.

Notes
-----
* Inconsistency: ``lmsiq.py`` and ``lmsiq_cuber.py`` still call
  ``set_configuration(analysis_type, opticon)`` with two arguments; the
  signature requires a third (``is_ait_data``), so the ``lmsiq`` program
  raises ``TypeError`` at start-up.
* Clean-up: :meth:`Filer.get_folder` creates every missing directory of the
  path it is given, so ``set_configuration`` creates ``../data/...`` and
  ``../output/...`` trees as a side effect even when the caller only reads.
* Clean-up: this module imports ``lms_transform.Transform`` (used by
  ``write_affine_transform`` and ``read_svd_transforms``); ``Transform``
  carries its own copies of ``read_pickle``, ``write_pickle`` and
  ``_get_results_path``.  There is no import cycle (``lms_transform``
  imports only ``lms_globals``).
* Clean-up: ``test_data_folder`` exists both as a class attribute (set by
  ``set_test_data_folder`` and read as ``Filer.test_data_folder`` by the
  ``lmsaiv`` modules) and as an instance attribute initialised to None in
  ``__init__``, which shadows the class value on instances.
* Clean-up: instance attributes ``sim_folder``, ``ait_dist_folder``,
  ``poly_file``, ``wcal_file``, ``stats_file`` and ``tf_fit_file`` are
  assigned but never read by any module.
* Clean-up: transform matrices are written with FITS column format ``'E'``
  (float32) in ``write_affine_transform`` and, through
  ``Transform.make_hdu_ext``, in ``write_svd_transforms``; ``'D'`` would
  preserve the sub-pixel precision of cubic terms over 2048 pixels.
"""
import os
from os import listdir
import pickle
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Card, HDUList, PrimaryHDU
from lms_globals import Globals
from lms_transform import Transform
from lms_mosaic import Mosaic
import numpy as np


class Filer:
    """Path conventions and file I/O for lmsiq.

    Most methods are static and stateless.  The instance methods
    (:meth:`set_configuration` and the transform read/write methods) rely on
    the folder attributes that :meth:`set_configuration` sets.

    Class attributes: ``analysis_types`` (``['distortion', 'iq']``,
    informational only, not used for validation) and ``test_data_folder``
    (set by :meth:`set_test_data_folder` to ``../data/{simulator_name}``).

    Instance attributes, all None until :meth:`set_configuration` is called:

    * ``model_configuration``: ``(analysis_type, opticon, date_stamp,
      fov_text, coord_in, coord_out)`` from ``Globals.model_configurations``.
    * ``psf_folder``, ``ray_trace_folder``: input data folder (both hold the
      same ``../data/...`` path).
    * ``output_folder``: ``../output/{analysis_type}/{opticon}/{date_stamp}/``.
    * ``tf_dir``: transform FITS folder ``{output_folder}fits/``.
    * ``trace_file``, ``poly_file``, ``wcal_file``, ``stats_file``,
      ``tf_fit_file``: output file stems (see the module docstring).
    * ``cube_folder``: ``{output_folder}/cube/`` (``lmsiq_cuber``).
    * ``sim_folder``, ``ait_dist_folder``: never assigned a value.
    """

    analysis_types = ['distortion', 'iq']
    test_data_folder = None

    def __init__(self):
        """Create a Filer with every folder and file attribute set to None.

        Call :meth:`set_configuration` before using the instance methods.
        """
        self.model_configuration = None
        self.test_data_folder = None
        self.psf_folder, self.sim_folder, self.output_folder = None, None, None
        self.tf_dir, self.trace_file, self.poly_file = None, None, None
        self.wcal_file, self.stats_file, self.tf_fit_file, self.cube_folder = None, None, None, None
        self.ray_trace_folder, self.ait_dist_folder = None, None
        return

    def set_configuration(self, analysis_type, opticon, is_ait_data):
        """Select a model configuration and derive all data and output folders.

        Parameters
        ----------
        analysis_type : str
            ``'distortion'`` or ``'iq'``.  ``'_ait'`` is appended when
            ``is_ait_data`` is True, selecting the AIT (measured) configuration.
        opticon : str
            ``'nominal'`` or ``'extended'`` (``Globals.nominal`` / ``Globals.extended``).
        is_ait_data : bool
            True to use AIT data (``../data/distortion_ait/{opticon}/ait/``) instead
            of the Zemax data set.

        Returns
        -------
        None

        Notes
        -----
        Sets ``model_configuration``, ``psf_folder``, ``ray_trace_folder``,
        ``output_folder``, ``tf_dir``, ``trace_file``, ``poly_file``, ``wcal_file``,
        ``stats_file``, ``tf_fit_file`` and ``cube_folder`` (paths in the module
        docstring).  Every folder is created if missing.

        * Inconsistency: ``lmsiq.py`` and ``lmsiq_cuber.py`` call this with two
          positional arguments and raise ``TypeError``.
        * Clean-up: ``psf_folder`` and ``ray_trace_folder`` are the same path;
          ``cube_folder`` is built with a doubled slash (``.../{date}//cube/``).
        * Clean-up: ``Globals.model_configurations`` has no ``'iq_ait'`` entry, so
          ``analysis_type='iq'`` with ``is_ait_data=True`` raises ``KeyError``.
        """
        if is_ait_data:
            analysis_type += '_ait'
        model_configuration = Globals.model_configurations[analysis_type][opticon]
        analysis_type, opticon, data_set, _, _, _ = model_configuration
        self.model_configuration = model_configuration
        sub_folder = "{:s}/{:s}/{:s}".format(analysis_type, opticon, data_set)

        # self.test_data_folder = Filer.get_folder('./data/' + sub_folder)
        self.psf_folder = Filer.get_folder('../data/' + sub_folder)
        self.ray_trace_folder = Filer.get_folder('../data/' + sub_folder)
        self.output_folder = Filer.get_folder('../output/' + sub_folder)
        self.tf_dir = Filer.get_folder(self.output_folder + 'fits')
        file_leader = self.output_folder + sub_folder.replace('/', '_')
        self.trace_file = file_leader + '_trace'                # All ray coordinates
        self.poly_file = file_leader + '_dist_poly.txt'
        self.wcal_file = file_leader + '_dist_wcal.txt'        # Echelle angle as function of wavelength
        self.stats_file = file_leader + '_dist_stats.txt'
        self.tf_fit_file = file_leader + '_dist_tf_fit'
        self.cube_folder = Filer.get_folder(self.output_folder + '/cube')
        return

    @staticmethod
    def read_zemax_fits(path):
        """Open a FITS file read-only and return its HDU list.

        Parameters
        ----------
        path : str
            Path of the FITS file (a Zemax PSF image with the data in the primary
            HDU, or any other FITS file: ``lmssim_toy`` uses it to read the
            ``../config/sim_template.fits`` header template).

        Returns
        -------
        astropy.io.fits.HDUList
            The open HDU list; the caller is responsible for closing it.
        """
        hdu_list = fits.open(path, mode='readonly')
        return hdu_list

    @staticmethod
    def write_zemax_fits(path, header, img_hdu):
        """Write an image into the primary HDU of a new FITS file.

        Parameters
        ----------
        path : str
            Output path; an existing file is overwritten.
        header : astropy.io.fits.Header
            Header for the primary HDU.
        img_hdu : numpy.ndarray
            Image data (despite the name, an array rather than an HDU).

        Returns
        -------
        None

        Notes
        -----
        Written with ``checksum=True``.

        * Clean-up: not called by any module in the package.
        """
        pri_hdu = PrimaryHDU(header=header, data=img_hdu)
        hdu_list = HDUList([pri_hdu])
        hdu_list.writeto(path, overwrite=True, checksum=True)
        return

    @staticmethod
    def read_mosaic(folder, file_name):
        """Read a four-detector LMS FITS file into a :class:`lms_mosaic.Mosaic`.

        Parameters
        ----------
        folder : str
            Folder containing the file.
        file_name : str
            File name, joined to ``folder`` with ``'/'``.

        Returns
        -------
        lms_mosaic.Mosaic
            ``Mosaic(file_name, primary_header, hdu_list)``; ``hdu_list`` is a
            4-element list of detector image HDUs ordered by mosaic position,
            index 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.

        Notes
        -----
        Header keywords read: primary ``ORIGIN``; per extension ``ID`` (detector
        number; ``int()`` accepts both the ScopeSim integer and the ToySim string),
        ``CRVAL1D`` and ``CRVAL2D`` (detector centre in mm, used only in the debug
        print).  When ``ORIGIN == 'SCOPESIM'`` the detector number is mapped
        through ``Globals.mos_idx`` = {2: 0, 1: 1, 3: 2, 4: 3}; otherwise (ToySim,
        ``ORIGIN == 'TOYSIM'``) the list index is ``ID - 1``.

        * Open question: for ScopeSim data detector 2 is placed at index 0 and
          detector 1 at index 1, the reverse of the ToySim ordering and of the irdb
          layout (``x_cen`` = -19.547 mm for detector 1).  Whether ScopeSim numbers
          its detectors differently from the irdb table should be checked with a
          real ScopeSim frame.
        * Clean-up: ``CRVAL1D`` / ``CRVAL2D`` must exist even though they are only
          printed; extensions without them raise ``KeyError``.
        * Clean-up: the file is opened read-only and never closed.
        """
        path = folder + '/' + file_name
        if Globals.is_debug('high'):
            print("Reading {:s}".format(path))
        hdu_in_list = fits.open(path, mode='readonly')
        primary_hdr = hdu_in_list[0].header
        origin = primary_hdr['ORIGIN']
        hdu_list = [None]*4       # Re-order hdus
        for hdu_in in hdu_in_list[1:]:
            det_no = int(hdu_in.header['ID'])
            x = float(hdu_in.header['CRVAL1D'])
            y = float(hdu_in.header['CRVAL2D'])
            if origin == 'SCOPESIM':
                mos_idx = Globals.mos_idx[det_no]
            else:
                mos_idx = det_no - 1
            if Globals.is_debug('high'):
                fmt = "- storing ScopeSim det_no= {:d} (x, y) = ({:6.3f}, {:6.3f}), at mosaic list index= {:d}"
                print(fmt.format(det_no, x, y, mos_idx))
            hdu_list[mos_idx] = hdu_in
        mosaic = Mosaic(file_name, primary_hdr, hdu_list)
        return mosaic

    @staticmethod
    def read_mosaic_list(folder, *args):
        """Read every mosaic file in a folder whose name contains the given tags.

        Parameters
        ----------
        folder : str
            Folder to search (typically ``Filer.test_data_folder``).
        *args
            ``args[0]``: list of substrings that must all occur in the file name
            (``inc_tags``); optional ``args[1]``: list of substrings that must not
            occur (``exc_tags``).

        Returns
        -------
        list of lms_mosaic.Mosaic
            One mosaic per matching file in ``os.listdir`` order; empty if nothing
            matches.

        Notes
        -----
        ScopeSim files hold the detector number in each extension header ``ID``.
        The original docstring states that ScopeSim extensions 1-4 contain
        detectors 2 (TR), 1 (TL), 3 (BL), 4 (BR), where T = top (slices 15-28),
        B = bottom (slices 1-14), L = left (short wavelength), R = right (long
        wavelength), and that the mosaic list is ordered 0 (TL), 1 (TR), 2 (BL),
        3 (BR).

        * Inconsistency: with ``Globals.mos_idx`` = {2: 0, 1: 1, 3: 2, 4: 3}
          detector 2 is stored at index 0, which the same description calls TL
          while naming detector 2 TR; one of the two statements is wrong.
        * Bug: when no file matches, a "not found" message is assembled in
          ``text`` but never printed; the empty list is returned silently.
        * Clean-up: ``*args`` hides the two-list signature; ``inc_tags`` is
          mandatory (``args[0]`` raises ``IndexError`` if omitted).
        """
        n_args = len(args)
        inc_tags = args[0]
        exc_tags = args[1] if n_args > 1 else []
        mosaic_list = []
        file_list = Filer.get_file_list(folder, inc_tags=inc_tags, exc_tags=exc_tags)
        if len(file_list) == 0:
            text = "Files in {:s} including tags ".format(folder)
            for tag in inc_tags:
                text += "{:s}, ".format(tag)
            text += 'not found'
            return mosaic_list
        for file_name in file_list:
            mosaic = Filer.read_mosaic(folder, file_name)
            mosaic_list.append(mosaic)
        return mosaic_list

    @staticmethod
    def write_mosaic(folder, mosaic):
        """Write a mosaic (name, primary header, detector HDUs) to a FITS file.

        Parameters
        ----------
        folder : str
            Output folder (must exist; joined to the file name with ``'/'``).
        mosaic : tuple
            Unpacked as ``file_name, primary_header, hdu_list``.  ``lmssim_toy``
            passes a plain 3-tuple.

        Returns
        -------
        None

        Notes
        -----
        Writes ``PrimaryHDU(header)`` followed by the HDUs in ``hdu_list`` with
        ``overwrite=True, checksum=True``.

        * Clean-up: a :class:`lms_mosaic.Mosaic` instance cannot be passed here
          (the class defines no ``__iter__``); despite the docstring wording only
          the tuple form works.
        """
        file_name, primary_header, hdu_list_in = mosaic
        path = folder + '/' + file_name
        primary_hdu = PrimaryHDU(header=primary_header)
        hdu_list = HDUList([primary_hdu])
        for hdu in hdu_list_in:
            hdu_list.append(hdu)
        hdu_list.writeto(path, overwrite=True, checksum=True)
        return

    @staticmethod
    def set_test_data_folder(simulator_name):
        """Set the class-level folder for simulated or measured test mosaics.

        Parameters
        ----------
        simulator_name : str
            Sub-folder of ``../data``, e.g. ``'test_toysim'`` (``lmsaiv``) or
            ``'scopesim'`` (``lmssim_scope``).

        Returns
        -------
        None

        Notes
        -----
        Sets ``Filer.test_data_folder = '../data/{simulator_name}'``; the folder is
        neither created nor checked.

        * Inconsistency: ``lmssim_scope.Scope`` sets ``'scopesim'``
          (``../data/scopesim``) but writes its frames to ``../data/test_scopesim``;
          ``lmsaiv`` reads ``../data/test_toysim``.
        """
        Filer.test_data_folder = "../data/{:s}".format(simulator_name)
        return

    @staticmethod
    def get_file_list(folder, inc_tags=None, exc_tags=None):
        """List the entries of a folder filtered by name substrings.

        Parameters
        ----------
        folder : str
            Folder to list.
        inc_tags : list of str, optional
            Every tag must occur in the entry name.
        exc_tags : list of str, optional
            No tag may occur in the entry name.

        Returns
        -------
        list of str
            Entry names (not paths) in ``os.listdir`` order; sub-directory names
            are included.
        """
        if exc_tags is None:
            exc_tags = []
        if inc_tags is None:
            inc_tags = []
        file_list = listdir(folder)
        for tag in inc_tags:
            file_list = [f for f in file_list if tag in f]
        for tag in exc_tags:
            file_list = [f for f in file_list if tag not in f]
        return file_list

    @staticmethod
    def get_folder(in_path):
        """Return a folder path with a trailing slash, creating it if necessary.

        Parameters
        ----------
        in_path : str
            Folder path with ``'/'`` separators.

        Returns
        -------
        str
            ``in_path`` with every component followed by ``'/'`` (an empty
            component in the input yields a doubled slash).

        Notes
        -----
        * Clean-up: every missing path component is created with ``os.mkdir`` as a
          side effect, including when the caller only intends to read.
        """
        tokens = in_path.split('/')
        out_path = ''
        for token in tokens:
            out_path = out_path + token + '/'
            if not os.path.exists(out_path):
                os.mkdir(out_path)
        return out_path

    @staticmethod
    def read_pinholes(file_name, efp_xy_filter=(1.6, 2.7)):
        """Read a WCU focal-plane mask file and return pinhole positions in EFP mm.

        Parameters
        ----------
        file_name : str
            Mask file stem; the file read is ``./inst_pkgs/METIS/wcu/{file_name}.dat``
            (irdb WCU mask files ``'fp_mask_pinhole_lm'`` and ``'fp_mask_grid_lm'``,
            downloaded by ScopeSim into ``src/inst_pkgs``).
        efp_xy_filter : tuple of float, optional
            ``(x_max, y_max)`` in mm; only pinholes with ``|efp_x| < x_max`` and
            ``|efp_y| < y_max`` are kept.  Default ``(1.6, 2.7)``.

        Returns
        -------
        list of [float, float]
            ``[efp_x, efp_y]`` pairs in mm, scaled from the WCU arcsecond
            coordinates in the file by ``Globals.efp_mm_wcu_as`` (4.538 mm/arcsec).

        Notes
        -----
        Lines containing ``'#'`` (comments) or ``'x'`` (the column-label line), or
        with fewer than two tokens, are skipped; the first two tokens of the other
        lines are ``x`` and ``y``.

        * Open question: the default filter is ``|x| < 1.6``, ``|y| < 2.7`` mm while
          the EFP half-field is 2.48 mm along x and 1.60 mm along y
          (``Globals.efp_x_fov_mm`` / ``efp_y_fov_mm``); the two limits look
          transposed.
        * Open question: the relative ``./inst_pkgs`` path resolves only when the
          ScopeSim instrument packages have been downloaded into ``src/``.
        * Clean-up: any data line containing the letter ``'x'`` is discarded.
        """
        path = './inst_pkgs/METIS/wcu/' + file_name + '.dat'
        efp_xy_list = []
        k_efp_wcu = Globals.efp_mm_wcu_as
        if Globals.is_debug('high'):
            fmt = "Reading pinhole mask from {:s}, coords scaled by {:10.3f} EFP(mm) per WCU(arcsecond"
            print(fmt.format(file_name, k_efp_wcu))
        with open(path, 'r') as text_file:
            records = text_file.read().splitlines()
            for record in records:
                if Globals.is_debug('high'):
                    print(record)
                if '#' in record:  # Skip comment lines
                    continue
                if 'x' in record:  # Skip column label line
                    continue
                tokens = record.split()
                if len(tokens) < 2:
                    continue
                wcu_x, wcu_y = float(tokens[0]), float(tokens[1])
                efp_x, efp_y = k_efp_wcu * wcu_x, k_efp_wcu * wcu_y
                if abs(efp_x) < efp_xy_filter[0] and abs(efp_y) < efp_xy_filter[1]:
                    efp_xy_list.append([efp_x, efp_y])
        return efp_xy_list

    def write_fit_parameters(self, wpa_fit, wxo_fit, wxo_hdr, term_fits):
        """Write the (prism angle, echelle angle) polynomial surface fits to FITS.

        Output file:
        ``{tf_dir}lms_dist_efp_mfp_{nom|ext}_fit_parameters_v{date_stamp}.fits``
        with ``opticon`` and ``date_stamp`` from ``model_configuration``.

        Parameters
        ----------
        wpa_fit : dict
            Wavelength -> prism angle polynomial: ``'n_coeffs'`` and ``'wpa_opt'``
            (coefficient list).
        wxo_fit : dict
            Wavelength x order surface fit for slice 13: ``'slice_no'``,
            ``'wxo_opt'`` (coefficients), ``'order'``, ``'n_coefficients'``.
        wxo_hdr : list of str
            Surface-term names ``PA{i}_EA{j}`` from
            ``lmsdist_polyfit.PolyFit.make_surface_model_header``.
        term_fits : list of tuple
            One ``(slice_no, matrices)`` per slice; ``matrices`` is a dict with keys
            ``'a', 'b', 'ai', 'bi'`` of arrays shaped ``(svd_order, svd_order, n_coeffs)``.

        Returns
        -------
        None

        Notes
        -----
        HDU layout: primary (``OPTICON``); HDU 1 table with columns
        ``WP0 .. WP{n-1}`` (header ``DESCR``, ``N_COEFFS``); HDU 2 one-row table
        ``SLICE_NO, PA0_EA0, ...`` (``DESCR``, ``ORDER``, ``N_COEFFS``); HDU 3 table
        with ``svd_order**2`` rows per slice and columns ``SLICE_NO``, ``ROW``,
        ``COL``, then ``A_PA{i}_EA{j}``, ``B_...``, ``AI_...``, ``BI_...``.

        * Bug: the one-row HDU 2 table stores ``wxo_opt`` in columns 1..n, but
          :meth:`read_fit_parameters` reads ``wxo_data[0][2:]``, dropping the first
          coefficient (``PA0_EA0``), unless ``wxo_opt`` carries an undocumented extra
          leading element.
        * Clean-up: ``wxo_fit['n_coefficients']`` here versus ``'n_coeffs'`` in the
          reader (different key spelling).
        * Clean-up: the HDU 3 fill loop uses a literal ``4`` for the matrix column
          count instead of ``nc``.
        """

        _, opticon, date_stamp, _, _, _ = self.model_configuration
        fmt = "lms_dist_efp_mfp_{:s}_fit_parameters_v{:s}"
        fits_name = fmt.format(opticon[0:3], date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'

        n_slices = len(term_fits)
        primary_cards = [Card('OPTICON', opticon, 'Optical configuration (nominal/extended)')]
        primary_header = fits.Header(primary_cards)
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        hdu_list = HDUList([primary_hdu])

        wpa_col_names = []
        n_coeffs = wpa_fit['n_coeffs']
        for i in range(0, n_coeffs):
            wpa_col_names.append("WP{:d}".format(i))
        wpa_data = wpa_fit['wpa_opt']     # All slices have the same wxo fit parameters...
        wpa_table = Table(data=np.array(wpa_data), names=wpa_col_names)
        wpa_cards = [Card('DESCR', 'Fit parms to map wavelength to prism angle', ''),
                     Card('N_COEFFS', n_coeffs, 'No. of polynomial fit coefficients')
                     ]
        wpa_hdr = fits.Header(wpa_cards)

        wpa_hdu = fits.BinTableHDU(data=wpa_table, header=wpa_hdr)
        hdu_list.append(wpa_hdu)

        wxo_column_names = ['SLICE_NO'] + wxo_hdr
        n_wxo_columns = len(wxo_column_names)
        wxo_data = np.zeros(n_wxo_columns)
        # Write wavelength x echelle order fit parameters to second HDU
        wxo_data[0] = wxo_fit['slice_no']
        wxo_data[1:n_wxo_columns] = wxo_fit['wxo_opt']     # All slices have the same wxo fit parameters...
        order = wxo_fit['order']
        n_coeffs = wxo_fit['n_coefficients']
        wxo_cards = [Card('DESCR', 'Fit parms to map prism and echelle angle to slice 13 wavelength', ''),
                     Card('ORDER', order, 'Surface fit matrix order'),
                     Card('N_COEFFS', n_coeffs, 'No. of coefficients in surface fit matrix')
                     ]
        wxo_hdr = fits.Header(wxo_cards)
        wxo_table = Table(data=wxo_data, names=wxo_column_names)
        wxo_hdu = fits.BinTableHDU(data=wxo_table, header=wxo_hdr)
        hdu_list.append(wxo_hdu)

        term_cards = [Card('DESCR', 'Fit parms to map prism and echelle angle to transform terms.', ''),
                      Card('ORDER', order, 'Surface fit matrix order'),
                      Card('N_COEFFS', n_coeffs, 'No. of coefficients in surface fit matrix')
                      ]
        term_hdr = fits.Header(term_cards)
        mat_tags_uc = ['A', 'B', 'AI', 'BI']
        term_names = ['SLICE_NO', 'ROW', 'COL']
        for mat_tag in mat_tags_uc:
            for tag in wxo_column_names[1:]:
                term_names.append(mat_tag + '_' + tag)
        n_columns = len(term_names)
        n_records_slice = Globals.svd_order * Globals.svd_order
        term_data = np.zeros((n_slices * n_records_slice, n_columns))
        data_row = 0                           # Row counter in term_data array
        for i, term_row in enumerate(term_fits):
            matrices = term_row[1]
            nr, nc, nvals = matrices['a'].shape
            nrc = nr * nc
            term_data[data_row:data_row + nrc, 0] = term_row[0]
            rows, cols = [], []
            for rc in range(0, nr*nc):
                rows.append(int(rc / nr))
                cols.append(int(rc % nc))
            term_data[data_row:data_row + nrc, 1] = rows  # matrix row and column nos.
            term_data[data_row:data_row + nrc, 2] = cols
            data_col = 3
            for key in matrices:
                data_row = i * nrc
                matrix = matrices[key]
                for row in range(0, nr):
                    data_block = matrix[row, :]
                    term_data[data_row:data_row+4, data_col:data_col+nvals] = data_block
                    data_row += nc
                data_col += nvals

        term_table = Table(data=term_data, names=term_names)
        term_hdu = fits.BinTableHDU(data=term_table, header=term_hdr)
        hdu_list.append(term_hdu)
        hdu_list.writeto(fits_path, overwrite=True)
        return

    def read_fit_parameters(self, opticon):
        """Read the surface-fit parameter file written by :meth:`write_fit_parameters`.

        Parameters
        ----------
        opticon : str
            ``'nominal'`` or ``'extended'``; only the first three characters enter
            the file name.  The date stamp comes from ``model_configuration``.

        Returns
        -------
        wpa_fit : dict
            ``{'n_coeffs', 'wpa_opt'}`` wavelength -> prism angle polynomial.
        wxo_fit : dict
            ``{'order', 'n_coeffs', 'slice_no', 'wxo_opt'}`` wavelength x order
            surface fit (slice 13 by construction).
        term_fits : dict
            ``{slice_no: {'a': arr, 'b': arr, 'ai': arr, 'bi': arr}}`` with arrays
            of shape ``(ORDER, ORDER, N_COEFFS)``; 18 slices expected in extended
            mode, 28 in nominal.

        Notes
        -----
        Header keywords read: ``N_COEFFS`` (HDUs 1-3), ``ORDER`` (HDUs 2-3); column
        ``SLICE_NO``.

        * Bug: ``wxo_opt`` is taken from ``wxo_data[0][2:]``, i.e. from the third
          column onwards, so the first surface coefficient is skipped (see
          :meth:`write_fit_parameters`).
        * Clean-up: the ``ORDER`` keyword of HDU 3 is used as the transform matrix
          order although the writer stores the surface-fit order there; both are 4.
        * Clean-up: the FITS file is never closed.
        """
        _, _, date_stamp, _, _, _ = self.model_configuration
        fmt = "lms_dist_efp_mfp_{:s}_fit_parameters_v{:s}"
        fits_name = fmt.format(opticon[0:3], date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'
        hdu_list = fits.open(fits_path, mode='readonly')
        if Globals.is_debug('high'):
            print("Reading global fit parameters from file {:s}".format(fits_name))

        wpa_hdr = hdu_list[1].header
        n_wpa_coeffs = wpa_hdr['N_COEFFS']
        wpa_data = hdu_list[1].data
        wpa_fit = {'n_coeffs': n_wpa_coeffs,  'wpa_opt': list(wpa_data[0][:])}

        wxo_header = hdu_list[2].header             # Map from wavelength to echelle angle for slice 13.
        wxo_fit_order, n_wxo_coeffs = wxo_header['ORDER'], wxo_header['N_COEFFS']
        wxo_data = hdu_list[2].data
        slice_no = int(wxo_data['SLICE_NO'][0])
        wxo_fit = {'order': wxo_fit_order, 'n_coeffs': n_wxo_coeffs,
                   'slice_no': slice_no,
                   'wxo_opt': list(wxo_data[0][2:])}

        term_header = hdu_list[3].header
        n_term_coeffs = term_header['N_COEFFS']
        term_data = np.array(hdu_list[3].data)
        svd_order = term_header['ORDER']
        mat_shape = svd_order, svd_order, n_term_coeffs,
        term_fits = {}

        # Should have 18 elements for ext, 28 for nominal
        for data_rec in np.array(term_data):
            data_array = list(data_rec)
            slice_no = int(data_array[0])
            if slice_no not in term_fits.keys():
                term_fits[slice_no] = {}
                for mat_name in Globals.matrix_names:
                    term_fits[slice_no][mat_name] = np.zeros(mat_shape)
            mat_row, mat_col = int(data_array[1]), int(data_array[2])
            data_col = 3
            for mat_name in Globals.matrix_names:
                mat = term_fits[slice_no][mat_name]
                mat[mat_row, mat_col] = data_array[data_col: data_col + n_term_coeffs]
                term_fits[slice_no][mat_name] = mat
                data_col += n_term_coeffs
        return wpa_fit, wxo_fit, term_fits

    def write_affine_transform(self, version_tag):
        """Write the four MFP <-> detector-pixel affine matrices to FITS.

        Output file: ``{tf_dir}lms_dist_mfp_dfp_{version_tag}.fits``.

        Parameters
        ----------
        version_tag : str
            Version suffix; ``lmsdist`` passes ``'v{date_stamp}'``.

        Returns
        -------
        None

        Notes
        -----
        The matrices come from the class attribute
        ``lms_transform.Transform.affines`` (``{'m': (4, 3, 3), 'mi': (4, 3, 3)}``,
        built by ``Transform.create_affines`` from ``Globals.det_size``, ``det_gap``
        and ``nom_pix_pitch``).  Primary header: ``N_MATS``, ``MAT_ORD``.  HDU 1 is
        a binary table with one column per matrix, ``MFP>DET1 .. MFP>DET4``
        (forward, flattened 3 x 3) then ``DET1>MFP .. DET4>MFP`` (inverse), format
        ``'E'`` (float32).

        * Inconsistency: :meth:`read_fits_affine_transform` prepends ``'v'`` to its
          ``date_stamp`` argument, whereas here the ``'v'`` must be part of
          ``version_tag``.
        * Bug: raises ``TypeError`` if ``Transform.affines`` is still None (no
          ``Transform`` instantiated yet).
        * Clean-up: float32 (``'E'``) storage of the affine terms.
        """
        affines = Transform.affines
        n_mats, mat_order, _ = affines['m'].shape

        primary_cards = [Card('N_MATS', n_mats, 'MFP <-> DFP transform matrices'),
                         Card('MAT_ORD', mat_order, 'Transform matrix dimensions')
                         ]
        hdr = fits.Header(primary_cards)
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdu_list = HDUList([primary_hdu])

        fmt = "lms_dist_mfp_dfp_{:s}"
        fits_name = fmt.format(version_tag)
        fits_path = self.tf_dir + fits_name + '.fits'
        cards = []

        col_list = []
        for key in affines.keys():
            col_fmt = "MFP>DET{:d}" if key == 'm' else "DET{:d}>MFP"
            for m in range(0, n_mats):
                col_name = col_fmt.format(m+1)
                array = affines[key][m]
                col = fits.Column(name=col_name, array=array.flatten(), format='E')
                col_list.append(col)

        hdr = fits.Header(cards)
        bintable_hdu = fits.BinTableHDU.from_columns(col_list, header=hdr)
        hdu_list.append(bintable_hdu)
        hdu_list.writeto(fits_path, overwrite=True)
        return

    def read_fits_affine_transform(self, date_stamp):
        """Read the affine matrices written by :meth:`write_affine_transform`.

        Parameters
        ----------
        date_stamp : str
            Model date stamp; the file read is
            ``{tf_dir}lms_dist_mfp_dfp_v{date_stamp}.fits``.

        Returns
        -------
        dict
            ``{'m': ndarray (n_mats, mat_order, mat_order), 'mi': ndarray}``:
            forward (MFP mm -> detector pixel) and inverse matrices, one per
            detector, in the layout of ``lms_transform.Transform.affines``.

        Notes
        -----
        Header keywords read: ``N_MATS``, ``MAT_ORD`` (primary).  Table columns 0-3
        are taken as the forward matrices and 4-7 as the inverses (the writer's
        dictionary order).

        * Clean-up: the previous docstring was a truncated sentence ("Read fits
          file into a 'trace' object, which"); the method returns a dictionary.
        * Clean-up: the file is never closed.
        """
        fmt = "lms_dist_mfp_dfp_v{:s}"
        fits_name = fmt.format(date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'
        hdu_list = fits.open(fits_path, mode='readonly')

        primary_hdr = hdu_list[0].header
        n_mats = primary_hdr['N_MATS']
        mat_order = primary_hdr['MAT_ORD']
        aff_shape = n_mats, mat_order, mat_order
        m, mi = np.zeros(aff_shape), np.zeros(aff_shape)

        hdu = hdu_list[1]
        table, hdr = hdu.data, hdu.header
        n_cols = len(hdu.columns)
        for i in range(0, n_cols):
            col_vals = table.field(i)
            if i < 4:
                m[i] = np.reshape(col_vals, (mat_order, mat_order))
            else:
                mi[i-4] = np.reshape(col_vals, (mat_order, mat_order))
        return {'m': m, 'mi': mi}

    def write_svd_transforms(self, trace):
        """Write all per-slice SVD distortion transforms of a ray trace to one FITS file.

        This is the data product delivered to ScopeSim and the pipeline:
        ``{tf_dir}lms_efp_mfp{_nom|_ext}_pa#####_ea[p|n]#####_v{date_stamp}.fits``.

        Parameters
        ----------
        trace : lmsdist_trace.RayTrace
            Object with ``transforms`` (list of ``lms_transform.Transform``, one per
            slice, sharing one ``lms_configuration``) and ``model_config`` (6-tuple
            whose third element is the date stamp).

        Returns
        -------
        str
            File name stem (without ``.fits``); ``lmsdist`` records it in the
            ``svd_fits_index_*`` pickle.

        Notes
        -----
        The primary HDU (``Transform.make_hdu_primary``: ``OPTICON``, ``PRI_ANG``,
        ``ECH_ANG``, ``N_MATS``, ``MAT_ORD``) comes from the first transform; each
        transform then contributes a ``SLICE_{slice_no}`` ``BinTableHDU`` with
        columns ``A``, ``B``, ``AI``, ``BI`` (16 float32 values each) and header
        ``SLICE_NO``, ``ECH_ORD``, ``W_MIN``, ``W_MAX``.  The file name is built
        with ``Globals.make_lms_config_id``.

        * Bug: an empty ``trace.transforms`` leaves ``hdu_list`` as None and raises
          ``AttributeError`` at ``writeto``.
        * Clean-up: matrices are stored as ``'E'`` (float32); ``'D'`` would keep the
          fitted precision.
        """
        hdu_list = None
        fits_path, fits_name = None, None
        write_primary = True
        for transform in trace.transforms:
            lms_config = transform.lms_configuration
            if write_primary:
                primary_hdu = transform.make_hdu_primary()
                hdu_list = HDUList([primary_hdu])
                lms_config_id = Globals.make_lms_config_id(lms_config)
                # Create fits file with primary HDU only
                _, _, date_stamp, _, _, _ = trace.model_config
                version = "_v{:s}".format(date_stamp)
                fits_name = 'lms_efp_mfp' + lms_config_id + version
                fits_path = self.tf_dir + fits_name + '.fits'
                write_primary = False
            bintable_hdu = transform.make_hdu_ext()
            hdu_list.append(bintable_hdu)
        hdu_list.writeto(fits_path, overwrite=True)
        return fits_name

    def read_svd_transforms(self, inc_tags=None, exc_tags=None):
        """Read SVD distortion transforms from every matching FITS file in ``tf_dir``.

        Parameters
        ----------
        inc_tags : list of str, optional
            Substrings that must all occur in the file name.
        exc_tags : list of str, optional
            Substrings that must not occur.  Callers pass ``['fit_parameters',
            'mfp_dfp']`` to skip the companion products.

        Returns
        -------
        list of lms_transform.Transform
            One transform per ``SLICE_n`` extension of every matching file, in file
            then extension order.

        Notes
        -----
        Each extension is ingested with ``Transform(hdu_list=..., ext_no=...)``,
        which reads ``OPTICON``, ``PRI_ANG``, ``ECH_ANG`` from the primary header
        and ``SLICE_NO``, ``ECH_ORD``, ``W_MIN``, ``W_MAX`` plus columns ``A``, ``B``,
        ``AI``, ``BI`` from the extension.

        * Clean-up: with no tags every file in ``tf_dir`` is read, including the
          ``fit_parameters`` and ``mfp_dfp`` files whose headers lack these keywords
          (``KeyError``); filtering is left to the caller.
        """
        if exc_tags is None:
            exc_tags = []
        if inc_tags is None:
            inc_tags = []
        fits_file_list = Filer.get_file_list(self.tf_dir, inc_tags=inc_tags, exc_tags=exc_tags)
        transform_list = []
        for fits_name in fits_file_list:
            fits_path = self.tf_dir + fits_name
            hdu_list = fits.open(fits_path, mode='readonly')
            n_ext = len(hdu_list)
            for ext_no in range(1, n_ext):
                transform = Transform(hdu_list=hdu_list, ext_no=ext_no)
                transform_list.append(transform)
            hdu_list.close()
        return transform_list

    @staticmethod
    def read_pickle(pickle_path):
        """Load a pickled Python object.

        Parameters
        ----------
        pickle_path : str
            Path with or without the ``.pkl`` extension (appended if absent).

        Returns
        -------
        object
            The unpickled object.
        """
        if pickle_path[-4:] != '.pkl':
            pickle_path += '.pkl'
        pickle_file = open(pickle_path, 'rb')
        python_object = pickle.load(pickle_file)
        pickle_file.close()
        return python_object

    @staticmethod
    def write_pickle(pickle_path, python_object):
        """Pickle a Python object to ``pickle_path + '.pkl'``.

        Parameters
        ----------
        pickle_path : str
            Path stem; ``'.pkl'`` is always appended (unlike :meth:`read_pickle`, no
            check for an existing extension is made).
        python_object : object
            Object to pickle.

        Returns
        -------
        None
        """
        pickle_file = open(pickle_path + '.pkl', 'wb')
        pickle.dump(python_object, pickle_file)
        pickle_file.close()
        return

    def _get_results_path(self, data_id, data_type):
        """Build the CSV path for an ``lmsiq`` image-quality result table.

        Parameters
        ----------
        data_id : tuple
            ``(dataset, slice_subfolder, ipc_tag, process_level, config_folder,
            mcrun_tag, axis)``; ``slice_subfolder`` and ``config_folder`` end with
            ``'/'``, which is stripped for the file name.
        data_type : str
            One of ``'xcentroids'``, ``'ycentroids'``, ``'xfwhm_gau'``,
            ``'photometry'``, ``'ee_spectral'``, ``'ee_spatial'``, ``'lsf_spectral'``,
            ``'lsf_spatial'``, ``'ee_dfp_spectral'``, ``'ee_dfp_spatial'``,
            ``'lsf_dfp_spectral'``, ``'lsf_dfp_spatial'``.

        Returns
        -------
        str
            ``{output_folder}{slice_subfolder}{ipc_tag}/{process_level}/{data_type}/``
            ``{slice}_{ipc_tag}_{process_level}{type_tag}_wav_{config}.csv``; the
            folder is created if missing.

        Notes
        -----
        Used by ``lmsiq_summariser`` only.  ``dataset``, ``mcrun_tag`` and ``axis``
        are unpacked but unused.

        * Clean-up: ``lms_transform.Transform._get_results_path`` duplicates this
          method.
        """
        dataset, slice_subfolder, ipc_tag, process_level, config_folder, mcrun_tag, axis = data_id
        folder = self.output_folder + slice_subfolder
        folder += ipc_tag + '/' + process_level + '/' + data_type
        folder = self.get_folder(folder)

        type_tags = {'xcentroids': '_xcen', 'ycentroids': '_ycen',
                     'xfwhm_gau': '_xfwhm', 'photometry': '_phot',
                     'ee_spectral': '_eex', 'ee_spatial': '_eey',
                     'lsf_spectral': '_lsx', 'lsf_spatial': '_lsy',
                     'ee_dfp_spectral': '_eex_dfp', 'ee_dfp_spatial': '_eey_dfp',
                     'lsf_dfp_spectral': '_lsx_dfp', 'lsf_dfp_spatial': '_lsy_dfp',
                     }
        type_tag = type_tags[data_type]
        config_tag = config_folder[0:-1]
        slice_tag = slice_subfolder[:-1] + '_'
        file_name = slice_tag + ipc_tag + '_' + process_level + type_tag + '_wav_' + config_tag + '.csv'
        path = folder + file_name
        return path
