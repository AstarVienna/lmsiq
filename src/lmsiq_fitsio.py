"""
Minimal FITS read/write helpers for the ``lmsiq`` program.

Purpose
-------
Provides the cube writer used by :meth:`lmsiq_cuber.Cuber.build`
(:meth:`FitsIo.write_cube`), a Zemax PSF reader, a synthetic flat 'extended source'
image and a legacy cube reader.  Most of the module is legacy: only ``write_cube`` is
called at HEAD, and the class-level ``data_path`` / ``zip_path`` are set by the
constructor but never read.

Input data
----------
Zemax PSF FITS files (image in the primary HDU) under
``../data/iq/<opticon>/<date_stamp>/<config folder>/``; 128 x 128 pixels at 4.5 um
(4x oversampled H2RG pixels).  ``read_zemax`` takes an explicit folder and file name.

Outputs
-------
``write_cube`` writes ``../output/iq/<opticon>/<date_stamp>/cube/fits/<cube_name>.fits``
as a bare ``PrimaryHDU`` (no header keywords beyond the mandatory ones).

Notes
-----
* Inconsistency: :meth:`FitsIo.read_cube` expects header keywords ``ALPHA_OS`` and
  ``WAV0..WAVn`` and a 4-D cube, whereas :meth:`FitsIo.write_cube` writes a 3-D cube
  with no header cards; the two methods are not compatible.
* Bug: ``read_cube`` calls ``fits.open(file_path, readonly=True)``; ``readonly`` is not
  a valid keyword (``mode='readonly'`` is), so it raises ``TypeError``.  It also uses
  ``iq_filer.cube_path``, which :class:`lms_filer.Filer` does not define
  (``cube_folder`` exists).  Dead code.
* Clean-up: ``FitsIo.data_path`` (``'../data/iq/<optical_path>/'``) and ``zip_path``
  (``'../zip/'``) are set but never used; ``_copy`` and ``make_extended`` are not
  called anywhere at HEAD.
"""
from os.path import join
import shutil
import numpy as np
from astropy.io import fits


class FitsIo:
    """
    Static FITS helpers; the constructor only records the optical path folder.

    Attributes
    ----------
    data_path : str
        ``'../data/iq/<optical_path>/'`` after construction; unused.
    zip_path : str
        ``'../zip/'`` after construction; unused.
    """

    data_path, zip_path = '', ''

    def __init__(self, optical_path):
        """
        Set the class-level data and zip folder paths.

        Parameters
        ----------
        optical_path : str
            ``'nominal'`` or ``'extended'``; appended to ``'../data/iq/'``.

        Notes
        -----
        * Clean-up: neither path is read by any method.
        """
        FitsIo.data_path = '../data/iq/' + optical_path + '/'
        FitsIo.zip_path = '../zip/'
        return

    @staticmethod
    def _copy(path, folder):
        """
        Copy a file into a folder and print what was copied.

        Parameters
        ----------
        path : str
            Source file path.
        folder : str
            Destination folder.

        Returns
        -------
        None

        Notes
        -----
        * Clean-up: unused at HEAD.
        """
        shutil.copy(path, folder)
        print("Copying {:s} to {:s}".format(path, folder))
        return

    @staticmethod
    def read_zemax(path, f):
        """
        Read a Zemax PSF image and header from the primary HDU of a FITS file.

        Parameters
        ----------
        path : str
            Folder containing the file.
        f : str
            File name.

        Returns
        -------
        data : ndarray
            Primary HDU image (128 x 128 for the standard Zemax PSFs).
        header : astropy.io.fits.Header
            Primary HDU header.

        Notes
        -----
        * Clean-up: the HDU list is not closed.  :meth:`lmsiq_image_manager.ImageManager.load_dataset`
          reads the files itself and does not use this method.
        """
        file_path = join(path, f)
        hdu_list = fits.open(file_path, mode='readonly')
        hdu = hdu_list[0]
        return hdu.data, hdu.header

    @staticmethod
    def make_extended():
        """
        Create a synthetic uniformly illuminated 512 x 512 image.

        Returns
        -------
        image : ndarray, shape (512, 512)
            All pixels equal to 1.0.
        params : tuple
            ``('Extended', 0.018 / 16.)`` -- a label and a pixel size in mm (18 um / 16,
            i.e. 16x oversampling).

        Notes
        -----
        * Inconsistency: the implied 16x oversampling does not match the 4x sampling of
          the Zemax data used elsewhere.  Unused at HEAD.
        """
        image = np.full((512, 512), 1.0)
        params = 'Extended', 0.018 / 16.
        return image, params

    @staticmethod
    def write_cube(cube, cube_name, iq_filer):
        """
        Write a cube array to ``<output_folder>/cube/fits/<cube_name>.fits``.

        Parameters
        ----------
        cube : ndarray, shape (n_models, n_rows, n_slices)
            Pseudo-cube from :meth:`lmsiq_cuber.Cuber.build`: axis 0 indexes the optical
            model (0 = perfect, 1 = as-designed, 2.. = Monte-Carlo instances), axis 1 the
            along-slice detector rows (32) and axis 2 the slice columns.
        cube_name : str
            File stem, e.g. ``'cube_ipc_on_field_1_spifu_0_cfg_3_focus_0_slices_9_to_17'``.
        iq_filer : lms_filer.Filer
            Provides ``output_folder`` and ``get_folder`` (creates the folder).

        Returns
        -------
        None

        Notes
        -----
        * Inconsistency: the ``PrimaryHDU`` carries no descriptive header (no
          wavelength, model tags, IPC state or pixel scale), so the FITS file cannot be
          interpreted without the companion pickle written by ``Cuber.build``.
        """
        fits_folder = iq_filer.get_folder(iq_filer.output_folder + 'cube/fits')
        fits_path = fits_folder + cube_name
        hdu = fits.PrimaryHDU(cube)
        hdu.writeto(fits_path + '.fits', overwrite=True)
        return

    @staticmethod
    def read_cube(process_level, ipc_tag, iq_filer):
        """
        Read a legacy 4-D cube with ``ALPHA_OS`` and ``WAVn`` header keywords (dead code).

        Parameters
        ----------
        process_level : str
            Legacy processing level tag (``'raw_zemax'``, ``'proc_zemax'`` or
            ``'proc_detector'`` in ``Globals.process_levels``).
        ipc_tag : str
            ``'ipc_on'`` or ``'ipc_off'``.
        iq_filer : lms_filer.Filer
            Expected to provide ``cube_path``.

        Returns
        -------
        data : ndarray, 4-D
            Cube with wavelength as the first axis.
        waves : list of float
            Wavelengths read from ``WAV0 .. WAV<n-1>``.
        alpha_oversampling : number
            Value of ``ALPHA_OS``.

        Raises
        ------
        AttributeError
            ``Filer`` has no ``cube_path`` attribute.
        TypeError
            ``fits.open(..., readonly=True)`` is not a valid call.

        Notes
        -----
        * Bug: see Raises; the method cannot run at HEAD and is not called.
        * Inconsistency: the expected file name ``<level>_<ipc>_cube.fits`` and header
          layout do not match what :meth:`write_cube` produces.
        """
        file_path = iq_filer.cube_path + process_level + '_' + ipc_tag + '_cube.fits'
        hdul = fits.open(file_path, readonly=True)
        hdu = hdul[0]
        data = hdu.data
        n_waves, _, _, _ = data.shape
        waves = []
        alpha_oversampling = hdu.header['ALPHA_OS']
        for i in range(n_waves):
            kw = "WAV{:d}".format(i)
            wave = hdu.header[kw]
            waves.append(wave)
        return data, waves, alpha_oversampling
