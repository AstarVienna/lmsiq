#!/usr/bin/env python
"""
LMS-RAD-01 analysis: dark current and read noise (lmsaiv program).

Registered as 'lms_rad_01_dark' -> ``Rad01.dark`` ('Dark current') in
``lmsaiv_opt_tests.OptTests``.  Added in upstream commit 9b6143a, taking
over the dark-frame statistics that previously ran at the start of
LMS-OPT-01 (``Opt01._analyse_darks`` and ``OptTools.dark_stats``).

**Input data**: FITS mosaics in ``Filer.test_data_folder`` whose names
contain both 'lms_rad_01' and '_dark' (``lms_obs_map`` names darks
'{test}_{step}_{nom|ext}_dark').  Header keywords read: primary 'HIERARCH
ESO DET DIT' (s) and 'HIERARCH ESO DET NDIT'; per detector HDU 'HIERARCH
ESO DET3 CHIP GAIN' (el/ADU, a string in ToySim output).

**Output**: a printed per-detector table (median and standard deviation in
DN, dark current in el/s, read noise in el) for every dark; at debug level
'low' a mosaic image and pixel histograms of the first dark.  Nothing is
added to ``as_built``.

Notes
-----
- Open question: ``dark_stats`` divides the standard deviation by
  sqrt(2 / ndit), which is the read-noise estimator for the difference
  of two frames; it is applied to single dark frames here.
- Open question: the ToySim writer (``lmssim_toy``) does not set
  'HIERARCH ESO DET DIT' / 'NDIT'; the values appear to come unchanged
  from the '../config/sim_template.fits' header rather than from the
  simulated ``lms_dit``.
- Clean-up: ``_find_rrf`` and ``_parse_abo`` are verbatim copies of the
  ``lmsaiv_opt01`` helpers and are not called here; ``OptTools`` is
  imported but unused.
- Clean-up: the gain keyword name 'HIERARCH ESO DET3 CHIP GAIN' is
  hard-coded for all four detectors.

@author: Alistair Glasse
"""
import math
import astropy.units as u
import numpy as np

from lmssim_model import Model
from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer

class Rad01:
    """LMS-RAD-01 dark analysis (static methods; entry point ``dark``)."""

    def __init__(self):
        """Instantiate ``lmssim_model.Model`` (result discarded), as in ``Opt01``.
        """
        _ = Model()
        return

    @staticmethod
    def dark(title, as_built, **kwargs):
        """Run the LMS-RAD-01 dark analysis (``analyse_darks``) and return ``as_built`` unchanged.

        Parameters
        ----------
        title : str
            Registry title; unused.
        as_built : dict
            'As built' dictionary; returned unchanged.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            ``as_built``.

        Notes
        -----
        - Inconsistency: the upstream docstring was copied from ``Opt01.fov``
          ("Field of view calculation ... Populates the slice bounds map"); it has
          been replaced here.
        """
        Rad01.analyse_darks()
        print('Done')
        return as_built

    @staticmethod
    def analyse_darks():
        """Read the LMS-RAD-01 dark frames and print their statistics.

        Reads all mosaics whose file names contain ['lms_rad_01', '_dark'] from
        ``Filer.test_data_folder`` and passes them to ``dark_stats``.  At debug
        level 'low' or higher the first dark is shown with ``Plot.mosaic`` and
        ``Plot.histograms``.

        Returns
        -------
        None

        Notes
        -----
        - Bug: if no dark files are found ``darks`` is empty and ``darks[0]``
          raises IndexError at debug level 'low'.
        """
        inc_tags = ['lms_rad_01', '_dark']
        darks = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags)
        Rad01.dark_stats(darks)
        if Globals.is_debug('low'):
            title = 'Dark'
            Plot.mosaic(darks[0], title=title)
            Plot.histograms(darks[0])
        return

    @staticmethod
    def dark_stats(mosaics):
        """Print dark-frame statistics for each detector of each mosaic.

        For every mosaic the integration parameters are read from the primary
        header ('HIERARCH ESO DET DIT' seconds, 'HIERARCH ESO DET NDIT'), and for
        each detector HDU the gain is read from 'HIERARCH ESO DET3 CHIP GAIN'
        (el/ADU, stored as a string by ToySim and cast with ``float``).  The
        printed columns are::

            median [DN]      = np.median(data)
            stdev  [DN]      = np.std(data)
            median [el/sec]  = median * gain / dit
            Rd_Noise [el]    = stdev * gain / sqrt(2 / ndit)

        Parameters
        ----------
        mosaics : list of Mosaic
            Dark exposures.

        Returns
        -------
        None
            Results are printed only.

        Notes
        -----
        - Open question: the dark current uses ``dit`` alone; if the frame is a
          sum over ``ndit`` sub-integrations the divisor should be ``dit * ndit``
          (``t_int`` is computed but only printed).
        - Open question: the sqrt(2 / ndit) factor is appropriate for the
          difference of two frames, not for a single frame's standard deviation.
        - Inconsistency: the header line formats ``ndit`` with '{:d}', which
          raises ValueError if the keyword is stored as a float.
        - Clean-up: return the statistics (per detector) instead of printing, so
          they can be used as QC values.
        """
        for mosaic in mosaics:
            dit = mosaic.primary_hdr['HIERARCH ESO DET DIT']
            ndit = mosaic.primary_hdr['HIERARCH ESO DET NDIT']
            t_int = dit * ndit

            print()
            print("File = {:s}".format(mosaic.name))
            fmt = "Signal distribution statistics, ndit= {:d} x dit= {:3.1f} sec, integration time = {:6.1f}"
            print(fmt.format(ndit, dit, t_int))
            fmt = "{:>8s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('Detector', 'median', 'stdev', 'median', 'Rd_Noise'))
            print(fmt.format('No.', 'DN', 'DN', 'el/sec.', 'el.'))
            fmt = "{:8d},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"

            for i, hdu in enumerate(mosaic.hdu_list):
                el_adu = float(hdu.header['HIERARCH ESO DET3 CHIP GAIN'])
                median = np.median(hdu.data)
                stdev = np.std(hdu.data)
                median_current = median * el_adu / dit
                rd_noise = stdev * el_adu / math.sqrt(2. / ndit)
                text = fmt.format(i + 1, median, stdev, median_current, rd_noise)
                print(text)
        return

    @staticmethod
    def _find_rrf(flood, slice_map):
        """Copy of ``lmsaiv_opt01.Opt01._find_rrf`` (relative response function); not called.

        See ``Opt01._find_rrf`` for the algorithm and its pixel-pitch unit bug.
        """
        # Generate relative response tuple.
        cols = np.arange(0, 4096, 1)
        rrf = slice_map.copy(copy_name='rel_res_function')

        wave_mosaic_cen = flood.primary_hdr['HIERARCH ESO INS WLEN CEN'] * u.micron
        for i in range(0, 4):
            slice_map_data = slice_map.hdu_list[i].data
            slice_mask = np.where(slice_map_data > 0., 1., 0.)
            # Very approximate dispersion...!
            hdr = flood.hdu_list[i].header
            flood_image = flood.hdu_list[i].data
            x_det_cen = float(hdr['X_CEN']) * u.mm
            n_det_rows, n_det_cols = flood_image.shape
            pix_size = float(hdr['HIERARCH AIT PIXEL_PITCH']) * u.mm
            c_det_cen = x_det_cen / pix_size
            c_det_org = c_det_cen - n_det_cols / 2
            disp = .08 * u.micron / (2. * n_det_cols)
            waves = wave_mosaic_cen + disp * (c_det_org + cols)
            flux = Model.black_body(waves, tbb=1000.)
            rrf_image = rrf.hdu_list[i].data
            for row in range(0, n_det_rows):
                idx = np.argwhere(slice_mask[row] > 0.)
                rrf_image[row, idx] = flood_image[row, idx] / flux[idx]
            rrf.hdu_list[i].data = rrf_image
        Plot.mosaic(rrf, title='Rel Response Function', cmap='grey', mask=(0.0, 'black'))
        return

    @staticmethod
    def _parse_abo(file_name):
        """Copy of ``lmsaiv_opt01.Opt01._parse_abo``; not called.

        Extracts the alpha, beta and observation number from a legacy
        '..._grid..._a{p|n}NNN_b{p|n}NNN_oNNN' file name.

        Returns
        -------
        tuple of float
            ``(alpha, beta, obs)``.
        """
        signed = {'_a': True, '_b': True, '_o': False}
        ip = file_name.find('_grid') + 5        # Get start position of wavelength, alpha and beta sub strings
        abo = []
        for tag in signed:
            ip = file_name.find(tag, ip) + len(tag)
            sign = 1.0
            if signed[tag]:
                sign = 1. if file_name[ip] == 'p' else -1.
                ip += 1
            mag = float(file_name[ip: ip + 3])
            val = sign * mag
            abo.append(val)
        return tuple(abo)

