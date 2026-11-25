#!/usr/bin/env python
"""
LMS-OPT-03 analysis stub: monochromatic point / line spread function (lmsaiv program).

Registered as 'lms_opt_03_psf' -> ``Opt03.psf`` in
``lmsaiv_opt_tests.OptTests`` with the title 'Monochomatic Point and line
spread function'.  As committed the module is non-functional: it is a
partial copy of an early version of the LMS-OPT-01 flood analysis (dark
statistics, legacy ``flood_stats`` slice map, approximate relative
response) written for the pre-``Mosaic`` tuple layout, and no PSF or LSF
measurement is implemented.

**Input data (as written)**: FITS mosaics tagged [test_name, 'nom_dark']
and [test_name, 'flood'] in ``Filer.test_data_folder``; header keywords
'HIERARCH ESO INS WLEN CEN', 'X_CEN', 'X_SIZE', 'Y_SIZE',
'HIERARCH pixel_size'.

**Output**: intended ``as_built['slice_map']`` and a relative response
plot.  In practice the method raises before producing anything.

Notes
-----
- Inconsistency: ``psf(test_name, title, as_built, **kwargs)`` does not
  match the ``method(title, as_built, **kwargs)`` call in
  ``OptTests.run`` (TypeError: missing 'as_built').
- Bug: ``Filer.read_mosaic_list(inc_tags)`` passes one positional
  argument where ``(folder, inc_tags)`` is required (IndexError on
  ``args[0]`` inside ``read_mosaic_list``), and
  ``Filer.read_mosaic_list(inc_tags=[...])`` uses a keyword the function
  does not accept (TypeError).
- Bug: ``OptTools.flood_stats`` does not exist (the legacy routine is
  ``lmsaiv_opt01.Opt01.flood_stats``).
- Bug: ``u`` (astropy.units) and ``Model`` (lmssim_model) are used but not
  imported.
- Bug: keywords 'X_SIZE', 'Y_SIZE' and 'HIERARCH pixel_size' are not
  written by ``lmssim_toy`` (which writes 'X_CEN', 'Y_CEN' and
  'HIERARCH AIT PIXEL_PITCH').
- Inconsistency: file tag 'flood' does not match the 'flat_bb' naming of
  ``lms_obs_map``; tuple unpacking of mosaics is incompatible with
  ``lms_mosaic.Mosaic``; ``as_built['slice_map']`` differs from the
  ``slice_map_...`` keys used by LMS-OPT-01 / LMS-OPT-02.
- Open question: LMS-OPT-03 needs laser + single-pinhole frames
  (``lms_obs_map`` tag ``las_psf_``) and a PSF / LSF centroid-and-width
  measurement; none of that exists yet.

@author: Alistair Glasse
"""
import math

import numpy as np
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt03:
    """LMS-OPT-03 point / line spread function analysis (stub, non-functional)."""

    def __init__(self):
        """Do nothing; all methods are static."""
        return

    @staticmethod
    def psf(test_name, title, as_built, **kwargs):
        """Intended LMS-OPT-03 analysis of laser + single-pinhole data (does not run).

        As written: read the darks, print ``OptTools.dark_stats``, plot them,
        then for each 'flood' mosaic compute the legacy slice map with
        ``OptTools.flood_stats`` (non-existent) and a 1000 K black-body relative
        response with the same approximate 0.08 um dispersion as
        ``lmsaiv_opt01.Opt01._find_rrf``.

        Parameters
        ----------
        test_name : str
            File-name tag (e.g. 'lms_opt_03').  Not supplied by ``OptTests.run``.
        title : str
            Plot title.
        as_built : dict
            'As built' dictionary; would receive 'slice_map'.
        **kwargs
            ``do_plot`` : bool, default True (later overwritten by a hard-coded
            True).

        Returns
        -------
        dict
            ``as_built`` (never reached).

        Notes
        -----
        - See the module notes for the list of defects; the first statement,
          ``Filer.read_mosaic_list(inc_tags)``, already fails with IndexError
          inside ``read_mosaic_list`` (``args[0]`` on an empty ``args``).
        - Clean-up: the hard-coded ``do_plot = True`` overrides the keyword.
        """
        inc_tags = [test_name, 'nom_dark']
        darks = Filer.read_mosaic_list(inc_tags)
        do_plot = kwargs.get('do_plot', True)
        if do_plot:
            Plot.mosaic(darks[0], title=title)
            Plot.histograms(darks[0])
        OptTools.dark_stats(darks)

        floods = Filer.read_mosaic_list(inc_tags=[test_name, 'flood'])
        for flood in floods:
            slice_map, profiles = OptTools.flood_stats(flood)
            do_plot = True
            if do_plot:
                Plot.profiles(profiles)
                Plot.mosaic(flood, title=title, cmap='hot')        # Use cmap='hot', 'gray' etc.
                Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
            as_built['slice_map'] = slice_map

        # Generate relative response tuple.
        cols = np.arange(0, 4096, 1)
        for flood in floods:
            slice_map = as_built['slice_map']
            rrf = OptTools.copy_mosaic(slice_map, copy_name='rel_res_function')
            rrf_name, rrf_primary_header, rrf_hdus = rrf
            Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
            name, primary_hdr, hdus = flood
            wave_mosaic_cen = primary_hdr['HIERARCH ESO INS WLEN CEN'] * u.micron
            _, _, slice_map_hdus = slice_map
            for i in range(0, 4):
                slice_map_data = slice_map_hdus[i].data
                slice_mask = np.where(slice_map_data > 0., 1., 0.)
                # Very approximate dispersion...!
                hdr = hdus[i].header
                flood_image = hdus[i].data
                x_det_cen = float(hdr['X_CEN']) * u.mm
                n_det_cols = float(hdr['X_SIZE'])
                pix_size = hdr['HIERARCH pixel_size'] * u.mm
                c_det_cen = x_det_cen / pix_size
                c_det_org = c_det_cen - n_det_cols / 2
                disp = .08 * u.micron / (2. * n_det_cols)
                waves = wave_mosaic_cen + disp * (c_det_org + cols)
                flux = Model.black_body(waves, tbb=1000.)
                n_det_rows = int(hdr['Y_SIZE'])
                rrf_image = rrf_hdus[i].data
                for row in range(0, n_det_rows):
                    idx = np.argwhere(slice_mask[row] > 0.)
                    rrf_image[row, idx] = flood_image[row, idx] / flux[idx]
                rrf_hdus[i].data = rrf_image
            Plot.mosaic(rrf, title='Rel Response Function', cmap='grey', mask=(0.0, 'black'))
        print('Done')
        return as_built

    @staticmethod
    def _parse_abo(file_name):
        """Parse alpha, beta and observation number from a legacy '_grid' file name.

        Searches for '_grid' and then the tags '_a', '_b' and '_o'; the first two
        are signed ('p' / 'n' followed by three digits), the last unsigned.

        Parameters
        ----------
        file_name : str
            File name such as '..._grid_ap100_bn050_o001.fits'.

        Returns
        -------
        tuple of float
            ``(alpha, beta, obs_no)``.

        Notes
        -----
        - Inconsistency: current file names from ``lms_obs_map`` use
          'bb_iso_alpha_a{p|n}NNN_b{p|n}NNN' with no '_grid' or '_o' tags, so
          ``str.find`` returns -1 and the parse is meaningless for them.
        - Clean-up: duplicated verbatim in ``lmsaiv_opt01.Opt01._parse_abo`` and
          ``lmsaiv_opt03.Opt03._parse_abo``; neither copy is called.
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
