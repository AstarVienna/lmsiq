#!/usr/bin/env python
"""
LMS-RAD-04 analysis: pixel flat field (lmsaiv program).

Registered as 'lms_rad_04_pix_flat' -> ``Rad04.pix_flat`` ('Pixel flat
field') in ``lmsaiv_opt_tests.OptTests``.  Added in upstream commit
9b6143a.  The pixel flat (pixel-to-pixel relative gain) is the ratio of
each pixel's signal to the mean of the 5 x 5 box centred on it, computed
for each flood-illuminated frame and averaged over frames.  Large-scale
structure is meant to be captured separately by a 'sky' flat.

**Input data**: FITS mosaics in ``Filer.test_data_folder`` whose names
contain 'lms_rad_04', '_flat' and '_nom' (nominal mode only); at most the
first four are used.  Per-HDU keyword 'ID' (detector number).

**Self test**: with ``simulate_pix_flat = True`` (hard-coded) every input
frame is first multiplied by a synthetic gain pattern: on detector N, N
square 'spots' of 400 x 400 pixels centred on row 1024 and columns
1024 + 400 (i - 1.5) (i = 0..N-1), in which every 30th pixel in both
directions has gain 1.1.  The recovered flat should show this dot pattern.

**Output**: blocking ``Plot.mosaic`` and ``Plot.histograms`` figures of the
mean pixel flat ('LMS_RAD_04_pixel_flat_nom', display range 0.99-1.01);
per-frame figures at debug level 'high'.  Nothing is written to disk or
added to ``as_built``.

Notes
-----
- Bug: ``pix_flat`` returns None, so running it through ``lmsaiv.main``
  pickles None as the 'as built' object.
- Clean-up: ``simulate_pix_flat`` is hard-coded True, so real data would
  also be multiplied by the synthetic pattern.
- Clean-up: ``Model``, ``OptTools``, ``Util``, ``u`` and ``Box1DKernel``
  are imported but unused.

@author: Alistair Glasse
"""
import astropy.units as u
import numpy as np
from astropy.convolution import Box1DKernel
from scipy.signal import convolve2d
from lmssim_model import Model
from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer
from lmsdist_util import Util

class Rad04:
    """LMS-RAD-04 pixel-flat analysis (static methods; entry point ``pix_flat``)."""

    def __init__(self):
        """Instantiate ``lmssim_model.Model`` (result discarded), as in ``Opt01``.
        """
        _ = Model()
        return

    @staticmethod
    def pix_flat(title, as_built, **kwargs):
        """Derive a pixel flat from nominal-mode flood frames and display it.

        Upstream description: "The pixel flat is derived from flood illuminated
        images by calculating the ratio of the per pixel signal to the mean 5x5
        box smoothed value centred on it.  We may look at using a best fit 2d
        surface.  Valid pixels are identified by setting all pixels below 10 % of
        the image median to Nan."

        Per frame (first four matching mosaics) and detector:

        1. optionally multiply by the synthetic gain pattern (module docstring);
        2. set pixels below ``cut_level * nanmedian`` (``cut_level`` = 0.05) to
           NaN;
        3. box-sum with a 5 x 5 boxcar (``scipy.signal.convolve2d``, 'full',
           cropped back to 2048 x 2048);
        4. pixel flat = ``cnv_weights * pixel / box_sum``, where ``cnv_weights``
           is the number of box pixels inside the detector (25 in the interior,
           reduced along the edges).

        The per-frame flats are summed with ``Mosaic.add`` and divided by their
        number with ``Mosaic.scale``.

        Parameters
        ----------
        title : str
            Registry title; unused.
        as_built : dict
            Not used or returned.
        **kwargs
            Ignored.

        Returns
        -------
        None
            Bug: the ``lmsaiv`` caller expects the 'as built' dict.

        Notes
        -----
        - Inconsistency: the docstring says pixels below 10 % of the median are
          rejected; the code uses ``cut_level = 0.05`` (5 %).
        - Inconsistency: ``convolve2d`` propagates NaN, so every pixel within two
          pixels of a rejected one (slice gaps, detector edges of the illuminated
          area) becomes NaN in the flat, not just the rejected pixels.
        - Bug: the edge correction of ``cnv_weights`` subtracts a full row and a
          full column of the box at the corners, so corner weights are too small
          (e.g. 5 instead of 9 at the detector corner); the column correction also
          uses the row count ``nr`` (harmless for square detectors).
        - Open question: the box mean includes the pixel itself, so a pixel with
          gain g is recovered as g / (1 + (g - 1) / 25) rather than g; a
          leave-one-out mean or median filter would avoid the bias.
        - Inconsistency: ``pix_flat.hdu_list[det_idx]`` is indexed by the 'ID'
          keyword while the input HDUs are iterated in list order; the two agree
          for ToySim files only (see ``Filer.read_mosaic``).
        - Clean-up: only ``mosaics[0:4]`` are used; there is no check that any
          file was found (``super_pix_flat`` stays None -> AttributeError).
        """
        opticon = Globals.nominal
        cut_level = 0.05
        box_hw = 2                          # Smoothing box half width, width = 2 x box_hw + 1
        simulate_pix_flat = True            # True= generate fake dot pattern flat to test

        bs = 2 * box_hw + 1
        boxcar = np.ones((bs, bs))
        opticon_tag = '_' + opticon[0:3]
        inc_tags = ['lms_rad_04', '_flat', opticon_tag]
        mosaics = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags)
        # pix_flat_tag = 'pix_flat' + opticon_tag
        box_area = bs * bs
        cnv_weights = np.full(Globals.det_format, box_area)     # Weights for boxcar smooth
        nr, nc = Globals.det_format
        for b in range(0, box_hw):
            cnv_weights[0:b+1, :] -= bs
            cnv_weights[nr-b-1:nr, :] -= bs
            cnv_weights[:, 0:b+1] -= bs
            cnv_weights[:, nr-b-1:nr] -= bs

        # We generate a test pixel flat here.  Note that the pixel flat is the pixel to pixel relative gain, so
        # we inject a pattern of individual pixels with specified gain.  Large scale structures need to be captured in
        # the 'sky' flat. Here, we create a 'spot' containing a scatterring of 'n_sgp' pixels with specific gain.
        sim_flat = None
        if simulate_pix_flat:
            sim_flat = {}
            spot_radius = 200
            spot_size = 2*spot_radius, 2*spot_radius
            sgp_gain = 1.1

            spot = np.full(spot_size, 1.0)
            for row in range(0, spot_size[0], 30):
                for col in range(0, spot_size[1], 30):
                    spot[row, col] = sgp_gain

            for det_idx in range(0, 4):
                det_no = det_idx + 1
                sim_flat[det_no] = np.full((nr, nc), 1.0)
                for i_det in range(0, det_no):      # One spot per detector
                    row_cen = int(nr / 2)
                    col_cen = int(nc / 2) + int(400 * (i_det - 1.5))
                    r1, r2 = row_cen - spot_radius, row_cen + spot_radius
                    c1, c2 = col_cen - spot_radius, col_cen + spot_radius
                    sim_flat[det_no][r1:r2, c1:c2] = spot

        pix_flat_stack = []
        for mosaic in mosaics[0:4]:
            if Globals.is_debug('high'):
                Plot.mosaic(mosaic)

            pix_flat_name = mosaic.name + '_pix_flat'
            pix_flat = mosaic.copy(clear_data=True, copy_name=pix_flat_name)
            print('Analysing image file= ', mosaic.name)
            for hdu in mosaic.hdu_list:
                det_no = int(hdu.header['ID'])
                det_idx = det_no - 1

                img = np.array(hdu.data)            # Copy the image data
                if sim_flat is not None:
                    img *= sim_flat[det_no]

                img_median = np.nanmedian(img)
                z_cut = cut_level * img_median
                img_thresh = np.where(img < z_cut, np.nan, img)

                img_cnv = convolve2d(img_thresh, boxcar, mode='full')
                nr_cnv, nc_cnv = img_cnv.shape
                img_pf = cnv_weights * img_thresh / img_cnv[box_hw:nr_cnv-box_hw, box_hw:nc_cnv-box_hw]
                pix_flat.hdu_list[det_idx].data = img_pf
                print('Calculated pixel flat for detector ', det_no)
            if Globals.is_debug('high'):
                Plot.mosaic(pix_flat, vmin=0.99, vmax=1.01)
                Plot.histograms(pix_flat, range=(0.99, 1.01), make_labels=True)
            pix_flat_stack.append(pix_flat)
        # Average all pixel flats
        super_pix_flat = None
        n_pix_flats = len(pix_flat_stack)
        for pix_flat in pix_flat_stack:
            if super_pix_flat is None:
                super_pix_flat = pix_flat.copy(copy_name='LMS_RAD_04_pixel_flat' + opticon_tag)
            else:
                super_pix_flat.add(pix_flat)
        super_pix_flat.scale(1./ n_pix_flats)
        Plot.mosaic(super_pix_flat, vmin=0.99, vmax=1.01)
        Plot.histograms(super_pix_flat, range=(0.99, 1.01), make_labels=True)

        return
