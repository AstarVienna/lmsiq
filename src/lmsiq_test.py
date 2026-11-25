"""
Visual sanity check of the detector diffusion / IPC model on synthetic images.

Purpose
-------
Builds two 32 x 32 oversampled test images (a 2 x 2 image-pixel point source at image
pixels ``[5:7, 5:7]``, and a uniform flat), passes each through
:meth:`lms_ipc.Ipc.apply` and :meth:`lms_detector.Detector.down_sample`, and saves a
2 x 3 collage of the six images, once on a linear scale clipped to [0.95, 1.05] and once
on a log scale.  It is a by-eye check that the flat is preserved (apart from the
intra-pixel screen) and that the point source spreads as expected; it asserts nothing.

Belongs to the ``lmsiq`` program; run from :func:`lmsiq.main` when ``run_test`` is True.

Input data
----------
None; the images are generated in memory at ``im_pix_size = 4.5`` um, giving
``oversampling = int(18 / 4.5) = 4`` and 8 x 8 detector-pixel output images.

Outputs
-------
``../output/iq/<opticon>/<date_stamp>/test/test_illumination.png`` and
``test_illumination_log.png``.

Notes
-----
* Inconsistency: this is not a unit test (no assertions, no pytest); the
  ``lmsiq.py`` comment describing it as "50 % IPC and an image with one bright pixel"
  does not match the code, which uses the nominal 1.3 % IPC and a 2 x 2 source.
* Inconsistency: the in-line comment says the 2 x 2 source is "centred at the boundary
  between sub-pixel 0 and 1", but image pixels 5-6 lie wholly inside detector pixel 1
  (image pixels 4-7 at 4x oversampling), occupying its sub-pixels 1 and 2; the source
  does not straddle a detector pixel boundary.
* Clean-up: :meth:`lmsiq_plot.Plot.images` calls ``plt.show()`` after ``savefig``, so
  the run blocks on two interactive windows in a non-Agg backend.
"""
import numpy as np
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot


class Test:
    """
    Namespace for the IPC / down-sampling visual check.

    Notes
    -----
    * Clean-up: the class holds no state; ``run`` could be a module function.
    """

    def __init__(self):
        """No-op constructor."""
        return

    @staticmethod
    def run(iq_filer):
        """
        Generate the point-source and flat test collages.

        Parameters
        ----------
        iq_filer : lms_filer.Filer
            Provides ``output_folder`` and ``get_folder``; the PNGs are written to
            ``<output_folder>/test/``.

        Returns
        -------
        None

        Notes
        -----
        * Inconsistency: ``Ipc.apply`` uses whatever ``Ipc.ipc_factor`` is set at the
          time of the first call; ``lmsiq.main`` calls ``Test.run`` before
          ``Ipc.set_inter_pixel``, so the kernel is built with ``ipc_factor = 0.0``
          (diffusion only) and, because the kernel is cached, the subsequent cube build
          in the same process also runs with IPC off.
        * Open question: the panel titles 'Zem point' / 'Zem flat' suggest the inputs
          stand in for Zemax images, but they are 32 x 32 rather than 128 x 128.
        """
        im_pix_size = 4.5

        oversampling = int(Detector.det_pix_size / im_pix_size)
        im_ps = np.zeros((32, 32))
        im_ps[5:7, 5:7] = 1.0          # 2x2 artificial source, centred at the boundary between sub-pixel 0 and 1
        im_ipc_ps = Ipc.apply(im_ps, oversampling)
        im_det_ps = Detector.down_sample(im_ipc_ps, im_pix_size)

        im_fl = np.zeros((32, 32))
        im_fl[:, :] = 1.0
        im_ipc_fl = Ipc.apply(im_fl, oversampling)
        im_det_fl = Detector.down_sample(im_ipc_fl, im_pix_size)
        collage = [im_ps, im_ipc_ps, im_det_ps,
                   im_fl, im_ipc_fl, im_det_fl]
        pane_titles = ['Zem point', 'point + diffusion', 'det point',
                       'Zem flat', 'flat + diffusion', 'det flat']

        png_folder = iq_filer.output_folder + '/test'
        png_folder = iq_filer.get_folder(png_folder)
        png_name = 'test_illumination'
        title = png_name
        png_path = png_folder + png_name
        Plot.images(collage,
                    nrowcol=(2, 3), title=title, pane_titles=pane_titles, aspect='equal',
                    do_log=False, vlim=[0.95, 1.05], png_path=png_path)
        png_path = png_folder + png_name + '_log'
        Plot.images(collage,
                    nrowcol=(2, 3), title=title, pane_titles=pane_titles, aspect='equal',
                    do_log=True, png_path=png_path)
        return
