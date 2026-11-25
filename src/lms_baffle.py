#!/usr/bin/env python
"""
Coronagraph PSF display for LMS baffle (beam-dump) location study.

Purpose
-------
One-off study script that loads a METIS apodising-phase-plate (APP) coronagraph PSF
image, overlays the LMS IFU field-of-view rectangle and displays the result on a
log10 colour scale so that a beam dump (baffle) position can be chosen by eye.  A
beam-dump placement scan is sketched but commented out, so the script only produces
the display and a single (unused) aperture photometry value.

This module is a stand-alone script and is not part of the ``lmsiq``, ``lmsdist``,
``lmssim`` or ``lmsaiv`` program flows.  Nothing executes on import; run :func:`main`.

Input data
----------
``../data/baffle/APP_METIS_PSF.fits`` (relative to ``src/``): a single-extension FITS
image, assumed square, with the PSF in the primary HDU.  The plate scale is hard-coded
as 0.0013 arcsec per pixel.  The file is not in the repository.

Outputs
-------
None written to disk.  One interactive matplotlib window (``Plot.show()``).

Notes
-----
* Open question: the plate scale ``scale = 0.0013`` arcsec/pixel is hard-coded (a
  commented alternative ``1.6 / nrc`` suggests a 1.6 arcsec field); the provenance of
  the value and of ``APP_METIS_PSF.fits`` is not recorded.
* Open question: the IFU field is drawn as 0.577 x 0.897 arcsec offset by 0.055 arcsec
  in x, which does not match ``Globals.alpha_fov`` (0.897) x ``Globals.beta_fov``
  (28 x 20.7 mas = 0.580) in orientation or offset; it is unclear which axis is alpha.
* Clean-up: the beam-dump scan loop is commented out and ``dump``, ``drb``, ``gap_xy``
  and the photometry value ``phot`` (16 x 16 pixel rectangular aperture at pixel (0, 0),
  i.e. the image corner rather than the PSF centre) are computed but never used.
* Clean-up: this script imports :class:`lmsdist_plot.Plot`, not
  :class:`lmsiq_plot.Plot`; ``Filer()`` is instantiated but unused.
* Inconsistency: ``Analyse()`` is instantiated only for the side effect of setting
  ``Analyse.fwhm_sigma``, which :meth:`lmsiq_analyse.Analyse.exact_rectangular` does
  not need.
"""
import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import RectangularAperture
from astropy.io import fits
from lms_filer import Filer
from lmsdist_plot import Plot
from lmsiq_analyse import Analyse


def main():
    """
    Load the APP coronagraph PSF, overlay the IFU field and display it.

    Flow:

    1. Instantiate ``Filer``, ``Plot`` and ``Analyse`` (side effects only).
    2. Open ``../data/baffle/APP_METIS_PSF.fits`` and take the primary HDU image and
       header; derive the angular extent from the image size and the hard-coded plate
       scale 0.0013 arcsec/pixel.
    3. Define the IFU field rectangle (0.577 x 0.897 arcsec, x offset 0.055 arcsec) as
       a closed polygon.
    4. Define beam-dump parameters (attenuation 1e-7, 1000 x 300 pixel footprint); the
       column scan applying the dump is commented out.
    5. Compute a 16 x 16 pixel rectangular aperture sum at (0, 0) with
       :meth:`lmsiq_analyse.Analyse.exact_rectangular` (result unused).
    6. Plot ``log10(image)`` with a 'hot' colour map clipped to [-10, 0], axes in
       arcsec, colour bar labelled ``log10(Tx)``, the IFU polygon in green, and call
       ``Plot.show()``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If ``../data/baffle/APP_METIS_PSF.fits`` is absent (it is not in the repo).

    Notes
    -----
    * Bug: ``np.log10(image)`` is evaluated on the raw image without clipping, so
      zero or negative pixels produce ``-inf``/``nan`` with runtime warnings.
    * Bug: ``Plot.set_plot_area('FP1 APP PSF')`` passes a positional argument to a
      ``**kwargs``-only method in :mod:`lmsdist_plot`, which raises ``TypeError``.
      Even with that fixed, ``set_plot_area`` returns ``(fig, ax_list)`` and the
      script indexes the tuple as ``ax_list[0, 0]``, which also fails.  The script
      therefore does not run at HEAD.
    """
    print('lms_baffle - Starting')
    print('- Program to analyse coronagraph image for baffle location optimisation ')
    filer = Filer()
    plot = Plot()
    analyse = Analyse()

    # File locations and names
    # output_folder = "../output/distortion/{:s}".format(optical_configuration)
    # output_folder = Filer.get_folder(output_folder)
    # zem_folder = "../data/distortion/{:s}".format(optical_configuration)
    # zem_folder = Filer.get_folder(zem_folder)

    baffle_image_file = '../data/baffle/APP_METIS_PSF.fits'
    print("- baffle image  = {:s}".format(baffle_image_file))

    hdu_list = fits.open(baffle_image_file, mode='readonly')
    image, header = hdu_list[0].data, hdu_list[0].header
    nrc, _ = image.shape
    scale = 0.0013      # 1.6 / nrc
    xy_range = nrc * scale
    xy1, xy2 = -xy_range / 2., xy_range / 2.

    # Define IFU field of view
    ifu_x_fov, ifu_y_fov = 0.577, 0.897
    ifu_x_off = 0.055
    xl, xr = ifu_x_off, ifu_x_off + ifu_x_fov
    yb, yt = -ifu_y_fov / 2., ifu_y_fov / 2.
    ifu_x = [xl, xl, xr, xr, xl]
    ifu_y = [yb, yt, yt, yb, yb]

    # Define gap between IFU field of view and beam dump
    gap_xy = 1.

    # Define beam dump as non-integer rectangle at a non-integer position
    dump_attenuation = 1.E-7
    dump_c_cov, dump_r_cov = 300, 1000      # Column and row coverage of beam dump
    dump = np.full((dump_r_cov, dump_c_cov), dump_attenuation)
    drb = int((nrc - dump_r_cov) / 2)

    # for dcl in range(0, 1200):      # Iterate over column of left edge of beam dump
    #     dcr = dcl + dump_c_cov
    #
    #     im = np.array(image)        # Copy image
    #     im[drb:drt, dcl:dcr] *= dump

    xcen, ycen = 0., 0.
    ap_pos, ap_width, ap_height = (xcen, ycen), 16., 16.
    rect_aper = RectangularAperture(ap_pos, w=ap_width, h=ap_height)
    phot = Analyse.exact_rectangular(image, rect_aper)

    ax_list = Plot.set_plot_area('FP1 APP PSF')
    ax = ax_list[0, 0]
    # ax.imshow(image)
    # ax.set_xlim([xy1, xy2])
    # ax.set_ylim([xy1, xy2])
    log_image = np.log10(image)
    vpmin, vpmax = -10, 0.

    im_map = ax.imshow(log_image, extent=(xy1 - 0.005, xy2 + 0.005, xy1 - 0.005, xy2 + 0.005),
                       interpolation='nearest', cmap='hot', vmin=vpmin, vmax=vpmax, origin='lower')
    ax.set_xlabel('arcsec')
    ax.set_ylabel('arcsec')
    plt.colorbar(mappable=im_map, ax=ax_list, label='log10(Tx)', shrink=0.75)

    ax.plot(ifu_x, ifu_y, color='green')
    Plot.show()

    print('lms_baffle - Done')


if __name__ == '__main__':
    main()
