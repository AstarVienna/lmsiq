#!/usr/bin/env python
"""
Worked example of the legacy LMS distortion forward model for a single test point.

Purpose
-------
Demonstration script that takes one (wavelength, alpha, beta) test point, finds the
echelle order and angle that place it on the detector, prints the grating blaze / slow
optics / combined efficiencies, plots efficiency curves for the adjacent orders, and
pushes the point through a polynomial-fit distortion transform to a detector (x, y)
position in mm and pixels.

This module is a stand-alone script exercising the legacy ``lmsdist`` machinery
(:class:`lms_wcal.Wcal`, :class:`lms_efficiency.Efficiency`,
:class:`lmsdist_util.Util`).  It is not called by any of the four programs.  Nothing
executes on import; run :func:`main`.

Input data
----------
``../output/lms_dist_poly_old.txt`` (relative to ``src/``): a legacy polynomial-fit
file produced by an earlier ``lmsdist`` version.  Also requires the wavelength /
echelle-angle polynomial file read by :meth:`lms_wcal.Wcal.read_poly`.  Neither file is
in the repository.

Outputs
-------
Console text and one interactive matplotlib window (efficiency vs wavelength).

Notes
-----
* Bug: ``Wcal.read_poly()`` is called without the required ``wcal_file`` argument, so
  the script raises ``TypeError`` immediately after printing the slice number.
* Bug: ``Globals.mm_pix`` does not exist in :class:`lms_globals.Globals` (the pixel
  pitch is ``Globals.nom_pix_pitch`` in microns), so the final pixel conversion would
  raise ``AttributeError``.
* Bug: ``util.get_polyfit_transform`` and ``util.apply_distortion`` are not methods of
  :class:`lmsdist_util.Util` at HEAD (``get_polyfit_transform`` lives on
  :class:`lms_wcal.Wcal` with a different signature and the transform is applied by
  ``Util.apply_svd_distortion``), so those calls would raise ``AttributeError``.
* Inconsistency: the polynomial file name ``lms_dist_poly_old.txt`` and the
  ``(poly, ech_bounds)`` return signature belong to a superseded ``lmsdist`` output
  format; the current products are the ``lms_efp_mfp_*.fits`` transform files.
* Inconsistency: ``fp2_x = alpha / Globals.efp_arcsec_mm`` divides a float by an
  ``astropy`` Quantity, giving a Quantity in mm/arcsec**-1 rather than a plain float.
* Clean-up: the local name ``slice`` shadows the Python built-in.
"""
from lmsdist_util import Util
from lms_wcal import Wcal
from lms_globals import Globals
from lms_efficiency import Efficiency
from lmsdist_plot import Plot


def main():
    """
    Trace one test point (4.70 um, alpha = 0.3 arcsec, beta = 0.1 arcsec) to the detector.

    Flow:

    1. Instantiate ``Util`` and ``Plot``; set ``poly_file`` to
       ``'../output/lms_dist_poly_old.txt'``.
    2. Convert alpha to an entrance-focal-plane x coordinate with
       ``Globals.efp_arcsec_mm`` and beta to a slice number as
       ``int(15 + 28 * beta / Globals.beta_fov)``.
    3. Read the wavelength <-> echelle-angle polynomials (``Wcal.read_poly()``) and find
       the optimum echelle order and angle for the wavelength
       (:meth:`lms_wcal.Wcal.find_echelle_setting`).
    4. Print blaze, slow-optics and combined efficiencies from
       :class:`lms_efficiency.Efficiency` and plot efficiency curves over 3.0-5.0 um
       (``Efficiency.test`` -> ``Plot.efficiency_v_wavelength``).
    5. Compute the grating phase ``wave * order / (2 * Globals.rule_spacing)``.
    6. Read the polynomial transform file, build the (A, B, AI, BI) transform for this
       order, slice and echelle angle, print it, apply (A, B) to (phase, fp2_x) and
       print the detector position in mm and in pixels (``Globals.mm_pix``).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        From ``Wcal.read_poly()``, which requires a file-name argument.
    AttributeError
        From ``util.get_polyfit_transform``, ``util.apply_distortion`` and
        ``Globals.mm_pix`` if execution got that far (see module Notes).

    Notes
    -----
    * Bug: see module Notes; the script cannot complete at HEAD.
    * Open question: ``int(15 + 28 * beta / Globals.beta_fov)`` mixes a float with an
      ``astropy`` arcsec Quantity; whether ``int()`` of the resulting Quantity is
      intended (it works only because the result is dimensionless after conversion) is
      not documented.
    """
    print('lms_test - Starting')
    util = Util()
    plot = Plot()
    poly_file = '../output/lms_dist_poly_old.txt'

    # Define test point(s) (wavelength/um, alpha/as, beta/as)
    wave = 4.70
    alpha = 0.3
    fp2_x = alpha / Globals.efp_arcsec_mm
    beta = 0.1
    slice = int(15 + 28 * beta / Globals.beta_fov)
    print("Slice= {:d}".format(slice))

    # Find configuration for this wavelength
    wcal = Wcal()
    Wcal.read_poly()
    ech_order, ech_angle = wcal.find_echelle_setting(wave)
    print("Echelle order= {:d}, Echelle angle = {:10.3f}".format(ech_order, ech_angle))

    blaze = Efficiency()
    eff_blaze = Efficiency.blaze(wave, ech_order)
    eff_slow = Efficiency.slow(wave)
    eff = Efficiency.combined(wave, ech_order)
    print('At this setting..')
    print('Blaze efficiency =    {:10.3f}'.format(eff_blaze))
    print('Slow efficiency =     {:10.3f}'.format(eff_slow))
    print('Combined efficiency = {:10.3f}'.format(eff))

    w_test_lim = [3.0, 5.0]
    weoas = Efficiency.test(w_test_lim)                         # Generate efficiency curves for adjacent orders
    plot.efficiency_v_wavelength(weoas, xlim=w_test_lim)        # Plot blaze profiles (cf SPIE 11451_49)

    phase = wave * ech_order / (2.0 * Globals.rule_spacing)
    print("Phase= {:10.3f}, FP2_X = {:10.3f}".format(phase, fp2_x))

    poly, ech_bounds = util.read_polyfits_file(poly_file)
    transform = util.get_polyfit_transform(poly, ech_bounds, ech_order, slice, ech_angle)
    util.print_poly_transform(transform)

    a, b, ai, bi = transform
    det_x, det_y = util.apply_distortion([phase], [fp2_x], a, b)
    print("Det_x= {:10.3f}, Det_y= {:10.3f}".format(det_x[0], det_y[0]))
    pix_x = det_x[0] / Globals.mm_pix
    pix_y = det_y[0] / Globals.mm_pix

    print("Pix_x= {:10.3f}, Pix_y= {:10.3f}".format(pix_x, pix_y))
    print("lms_test - Done.")


if __name__ == '__main__':
    main()
