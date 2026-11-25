#!/usr/bin/env python
"""Toy throughput model for the METIS LMS (echelle blaze and "slow" optics).

Purpose
-------
Provides a rough analytic estimate of the LMS optical efficiency as a function
of wavelength and echelle diffraction order.  It has two parts, a wavelength-
independent "slow" term for the mirror and prism transmission
(:meth:`Efficiency.slow`) and a sinc-squared echelle blaze term
(:meth:`Efficiency.blaze`) whose argument is the echelle incidence angle
obtained from the legacy :class:`lms_wcal.Wcal` wavelength calibration.

Program
-------
Not part of the ``lmsdist`` processing chain.  It is exercised only by the
one-off study script ``lms_test.py`` (together with
``lmsdist_plot.Plot.efficiency_v_wavelength``).

Input data
----------
None directly.  :meth:`Efficiency.blaze` and :meth:`Efficiency.test` call
``Wcal.find_echelle_angle`` / ``Wcal.find_echelle_setting`` /
``Wcal.find_wavelength``, which read the class attributes ``Wcal.p_wtoa`` and
``Wcal.p_atow``.  Those are populated only by ``Wcal.read_poly`` from a CSV of
linear wavelength <-> echelle-angle polynomials (``*_dist_wcal.txt`` under
``../output/distortion/...`` relative to ``src/``).  That file is not in the
repository, so the model cannot be evaluated from a clean checkout.

Outputs
-------
Plain floats / numpy arrays; nothing is written to disk.

Notes
-----
- Open question: no design document is cited for the constants (mirror
  reflectivity 0.99, ZnSe transmission 0.98, blaze peak 0.75, angular width
  0.23, reflection counts 4+4+5+6).  Their provenance should be confirmed with
  the author before the model is used for anything quantitative.
- Bug: ``Efficiency.blaze`` divides by ``phase`` which is exactly zero when the
  echelle angle is 0.0 deg, raising ``ZeroDivisionError``.
- Inconsistency: the ``wave`` argument of ``Efficiency.slow`` is unused; the
  "slow" transmission is wavelength independent.
- Clean-up: depends on hidden class state in ``Wcal`` (``p_wtoa`` / ``p_atow``)
  that must be loaded by the caller beforehand.
"""
import math
import numpy as np
from lms_wcal import Wcal

class Efficiency:
    """Analytic LMS efficiency model (blaze x slow optics).

    All methods are static; the class carries no state of its own.  The blaze
    term depends on the class-level wavelength calibration held in
    :class:`lms_wcal.Wcal`, which must have been loaded with
    ``Wcal.read_poly`` first.

    Notes
    -----
    - Clean-up: the ``__init__`` method does nothing; the class is used purely
      as a namespace for static methods.
    """


    def __init__(self):
        """Create an ``Efficiency`` object.  No state is initialised."""
        return

    @staticmethod
    def combined(wave, ech_order):
        """Return the product of the slow-optics and blaze efficiencies.

        Parameters
        ----------
        wave : float
            Wavelength in microns.
        ech_order : int
            Echelle diffraction order.

        Returns
        -------
        float
            ``Efficiency.slow(wave) * Efficiency.blaze(wave, ech_order)``.

        Notes
        -----
        - Bug: inherits the ``ZeroDivisionError`` of :meth:`blaze` at an
          echelle angle of exactly 0.0 deg.
        """
        eff_slow = Efficiency.slow(wave)
        eff_blaze = Efficiency.blaze(wave, ech_order)
        eff_combi = eff_slow * eff_blaze
        return eff_combi

    @staticmethod
    def slow(wave):
        """Return the wavelength-independent transmission of the non-grating optics.

        The model is ``t_slow = r_mirror**n_mirrors * t_znse**n_prism`` with
        ``r_mirror = 0.99``, ``n_mirrors = 4 + 4 + 5 + 6 = 19`` (transfer
        optics, IFU, pre-disperser and main-disperser reflections),
        ``t_znse = 0.98`` and ``n_prism = 4`` prism surfaces, giving
        ``0.99**19 * 0.98**4 ~= 0.76``.

        Parameters
        ----------
        wave : float
            Wavelength in microns.  Not used.

        Returns
        -------
        float
            Slow-optics transmission (about 0.76).

        Notes
        -----
        - Inconsistency: ``wave`` is accepted but ignored.
        - Open question: the reflection counts and coating values are not
          referenced to any document.
        """
        r_mirror = 0.99         # Mirror reflectivity
        n_tfer = 4              # Transfer optics reflections
        n_ifu = 4               # IFU reflections
        n_pre = 5           # Pre-disperser reflections
        n_main = 6              # Main disperser reflections
        n_mirrors = n_tfer + n_ifu + n_pre + n_main
        t_mirrors = math.pow(r_mirror, n_mirrors)
        t_znse = 0.98          # Transmission of AT coated ZnSe prism
        n_prism = 4             # No of prism surfaces transited
        t_prism = math.pow(t_znse, n_prism)
        t_slow = t_mirrors * t_prism
        return t_slow

    @staticmethod
    def blaze(wave, order):
        """Return the echelle blaze efficiency for a wavelength and order.

        The echelle incidence angle ``ea`` (deg) is taken from
        ``Wcal.find_echelle_angle(wave, order)`` and the efficiency is a
        sinc-squared blaze profile

        .. math::

            \\theta = n \\pi \\sin(ea) \\cdot 0.23, \\qquad
            eff = 0.75 \\left( \\frac{\\sin \\theta}{\\theta} \\right)^2

        where ``n`` is the diffraction order, ``0.75`` is ``eff_max`` and
        ``0.23`` is ``eff_width`` (the incidence angle at which adjacent
        orders have equal efficiency, per the in-code comment).

        Parameters
        ----------
        wave : float
            Wavelength in microns.
        order : int
            Echelle diffraction order.

        Returns
        -------
        float
            Blaze efficiency in the range 0 to 0.75.

        Raises
        ------
        ZeroDivisionError
            If the echelle angle returned by ``Wcal`` is exactly 0.0 deg
            (``phase`` is then 0 and ``sin(phase) / phase`` is evaluated with
            Python scalars, not numpy).
        TypeError
            If ``Wcal.p_wtoa`` has not been loaded (``Wcal.read_poly`` not
            called), because ``None`` is then indexed.

        Notes
        -----
        - Bug: division by zero at ``ea == 0``; the sinc limit of 1 is not
          handled.
        - Inconsistency: the previous docstring described the model as
          ``eff_max cos(k x echelle_angle)``; the implementation is a
          sinc-squared profile.
        """
        eff_max = 0.75
        eff_width = 0.23            # Incidence angle where efficiency of adjacent orders is equal

        ech_angle = Wcal.find_echelle_angle(wave, order)
        ea = ech_angle * math.pi / 180.0
        phase = order * math.pi * math.sin(ea) * eff_width
        amp = math.sin(phase) / phase
        eff = eff_max * amp * amp
        return eff

    @staticmethod
    def test(w_test_lim):
        """Tabulate blaze efficiency versus wavelength for a range of orders.

        The order range is taken from ``Wcal.find_echelle_setting`` at the two
        wavelength limits (the shorter wavelength gives the higher order
        ``omax``).  The echelle angle is then scanned over 50 samples from
        -6.0 to +6.0 deg (the values returned by ``find_echelle_setting`` are
        overwritten) and, for each order, the wavelength at that angle is
        found with ``Wcal.find_wavelength`` and the blaze efficiency with
        :meth:`blaze`.

        Parameters
        ----------
        w_test_lim : sequence of float
            Two-element ``(w_short, w_long)`` wavelength limits in microns.

        Returns
        -------
        wavs : numpy.ndarray
            Shape ``(50, n_orders)`` wavelengths in microns.
        effs : numpy.ndarray
            Shape ``(50, n_orders)`` blaze efficiencies (slow-optics term not
            applied; the multiplication is commented out).
        orders : numpy.ndarray
            The orders ``numpy.arange(omin, omax)``; note ``omax`` itself is
            excluded.
        angs : numpy.ndarray
            The 50 echelle angles in degrees.

        Notes
        -----
        - Inconsistency: ``n_orders = omax - omin`` and ``arange(omin, omax)``
          drop the highest order returned for ``w_test_lim[0]``.
        - Inconsistency: the ``amin`` / ``amax`` echelle angles returned by
          ``Wcal.find_echelle_setting`` are immediately replaced by the fixed
          values -6.0 and +6.0.
        - Bug: the angle scan includes ``ang == 0.0`` only if 50 samples over
          [-6, 6] land exactly on zero (they do not, the step is 12/49), so
          the ``ZeroDivisionError`` in :meth:`blaze` is avoided by accident.
        - Clean-up: the result tuple is consumed only by
          ``lmsdist_plot.Plot.efficiency_v_wavelength``, which is itself broken
          (it calls a non-existent ``Axes.plot_focal_planes``).
        """
        omax, amax = Wcal.find_echelle_setting(w_test_lim[0])
        omin, amin = Wcal.find_echelle_setting(w_test_lim[1])

        n_orders = int(omax - omin)
        amin = -6.0
        amax = 6.0
        n_pts = 50
        da = (amax - amin) / (n_pts - 1)
        orders = np.arange(omin, omax, 1)
        angs = np.zeros(n_pts)
        wavs, effs, = np.zeros((n_pts, n_orders)), np.zeros((n_pts, n_orders))
        for i in range(0, n_pts):
            ang = amin + i * da
            angs[i] = ang
            for j in range(0, n_orders):
                order = orders[j]
                wav = Wcal.find_wavelength(ang, order)
                slow_eff = Efficiency.slow(wav)
                blaze_eff = Efficiency.blaze(wav, order)
                wavs[i,j] = wav
                effs[i,j] = blaze_eff   # * slow_eff
        return wavs, effs, orders, angs
