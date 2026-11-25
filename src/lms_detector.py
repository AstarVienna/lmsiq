#!/usr/bin/env python
"""
H2RG detector geometry constants and the detector sampling / noise forward model.

Purpose
-------
Holds the LMS detector constants (18 um pixel pitch, 2048 x 2048 format, 2 x 2
mosaic, QE, dark current, read noise, well depth), provides the block-mean rebinning
that converts an oversampled Zemax PSF image to detector pixels
(:meth:`Detector.down_sample`), the mosaic corner coordinates
(:meth:`Detector.get_bounds`) and a simple noise model (:meth:`Detector.detect`).

Shared by the ``lmsiq`` program (down-sampling of PSF images in
:mod:`lmsiq_cuber`, :mod:`lmsiq_phase` and :mod:`lmsiq_test`) and by ``lmssim``
(``detect``).

Input data
----------
Images are passed in memory.  In ``lmsiq`` they are 128 x 128 pixel Zemax PSFs sampled
at 4.5 um (4x oversampled with respect to the 18 um pixel), read from
``../data/iq/<opticon>/<date_stamp>/``; ``down_sample`` therefore returns 32 x 32
detector-pixel images.  No files are read or written by this module.

Outputs
-------
None on disk.  The constructor prints the pixel size.

Notes
-----
* Inconsistency: ``Detector.mosaic_gap`` is 2.23 mm whereas ``Globals.det_gap`` is
  3.0 mm; the two constants describe the same inter-detector gap and are used in
  different parts of the package (the irdb uses 2.23 mm).
* Bug: :meth:`Detector.detect` adds ``rng.poisson(sqrt(image))`` to the image instead
  of replacing the image by ``rng.poisson(image)``; this biases the mean signal upward
  by ``sqrt(signal)`` and gives the wrong variance.
* Inconsistency: the original module docstring read "Inter pixel capacitance object",
  which describes :mod:`lms_ipc`, not this module.
* Clean-up: ``det_pix_size`` and ``nom_pix_pitch`` are the same number; the constructor
  message that distinguishes a 'true' pixel size from the one in use is a remnant of a
  configurable pixel size that no longer exists.
"""
import numpy as np
from lms_globals import Globals


class Detector:
    """
    LMS H2RG detector constants and static sampling / noise helpers.

    All members are static or class attributes; instantiation is only needed for the
    informational print in ``__init__``.

    Attributes
    ----------
    nom_pix_pitch : float
        Nominal pixel pitch in microns, from ``Globals.nom_pix_pitch`` (18.0).
    det_pix_size : float
        Pixel size used for sampling, equal to ``nom_pix_pitch``.
    det_size : int
        Pixels per side of one H2RG (2048).
    mosaic_format : tuple of int
        Detector mosaic layout, (2, 2).
    mosaic_gap : float
        Gap between active areas of adjacent detectors in mm (2.23).
    detector_edge_mm : float
        Physical size of one detector in mm (``det_size * det_pix_size / 1000``).
    mosaic_edge_mm : float
        Full mosaic width in mm (two detectors plus one gap).
    qe : float
        Quantum efficiency, electrons per photon (0.7).
    idark : float
        Dark current in electrons per second per pixel (0.05).
    rnoise : float
        Read noise in electrons (10).
    q_well : float
        Well depth in electrons (1e5).

    Notes
    -----
    * Inconsistency: ``mosaic_gap`` (2.23 mm) disagrees with ``Globals.det_gap``
      (3.0 mm).
    """

    nom_pix_pitch = Globals.nom_pix_pitch
    det_pix_size = nom_pix_pitch
    det_size = 2048             # H2RG format
    mosaic_format = 2, 2
    mosaic_gap = 2.23           # Gap in mm
    detector_edge_mm = 0.001 * det_size * det_pix_size
    mosaic_edge_mm = detector_edge_mm * mosaic_format[0] + mosaic_gap
    qe = 0.7                                    # QE (el/photon)
    idark = 0.05        # Dark current approx, from Roy (el / second). (Finger quotes 0.01)
    rnoise = 10.        # Very approx read noise (Roy model) (Finger/Rauscher use 10 el.)
    q_well = 1.E+5      # Well depth (el.)

    def __init__(self):
        """
        Print the detector pixel size; no state is created.

        Notes
        -----
        * Clean-up: the message contrasts ``det_pix_size`` with the 'true' pitch
          ``nom_pix_pitch`` although both are the same class attribute.
        """
        fmt = "Detector pixel size = {:10.3f} micron (Note 'true' LMS det pixel size = {:10.3f}"
        print(fmt.format(Detector.det_pix_size, Detector.nom_pix_pitch))
        return

    @staticmethod
    def get_bounds():
        """
        Return the corner coordinates of the four detectors in the mosaic focal plane.

        The mosaic is centred on the origin.  Each detector has edge length
        ``nom_pix_pitch * det_size / 1000`` mm and the detectors are separated by
        ``mosaic_gap`` mm, so the inner edges lie at +-``mosaic_gap / 2``.

        Returns
        -------
        x : ndarray, shape (16,)
            x coordinates (mm) of the four corners of each detector, in blocks of four:
            bottom-left detector, then +x, then +y, then +x +y.  Within a block the
            corner order is (left, bottom), (left, top), (right, bottom), (right, top).
        y : ndarray, shape (16,)
            Matching y coordinates (mm).

        Notes
        -----
        * Inconsistency: uses ``Detector.mosaic_gap`` (2.23 mm); the affine transforms
          in :mod:`lms_transform` use ``Globals.det_gap`` (3.0 mm).
        """
        det_edge = Detector.nom_pix_pitch * Detector.det_size / 1000.
        det_gap = Detector.mosaic_gap
        xy_lim = det_edge + det_gap / 2.
        xbl = np.array([-xy_lim, -xy_lim, -xy_lim + det_edge, -xy_lim + det_edge])
        ybl = np.array([-xy_lim, -xy_lim + det_edge, -xy_lim, -xy_lim + det_edge])

        x, y = np.zeros(16), np.zeros(16)
        xy_offset = det_edge + det_gap
        x[0:4], y[0:4] = xbl, ybl
        x[4:8], y[4:8] = xbl + xy_offset, ybl
        x[8:12], y[8:12] = xbl, ybl + xy_offset
        x[12:16], y[12:16] = xbl + xy_offset, ybl + xy_offset
        return x, y

    @staticmethod
    def down_sample(image_in, im_pix_pitch):
        """
        Rebin an oversampled image to detector pixels by block averaging.

        The integer sampling factor is ``int(det_pix_size / im_pix_pitch)`` (4 for the
        4.5 um Zemax images).  The image is reshaped to
        ``(n_rows / s, s, n_cols / s, s)`` and the mean is taken over both ``s`` axes,
        so each output pixel is the mean (not the sum) of an ``s x s`` block.  Because
        the block mean is used, the total flux of the output is ``1 / s**2`` times the
        input total; the metrics in :mod:`lmsiq_analyse` are all normalised so this
        scaling cancels.

        Parameters
        ----------
        image_in : ndarray, shape (n_rows, n_cols)
            Oversampled image.  ``n_rows`` and ``n_cols`` must be multiples of the
            sampling factor (128 -> 32 for the standard Zemax images).
        im_pix_pitch : float
            Pixel pitch of ``image_in`` in microns (4.5 for Zemax PSFs).

        Returns
        -------
        image_out : ndarray, shape (n_rows // s, n_cols // s)
            Block-mean rebinned image.

        Notes
        -----
        * Open question: if ``n_rows`` or ``n_cols`` is not a multiple of ``s`` the
          ``reshape`` raises ``ValueError``; there is no check or padding.
        * Inconsistency: the block *mean* is used, so ``down_sample`` does not conserve
          flux; the docstring of the original called this "measuring" an observation.
        """
        sampling = int(Detector.det_pix_size / im_pix_pitch)        # Image pixels per detector pixel
        nr, nc = image_in.shape
        n_frame_rows, n_frame_cols = int(nr/sampling), int(nc/sampling)
        image_out = image_in.reshape(n_frame_rows, sampling, n_frame_cols, -1).mean(axis=3).mean(axis=1)
        return image_out

    @staticmethod
    def detect(frame_in, dit, ndit):
        """
        Convert a photocurrent frame to a noisy detector image (electrons).

        Steps: copy the frame; add the dark current ``idark`` (el/s) to every pixel;
        multiply by ``dit`` to get electrons per CDS frame; add a Poisson-distributed
        'shot' term drawn with mean ``sqrt(image)``; add Gaussian read noise of sigma
        ``rnoise`` electrons.

        Parameters
        ----------
        frame_in : ndarray, shape (2048, 2048)
            Photocurrent map in electrons per second per pixel.  The shape must equal
            ``(det_size, det_size)`` because the dark and noise arrays are built with
            that shape.
        dit : float
            Detector integration time in seconds.
        ndit : int
            Number of integrations.  Only used to compute ``t_int = dit * ndit``, which
            is itself unused.

        Returns
        -------
        image : ndarray, shape (2048, 2048)
            Detector image in electrons, single DIT, with noise.

        Notes
        -----
        * Bug: ``shot = rng.poisson(np.sqrt(image))`` is *added* to ``image``.  A
          correct shot-noise realisation is ``image = rng.poisson(image)``; the current
          code biases every pixel upward by about ``sqrt(signal)`` and gives the wrong
          variance.  Not used by ``lmsiq``; used by ``lmssim``.
        * Inconsistency: the original docstring described a returned mosaic dictionary
          with 'images', 'cartoons' and 'reads' entries; the method returns a single
          array.
        * Clean-up: ``ndit`` and ``t_int`` have no effect; QE (``qe``) and well depth
          (``q_well``) are not applied.
        """
        t_int = dit * ndit
        det_shape = Detector.det_size, Detector.det_size
        frame = np.array(frame_in)                          # Copy frame
        dark = np.full(det_shape, Detector.idark)           # Dark photocurrent (el / sec)
        frame += dark
        image = frame * dit                                 # Convert from photocurrent to electrons per CDS frame

        rng = np.random.default_rng()
        shot = rng.poisson(np.sqrt(image), det_shape)
        image += shot
        read = rng.normal(loc=0., scale=Detector.rnoise, size=det_shape)
        image += read
        # print('Det.detect')
        return image

    @staticmethod
    def get_tag(det_pix_size):
        """
        Build a file-name tag encoding a pixel size in microns.

        Parameters
        ----------
        det_pix_size : float
            Pixel size in microns, e.g. 18.0.

        Returns
        -------
        tag : str
            ``'_pix_<int>_<hundredths>'``, e.g. ``'_pix_18_00'`` for 18.0 um.

        Notes
        -----
        * Clean-up: not called anywhere in the package at HEAD.
        """
        num = int(det_pix_size)
        dec = int(100. * (det_pix_size - num) + 0.5)
        tag = "_pix_{:02d}_{:02d}".format(num, dec)
        return tag
