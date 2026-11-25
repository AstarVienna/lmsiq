#!/usr/bin/env python
"""
Detector charge-diffusion, inter-pixel-capacitance and intra-pixel response forward model.

Purpose
-------
Builds a 3 x 3 detector-pixel kernel (sampled at the oversampled Zemax image
resolution) that combines a radial Gaussian charge-diffusion term with an
inter-pixel-capacitance (IPC) coupling to the four edge-adjacent pixels, convolves
oversampled PSF images with it, and applies a multiplicative intra-pixel transmission
screen that removes 3.3 % of the signal from the outer sub-pixels of each detector
pixel.  The result models what an H2RG would record before rebinning to detector
pixels with :meth:`lms_detector.Detector.down_sample`.

Belongs to the ``lmsiq`` program (called from :mod:`lmsiq_cuber`, :mod:`lmsiq_phase`
and :mod:`lmsiq_test`).  All state is held in class attributes.

Input data
----------
Oversampled images in memory: 128 x 128 pixel Zemax PSFs at 4.5 um sampling (4x
oversampled with respect to the 18 um H2RG pixel), loaded by
:class:`lmsiq_image_manager.ImageManager` from
``../data/iq/<opticon>/<date_stamp>/``.  The oversampling factor is passed in; it
should be an even integer (in practice always 4).

Outputs
-------
None written to disk by the working methods.  The two plotting methods would write
PNGs under ``iq_filer.iq_png_folder + 'kernel/'`` but are broken (see Notes).

Notes
-----
* Inconsistency: the ``make_kernel`` comment says the diffusion term is
  ``A exp(-B x^2)`` with ``B = 2.4``; :meth:`Ipc.add_diffusion` actually evaluates
  ``exp(-d^2 / b^2)`` with ``b = 0.65`` detector pixels (equivalent to ``B = 1 / b^2
  = 2.37`` per pixel^2).
* Inconsistency: the kernel is built once, on the first call to :meth:`Ipc.apply`,
  using whatever ``Ipc.ipc_factor`` is set at that time.  Later calls to
  :meth:`Ipc.set_inter_pixel` change ``ipc_factor`` and the tag but do *not* rebuild
  the kernel, so toggling IPC on/off within one process has no effect on the images.
* Clean-up: :meth:`Ipc.apply` convolves by a pure-Python double loop over image pixels
  (O(N_pix * kernel_size^2)), acceptable for 128 x 128 images but infeasible for a
  2048 x 2048 frame; ``scipy.signal.fftconvolve`` or ``scipy.ndimage.convolve`` would
  be equivalent.
* Bug: :meth:`Ipc.plot_kernel` and :meth:`Ipc.plot_kernel_profile` reference
  ``iq_filer.iq_png_folder`` (no such Filer attribute), pass ``(kernel, params)`` tuples
  where :meth:`lmsiq_plot.Plot.images` expects arrays, and call
  ``ax.plot_focal_planes`` (not a matplotlib Axes method), so neither runs.
* Open question: the IPC coupling of 1.3 % is attributed to "Rauscher"; the diffusion
  width to "Hardy et al. 2014"; the 3.3 % corner loss has no reference.  No wavelength
  dependence is modelled.
* Clean-up: ``Globals`` is imported but unused; the commented-out ``test`` method
  refers to a ``convolve`` method that no longer exists.
"""
import math
import numpy as np
from lms_globals import Globals
from lmsiq_plot import Plot


class Ipc:
    """
    Static container for the diffusion + IPC kernel and intra-pixel screen.

    Attributes
    ----------
    kernel : ndarray or None
        Normalised convolution kernel of shape ``(3*os + 1, 3*os + 1)`` where ``os`` is
        the oversampling; ``None`` until :meth:`make_kernel` has run.
    ipc_factor : float
        IPC coupling fraction currently in force (0.0 or ``ipc_factor_nominal``).
    ipc_factor_nominal : float
        Nominal IPC coupling, 0.013 (1.3 %).
    intra_screen : ndarray or None
        ``(os, os)`` multiplicative transmission stamp; ``None`` until built.
    tag : str or None
        ``'ipc_on'`` or ``'ipc_off'``, used in output file names.
    folder : str or None
        ``tag + '/'``.
    oversampling : int or None
        Oversampling factor the kernel was built for.
    det_kernel_size : int or None
        Kernel size in detector pixels (3).

    Notes
    -----
    * Inconsistency: ``ipc_factor`` is listed twice in the class body; the second
      assignment (``ipc_factor_nominal, ipc_factor = 0.013, 0.0``) wins.
    * Clean-up: the constructor accepts ``**kwargs`` but ignores them.
    """

    kernel, ipc_factor, intra_screen = None, None, None
    tag, folder = None, None
    oversampling, det_kernel_size = None, None
    ipc_factor_nominal, ipc_factor = 0.013, 0.0

    def __init__(self, **kwargs):
        """
        No-op constructor; all state lives in class attributes.

        Parameters
        ----------
        **kwargs
            Ignored.
        """
        return

    @staticmethod
    def make_kernel(oversampling, **kwargs):
        """
        Build and store the normalised diffusion + IPC kernel.

        The kernel covers 3 x 3 detector pixels at image sampling, with
        ``kernel_size = 3 * oversampling + 1`` (odd, so that the central image pixel is
        the peak).  The diffusion profile is written into it by :meth:`add_diffusion`,
        the IPC coupling is added by :meth:`add_ipc`, and finally the kernel is divided
        by its sum so that convolution conserves flux.  The result is stored in
        ``Ipc.kernel`` together with ``Ipc.oversampling`` and ``Ipc.det_kernel_size``.

        Parameters
        ----------
        oversampling : int
            Image pixels per detector pixel (4 for the Zemax data).
        **kwargs
            Ignored.

        Returns
        -------
        None

        Notes
        -----
        * Inconsistency: the in-line comment quotes a diffusion term ``A exp(-B x^2)``
          with ``B = 2.4``; the code in :meth:`add_diffusion` uses ``exp(-d^2 / 0.65^2)``.
        * Inconsistency: the ``+1`` makes the kernel one image pixel larger than three
          detector pixels, so the IPC blocks added by :meth:`add_ipc` (which span
          ``[0:os], [os:2os], [2os:3os]``) are offset by half an image pixel from the
          kernel centre.
        * Clean-up: a commented line shows the oversampling was once taken from
          ``Globals.get_im_oversampling('raw_zemax')``, which no longer exists.
        """
        kernel_det_size = 3
        # oversampling = Globals.get_im_oversampling('raw_zemax')
        kernel_size = kernel_det_size * oversampling + 1        # Force odd number, central pixel = peak transmission
        kernel_shape = kernel_size, kernel_size
        kernel = np.zeros(kernel_shape)
        # Generate diffusion component  A exp(-B x^2) where A = 1 and B = 2.4 to minimise transmission losses
        kernel = Ipc.add_diffusion(kernel, oversampling)
        kernel = Ipc.add_ipc(kernel, oversampling, kernel_det_size)
        # Finally, re-normalise the kernel to unity (no signal lost from image)
        renorm = np.sum(kernel) # * (1. + loss)
        kernel /= renorm
        Ipc.kernel = kernel
        Ipc.oversampling, Ipc.det_kernel_size = oversampling, kernel_det_size
        return

    @staticmethod
    def make_intra_pixel_transmission_screen(oversampling):
        """
        Build the ``(oversampling, oversampling)`` intra-pixel transmission stamp.

        The stamp is 1.0 in the interior sub-pixels and ``1 - 0.033`` on the one
        sub-pixel wide border, modelling a 3.3 % loss of signal at detector pixel
        edges/corners.  It is stored in ``Ipc.intra_screen`` and tiled over the image by
        :meth:`apply`.

        Parameters
        ----------
        oversampling : int
            Image pixels per detector pixel.  Must be >= 3 for an interior to exist; the
            original comment says it should be even and is "really always equal to 4".

        Returns
        -------
        None

        Notes
        -----
        * Open question: with ``oversampling = 4`` the border is 12 of 16 sub-pixels,
          so the mean pixel transmission is ``1 - 0.033 * 12 / 16 = 0.975``, not
          ``1 - 0.033``; whether the 3.3 % figure is meant per border sub-pixel or per
          pixel is not documented.
        """
        stamp_shape = oversampling, oversampling
        core, edge = 1., 1. - 0.033
        screen = np.full(stamp_shape, edge)
        screen[1:-1, 1:-1] = core
        Ipc.intra_screen = screen
        return

    # @staticmethod
    # def test(iq_filer):
    #     params = 'file_id', 1.
    #     image1 = np.zeros((128, 128)) + 1.0      # make an extended image
    #     obs1 = image1, params
    #     obs_con1 = Ipc.convolve(obs1)
    #     image2 = np.zeros((128, 128))
    #     image2[64, 64] = 1.0
    #     obs2 = image2, params
    #     obs_con2 = Ipc.convolve(obs2)
    #     sum2 = np.sum(obs_con2[0])
    #     image3 = np.zeros((128, 128))
    #     image3[66, 66] = 1.0
    #     obs3 = image3, params
    #     obs_con3 = Ipc.convolve(obs3)
    #     sum3 = np.sum(obs_con3[0])
    #     png_folder = iq_filer.iq_png_folder + 'kernel/'
    #     png_folder = iq_filer.get_folder(png_folder)
    #     png_file = 'convolution_test'
    #     png_path = png_folder + png_file
    #
    #     Plot.images([obs1, obs2, obs3, obs_con1, obs_con2, obs_con3],
    #                 nrowcol=(2, 3), title='Ipc Test', shrink=0.1, png_path=png_path)
    #
    #     return
    #
    @staticmethod
    def set_inter_pixel(inter_pixel):
        """
        Switch the IPC coupling on or off and set the file-name tag.

        Parameters
        ----------
        inter_pixel : bool
            True -> ``Ipc.ipc_factor = ipc_factor_nominal`` (0.013), ``tag = 'ipc_on'``;
            False -> ``ipc_factor = 0.0``, ``tag = 'ipc_off'``.  ``Ipc.folder`` is set to
            ``tag + '/'``.

        Returns
        -------
        None

        Notes
        -----
        * Bug: this does not reset ``Ipc.kernel``, so if a kernel has already been
          built by :meth:`apply` the new ``ipc_factor`` is never used.  In
          :meth:`lmsiq_cuber.Cuber.build` the flag is set before the first ``apply``
          call, so the single-setting case works; iterating over
          ``inter_pixels = [True, False]`` would silently reuse the first kernel.
        * Inconsistency: the tag controls only the IPC term; the diffusion term and the
          intra-pixel screen are always applied when :meth:`apply` is called, so
          ``'ipc_off'`` images still include diffusion.  (Callers skip ``apply``
          entirely when ``inter_pixel`` is False, which is the actual 'off' path.)
        """
        Ipc.ipc_factor = Ipc.ipc_factor_nominal if inter_pixel else 0.0
        Ipc.tag = 'ipc_on' if inter_pixel else 'ipc_off'
        Ipc.folder = Ipc.tag + '/'
        return

    @staticmethod
    def plot_kernel(iq_filer):
        """
        Plot the kernel image to ``<iq_png_folder>/kernel/kernel.png`` (broken).

        Parameters
        ----------
        iq_filer : lms_filer.Filer
            Filer expected to provide ``iq_png_folder``.

        Returns
        -------
        None

        Notes
        -----
        * Bug: ``iq_filer.iq_png_folder`` is not an attribute of
          :class:`lms_filer.Filer`, and ``Plot.images`` is passed a list containing a
          ``(kernel, params)`` tuple rather than an array; the method raises before
          plotting.  Not called at HEAD.
        * Inconsistency: the title prints ``ipc_factor`` (a fraction, 0.013) with a
          '%' sign.
        """
        title = "Diffusion + IPC ({:6.3f} %)".format(Ipc.ipc_factor)
        params = title, None
        png_folder = iq_filer.iq_png_folder + 'kernel/'
        png_folder = iq_filer.get_folder(png_folder)
        png_file = 'kernel'
        png_path = png_folder + png_file
        Plot.images([(Ipc.kernel, params)],
                    nrowcol=(1, 1), title=title, png_path=png_path)
        return

    @staticmethod
    def add_diffusion(im_kernel, oversampling):
        """
        Overwrite a kernel array with a radial Gaussian charge-diffusion profile.

        For every kernel element the distance ``d`` (in detector pixels) from the array
        centre ``((nrows - 1) / 2, (ncols - 1) / 2)`` is computed as
        ``sqrt(x^2 + y^2)`` with ``x = (r - rc) / oversampling``,
        ``y = (c - cc) / oversampling``, and the element is set to
        ``a * exp(-d^2 / b^2)`` with ``a = 1.0`` and ``b = 0.65`` detector pixels.  This
        corresponds to a Gaussian with sigma ``b / sqrt(2) = 0.46`` pixels, FWHM
        ``1.08`` pixels.  Existing kernel values are replaced, not added to.

        Parameters
        ----------
        im_kernel : ndarray, shape (n, n)
            Kernel array; its contents are overwritten in place.
        oversampling : int
            Image pixels per detector pixel.

        Returns
        -------
        im_kernel : ndarray
            The same array, filled with the diffusion profile (unnormalised, peak 1).

        Notes
        -----
        * Inconsistency: the ``make_kernel`` comment describes this as
          ``A exp(-B x^2)`` with ``B = 2.4``; ``1 / b^2 = 2.37`` per pixel^2 is close but
          the code is written in terms of ``b = 0.65``.  The attribution "Hardy et al.
          2014" is given in :meth:`make_kernel`.
        * Clean-up: pure-Python double loop; ``np.meshgrid`` would be equivalent.
        """
        nrows, ncols = im_kernel.shape
        a, b = 1.0, 0.65
        bsq = b * b
        rc, cc = (nrows / 2.0) - 0.5, (ncols / 2.0) - 0.5
        for r in range(0, nrows):
            x = (r - rc) / oversampling
            for c in range(0, ncols):
                y = (c - cc) / oversampling
                d = math.sqrt(x*x + y*y)
                diff = a * math.exp(-d * d / bsq)
                im_kernel[r, c] = diff
        return im_kernel

    @staticmethod
    def add_ipc(kernel, oversampling, det_kernel_size):
        """
        Add the inter-pixel-capacitance coupling to the kernel.

        The signal in the central detector pixel of the kernel,
        ``sig_cen = sum(kernel[os:2os, os:2os])``, is scaled by ``Ipc.ipc_factor`` and
        divided by ``os^2`` to give a per-image-pixel value ``ipc_sig``.  A 3 x 3
        detector-pixel array is built with ``ipc_sig`` in the four edge-adjacent
        positions (up, down, left, right) and zero in the centre and corners, and each
        detector pixel value is added uniformly to the corresponding ``os x os`` block
        of the image-sampled kernel.

        Parameters
        ----------
        kernel : ndarray, shape (3*os + 1, 3*os + 1)
            Kernel already containing the diffusion profile; modified in place.
        oversampling : int
            Image pixels per detector pixel (``os``).
        det_kernel_size : int
            Kernel size in detector pixels (3).

        Returns
        -------
        kernel : ndarray
            The same array with the IPC blocks added.

        Notes
        -----
        * Open question: the coupling is not subtracted from the central pixel; flux
          conservation is restored only by the global renormalisation in
          :meth:`make_kernel`, so the effective central-pixel fraction is slightly lower
          than ``1 - 4 * ipc_factor``.
        * Inconsistency: the blocks span rows/cols ``[0:os], [os:2os], [2os:3os]`` of a
          ``3os + 1`` wide kernel, so the last row and column receive no IPC and the
          blocks are not centred on the diffusion peak (see :meth:`make_kernel`).
        * Clean-up: with ``Ipc.ipc_factor = 0.0`` (``set_inter_pixel(False)``) the
          method adds zeros.
        """
        rc1, rc2 = oversampling, 2 * oversampling
        sig_cen = np.sum(kernel[rc1:rc2, rc1:rc2])
        ovs2 = oversampling * oversampling          # Normalise IPC to image pixels
        ipc_sig = Ipc.ipc_factor * sig_cen / ovs2
        det_kernel_shape = det_kernel_size, det_kernel_size
        det_kernel = np.zeros(det_kernel_shape)
        det_kernel[0, 1], det_kernel[1, 0], det_kernel[1, 2], det_kernel[2, 1] = ipc_sig, ipc_sig, ipc_sig, ipc_sig

        for r in range(0, det_kernel_size):
            imr1 = r * oversampling
            imr2 = imr1 + oversampling
            for c in range(0, det_kernel_size):
                imc1 = c * oversampling
                imc2 = imc1 + oversampling
                kernel[imr1:imr2, imc1:imc2] += det_kernel[r, c]
        return kernel

    @staticmethod
    def plot_kernel_profile(title, **kwargs):
        """
        Plot the column-summed kernel profile against detector pixel (broken).

        Intended for comparison with the Hardy et al. diffusion measurements: the kernel
        is summed over rows, optionally rescaled so its peak equals ``max_val``, and
        plotted against ``x = index / oversampling`` with vertical lines at integer
        detector pixel boundaries.

        Parameters
        ----------
        title : str
            Plot title.
        **kwargs
            ``max_val`` : float, optional -- rescale the profile peak to this value.

        Returns
        -------
        None

        Notes
        -----
        * Bug: ``Plot.set_plot_area`` returns ``(fig, ax_list)`` but the result is
          indexed as ``axs[0, 0]``; and ``ax.plot_focal_planes`` is not a matplotlib
          Axes method.  The method raises ``TypeError``/``AttributeError``.  Not called
          at HEAD.
        """
        kernel = Ipc.kernel
        max_val = kwargs.get('max_val', None)
        # Plot column combined profile for comparison with Hardy et al.
        im_k = np.sum(kernel, axis=0)
        if max_val is not None:
            f = max_val / np.amax(im_k)
            im_k *= f
        im_x = np.arange(0, len(im_k)) / Ipc.oversampling
        axs = Plot.set_plot_area(title=title)
        ax = axs[0, 0]
        ax.plot_focal_planes(im_x, im_k, ls='none', marker='o', ms=5.0, color='black')
        for x in range(0, Ipc.det_kernel_size):
            ax.plot_focal_planes([x, x], [0., np.amax(im_k)], lw=0.5, color='grey')
        Plot.show()
        return

    @staticmethod
    def apply(im1, oversampling):
        """
        Convolve an oversampled image with the diffusion + IPC kernel and apply the screen.

        Algorithm:

        1. If ``Ipc.kernel`` is ``None`` build it with :meth:`make_kernel` (using the
           current ``Ipc.ipc_factor``).
        2. For every image pixel ``(ri, ci)`` form ``patch = im1[ri, ci] * kernel``,
           clip the patch to the image boundary, renormalise the clipped patch so its
           weights sum to 1 (i.e. flux that would fall off the image is redistributed
           inside it), and accumulate it into the output image centred on ``(ri, ci)``.
        3. If ``Ipc.intra_screen`` is ``None`` build it with
           :meth:`make_intra_pixel_transmission_screen`, then multiply every
           ``oversampling x oversampling`` block of the output (aligned to the image
           origin) by the screen.

        Parameters
        ----------
        im1 : ndarray, shape (n_rows, n_cols)
            Oversampled input image (128 x 128 Zemax PSF).  Not modified.
        oversampling : int
            Image pixels per detector pixel (4).  Used only when the kernel or screen
            still has to be built and for the screen tiling stride.

        Returns
        -------
        im2 : ndarray, shape (n_rows, n_cols)
            Diffused, IPC-coupled and screened image.

        Notes
        -----
        * Clean-up: pure-Python double loop, O(n_rows * n_cols) Python iterations each
          with a ``(3os+1)^2`` array multiply; about 16 k iterations for 128 x 128 but
          4 M for a full 2048 x 2048 frame.  ``scipy.signal.fftconvolve(im1, kernel,
          mode='same')`` gives the same result apart from the edge renormalisation.
        * Inconsistency: the edge renormalisation (step 2) conserves flux at the image
          border instead of letting it leave the frame, which is a different boundary
          condition from a physical detector.
        * Bug: the kernel is cached after the first call and never rebuilt, so a
          different ``oversampling`` or ``ipc_factor`` on a later call is ignored (see
          :meth:`set_inter_pixel`).
        * Open question: the screen tiling assumes detector pixel boundaries coincide
          with image pixel ``0, os, 2os, ...``; :class:`lmsiq_test.Test` places a source
          across such a boundary to check this.
        """
        im2 = np.zeros(im1.shape)           # Create new image for output
        if Ipc.kernel is None:
            Ipc.make_kernel(oversampling)
        kernel = Ipc.kernel
        nrk, nck = kernel.shape

        rck_half = int(nrk / 2.0)
        # Apply the kernel first by brute force convolution.
        nri, nci = im1.shape
        for ri in range(0, nri):
            for ci in range(0, nci):
                patch = im1[ri, ci] * kernel
                ri1 = ri - rck_half
                rk1 = 0
                if ri1 < 0:
                    rk1 = -ri1
                    ri1 = 0
                ri2 = ri + rck_half
                rk2 = nrk - 1
                if ri2 > nri - 1:
                    rk2 -= ri2 - nri + 1
                    ri2 = nri - 1

                ci1 = ci - rck_half
                ck1 = 0
                if ci1 < 0:
                    ck1 = -ci1
                    ci1 = 0
                ci2 = ci + rck_half
                ck2 = nck - 1
                if ci2 > nci - 1:
                    ck2 -= ci2 - nci + 1
                    ci2 = nci - 1

                weight = 1. / np.sum(kernel[rk1:rk2+1, ck1:ck2+1])
                patch_cal = patch[rk1:rk2+1, ck1:ck2+1] * weight
                im2[ri1:ri2+1, ci1:ci2+1] += patch_cal

        if Ipc.intra_screen is None:
            Ipc.make_intra_pixel_transmission_screen(oversampling)
        screen = Ipc.intra_screen
        for ri in range(0, nri, oversampling):
            for ci in range(0, nci, oversampling):
                im2[ri:ri+oversampling, ci:ci+oversampling] *= screen
        return im2
