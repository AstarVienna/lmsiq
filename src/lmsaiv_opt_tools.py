"""
Shared image-processing helpers for the LMS AIT analyses (lmsaiv program).

Used by ``lmsaiv_opt01`` (LMS-OPT-01), ``lmsaiv_opt02`` (LMS-OPT-02) and the
non-functional ``lmsaiv_opt03``.  The helpers operate on
``lms_mosaic.Mosaic`` objects (four detector HDUs plus a primary header),
except ``copy_mosaic`` which is a legacy tuple-based routine.

**Input data**: ``Mosaic`` objects read by ``lms_filer.Filer.read_mosaic``.
``dark_stats`` reads the primary header keywords 'HIERARCH ESO DET DIT' and
'HIERARCH ESO DET NDIT' and, per detector HDU, 'HIERARCH ESO DET3 CHIP
GAIN' (electrons per ADU).  Nothing is read from disk directly.

**Output**:

- ``median_subtract`` returns the input list with every mosaic
  background-subtracted in place.
- ``dark_stats`` prints a per-detector table (median, standard deviation,
  dark current in el/s, read noise in electrons) and returns None.
- ``transform_detector_image`` returns the (modified) input mosaic.
- ``_find_thetas`` adds 'theta' and 'theta_err' keys to each trace
  dictionary and returns the list.

Notes
-----
- Inconsistency: ``copy_mosaic`` unpacks the legacy ``(file_name, hdr,
  hdus)`` tuple and cannot accept a ``Mosaic`` (use ``Mosaic.copy``
  instead); it is still called by ``Opt01.flood_stats`` and ``Opt03.psf``.
- Bug: ``transform_detector_image`` accepts an ``xy_pix`` offset but never
  uses it; only the rotation is applied (about the array origin, not the
  detector centre).
- Open question: ``dark_stats`` divides the standard deviation by
  sqrt(2 / ndit), which is the read-noise estimator for the difference
  of two frames; it is applied to single dark frames here.
- Open question: the ToySim writer (``lmssim_toy``) does not set
  'HIERARCH ESO DET DIT' / 'NDIT'; the values appear to come unchanged
  from the '../config/sim_template.fits' header rather than from the
  simulated ``lms_dit``.
- Clean-up: the gain keyword name 'HIERARCH ESO DET3 CHIP GAIN' is
  hard-coded for all four detectors.
"""
import math
import numpy as np
import scipy
from scipy.optimize import curve_fit, OptimizeWarning
from lms_globals import Globals


class OptTools:
    """Namespace of static helper methods for the LMS AIT analyses.

    All methods are static; the class is never meaningfully instantiated.

    Notes
    -----
    - Clean-up: could be a plain module of functions.
    """


    def __init__(self):
        """Do nothing; all methods are static."""
        return

    @staticmethod
    def median_subtract(raw_mosaics):
        """Subtract the pixel-wise median of a list of mosaics from each of them.

        The median over the stack of exposures is used as a background estimate:
        for a tunable-laser series (LMS-OPT-02 iso-lambda frames, where each
        exposure has its lines in different columns) the per-pixel median is the
        line-free background, which is then subtracted from every exposure.

        Parameters
        ----------
        raw_mosaics : list of Mosaic
            Exposures of identical geometry.  Exactly four detector HDUs per
            mosaic are assumed (``range(0, 4)``).

        Returns
        -------
        list of Mosaic
            The same list object; each mosaic has been modified in place via
            ``Mosaic.subtract``.

        Notes
        -----
        - The median mosaic is built from ``raw_mosaics[0].copy(clear_data=True)``
          and is not returned, so the background estimate is not available to the
          caller.
        - Open question: with fewer than three exposures the median is not a
          robust background (for two frames it is their mean), and lines present
          at the same column in most frames survive into the background.
        """
        first_mosaic = raw_mosaics[0]
        first_hdu_list = first_mosaic.hdu_list
        median_mosaic = first_mosaic.copy(clear_data=True)
        for i in range(0, 4):
            median_image_stack = []
            for raw_mosaic in raw_mosaics:
                raw_image = raw_mosaic.hdu_list[i].data
                median_image_stack.append(raw_image)
            mis = np.array(median_image_stack)
            median_image = np.median(mis, axis=0)
            median_mosaic.hdu_list[i].data = median_image
        for raw_mosaic in raw_mosaics:
            raw_mosaic.subtract(median_mosaic)

        return raw_mosaics

    @staticmethod
    def copy_mosaic(mosaic, clear_data=False, copy_name=''):
        """Copy a legacy mosaic tuple, optionally zeroing the image data.

        Legacy routine predating ``lms_mosaic.Mosaic``; retained for
        ``Opt01.flood_stats`` and ``Opt03.psf``.

        Parameters
        ----------
        mosaic : tuple
            ``(file_name, primary_hdr, hdus)`` where ``hdus`` is a list of image
            HDUs.
        clear_data : bool, optional
            If True, multiply each copied data array by 0.  Default False.
        copy_name : str, optional
            Name of the copy; '' (default) keeps ``file_name``.

        Returns
        -------
        tuple
            ``(name, primary_hdr, copied_hdus)``.  The primary header is shared,
            not copied.

        Notes
        -----
        - Inconsistency: unpacking a ``Mosaic`` object as a 3-tuple raises
          TypeError; the modern equivalent is ``Mosaic.copy``.
        - Clean-up: remove once ``Opt01.flood_stats`` and ``Opt03.psf`` are
          ported or deleted.
        """
        file_name, hdr, hdus = mosaic
        moscopy_hdus = []
        for hdu in hdus:
            moscopy_hdu = hdu.copy()
            if clear_data:
                moscopy_hdu.data *= 0.
            moscopy_hdus.append(moscopy_hdu)
        moscopy_name = file_name if copy_name == '' else copy_name
        moscopy = moscopy_name, hdr, moscopy_hdus
        return moscopy

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
            Dark exposures (LMS-OPT-01 '_dark' files).

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
    def transform_detector_image(mosaic, det_no, xy_pix=(0, 0), angle=0.0):
        """Rotate the image of one detector in a mosaic (to emulate a misaligned detector).

        The image of detector ``det_no`` is replaced by
        ``scipy.ndimage.affine_transform`` applied with a pure rotation matrix
        ``[[cos a, sin a, 0], [-sin a, cos a, 0], [0, 0, 1]]``, cubic spline
        interpolation (``order=3``) and the image median as fill value.

        Parameters
        ----------
        mosaic : Mosaic
            Mosaic whose ``hdu_list[det_no - 1]`` image is modified in place.
        det_no : int
            Detector number 1..4; the HDU at list index ``det_no - 1`` is used.
        xy_pix : tuple of float, optional
            Intended (x, y) pixel offset.  Accepted but not used.  Default (0, 0).
        angle : float, optional
            Rotation angle in degrees.  Default 0.0.

        Returns
        -------
        Mosaic
            The input mosaic, modified in place.

        Notes
        -----
        - Bug: ``xy_pix`` is ignored; no translation is applied.
        - Open question: ``affine_transform`` is called without an ``offset``, so
          the rotation is about array element (0, 0), not the detector centre.
        - Inconsistency: indexes the HDU list with ``det_no - 1`` while
          ``Opt02._extract_det_traces`` uses ``Globals.mos_idx[det_no]``.
        - Called by ``Opt02.dist`` only when its hard-coded
          ``misalign_detector`` flag is True (and then with angle 0.0).
        """
        det_idx = det_no - 1
        cosa = math.cos(math.radians(angle))
        sina = math.sin(math.radians(angle))
        img = mosaic.hdu_list[det_idx].data
        cval = np.median(img)
        tr_mat = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
        rotimg = scipy.ndimage.affine_transform(img, tr_mat, cval=cval, order=3)
        mosaic.hdu_list[det_idx].data = rotimg
        return mosaic

    @staticmethod
    def _find_thetas(det_traces):
        """Add the local tilt angle of each fitted trace to the trace dictionaries.

        For each trace the gradient dv/du of the polynomial fit
        (``Globals.polynomial(u_mean, *popt, gradient=True)``) is evaluated at
        the trace's mean along-trace coordinate ``u_mean`` (columns for iso-alpha
        traces, rows for iso-lambda traces) and converted to an angle::

            theta     = atan(gradient) in degrees
            theta_err = sqrt(pcov[1][1]) * (180 / pi) / (1 + gradient ** 2)

        Parameters
        ----------
        det_traces : list of dict
            Trace dictionaries as built by ``Opt02._extract_det_traces``; keys
            'popt', 'pcov' and 'u_mean' are read.

        Returns
        -------
        list of dict
            The same list; each dict gains 'theta' and 'theta_err' (degrees).

        Notes
        -----
        - Open question: ``theta_err`` uses only the variance of the linear
          coefficient ``pcov[1][1]``; the contributions of the quadratic and cubic
          terms of the 4-coefficient fit to the gradient at ``u_mean`` are
          ignored.
        - Inconsistency: the original docstring mentioned a 'u_fiducial'
          location; the code evaluates at ``u_mean``.
        """
        for det_trace in det_traces:
            popt = det_trace['popt']
            pcov = det_trace['pcov']
            u_mean = det_trace['u_mean']
            gradient = Globals.polynomial(u_mean, *popt, gradient=True)
            gradient_err = np.sqrt(pcov[1][1])
            deg_rad = 180. / math.pi
            det_trace['theta'] = deg_rad * math.atan(gradient)
            det_trace['theta_err'] = gradient_err * deg_rad / (1 + gradient ** 2)
        return det_traces
