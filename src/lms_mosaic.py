#!/usr/bin/env python
"""
In-memory container for a four-detector LMS exposure (``Mosaic``).

Shared by the ``lmssim`` and ``lmsaiv`` programs: ``lms_filer.Filer.read_mosaic``
builds a ``Mosaic`` from a FITS file with a primary HDU plus one image
extension per detector, and the LMS-OPT-01 / LMS-OPT-02 analyses
(``lmsaiv_opt01``, ``lmsaiv_opt02``, ``lmsaiv_opt_tools``, ``lmsaiv_plot``)
and the LMS-RAD-04 pixel flat (``lmsaiv_rad04``) operate on these objects.

**Input data**: not read directly.  ``Filer.read_mosaic`` supplies ``name``
(the FITS file name), ``primary_hdr`` (the primary header, e.g. carrying
'HIERARCH AIT OPTICON', 'HIERARCH ESO INS WLEN CEN', 'HIERARCH ESO DET
DIT') and ``hdu_list``, a list of four ``astropy.io.fits.ImageHDU`` whose
headers carry per-detector keywords ('ID', 'CRVAL1D', 'CRVAL2D', 'X_CEN',
'HIERARCH ESO DET3 CHIP GAIN', 'HIERARCH AIT PIXEL_PITCH' in ToySim
output).  The list order is set by ``Filer.read_mosaic``: index
``det_no - 1`` for ToySim files, ``Globals.mos_idx[det_no]`` (= {2: 0,
1: 1, 3: 2, 4: 3}) for files with ORIGIN = 'SCOPESIM'.

**Output**: ``copy`` returns a new ``Mosaic``; ``add``, ``subtract`` and
``scale`` modify the receiver in place and return it.  Nothing is written to disk
here (``Filer.write_mosaic`` still expects the legacy ``(name, header,
hdus)`` tuple, not a ``Mosaic``).

Notes
-----
- Inconsistency: ``lmsaiv_opt_tools.OptTools.copy_mosaic``,
  ``Opt01.flood_stats``, ``Opt01.print_inter_slice``, ``Opt03.psf`` and
  ``Filer.write_mosaic`` still use the pre-``Mosaic`` tuple layout
  ``(file_name, primary_hdr, hdus)`` and cannot accept a ``Mosaic``.
- Open question: the HDU ordering differs between ToySim and ScopeSim
  files (see above) while ``Opt02._extract_det_traces`` always indexes
  the slice map with ``Globals.mos_idx``; for ToySim data this appears to
  pair detector 1 with the slice map of detector 2 and vice versa.

@author: Alistair Glasse
"""
import copy
from astropy.io.fits import ImageHDU


class Mosaic:
    """Four-detector LMS exposure held in memory.

    Parameters
    ----------
    name : str
        Identifier, normally the FITS file name; used in plot titles and
        print-outs.
    primary_hdr : astropy.io.fits.Header
        Primary header of the exposure.
    hdu_list : list of astropy.io.fits.ImageHDU
        One image HDU per detector (four expected; ``lmsaiv_opt_tools`` loops
        over ``range(0, 4)``).

    Attributes
    ----------
    name, primary_hdr, hdu_list
        As above.  They are also declared as class attributes initialised to
        None, which the instance attributes shadow.

    Notes
    -----
    - Clean-up: the class-level ``name, primary_hdr, hdu_list = None, None,
      None`` declaration is redundant with the instance assignments.
    """

    name, primary_hdr, hdu_list = None, None, None

    def __init__(self, name, primary_hdr, hdu_list):
        """Store the name, primary header and HDU list without copying.

        Parameters
        ----------
        name : str
            Mosaic identifier (FITS file name).
        primary_hdr : astropy.io.fits.Header
            Primary header.
        hdu_list : list of astropy.io.fits.ImageHDU
            Detector image HDUs, in the order chosen by the caller.
        """
        self.name = name
        self.primary_hdr = primary_hdr
        self.hdu_list = hdu_list
        return

    def copy(self, clear_data=False, copy_name=''):
        """Return a deep copy of the mosaic, optionally with zeroed image data.

        Parameters
        ----------
        clear_data : bool, optional
            If True the copied data arrays are multiplied by 0. (so the copy keeps
            the shape, dtype and header of each detector but holds zeros).  Default
            False.
        copy_name : str, optional
            Name of the copy.  If '' (default) the original name is kept.

        Returns
        -------
        Mosaic
            New object with deep-copied primary header, per-HDU headers and data,
            each wrapped in a fresh ``ImageHDU``.

        Notes
        -----
        - ``clear_data`` uses ``mos_data *= 0.``, so NaN pixels stay NaN and the
          operation raises a numpy casting TypeError for integer image arrays;
          ToySim frames are floats so this is not hit in the current chain.
        - Used by ``Opt01.calculate_fov`` (coadd buffer), ``Opt01._make_slice_map``
          and ``Opt01._find_rrf`` (zeroed template), and
          ``OptTools.median_subtract``.
        """
        mos_hdus = []
        for hdu in self.hdu_list:
            mos_hdr = copy.deepcopy(hdu.header)
            mos_data = copy.deepcopy(hdu.data)
            if clear_data:
                mos_data *= 0.
            mos_hdu = ImageHDU(data=mos_data, header=mos_hdr)
            mos_hdus.append(mos_hdu)
        mos_name = self.name if copy_name == '' else copy_name
        mos_pri_hdr = copy.deepcopy(self.primary_hdr)
        return Mosaic(mos_name, mos_pri_hdr, mos_hdus)

    def subtract(self, mos):
        """Subtract another mosaic's images from this one, in place.

        Parameters
        ----------
        mos : Mosaic
            Mosaic whose ``hdu_list`` data are subtracted HDU by HDU (paired by
            list position; ``zip`` truncates to the shorter list).

        Returns
        -------
        Mosaic
            ``self``, after modification.

        Notes
        -----
        - The receiver is modified: ``Opt02.dist`` writes
          ``tra_mosaic = sig_mosaic.subtract(bgd_mosaic)`` and afterwards plots
          ``sig_mosaic`` as 'Signal', which therefore already shows the
          background-subtracted image (see ``lmsaiv_opt02``).
        - Headers are not compared or updated; there is no check that ``mos``
          refers to the same detectors.
        """
        for self_hdu, hdu in zip(self.hdu_list, mos.hdu_list):
            self_hdu.data = self_hdu.data - hdu.data
        return self

    def add(self, mos):
        """Add another mosaic's images to this one, in place.

        Parameters
        ----------
        mos : Mosaic
            Mosaic whose ``hdu_list`` data are added HDU by HDU (paired by list
            position).

        Returns
        -------
        Mosaic
            ``self``, after modification.

        Notes
        -----
        - Used by ``Opt01.calculate_fov`` to coadd the flood-illuminated flats of
          one LMS configuration group, and by ``Rad04.pix_flat`` to sum the
          per-frame pixel flats.  No header bookkeeping (e.g. total DIT) is
          done.
        """
        for self_hdu, hdu in zip(self.hdu_list, mos.hdu_list):
            self_hdu.data = self_hdu.data + hdu.data
        return self

    def scale(self, factor):
        """Multiply every detector image by a constant, in place.

        Parameters
        ----------
        factor : float
            Multiplier applied to each ``hdu.data``.

        Returns
        -------
        Mosaic
            ``self``, after modification.

        Notes
        -----
        - Used by ``Rad04.pix_flat`` to turn the sum of pixel flats into a mean.
          No header bookkeeping is done.
        """
        for self_hdu in self.hdu_list:
            self_hdu.data = factor * self_hdu.data
        return self

