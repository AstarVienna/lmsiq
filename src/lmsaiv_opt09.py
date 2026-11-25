#!/usr/bin/env python
"""
LMS-OPT-09 analysis stub: sky / background difference image (lmsaiv program).

``Opt09.sky`` is a copy of ``lmsaiv_opt04.Opt04.ghost`` with
``test_name`` = 'lms_opt_09'.  It is registered in
``lmsaiv_opt_tests.OptTests.optical_test`` as 'lms_opt_09_sky' with the
title 'Leiden sky', but unlike the other stubs it has no early return,
so selecting it runs the draft code.  The intended analysis is a
target-minus-reference difference of two test steps shown with
``Plot.mosaic``.

**Input data (as written)**: FITS mosaics in ``Filer.test_data_folder``
tagged ['lms_opt_09', 'nom'] plus 'Step1' (reference) / 'Step2' (target).

**Output**: one blocking ``Plot.mosaic`` figure.  Nothing stored in
``as_built``.

Notes
-----
- Bug: ``inc_tags + ref_step`` adds a list and a str -> TypeError.
- Bug: ``sky`` returns None instead of ``as_built`` (were it to get that
  far, ``lmsaiv.main`` would pickle None as the 'as built' object).
- Inconsistency: capitalised 'Step1' / 'Step2' tags versus the lower-case
  'stepN' naming of ``lms_obs_map``.
- Open question: the registry title is 'Leiden sky', but the draft
  computes a step difference image; what the LMS-OPT-09 analysis should
  measure is not documented.
- Clean-up: duplicate of ``lmsaiv_opt04``; unused
  ``filer.set_configuration`` and ``flag_text``.

@author: Alistair Glasse
"""
from lms_globals import Globals
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt09:
    """LMS-OPT-09 difference-image analysis (stub, non-functional)."""


    def __init__(self):
        """Do nothing; all methods are static."""
        return

    @staticmethod
    def sky(title, as_built, **kwargs):
        """Intended LMS-OPT-09 difference-image analysis (does not run).

        Parameters
        ----------
        title : str
            Title; unused.
        as_built : dict
            'As built' dictionary; not modified.
        **kwargs
            Ignored.

        Returns
        -------
        None
            Bug: an ``OptTests``-style runner expects the updated ``as_built``
            dict.

        Notes
        -----
        - Bug: ``inc_tags + ref_step`` raises TypeError (list + str).
        - See the module notes.
        """
        test_name = 'lms_opt_09'
        opticon = Globals.nominal

        ref_step = 'Step1'
        tgt_step = 'Step2'

        inc_tags = [test_name, opticon[0:3]]         # Tokens to identify image files.
        filer = Filer()
        filer.set_configuration('distortion', opticon, is_ait_data=True)

        flag_text = 'Analysing'
        # Load offset/background images to subtract from ghost images
        ref_mosaic = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags + ref_step)[0]
        tgt_mosaic = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags + tgt_step)[0]
        sig_mosaic = tgt_mosaic.subtract(ref_mosaic)
        Plot.mosaic(sig_mosaic, title='tgt - ref mosaic', cmap='hot')
        return
