#!/usr/bin/env python
"""
LMS-OPT-04 analysis stub: deep imaging of ghosts (lmsaiv program).

Registered as 'lms_opt_04_ghost' -> ``Opt04.ghost`` in
``lmsaiv_opt_tests.OptTests`` ('Deep imaging of any ghosts detected in
LMS-OPT-03').  The intended analysis is a target-minus-reference
difference image: read the mosaics of two test steps, subtract the
reference from the target with ``Mosaic.subtract`` and display the result
with ``Plot.mosaic``.  No ghost detection or photometry is implemented and
the code does not run as committed.

**Input data (as written)**: FITS mosaics in ``Filer.test_data_folder``
tagged ['lms_opt_03', 'nom'] plus a step tag ('Step1' reference, 'Step2'
target).

**Output**: one blocking ``Plot.mosaic`` figure ('tgt - ref mosaic').
Nothing is stored in ``as_built``.

Notes
-----
- Bug: ``inc_tags + ref_step`` adds a list and a str -> TypeError before
  any file is read (should be ``inc_tags + [ref_step]``).
- Bug: ``ghost`` returns None, so ``OptTests.run`` pickles None as the
  'as built' object.
- Inconsistency: ``test_name`` is 'lms_opt_03', not 'lms_opt_04'; this
  may be intentional (ghosts found in LMS-OPT-03 data) but the registry
  key 'lms_opt_04_ghost' makes ``OptTests.run`` list 'lms_opt_04' files.
- Inconsistency: step tags 'Step1' / 'Step2' are capitalised while
  ``lms_obs_map`` generates lower-case 'step1' tags, and 'step1' also
  matches 'step10'.. as a sub-string.
- Clean-up: ``filer.set_configuration('distortion', ...)`` and
  ``flag_text`` are unused leftovers from ``lmsaiv_opt02``.  The module
  is nearly identical to ``lmsaiv_opt09``.

@author: Alistair Glasse
"""
from lms_globals import Globals
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt04:
    """LMS-OPT-04 ghost imaging analysis (stub, non-functional)."""


    def __init__(self):
        """Do nothing; all methods are static."""
        return

    @staticmethod
    def ghost(title, as_built, **kwargs):
        """Intended LMS-OPT-04 difference-image analysis (does not run).

        Parameters
        ----------
        title : str
            Registry title; unused.
        as_built : dict
            'As built' dictionary; not modified.
        **kwargs
            Ignored.

        Returns
        -------
        None
            Bug: ``OptTests.run`` expects the updated ``as_built`` dict.

        Notes
        -----
        - Bug: ``inc_tags + ref_step`` raises TypeError (list + str).
        - See the module notes.
        """
        test_name = 'lms_opt_03'
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
