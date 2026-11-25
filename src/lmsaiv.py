"""
Entry-point script of the lmsaiv program: run one LMS AIT performance-test analysis.

``lmsaiv`` is the fourth of the four lmsiq programs (lmsiq, lmsdist,
lmssim, lmsaiv).  This script selects the test data folder and the
analysis project, sets the debug level, loads (or creates) the 'as built'
object, looks the analysis up with
``lmsaiv_opt_tests.OptTests.get_test``, runs it and pickles the updated
'as built' object.  As committed it is wired for the ToySim data set:

- ``test_data_folder`` = 'test_toysim' -> ``Filer.test_data_folder`` =
  '../data/test_toysim' (relative to ``src/``; not in the repository).
  The comment lists 'test_scope_sim' and 'test_aiv' as the alternatives.
- ``cap_name`` = 'lms_opt_04_ghost' -> ``Opt04.ghost``, which is currently
  a stub that prints '!! Analysis not yet available !!'.  The full list of
  analysis projects is the ``OptTests.optical_test`` registry
  (LMS-RAD-01, LMS-RAD-04, LMS-OPT-01 to -04 and LMS-OPT-09).
- ``debug_level`` = 'low' (one of 'off', 'low', 'medium', 'high';
  controls plotting in the analyses).
- ``clear_as_built`` = False; set True to start from an empty 'as built'
  dictionary even if the pickle exists.

**Input data**: FITS mosaics in '../data/test_toysim' written by
``lmssim_toy``, and the optional 'as built' pickle
'../output/asbuilt/asbuilt_toysim.pkl' (``Globals.as_built_file``).

**Output**: the updated 'as built' pickle, printed statistics and
blocking matplotlib figures.

Notes
-----
- The executable statements are wrapped in ``main()`` with an
  ``if __name__ == '__main__'`` guard so the module can be imported (e.g.
  by Sphinx) without running the analysis; the statements and their order
  are unchanged from the upstream script.
- Open question: ``do_plot=False`` is passed to ``OptTests.get_test``,
  which ignores it, and the analysis method is then called as
  ``method(title, as_built)`` without keyword arguments; plotting is
  governed by ``Globals.debug_level`` only.
- Bug: ``Opt03.psf`` has the signature ``(test_name, as_built, **kwargs)``,
  so the ``method(title, as_built)`` call passes the registry title as
  ``test_name``.  Harmless while the method is a stub.
- Bug: ``Rad04.pix_flat`` returns None, so selecting 'lms_rad_04_pix_flat'
  pickles None as the 'as built' object and the next analysis that reads
  it fails.
- Clean-up: ``cap_name``, ``test_data_folder`` and ``debug_level`` are
  hard-coded; a command-line interface would let the same script drive
  every analysis and data folder.
"""
from lmsaiv_opt_tests import OptTests
from lms_globals import Globals
from lms_filer import Filer


def main():
    """
    Run the analysis project ``cap_name`` on ``test_data_folder`` and update the 'as built' pickle.

    Steps: instantiate ``Globals`` and ``Filer`` (for their initialisation
    side effects); set ``Filer.test_data_folder`` and
    ``Globals.debug_level``; print the analysis registry
    (``str(OptTests())``); read the 'as built' dictionary from
    ``Globals.as_built_file`` (an empty dict if the file is missing or
    ``clear_as_built`` is True); fetch ``(method, title)`` with
    ``OptTests.get_test(cap_name)``; call ``method(title, as_built)``; and
    write the returned object back with ``Filer.write_pickle``.

    Returns
    -------
    None
    """
    _ = Globals()
    _ = Filer()
    cap_name = 'lms_opt_04_ghost'            # Name of analysis project (see lmsaiv_opt_tests.py for a full list)
    test_data_folder = 'test_toysim'        # Location of test data (e.g. 'test_toysim', 'test_scope_sim' or 'test_aiv'
    clear_as_built = False

    Filer.set_test_data_folder(test_data_folder)

    debug_level = 'low'                    # Debug information level 'off', 'low', 'medium' or 'high'
    Globals.set_debug_level(debug_level)
    opt_tests = OptTests()
    print(opt_tests)

    print("Analysing data in folder data/{:s} for {:s}.".format(test_data_folder, cap_name))
    try:                                                        # Import or create new 'as built' object.
        as_built = Filer.read_pickle(Globals.as_built_file)
    except FileNotFoundError:
        clear_as_built = True
    if clear_as_built:
        as_built = {}

    method, title = opt_tests.get_test(cap_name, do_plot=False)
    as_built = method(title, as_built)
    Filer.write_pickle(Globals.as_built_file, as_built)

    print()
    print("Finished analysis of {:s}".format(cap_name))


if __name__ == '__main__':
    main()
