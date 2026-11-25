"""
Entry-point script of the lmsaiv program: run one LMS AIT performance-test analysis.

``lmsaiv`` is the fourth of the four lmsiq programs (lmsiq, lmsdist,
lmssim, lmsaiv).  This script selects the test data folder and the
analysis project, sets the debug level and calls
``lmsaiv_opt_tests.OptTests.run``.  As committed it is wired for the
ToySim data set only:

- ``test_data_folder`` = 'test_toysim' -> ``Filer.test_data_folder`` =
  '../data/test_toysim' (relative to ``src/``; not in the repository);
- ``cap_name`` = 'lms_opt_01_fov' -> ``Opt01.fov`` (LMS-OPT-01 field of
  view, slice map and relative response);
- ``debug_level`` = 'low' (controls plotting in the analyses).

**Input data**: FITS mosaics in '../data/test_toysim' written by
``lmssim_toy``, and the optional 'as built' pickle
'../output/asbuilt/asbuilt_toysim.pkl' (``Globals.as_built_file``).

**Output**: the updated 'as built' pickle (written by ``OptTests.run``),
printed statistics and blocking matplotlib figures.

Notes
-----
- The executable statements are wrapped in ``main()`` with an
  ``if __name__ == '__main__'`` guard so the module can be imported (e.g.
  by Sphinx) without running the analysis; the statements and their order
  are unchanged.
- Open question: ``do_plot=False`` is passed to ``OptTests.run`` and
  forwarded to ``Opt01.fov``, which ignores it; plots are governed by
  ``Globals.debug_level`` instead.
- Clean-up: ``cap_name``, ``test_data_folder`` and ``debug_level`` are
  hard-coded; a command-line interface would let the same script drive
  LMS-OPT-02 and other data folders.
"""
from lmsaiv_opt_tests import OptTests
from lms_globals import Globals
from lms_filer import Filer


def main():
    """
    Configure ``Globals`` / ``Filer`` and run the 'lms_opt_01_fov' analysis on the ToySim data.

    Instantiates ``Globals`` and ``Filer`` (for their initialisation side
    effects), sets ``Filer.test_data_folder`` to '../data/test_toysim' and
    ``Globals.debug_level`` to 'low', prints the registry of analyses
    (``str(OptTests())``) and calls ``OptTests.run('lms_opt_01_fov',
    do_plot=False)``.

    Returns
    -------
    None
    """
    _ = Globals()
    _ = Filer()
    cap_name = 'lms_opt_01_fov'             # Name of analysis project
    test_data_folder = 'test_toysim'
    Filer.set_test_data_folder(test_data_folder)

    debug_level = 'low'
    Globals.set_debug_level(debug_level)
    opt_tests = OptTests()
    print(opt_tests)

    print("Analysing data in folder data/{:s} for {:s}.".format(test_data_folder, cap_name))
    opt_tests.run(cap_name, do_plot=False)
    print()
    print("Finished analysis of {:s}".format(cap_name))


if __name__ == '__main__':
    main()
