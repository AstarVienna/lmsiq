"""
Registry and runner for the LMS AIT performance-test analyses (lmsaiv program).

This module belongs to the ``lmsaiv`` program (the fourth of the four lmsiq
programs: lmsiq, lmsdist, lmssim, lmsaiv).  It maps an "analysis project"
name (``cap_name``) onto the static analysis method that implements it and a
human readable title, and it owns the only persistent product of the AIV
analysis chain: the ``as_built`` dictionary, pickled at
``Globals.as_built_file`` ('../output/asbuilt/asbuilt_toysim', with '.pkl'
appended by ``Filer.write_pickle``).

**Registered analyses** (key -> method, AIT test):

- 'lms_opt_01_fov' -> ``Opt01.fov``: LMS-OPT-01, field of view, slice map
  and relative spectral response (``lmsaiv_opt01``).
- 'lms_opt_02_dist' -> ``Opt02.dist``: LMS-OPT-02, iso-alpha / iso-lambda
  traces and distortion coordinates (``lmsaiv_opt02``).
- 'lms_opt_03_psf' -> ``Opt03.psf``: LMS-OPT-03, point / line spread
  function (non-functional stub, ``lmsaiv_opt03``).
- 'lms_opt_04_ghost' -> ``Opt04.ghost``: LMS-OPT-04, ghost imaging
  (non-functional stub, ``lmsaiv_opt04``).

An 'lms_opt_08' (out-of-field straylight) entry is present but commented
out, and ``lmsaiv_opt09.Opt09`` is not registered at all.

**Input data**: FITS mosaics are read from ``Filer.test_data_folder`` (set
by the caller via ``Filer.set_test_data_folder``, e.g.
'../data/test_toysim'); the files are selected by sub-string tags derived
from the test name (see the individual Opt modules and ``lms_obs_map`` for
the naming convention).  The 'as built' pickle is read from
``Globals.as_built_file`` if it exists.

**Output**: ``run`` writes the updated ``as_built`` dictionary back to
``Globals.as_built_file`` + '.pkl'.  Everything else (tables, statistics)
is printed to stdout or shown as blocking matplotlib figures by the
individual analyses.

Notes
-----
- Inconsistency: ``OptTests.run`` calls the registered method as
  ``method(title, as_built, **kwargs)`` but ``Opt03.psf`` is declared as
  ``psf(test_name, title, as_built, **kwargs)``, so 'lms_opt_03_psf' fails
  with a TypeError before doing anything.
- Bug: ``Opt04.ghost`` returns None, so running 'lms_opt_04_ghost' pickles
  None as the 'as built' object and breaks every subsequent analysis that
  indexes ``as_built``.
- Open question: ``Filer.write_pickle`` does not create the
  '../output/asbuilt' folder; on a clean checkout the first ``run`` appears
  to fail with FileNotFoundError when writing the pickle.
- Clean-up: plotting is controlled by the global ``Globals.debug_level``;
  the ``do_plot`` keyword passed by ``lmsaiv.py`` is forwarded via
  ``**kwargs`` but ignored by ``Opt01.fov``.

@author: Alistair Glasse
"""
from lms_globals import Globals
from lms_filer import Filer
from lmsaiv_rad01 import Rad01
from lmsaiv_rad04 import Rad04
from lmsaiv_opt01 import Opt01
from lmsaiv_opt02 import Opt02
from lmsaiv_opt03 import Opt03
from lmsaiv_opt04 import Opt04
from lmsaiv_opt09 import Opt09


class OptTests:
    """Registry of LMS AIT optical-test analyses and their runner.

    The registry is held in the class attribute ``optical_test`` and is
    populated by ``__init__`` (so an instance must be created once before
    ``OptTests.run`` is used).  Keys are ``cap_name`` strings of the form
    '<data source test>_<analysis>' (e.g. 'lms_opt_01' + '_fov'); the first
    ten characters name the AIT test whose data files are read.  Values are
    ``(method, title)`` tuples.

    Attributes
    ----------
    optical_test : dict or None
        Mapping ``cap_name -> (method, title)``.  None until ``__init__`` has
        run.
    """

    optical_test = None

    def __init__(self):
        """Populate the class-level ``optical_test`` registry.

        Upstream note: the key comprises 'data_source_' + 'analysis_type' (e.g.
        'lms_opt_01_' + 'fov'), acknowledging that an analysis project (Jupyter
        notebook) may use data from several tests.

        The dictionary literal is assigned to ``OptTests.optical_test`` (a class
        attribute), so every instance shares the same registry and creating a
        second instance simply rebuilds it.

        Notes
        -----
        - Clean-up: the registry could be a class-level constant; building it in
          ``__init__`` forces callers (``lmsaiv.py``) to instantiate the class
          before calling the static ``get_test`` method.
        - Open question: an 'lms_rad_03_inst_bgd' (instrument background) entry
          is present but commented out; there is no ``Rad03`` module yet.
        """
        optical_test = {'lms_rad_01_dark': (Rad01.dark, 'Dark current'),
                        # 'lms_rad_03_inst_bgd': (Rad03.inst_bgd, 'Instrument background')
                        'lms_rad_04_pix_flat': (Rad04.pix_flat, 'Pixel flat field'),
                        'lms_opt_01_fov': (Opt01.fov, 'Field of view and RSRF'),
                        'lms_opt_02_dist': (Opt02.dist, 'Distortion transforms and enslitted profiles'),
                        'lms_opt_03_psf': (Opt03.psf, 'Monochromatic Point and line spread function'),
                        'lms_opt_04_ghost': (Opt04.ghost, 'Deep imaging of any ghosts detected in LMS-OPT-03'),
                        'lms_opt_09_sky': (Opt09.sky, 'Leiden sky')
                        }
        OptTests.optical_test = optical_test
        return

    def __str__(self):
        """Return a table of the registered analysis names and titles.

        Returns
        -------
        str
            One header line followed by one '<cap_name>,<title>' line per
            registry entry.
        """
        text = 'Valid analysis methods, \n'
        for cap_name in OptTests.optical_test:
            test = OptTests.optical_test[cap_name]
            text += "{:<20s},{:<20s} \n".format(cap_name, test[1])
        return text

    @staticmethod
    def get_test(cap_name, **kwargs):
        """Look up a registered analysis project and list its test data files.

        Parameters
        ----------
        cap_name : str
            Registry key, e.g. 'lms_opt_01_fov'.  Its first ten characters
            (``cap_name[0:10]``, e.g. 'lms_opt_01') are used as the file-name tag
            that selects the test data files listed for information.
        **kwargs
            Accepted and ignored (``lmsaiv.main`` passes ``do_plot=False``).

        Returns
        -------
        tuple
            ``(method, title)``: the static analysis method, to be called as
            ``method(title, as_built)``, and its title.  Loading and saving
            the 'as built' object is the caller's job (``lmsaiv.main``).

        Raises
        ------
        SystemExit
            Via ``quit()`` if ``cap_name`` is not in the registry.
        TypeError
            If ``Filer.test_data_folder`` has not been set (``listdir(None)``).

        Notes
        -----
        - The file list is only printed; the analysis methods re-read the data
          with their own tags.
        - Inconsistency: the ``cap_name[0:10]`` slice assumes the ten-character
          'lms_xxx_NN' test-name format.
        """
        if cap_name not in OptTests.optical_test.keys():
            print()
            print("!!! Analysis project {:s} not found !!!".format(cap_name))
            quit()

        test_name = cap_name[0:10]
        file_list = Filer.get_file_list(Filer.test_data_folder, inc_tags=[test_name])
        print("Test data for {:s} found in folder ".format(test_name))
        for file in file_list:
            print("- {:s}".format(file))
        print()

        method, title = OptTests.optical_test[cap_name]
        return method, title
