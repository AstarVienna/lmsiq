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
from lmsaiv_opt01 import Opt01
from lmsaiv_opt02 import Opt02
from lmsaiv_opt03 import Opt03
from lmsaiv_opt04 import Opt04


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

        The dictionary literal is assigned to ``OptTests.optical_test`` (a class
        attribute), so every instance shares the same registry and creating a
        second instance simply rebuilds it.

        Notes
        -----
        - Clean-up: the registry could be a class-level constant; building it in
          ``__init__`` forces callers (``lmsaiv.py``) to instantiate the class
          before calling the static ``run`` method.
        """
        # Test dictionary, note that the key comprises 'data_source_' + 'analysis' (eg 'lms_opt_01_' + 'fov'.  This
        # supports the fact that an analysis project (Jupyter notebook) may use data from several tests.
        optical_test = {'lms_opt_01_fov': (Opt01.fov, 'Field of view and RSRF'),
                        'lms_opt_02_dist': (Opt02.dist, 'Distortion transforms and enslitted profiles'),
                        'lms_opt_03_psf': (Opt03.psf, 'Monochomatic Point and line spread function'),
                        'lms_opt_04_ghost': (Opt04.ghost, 'Deep imaging of any ghosts detected in LMS-OPT-03'),
                        # 'lms_opt_08': (OptTests.lms_opt_08, 'Out-of-field straylight'),
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
    def run(cap_name, **kwargs):
        """Run one registered analysis project and persist the 'as built' object.

        Parameters
        ----------
        cap_name : str
            Registry key, e.g. 'lms_opt_01_fov'.  Its first ten characters
            (``cap_name[0:10]``, e.g. 'lms_opt_01') are used as the file-name tag
            that selects the test data files listed for information.
        **kwargs
            Forwarded unchanged to the registered analysis method (e.g.
            ``overwrite_as_built_traces`` for ``Opt02.dist``; ``do_plot`` is
            accepted but ignored by ``Opt01.fov``).

        Returns
        -------
        None
            The updated 'as built' dictionary is written to
            ``Globals.as_built_file`` + '.pkl' rather than returned.

        Raises
        ------
        SystemExit
            Via ``quit()`` if ``cap_name`` is not in the registry.
        TypeError
            If ``Filer.test_data_folder`` has not been set (``listdir(None)``).

        Notes
        -----
        - The 'as built' pickle is loaded from ``Globals.as_built_file``; a
          FileNotFoundError is caught and an empty dict is used instead, so the
          first analysis in the chain (LMS-OPT-01) creates the object and later
          analyses (LMS-OPT-02 requires ``as_built['slice_map_nominal']``) extend
          it.
        - The file list is only printed; the analysis methods re-read the data
          with their own tags.
        - Bug: the return value of ``method`` is pickled without checking that it
          is a dict; ``Opt04.ghost`` returns None (see module notes).
        - Inconsistency: the ``cap_name[0:10]`` slice assumes the ten-character
          'lms_opt_NN' test-name format.
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

        try:            # Import or create new 'as built' object.
            as_built = Filer.read_pickle(Globals.as_built_file)
        except FileNotFoundError:
            as_built = {}
        method, title = OptTests.optical_test[cap_name]
        as_built = method(title, as_built, **kwargs)
        Filer.write_pickle(Globals.as_built_file, as_built)
        return
