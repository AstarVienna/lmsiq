#!/usr/bin/env python
"""Entry point of the ``lmssim`` program: simulate LMS performance-test frames.

Running this file (``python lmssim.py`` from ``src/``) reads the observation
list of one AIT test from the Polarion export
``../config/METIS_Performance_Test_Sequence_LMS.csv`` (through
``lms_obs_map.ObsMap.get_configuration``) and hands every observation to one
of two simulators:

* ``lmssim_toy.Toy`` (``use_scope_sim = False``, the default): the bespoke
  "ToySim" geometric forward model built on the ``lmsdist`` distortion
  transforms and the Zemax PSFs; writes ``../data/test_toysim/{cfg_id}_000.fits``.
* ``lmssim_scope.Scope`` (``use_scope_sim = True``): ScopeSim with the METIS
  ``wcu_lms`` / ``wcu_lms_extended`` modes; writes
  ``../data/test_scopesim/{cfg_id}_o###.fits``.

The test name (``'lms_opt_01'``), the simulator switch and the debug level
are hard-coded inside :func:`run`; there is no command-line interface.
Optionally (``redownload_metis = True``) the ScopeSim instrument packages
METIS, ELT and Armazones are downloaded first.

No FITS headers are read or written directly here; see the two simulator
modules for their header contracts.

Notes
-----
* Clean-up: the module-level ``run()`` call is guarded by
  ``if __name__ == '__main__':`` so that the module can be imported (e.g. by
  Sphinx) without starting a simulation; behaviour when executed as a script
  is unchanged.
* Clean-up: ``scopesim`` is imported at module level even when ToySim is
  selected, so the ``lmssim`` program cannot run without a ScopeSim install.
* Inconsistency: ``Scope.run`` expects configuration keys that
  ``ObsMap.get_configuration`` does not produce (see ``lmssim_scope``), so
  only the ToySim path is currently functional.
"""
from lms_globals import Globals
from lms_obs_map import ObsMap
from lmssim_scope import Scope
from lmssim_toy import Toy
import scopesim as sim


def run():
    """Run the LMS performance-test simulation for one hard-coded test.

    Simulate the observations of ``test_name`` (``'lms_opt_01'``) with either
    ScopeSim or Alistair's bespoke 'ToySim' simulator.  ToySim uses the LMS
    entrance-focal-plane to detector transforms calculated by ``lmsdist.py``,
    the Zemax LMS PSFs (currently including the significant optical design
    aberrations, to be replaced by real wavefront-error maps) and models of the
    calibration sources available during AIT in Leiden (WCU black body and
    lasers, sea-level sky).

    Returns
    -------
    None
        Returns early (after ``ObsMap`` prints a message) when the test name is
        not found in the CSV.

    Notes
    -----
    Side effects: sets ``Globals.debug_level`` ('medium') and ``Globals.sim_id``;
    instantiates ``Scope`` (which calls ``Filer.set_test_data_folder('scopesim')``)
    and ``Toy`` regardless of which simulator is used.

    * Clean-up: ``_ = Globals()`` creates an unused instance; ``Scope`` and
      ``Toy`` are both constructed although only one is run.
    * Open question: the test-name match in ``ObsMap.get_configuration`` is a
      substring test on the first CSV column, so ``'lms_opt_01'`` would also
      match a hypothetical ``'lms_opt_010'``.
    """
    _ = Globals()
    scope = Scope()
    toy = Toy()
    obs_map = ObsMap()

    redownload_metis = False
    if redownload_metis:
        sim.download_packages(["METIS", "ELT", "Armazones"], release="latest")
    # ============================== Set simulator parameters here ====================================
    test_name = 'lms_opt_01'        # Name of test to simulate, e.g 'lms_opt_01'
    use_scope_sim = False

    base_debug_level = 'medium'
    Globals.set_debug_level(base_debug_level)

    Globals.sim_id = Globals.scopesim_id if use_scope_sim else Globals.toysim_id
    print('Simulating measurements for test {:s} using {:s}'.format(test_name, Globals.sim_id))

    sim_configs = obs_map.get_configuration(test_name)
    if sim_configs is None:
        return
    if use_scope_sim:
        scope.run(sim_configs)
    else:
        toy.run(sim_configs)
    return


if __name__ == '__main__':
    run()
