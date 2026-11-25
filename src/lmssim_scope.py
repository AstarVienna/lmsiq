#!/usr/bin/env python
"""ScopeSim back end of the ``lmssim`` program.

:class:`Scope` drives ScopeSim's METIS model in the ``wcu_lms`` (nominal) or
``wcu_lms_extended`` mode to produce simulated LMS detector frames for the
AIT performance tests, one FITS file per observation:
``../data/test_scopesim/{obs_name}_o{obs_idx:03d}.fits`` (the folder is not
created; ScopeSim's ``readout()`` HDUList is written with ``overwrite=True,
checksum=True``).

Requires the ScopeSim instrument packages METIS, ELT and Armazones (set
``reinstall_inst_pkgs = True`` in :meth:`Scope.__init__` to download them
into ``./inst_pkgs``).

Header keywords written (primary HDU, added to ScopeSim's own header):
``ACHG WCU X``, ``ACHG WCU Y`` (WCU mask shift) and ``ACHG LASER WAVE``
(observation wavelength, um).

Notes
-----
* Bug: :meth:`Scope.run` reads ``sim_config`` keys ``'opticon'``, ``'dit'``,
  ``'ndit'``, ``'wave_cen'``, ``'wcu_aper'``, ``'wcu_mask'``, ``'wcu_angle'``,
  ``'wcu_x'``, ``'wcu_y'``, ``'bgd_src'`` and ``'nobs'`` that
  ``lms_obs_map.ObsMap.get_configuration`` never produces (it yields the
  lower-cased CSV column names such as ``'lms_msa'``, ``'lms_dit'``,
  ``'lms_ndit'``, ``'wcu_fp2_1'``, ``'cfo_fp2'``, ``'lms_pp1'``); the first
  lookup raises ``KeyError``.  Only ``'lms_pp1'`` matches.  This back end is
  a stale code path; ``lmssim_toy`` is the working one.
* Inconsistency: ``lmsaiv_opt02`` reads ``HIERARCH AIT OPTICON`` /
  ``PRI_ANG`` / ``ECH_ANG``, ``ACHG REF_ECH_ORD``, ``ACHG CFO CHOP X`` / ``Y``
  and ``ACHG LASER WAVE`` from the primary header; frames written here carry
  only ``ACHG WCU X`` / ``Y`` and ``ACHG LASER WAVE``, so ScopeSim output
  cannot be analysed by ``lmsaiv_opt02``.
* Inconsistency: ``Filer.set_test_data_folder('scopesim')`` points the
  analysis at ``../data/scopesim`` while the frames are written to
  ``../data/test_scopesim``.
"""
import scopesim as sim
import astropy.units as u
from lms_filer import Filer


class Scope:
    """ScopeSim driver for simulated LMS WCU observations.

    Construction only sets ``Filer.test_data_folder``; all work happens in
    :meth:`run`.
    """

    def __init__(self):
        """Set the test-data folder and optionally download the ScopeSim packages.

        Notes
        -----
        Calls ``Filer.set_test_data_folder('scopesim')``.  With the hard-coded
        ``reinstall_inst_pkgs`` switched to True, the METIS, ELT and Armazones
        packages are fetched with ``scopesim.download_packages``.
        """
        reinstall_inst_pkgs = False
        if reinstall_inst_pkgs:
            sim.download_packages(["METIS", "ELT", "Armazones"], release="latest")
        Filer.set_test_data_folder('scopesim')
        return

    def run(self, sim_configs):
        """Simulate every observation of a test with ScopeSim.

        Parameters
        ----------
        sim_configs : dict
            ``{obs_name: sim_config}`` as returned by
            ``lms_obs_map.ObsMap.get_configuration``; ``sim_config`` maps lower-cased
            CSV column names to lower-cased values.

        Returns
        -------
        None

        Notes
        -----
        For each observation: build ``UserCommands(use_instrument='METIS',
        set_modes=[mode])`` with ``mode`` = ``'wcu_lms'`` (nominal) or
        ``'wcu_lms_extended'`` (extended); set ``!OBS.dit``, ``!OBS.ndit``,
        ``!OBS.wavelen`` (``wave_cen`` / 1000, i.e. the CSV value is taken to be in
        nm) and ``!OBS.pupil_transmission`` (0 when ``lms_pp1`` is ``'closed'``,
        else 1); configure the ``wcu_source`` effect (``set_lamp('laser')`` for
        ``bgd_src == 'wcu_lt'``; ``set_lamp('bb')`` plus ``set_temperature(bb_temp=
        1000 K, is_temp=320 K, wcu_temp=295 K)`` for ``'wcu_bb'``;
        ``set_bb_aperture``; ``set_fpmask(maskname, angle, shift=(wcu_x, wcu_y))``);
        then ``nobs`` times ``observe()`` and ``readout(dit, ndit)[0]``, add
        ``ACHG WCU X``, ``ACHG WCU Y`` and ``ACHG LASER WAVE`` to the primary header
        and write ``../data/test_scopesim/{obs_name}_o{idx:03d}.fits``.

        * Bug: the configuration keys used here do not exist in the ``ObsMap``
          output (see the module docstring) -> ``KeyError`` on the first
          observation.
        * Open question: ``ACHG LASER WAVE`` is set to the observation wavelength
          for every source, including black-body observations with no laser on.
        * Clean-up: ``wcu`` and ``metis['wcu_source']`` are the same effect;
          ``print(metis)``, ``metis.effects.pprint_all()`` and ``print(wcu)`` dump
          the whole optical train per observation; the output folder is not
          created if missing.
        """
        for obs_name in sim_configs:
            sim_config = sim_configs[obs_name]
            # print(sim_config)
            opticon = sim_config['opticon']
            mode = {'nominal': 'wcu_lms', 'extended': 'wcu_lms_extended'}[opticon]

            cmds = sim.UserCommands(use_instrument="METIS", set_modes=[mode])
            cmds['!OBS.dit'] = float(sim_config['dit'])     # Integration time per ramp (> 1.3 seconds)
            cmds['!OBS.ndit'] = int(sim_config['ndit'])     # No. of ramps
            wavelen = float(sim_config['wave_cen']) / 1000.
            cmds['!OBS.wavelen'] = wavelen                  # Observation wavelength (microns)
            pup_trans = 0. if sim_config['lms_pp1'] == 'closed' else 1.
            cmds["!OBS.pupil_transmission"] = pup_trans

            metis = sim.OpticalTrain(cmds)

            wcu = metis['wcu_source']
            wcu_aper = float(sim_config['wcu_aper'])
            wcu_mask = sim_config['wcu_mask']
            wcu_angle = float(sim_config['wcu_angle'])
            wcu_x = float(sim_config['wcu_x'])
            wcu_y = float(sim_config['wcu_y'])
            wcu_shift = wcu_x, wcu_y
            if sim_config['bgd_src'] == 'wcu_lt':
                metis["wcu_source"].set_lamp("laser")
            if sim_config['bgd_src'] == 'wcu_bb':
                metis["wcu_source"].set_lamp("bb")
                metis["wcu_source"].set_temperature(bb_temp=1000 * u.K, is_temp=320 * u.K, wcu_temp=295 * u.K)
            wcu.set_bb_aperture(value=wcu_aper)
            wcu.set_fpmask(maskname=wcu_mask, angle=wcu_angle, shift=wcu_shift)
            print(metis)
            metis.effects.pprint_all()
            print(wcu)
            n_obs = int(sim_config['nobs'])
            for obs_idx in range(0, n_obs):
                metis.observe()
                result = metis.readout(dit=cmds['!OBS.dit'], ndit=cmds['!OBS.ndit'])[0]
                obs_tag = "_o{:03d}.fits".format(obs_idx)
                # Add test configuration information explicitly to primary header.
                pri_hdr = result[0].header
                pri_hdr['ACHG WCU X'] = wcu_x
                pri_hdr['ACHG WCU Y'] = wcu_y
                pri_hdr['ACHG LASER WAVE'] = wavelen
                out_folder = '../data/test_scopesim'
                out_file = obs_name + obs_tag
                out_path = out_folder + '/' + out_file
                result.writeto(out_path, overwrite=True, checksum=True)

        print('Done simulating')
        return
