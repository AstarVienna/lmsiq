#!/usr/bin/env python
"""
Observation map: translate the LMS performance-test sequence into per-observation configurations and file-name tags.

Belongs primarily to the ``lmssim`` program (``lmssim.py`` calls
``ObsMap.get_configuration``; ``lmssim_toy`` and ``lmssim_model`` call
``ObsMap.is_dark``), and defines the file-name convention that the
``lmsaiv`` analyses (``lmsaiv_opt01``, ``lmsaiv_opt02``) rely on when
selecting files by sub-string tags ('flat_bb', 'iso_alpha', 'iso_lambda',
'_dark', 'stepN').

**Input data**: '../config/METIS_Performance_Test_Sequence_LMS.csv',
relative to the working directory (``src/``): a comma-separated export of
the METIS performance-test sequence (Polarion) that is not part of the
repository.  Lines whose first token contains '#' are comments; the first
uncommented line holds the column names.  Column names are normalised to
lower case with '-' -> '_', and these are used: 'lms_msa' (nominal /
extended), 'step_no', 'cfo_fp2', 'cfo_pp1', 'lms_pp1', 'wcu_fp2_1',
'wcu_laser_tune', 'wcu_laser_tune_woff', 'wcu_bb_ap_mask', 'wcu_per_arm',
'cfo_chop_off_x', 'cfo_chop_off_y'.  Columns whose name contains ``img_``
are dropped.  Rows are selected when ``test_name`` occurs in the first
column (lower-cased).

**Output**: ``get_configuration`` returns ``{cfg_id: sim_config}`` where
``sim_config`` holds the lower-cased CSV values plus 'cfg_id', and::

    cfg_id = '{test}_{step}_{nom|ext}_' + tag

    tag = 'dark'
        | [``sky_``] + ('bb_iso_alpha_a{p|n}NNN_b{p|n}NNN_' | ``las_psf_``)   (pinhole in)
        | [``sky_``] + (``flat_bb_`` | ``las_iso_lambda_``)                     (no pinhole)
          [+ 'woff_{p|m}NNNnm_']                                          (tunable laser)

with the trailing '_' removed.  ``lmssim_toy`` appends '_{obs:03d}.fits'
to make the file name.

Notes
-----
- Bug: ``is_sw_laser`` and ``is_lw_laser`` both test 'wcu_laser_tune'
  (copy-paste), so the fixed-wavelength laser columns are never consulted
  and every laser observation is treated as tunable (a ``woff_`` tag is
  always appended).
- Open question: rows with identical ``cfg_id`` (repeat observations)
  overwrite each other in the returned dict; observation numbering is
  left to ``lmssim_toy``.
- Open question: ``_get_chop_tag`` scales by 1000 and pads to three
  digits, so offsets >= 1.0 produce four-digit tags while
  ``get_configuration`` formats them with '{:4s}'.
- Inconsistency: the ``get_configuration`` docstring referred to
  '/config/lms-opt-config.csv'; the code reads
  'METIS_Performance_Test_Sequence_LMS.csv'.
- Clean-up: the CSV file handle is not closed; ``in_csv_file`` and the
  ``Globals`` import are unused.

@author: Alistair Glasse
"""
from lms_globals import Globals


class ObsMap:
    """Static helpers mapping performance-test sequence rows to simulation configurations and file tags."""

    def __init__(self):
        """Do nothing; all methods are static."""
        return

    @staticmethod
    def _get_chop_tag(cxy_str):
        """Encode a chopper offset as a signed three-digit tag.

        Parameters
        ----------
        cxy_str : str
            Offset as text (e.g. '0.5'); appears to be in the units of the
            'cfo_chop_off_x' / 'cfo_chop_off_y' CSV columns, which ``lmssim_toy``
            adds directly to entrance focal plane coordinates in mm.

        Returns
        -------
        str
            'p' or 'n' followed by ``abs(int(1000 * value))`` zero-padded to
            three digits, e.g. 'p500', 'n050'.

        Notes
        -----
        - Open question: values >= 1.0 give more than three digits.
        """
        cxy_int = int(1000. * float(cxy_str))
        cxy_sgn = 'n' if cxy_int < 0 else 'p'
        cxy_tag = "{:s}{:03d}".format(cxy_sgn, abs(cxy_int))
        return cxy_tag

    @staticmethod
    def is_dark(sim_config):
        """Decide whether a sequence row is a dark observation.

        Parameters
        ----------
        sim_config : dict
            Row dictionary from ``get_configuration``; keys 'cfo_fp2', 'cfo_pp1'
            and 'lms_pp1' are inspected.

        Returns
        -------
        bool
            True if 'dark' occurs in 'cfo_fp2' or 'cfo_pp1', or 'closed' in
            'lms_pp1'.
        """
        is_dark = 'dark' in sim_config['cfo_fp2']
        is_dark = True if 'dark' in sim_config['cfo_pp1'] else is_dark
        is_dark = True if 'closed' in sim_config['lms_pp1'] else is_dark
        return is_dark

    @staticmethod
    def get_configuration(test_name):
        """Read the test-sequence CSV and return the configurations of one test.

        Parameters
        ----------
        test_name : str
            Sub-string matched (case-insensitively) against the first column,
            e.g. 'lms_opt_01'.

        Returns
        -------
        dict or None
            ``{cfg_id: sim_config}`` (see the module docstring for the ``cfg_id``
            grammar), or None with a printed message if no row matched.

        Raises
        ------
        FileNotFoundError
            If '../config/METIS_Performance_Test_Sequence_LMS.csv' is absent
            (it is not in the repository).
        KeyError
            If one of the expected columns is missing from the header line.

        Notes
        -----
        - Bug: 'wcu_laser_tune' is used for all three laser flags (see module
          notes).
        - ``opticon[0:3]`` yields 'nom' / 'ext', the tag the ``lmsaiv`` modules
          match (``Globals.nominal[0:3]``).
        - Clean-up: unused ``in_csv_file``; file handle never closed.
        """
        in_csv_file = False
        cfg_path = '../config/METIS_Performance_Test_Sequence_LMS.csv'
        lines = open(cfg_path, 'r').read().splitlines()
        sim_configs, obs_keys, key_fmts = None, None, None
        for line in lines:
            tokens = line.split(',')
            if '#' in tokens[0]:        # Skip commented lines.
                continue
            if sim_configs is None:     # First uncommented line holds the observation keys
                sim_configs = {}
                obs_keys = line.split(',')
                continue
            if test_name not in tokens[0].lower():      # Select requested test
                continue
            sim_config = {}
            for obs_key, token in zip(obs_keys, tokens):
                obs_key = obs_key.replace('-', '_').lower()
                if obs_key == '' or 'img_' in obs_key:
                    continue
                sim_config[obs_key] = token.lower()

            opticon = sim_config['lms_msa']
            step_no = sim_config['step_no']
            is_dark = ObsMap.is_dark(sim_config)
            is_cfo_pnh = 'pnh-1' in sim_config['cfo_fp2']
            is_wcu_pnh = 'lm_pinhole' in sim_config['wcu_fp2_1'] or 'lm_grid' in sim_config['wcu_fp2_1']
            is_pnh = is_wcu_pnh or is_cfo_pnh
            is_tunable_laser = sim_config['wcu_laser_tune'] == 'true'
            is_sw_laser = sim_config['wcu_laser_tune'] == 'true'
            is_lw_laser = sim_config['wcu_laser_tune'] == 'true'
            is_laser = is_tunable_laser or is_sw_laser or is_lw_laser
            is_bb_on = sim_config['wcu_bb_ap_mask'] != 'closed'
            is_sky = sim_config['wcu_per_arm'] != 'out'

            cfg_tag = ''
            if is_dark:
                cfg_tag += 'dark_'
            else:
                if is_sky:
                    cfg_tag += 'sky_'
                if is_pnh:
                    if is_bb_on:
                        cx_off_tag = ObsMap._get_chop_tag(sim_config['cfo_chop_off_x'])
                        cy_off_tag = ObsMap._get_chop_tag(sim_config['cfo_chop_off_y'])
                        cfg_tag += "bb_iso_alpha_a{:4s}_b{:4s}_".format(cx_off_tag, cy_off_tag)
                    if is_laser:
                        cfg_tag += 'las_psf_'
                else:
                    if is_bb_on:
                        cfg_tag += 'flat_bb_'
                    if is_laser:
                        cfg_tag += 'las_iso_lambda_'
                if is_tunable_laser:
                    lt_woff_str = sim_config['wcu_laser_tune_woff']
                    lt_woff_nm = int(1000. * float(lt_woff_str))
                    lt_woff_sgn = 'm' if lt_woff_nm < 0 else 'p'
                    lt_woff_tag = "{:s}{:03d}".format(lt_woff_sgn, abs(lt_woff_nm))
                    cfg_tag += "woff_{:s}nm_".format(lt_woff_tag)

            cfg_id = test_name + '_' + step_no + '_' + opticon[0:3] + '_' + cfg_tag[0:-1]
            sim_config['cfg_id'] = cfg_id
            sim_configs[cfg_id] = sim_config

        if not sim_configs:         # No configurations found
            print("!! Test {:s} not found in csv file {:s}".format(test_name, cfg_path))
            return None
        return sim_configs
