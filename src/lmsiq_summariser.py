"""
Legacy CSV writers and readers for lmsiq summary tables, profiles and phase data.

Purpose
-------
In earlier versions of ``lmsiq`` the per-wavelength results (SRP, EE at reference
apertures, FWHM statistics), the EE / LSF profiles and the pixel-phase centroid tables
were exchanged between processing stages through fixed-width CSV files.  This module
holds those formats.  At HEAD the current flow (:mod:`lmsiq_cuber`) exchanges data by
pickle instead, and none of these methods is called; ``lmsiq.main`` instantiates the
class only for its side effect on ``summary_dict``.

Belongs to the ``lmsiq`` program (legacy).

Input data
----------
The CSV files this module itself wrote, under
``../output/iq/<opticon>/<date_stamp>/<slice_subfolder>/<ipc_tag>/...``.

Outputs
-------
CSV files:

* ``<process_level>_summary_<ipc_tag>.csv`` (:meth:`Summariser.write_summary`);
* ``<slice>_<ipc>_<level>_<type_tag>_wav_<config>.csv`` for the profile, centroid and
  phase-data writers via :meth:`Summariser._get_results_path`.

Notes
-----
* Bug: :meth:`Summariser._get_results_path` references ``Summariser.dataset_results_path``
  and ``Summariser._get_folder``, neither of which exists on the class, so every
  method that builds a path through it (``write_phase_data``, ``read_phase_data``,
  ``write_centroids``, ``read_centroids``, ``write_profiles`` without ``set_path``,
  ``read_profiles``) raises ``AttributeError``.  A working copy of the same function
  exists as :meth:`lms_filer.Filer._get_results_path`.
* Inconsistency: :meth:`Summariser.get_hdr_text` reads the third element of
  ``summary_dict`` entries as ``unit`` but the units are only used when passed back
  in explicitly by :meth:`create_summary_header`.
* Clean-up: the whole module is dead code at HEAD; ``Filer`` is imported but unused.
"""
import os
import numpy as np
from lms_globals import Globals
from lms_ipc import Ipc
from lms_filer import Filer


class Summariser:
    """
    Fixed-width CSV formats for legacy lmsiq summary products.

    Attributes
    ----------
    summary_dict : dict
        ``column key -> (width, decimals, unit)`` for the summary table.  Populated
        with the base columns at class level and extended per axis in ``Globals.axes``
        (``'spectral'``, ``'spatial'``) by the constructor with EE reference / perfect /
        design / mean and FWHM linear / Gaussian MC and error columns.

    Notes
    -----
    * Clean-up: the constructor mutates the class-level dict, so repeated
      instantiation is idempotent but order-dependent.
    """

    summary_dict = {'wave': (10, 6, '(um)'), 'order': (6, 0, '-'),
                    'srp_mc_lin': (12, 0, '-'), 'srp_mc_lin_err': (15, 0, '-'),
                    'srp_mc_gau': (12, 0, '-'), 'srp_mc_gau_err': (15, 0, '-'),
                    }

    def __init__(self):
        """
        Extend ``summary_dict`` with the per-axis EE and FWHM columns.

        For each axis in ``Globals.axes`` the keys ``ee_<ax4>_ref/per/des/mean`` and
        ``fwhm_<ax4>_lin_mc/lin_mc_err/gau_mc/gau_mc_err/lin_per/lin_des`` are added,
        where ``<ax4>`` is the first four letters of the axis name.
        """
        axes = Globals.axes
        for axis in axes:
            axis_tag = '_' + axis[0:4]
            ee_tag = 'ee' + axis_tag + '_'
            Summariser.summary_dict[ee_tag + 'ref'] = 12, 1, '-'
            Summariser.summary_dict[ee_tag + 'per'] = 12, 5, '-'
            Summariser.summary_dict[ee_tag + 'des'] = 12, 5, '-'
            Summariser.summary_dict[ee_tag + 'mean'] = 13, 5, '-'
            fwhm_mc_tag = 'fwhm' + axis_tag + '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_mc'] = 17, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_mc_err'] = 21, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'gau_mc'] = 17, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'gau_mc_err'] = 21, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_per'] = 21, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_des'] = 21, 5, '_'
        return

    @staticmethod
    def get_value_text(key_value_pairs):
        """
        Format ``(key, value)`` pairs as one fixed-width CSV record fragment.

        Parameters
        ----------
        key_value_pairs : iterable of (str, float)
            Keys must be in ``summary_dict``; width and decimals come from there.

        Returns
        -------
        text : str
            Concatenated ``'{:>w.df},'`` fields.
        """
        text = ''
        for kv_pair in key_value_pairs:
            key, value = kv_pair
            wid, dec, _ = Summariser.summary_dict[key]
            fmt = '{:>' + "{:d}".format(wid) + '.' "{:d}".format(dec) + 'f},'
            text += fmt.format(value)
        return text

    @staticmethod
    def get_hdr_text(key, **kwargs):
        """
        Right-justified header cell for a summary column (key name or its unit).

        Parameters
        ----------
        key : str
            Column key in ``summary_dict``.
        **kwargs
            ``unit`` : str or None -- if given, print this instead of the key.

        Returns
        -------
        text : str
            ``'{:>w s},'`` with ``w`` the column width.

        Notes
        -----
        * Clean-up: the local ``unit`` read from ``summary_dict`` shadows the keyword
          after it has been used; ``val_fmt`` is assigned twice to the same value.
        """
        unit = kwargs.get('unit', None)
        val, val_fmt = key, "{:d}"
        if unit is not None:
            val, val_fmt = unit, "{:d}"
        wid, _, unit = Summariser.summary_dict[key]
        fmt = '{:>' + val_fmt.format(wid) + 's},'
        text = fmt.format(val)
        return text

    @staticmethod
    def create_summary_header(axes):
        """
        Build the two header rows (column keys, then units) of the summary CSV.

        Parameters
        ----------
        axes : list of str
            Unused; the columns are taken from ``summary_dict`` as populated by the
            constructor.

        Returns
        -------
        headers : list of str
            ``[keys_row, units_row]``.

        Notes
        -----
        * Clean-up: ``axes`` parameter and ``dec`` local are unused.
        """
        hdr1, hdr2 = '', ''
        sd = Summariser.summary_dict
        for key in sd:
            wid, dec, unit = sd[key]
            hdr1 += Summariser.get_hdr_text(key)
            hdr2 += Summariser.get_hdr_text(key, unit=unit)
        return [hdr1, hdr2]

    @staticmethod
    def write_summary(process_level, slice_subfolder, rows, iq_filer):
        """
        Write text rows to ``<output_folder>/<slice_subfolder><ipc_folder><level>_summary_<ipc_tag>.csv``.

        Parameters
        ----------
        process_level : str
            One of ``Globals.process_levels``.
        slice_subfolder : str
            Sub-folder name including trailing ``/``.
        rows : list of str
            Lines to write.
        iq_filer : lms_filer.Filer

        Returns
        -------
        None

        Notes
        -----
        * Inconsistency: relies on ``Ipc.tag`` / ``Ipc.folder`` having been set by
          :meth:`lms_ipc.Ipc.set_inter_pixel`; they are None otherwise and the string
          concatenation raises ``TypeError``.
        """
        summary_file_name = process_level + '_summary_' + Ipc.tag
        folder = iq_filer.output_folder + slice_subfolder + Ipc.folder
        folder = iq_filer.get_folder(folder)
        path = folder + summary_file_name + '.csv'
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_summary(process_level, slice_subfolder, iq_filer):
        """
        Read a summary CSV back into a column-index dictionary and a float array.

        The first record gives the column names (lower-cased, left-stripped); the
        second (units) is skipped; the remaining records are parsed as floats, with
        tokens of length <= 1 left as zero.

        Parameters
        ----------
        process_level : str
        slice_subfolder : str
        iq_filer : lms_filer.Filer

        Returns
        -------
        profile_dict : dict or None
            ``column name -> column index``; None if the file does not exist (a hint to
            re-run ``lmsiq`` is printed).
        profiles : ndarray, shape (n_rows, n_cols) or None

        Notes
        -----
        * Inconsistency: the hint text refers to ``IPC = x.xxx`` and
          ``'reanalyse = True'`` options that no longer exist in ``lmsiq.py``.
        """
        summary_file_name = process_level + '_summary_' + Ipc.tag
        folder = iq_filer.output_folder + slice_subfolder + Ipc.folder
        folder = iq_filer.get_folder(folder)
        path = folder + summary_file_name + '.csv'

        # summary_file_name = process_level + '_summary_' + Ipc.tag
        # folder = iq_filer.get_slice_results_folder(slice_subfolder)
        # path = folder + summary_file_name + '.csv'
        if not os.path.exists(path):
            print("!! File {:s} not found".format(path))
            print("   - run lmsiq with IPC = x.xxx and 'reanalyse = True'")
            return None
        profile_dict, profiles = {}, None
        with open(path, 'r') as text_file:
            records = text_file.read().splitlines()
            tokens = records[0].split(',')
            n_rows = len(records) - 2
            n_cols = len(tokens)
            for col, token in enumerate(tokens):
                profile_dict[token.lower().lstrip()] = col
            profiles = np.zeros((n_rows, n_cols))
            for row, record in enumerate(records[2:]):
                tokens = record.split(',')
                for col, token in enumerate(tokens):
                    if len(token) > 1:
                        profiles[row, col] = float(token)
        return profile_dict, profiles

    @staticmethod
    def write_phase_data(data_type, data_id, det_shifts, data_block):
        """
        Write a (shift x model) table of phase-scan results to CSV.

        Format: one header record ``dataset=,<..>,slice_folder=,<..>,n_mcruns=,<..>,
        axis=,<..>,``; one column-label record ``Shift, Xcen_0000, Xcen_0001, ...``;
        then one record per shift with ``{:10.3f}`` shift and ``{:16.6e}`` values.

        Parameters
        ----------
        data_type : str
            Key of the ``type_tags`` table in :meth:`_get_results_path` (e.g.
            ``'xcentroids'``, ``'photometry'``).
        data_id : tuple
            ``(dataset, slice_folder, ipc_tag, process_level, config_tag,
            n_mcruns_tag, axis)``.
        det_shifts : ndarray, shape (n_shifts,)
        data_block : ndarray, shape (n_shifts, n_runs)

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            From :meth:`_get_results_path` (module Notes).

        Notes
        -----
        * Inconsistency: the header format string has four placeholders but the
          column labels are always ``Xcen_`` regardless of ``data_type``.
        """

        dataset, slice_folder, ipc_tag, process_level, config_tag, n_mcruns_tag, axis = data_id
        fmt = "dataset=,{:<16s},slice_folder=,{:<16s},n_mcruns=,{:<16s},axis=,{:<16s},"
        header = fmt.format(dataset, slice_folder, n_mcruns_tag, axis)
        xcen_rec_list = [header]

        n_shifts, n_runs = data_block.shape
        # Write header line to text block
        fmt = "{:>10s},"
        record = fmt.format('Shift', )
        for run_number in range(0, n_runs):
            fmt = "{:>12s}{:04d},"
            record += fmt.format('Xcen_', run_number)
        xcen_rec_list.append(record)

        for res_row, det_shift in enumerate(det_shifts):
            record = "{:10.3f},".format(det_shift)
            for res_col in range(0, n_runs):
                xcen = data_block[res_row, res_col]
                record += "{:16.6e},".format(xcen)
            xcen_rec_list.append(record)

        path = Summariser._get_results_path(data_id, data_type)
        with open(path, 'w', newline='') as text_file:
            for record in xcen_rec_list:
                print(record, file=text_file)
        return

    @staticmethod
    def read_phase_data(data_type, data_id, n_runs):
        """
        Read a phase-scan CSV written by :meth:`write_phase_data`.

        Parameters
        ----------
        data_type : str
        data_id : tuple
            See :meth:`write_phase_data`.
        n_runs : int
            Number of model columns to read (``n_runs + 1`` tokens including the shift).

        Returns
        -------
        values : ndarray, shape (n_shifts, n_runs + 1)
            Column 0 is the shift.

        Raises
        ------
        AttributeError
            From :meth:`_get_results_path` (module Notes).
        """
        path = Summariser._get_results_path(data_id, data_type)
        with open(path, 'r') as text_file:
            text_block = text_file.read()

        line_list = text_block.split('\n')
        value_list = []
        for line in line_list[2:]:
            tokens = line.split(',')
            row_values = []
            if len(tokens) > 2:
                for col in range(0, n_runs + 1):
                    row_values.append(float(tokens[col]))
                value_list.append(row_values)
        values = np.array(value_list)
        return values

    @staticmethod
    def write_centroids(data_id, det_shifts, xcen_block):
        """
        Legacy: write a centroid-versus-shift table to ``..._cen...csv``.

        Parameters
        ----------
        data_id : tuple
            Five-element ``(dataset, folder_tag, config_tag, n_mcruns_tag, axis)``.
        det_shifts : ndarray, shape (n_shifts,)
        xcen_block : ndarray, shape (n_shifts, n_runs)

        Returns
        -------
        None

        Notes
        -----
        * Bug: unpacks a five-element ``data_id`` but then passes it to
          :meth:`_get_results_path`, which unpacks seven elements
          (``ValueError``), and ``'centroids'`` is not a key of that method's
          ``type_tags`` (``KeyError``).  Dead code.
        * Clean-up: ``tag = '_cen'`` is unused.
        """

        dataset, folder_tag, config_tag, n_mcruns_tag, axis = data_id
        fmt = "{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},"
        xcen_rec = fmt.format('dataset=', dataset, 'n_mcruns=', n_mcruns_tag, 'axis=', axis)
        xcen_rec_list = [xcen_rec]

        n_shifts, n_runs = xcen_block.shape
        # Write header line to text block
        fmt = "{:>10s},"
        xcen_rec = fmt.format('Shift',)
        for run_number in range(0, n_runs):
            fmt = "{:>12s}{:04d},"
            xcen_rec += fmt.format('Xcen_', run_number)
        xcen_rec_list.append(xcen_rec)

        for res_row, det_shift in enumerate(det_shifts):
            xcen_rec = "{:10.6f},".format(det_shift)
            for res_col in range(0, n_runs):
                xcen = xcen_block[res_row, res_col]
                xcen_rec += "{:16.6f},".format(xcen)
            xcen_rec_list.append(xcen_rec)

        tag = '_cen'
        path = Summariser._get_results_path(data_id, 'centroids')
#        print("Writing centroids to {:s}".format(path))
        with open(path, 'w', newline='') as text_file:
            for xcen_rec in xcen_rec_list:
                print(xcen_rec, file=text_file)
        return

    @staticmethod
    def read_centroids(data_id, n_runs):
        """
        Legacy: read a centroid table and convert model columns to shifts relative to column 0.

        Parameters
        ----------
        data_id : tuple
        n_runs : int

        Returns
        -------
        phase_shifts : ndarray, shape (n_shifts, n_runs + 1)
            Column 0 unchanged; columns >= 1 have column 0 subtracted.

        Notes
        -----
        * Bug: same ``_get_results_path`` / ``'centroids'`` key problems as
          :meth:`write_centroids`.  Dead code.
        """
        # Read data from file
        tag = '_cen'
        path = Summariser._get_results_path(data_id, 'centroids')
#        print("Reading centroids from {:s}".format(path))

        with open(path, 'r') as text_file:
            text_block = text_file.read()

        line_list = text_block.split('\n')
        centroids = []
        for line in line_list[2:]:
            tokens = line.split(',')
            cen_row = []
            if len(tokens) > 2:
                for col in range(0, n_runs + 1):
                    cen_row.append(float(tokens[col]))
                centroids.append(cen_row)
        phase_shifts = np.array(centroids)
        _, n_runs = phase_shifts.shape
        for col in range(1, n_runs):
            phase_shifts[:, col] = phase_shifts[:, col] - phase_shifts[:, 0]
        return phase_shifts

    @staticmethod
    def write_profiles(data_type, data_id, xy_data, strehl_data, ipc_factor, **kwargs):
        """
        Legacy: write an EE or LSF profile set (perfect, design, mean, rms, all MC) to CSV.

        Record layout: ``Strehl=,<s>,+-,<err>,IPC factor=,<f>``; then
        ``type=,<axis>,n_points=,<n>,n_files=,<m>,dataset=,<ds>_<cfg>,x_max=,<x>``; a
        column-label record ``X/pix., perfect, design, Mean, RMS, 0000, 0001, ...``;
        then one record per sample point.

        Parameters
        ----------
        data_type : str
            e.g. ``'ee_spectral'``, ``'lsf_spatial'`` (keys of ``type_tags``).
        data_id : tuple
            Seven-element identifier (see :meth:`write_phase_data`).
        xy_data : tuple
            ``(x, y_perfect, y_design, y_mean, y_rms, y_mcs)`` with ``y_mcs`` shape
            (n_points, n_files).
        strehl_data : tuple
            ``(strehl, strehl_err)``.
        ipc_factor : float
        **kwargs
            ``set_path`` : str or None -- explicit output path; if None the path comes
            from :meth:`_get_results_path` (which fails, see module Notes).

        Returns
        -------
        None

        Notes
        -----
        * Inconsistency: ``x_max`` is taken from ``y_mean[-1]`` (the last *y* value),
          not from ``x``.
        * Inconsistency: the message says "Filer.write_profiles"; the method lives on
          ``Summariser``.
        """
        set_path = kwargs.get('set_path', None)

        dataset, slice_folder, ipc_tag, process_level, config_tag, n_mcruns_tag, axis = data_id

        x, y_perfect, y_design, y_mean, y_rms, y_mcs = xy_data
        n_points, n_files = y_mcs.shape
        x_max = y_mean[-1]
        rows = []

        strehl, strehl_err = strehl_data
        fmt = "Strehl=, {:12.6f},+-,{:12.6f},IPC factor=,{:12.3f}"
        row = fmt.format(strehl, strehl_err, ipc_factor)
        rows.append(row)

        fmt = "type=,{:s},n_points=,{:d},n_files=,{:d},dataset=,{:s},x_max=,{:16.3f}"
        run = "{:s}_{:s}".format(dataset, config_tag)
        hdr1 = fmt.format(axis, n_points, n_files, run, x_max)
        rows.append(hdr1)
        fmt = "{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},"
        row = fmt.format('X/pix.', 'perfect', 'design', 'Mean', 'RMS')
        for j in range(0, n_files):
            file_id = "{:04d}".format(j)
            row += "{:>16s},".format(file_id)

        rows.append(row)

        for i in range(0, n_points):
            fmt = "{:>16.6f},{:>16.8e},{:>16.8e},{:>16.8e},{:>16.8e},"
            row = fmt.format(x[i], y_perfect[i], y_design[i], y_mean[i], y_rms[i])
            for j in range(0, n_files):
                row += "{:16.8e},".format(y_mcs[i, j])
            rows.append(row)
        if set_path is None:
            path = Summariser._get_results_path(data_id, data_type)
        else:
            path = set_path
        print("Filer.write_profiles to {:s}".format(path))
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_profiles(data_id, profile_type):
        """
        Legacy: read a profile CSV written by :meth:`write_profiles`.

        Parameters
        ----------
        data_id : tuple
            Seven-element identifier.
        profile_type : str
            ``data_type`` used when writing.

        Returns
        -------
        xy_data : tuple
            ``(axis, x, y_perfect, y_design, y_mean, y_rms, y_mcs)`` -- the layout
            expected by :meth:`lmsiq_analyse.Analyse.find_ee_axis_references`.
        strehl_data : tuple
            ``(strehl, strehl_err)``.
        ipc_factor : float

        Raises
        ------
        AttributeError
            From :meth:`_get_results_path` (module Notes).

        Notes
        -----
        * Clean-up: ``tag`` is built and unused; commented debug prints remain.
        """
        # Read data from file
        dataset, slice_folder, ipc_tag, process_level, config_tag, n_mcruns_tag, axis = data_id
        tag = "_{:s}_{:s}".format(profile_type, axis)
        path = Summariser._get_results_path(data_id, profile_type)
        with open(path, 'r') as text_file:
            text_block = text_file.read()
        line_list = text_block.split('\n')
        tokens = line_list[0].split(',')
        strehl = float(tokens[1])
        strehl_err = float(tokens[3])
        ipc_factor = float(tokens[5])

        hdr1 = line_list[1]
        tokens = hdr1.split(',')
        n_points = int(tokens[3])
        n_files = int(tokens[5])

        x = np.zeros(n_points)
        y_per, y_des, y_mean = np.zeros(n_points), np.zeros(n_points), np.zeros(n_points)
        y_rms = np.zeros(n_points)
        y_mcs = np.zeros((n_points, n_files))
        for i in range(0, n_points):
            line = line_list[i+3]
#            if 'lsf' in profile_type and i == 0:
#                print('Filer.readprofiles ')
#                print(path)
#                print(line)
            tokens = line.split(',')
            x[i] = float(tokens[0])
            y_per[i] = float(tokens[1])
            y_des[i] = float(tokens[2])
            y_mean[i] = float(tokens[3])
            y_rms[i] = float(tokens[4])
            for j in range(0, n_files):
                y_mcs[i, j] = float(tokens[j+5])
        xy_data = axis, x, y_per, y_des, y_mean, y_rms, y_mcs
        strehl_data = strehl, strehl_err
        return xy_data, strehl_data, ipc_factor

    @staticmethod
    def _get_results_path(data_id, data_type):
        """
        Legacy: build the CSV path ``<results>/<slice>/<ipc>/<level>/<type>/<name>.csv``.

        File name: ``<slice_tag>_<ipc_tag>_<process_level><type_tag>_wav_<config_tag>.csv``
        with ``type_tag`` from a fixed table (``'xcentroids' -> '_xcen'``,
        ``'ee_spectral' -> '_eex'``, ``'lsf_dfp_spatial' -> '_lsy_dfp'`` ...).

        Parameters
        ----------
        data_id : tuple
            ``(dataset, slice_subfolder, ipc_tag, process_level, config_folder,
            mcrun_tag, axis)``.
        data_type : str
            Key of the ``type_tags`` table.

        Returns
        -------
        path : str

        Raises
        ------
        AttributeError
            ``Summariser.dataset_results_path`` and ``Summariser._get_folder`` do not
            exist.

        Notes
        -----
        * Bug: see Raises; the method (and everything that calls it) cannot run.
          :meth:`lms_filer.Filer._get_results_path` is the working equivalent using
          ``self.output_folder`` and ``Filer.get_folder``.
        * Clean-up: ``type_folder`` is computed and unused; the final ``os.mkdir``
          duplicates what the (missing) ``_get_folder`` would do.
        """
        dataset, slice_subfolder, ipc_tag, process_level, config_folder, mcrun_tag, axis = data_id

        slice_folder = Summariser._get_folder(Summariser.dataset_results_path + '/' + slice_subfolder)
        ipc_folder = Summariser._get_folder(slice_folder + Ipc.folder)
        proc_folder = Summariser._get_folder(ipc_folder + process_level + '/')
        type_folder = Summariser._get_folder(proc_folder + data_type) + '/'

        type_tags = {'xcentroids': '_xcen', 'ycentroids': '_ycen',
                     'xfwhm_gau': '_xfwhm', 'photometry': '_phot',
                     'ee_spectral': '_eex', 'ee_spatial': '_eey',
                     'lsf_spectral': '_lsx', 'lsf_spatial': '_lsy',
                     'ee_dfp_spectral': '_eex_dfp', 'ee_dfp_spatial': '_eey_dfp',
                     'lsf_dfp_spectral': '_lsx_dfp', 'lsf_dfp_spatial': '_lsy_dfp',
                     }
        type_tag = type_tags[data_type]
        config_tag = config_folder[0:-1]
        slice_tag = slice_subfolder[:-1] + '_'
        file_name = slice_tag + ipc_tag + '_' + process_level + type_tag + '_wav_' + config_tag + '.csv'
        folder = ipc_folder + process_level + '/' + data_type
        path = folder + '/' + file_name
        if not os.path.exists(folder):
            os.mkdir(folder)
        return path
