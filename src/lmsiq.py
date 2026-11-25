"""
Top-level driver script for the ``lmsiq`` program (LMS image-quality repeatability analysis).

Purpose
-------
``lmsiq`` ingests stacks of Zemax point-spread-function (PSF) images of the METIS LMS
(one 'perfect' model, one 'as-designed' model and N Monte-Carlo (MC) as-built instances
per slice, field point and spectral configuration), pushes them through a detector
diffusion / inter-pixel-capacitance forward model (:class:`lms_ipc.Ipc`), rebins them to
H2RG pixels (:class:`lms_detector.Detector`) and derives image-quality metrics: line
spread function (LSF) FWHM, spectral resolving power (SRP), enslitted energy (EE),
along- and across-slice FWHM and a relative 'Strehl' against the perfect model.  Results
are collected per field point as a function of wavelength (nominal mode) or spectral IFU
slice (extended mode) and plotted against requirements MET-2745 (SRP) and MET-3739
(spectral FWHM).

This module belongs to the ``lmsiq`` program.  It is a script: all work is performed by
:func:`main`, which must be invoked explicitly (``python lmsiq.py``) or imported and
called.

Input data
----------
All data live outside the repository, under ``../data/iq/<opticon>/<date_stamp>/``
relative to ``src/``, where ``<opticon>`` is ``'nominal'`` or ``'extended'`` and
``<date_stamp>`` is a 10-digit tag (8-digit date plus a 2-digit run counter, e.g.
``2024073000``).  Within that folder each Zemax configuration is a sub-folder whose name
carries the codes ``config<nnn>``, ``field<nnn>`` and optionally ``defoc<nnn>`` (three
digits each), and contains

* one ``.txt`` parameter file with colon-separated keys ``wavelength``, ``prism angle``,
  ``grating angle``, ``grating order`` and ``Pixel size (micron)``;
* FITS PSF images named ``<folder>_spat<ss>_spec<p>_det<tag>.fits`` where ``<ss>`` is the
  two-digit spatial slice number, ``<p>`` the one-digit spectral-IFU slice number and
  ``<tag>`` is ``perf`` (perfect), ``desi`` (as-designed) or a four-digit MC number;
  files with ``sli`` in the name are slicer-plane images and are skipped.

Images are 128 x 128 pixels at 4.5 um sampling, i.e. 4x oversampled with respect to the
18 um H2RG pixel.  Distortion (dispersion) information is read from the ``lmsdist`` trace
pickle for the matching optical path (``../output/distortion/<opticon>/<date>/...``).

Outputs
-------
Written under ``../output/iq/<opticon>/<date_stamp>/``:

* ``cube/fits/cube_<tags>.fits`` -- pseudo (model, along-slice, slice) cubes, no header;
* ``cube/pkl/cube_<tags>.pkl`` -- pickled ``(cube_ident, cube, field_series)`` packages;
* ``profiles/lsf/`` and ``profiles/ee/`` -- PNG LSF and EE profiles;
* ``cube/series/png/`` -- PNG series plots (metric vs wavelength / spifu slice per field);
* ``test/`` -- PNG collage from :class:`lmsiq_test.Test` when ``run_test`` is True.

Notes
-----
* Inconsistency: the data set actually analysed is not the one selected by
  ``iq_date_stamp`` in this file.  :class:`lms_filer.Filer` takes the date stamp from
  ``Globals.model_configurations['iq'][opticon]`` (``2024073000`` for nominal,
  ``2024032401`` for extended), so selecting ``'2024061403'`` here would still read the
  ``2024032401`` folder while labelling plots with the other stamp.
* Inconsistency: ``mc_bounds`` is described in a comment as ``None``, ``[mlo, mhi]`` or
  ``'all'``; the code only ever overrides the folder-derived bounds when the requested
  value is neither ``None`` nor ``'all'``, so ``'all'`` and ``None`` behave identically.
* Open question: the ``optics`` dictionary and ``analysis_type`` are built but never
  used; ``fts_dfp`` is only referenced by commented-out ``data_dictionary`` entries.
* Open question: the nominal ``cube_packages`` have configuration 21 removed with the
  comment "Filter out weird nominal configuration"; the reason is not recorded.
* Clean-up: :class:`lmsiq_analyse.Analyse` must be instantiated (``analyse =
  Analyse()``) because its constructor sets the class attribute ``fwhm_sigma`` used by
  ``Analyse._gauss``; the instances ``plot``, ``detector``, ``util``, ``fitsio``,
  ``summariser`` and ``ipc`` are otherwise unused.
* Clean-up: ``time``, ``os`` and ``np`` are imported but unused here.
"""
import os
import numpy as np
import time
from lms_globals import Globals
from lms_filer import Filer
from lms_wcal import Wcal
from lms_ipc import Ipc
from lms_detector import Detector
from lmsiq_analyse import Analyse
from lmsiq_fitsio import FitsIo
from lmsiq_plot import Plot
from lmsiq_phase import Phase
from lmsiq_cuber import Cuber
from lmsiq_summariser import Summariser
from lmsiq_image_manager import ImageManager
from lmsdist_util import Util
from lmsiq_test import Test


def main():
    """
    Run the lmsiq image-quality analysis end to end.

    The flow is:

    1. Select the data set.  ``data_dictionary`` maps a 10-digit iq date stamp to
       ``(optical_path, mc_bounds, dist_date_stamp, field_tgt_slice, slice_radius,
       label)``; ``iq_date_stamp`` is hard-coded to ``'2024073000'`` (nominal, spherical
       M12, EFP fields 1-9).  ``field_tgt_slice`` maps a Zemax field number to the slice
       on which the point source is centred (``fts_efp`` = {1-3: 13, 4-6: 5, 7-9: 24};
       ``fts_spifu`` = {f: 12 + (f-1) % 3} for f = 1..18); ``slice_radius`` is the number
       of neighbouring slices either side of the target slice included in the cube.
    2. Initialise the static helper classes (:class:`lmsiq_analyse.Analyse` must be
       instantiated so that ``Analyse.fwhm_sigma`` is set) and configure an ``'iq'``
       :class:`lms_filer.Filer`.
    3. Build the image dictionary with :meth:`lmsiq_image_manager.ImageManager.make_dictionary`.
    4. Optionally (``run_test``, default False) run the :class:`lmsiq_test.Test` IPC
       sanity check.
    5. Resolve the Monte-Carlo bounds and set ``inter_pixels = [True]`` (diffusion on).
    6. Optionally (``process_phase_data``, default False) run the sub-pixel phase study
       :meth:`lmsiq_phase.Phase.process`.
    7. If ``build_cubes`` (default True) build the cubes and metrics with
       :meth:`lmsiq_cuber.Cuber.build` using a second, ``'distortion'`` Filer for the
       dispersion traces.
    8. If ``plot_cubes`` (default True) reload the pickled cube packages, drop
       configuration 21 for the nominal path (:meth:`lmsiq_cuber.Cuber.remove_configs`)
       and plot the series with :meth:`lmsiq_cuber.Cuber.plot`.

    Returns
    -------
    None

    Notes
    -----
    * Both Filers are configured with ``is_ait_data=False`` (Zemax model data).
    * Inconsistency: the ``'extended'`` entry ``'2024061403'`` in ``data_dictionary``
      cannot be reached through :class:`lms_filer.Filer`, which resolves the extended
      iq data set to ``'2024032401'`` from :class:`lms_globals.Globals`.
    """
    print('LMS Repeatability (lmsiq.py) - started')
    print()

    analysis_type = 'iq'

    # Define the optical path, either nominal or with the spectral IFU inserted in the beam.
    nominal = Globals.nominal
    extended = Globals.extended
    optics = {nominal: ('Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)', ('phase', 'fp2_x'), ('det_x', 'det_y')),
              extended: ('Extended spectral coverage (fov = 1.0 x 0.054 arcsec)', ('phase', 'fp1_x'), ('det_x', 'det_y'))}

    # Mapping from field number to target slice
    fts_dfp = {10: 1, 11: 13, 12: 28}
    fts_efp = {1: 13, 2: 13, 3: 13, 4: 5, 5: 5, 6: 5, 7: 24, 8: 24, 9: 24}
    fts_spifu = {}
    for f in range(0, 18):
        fts_spifu[f+1] = 12 + f % 3

    # Data descriptor: optical_path, mc_bounds, dist_date_stamp, field_to_tgt_slice_map, slice_radius, label
    data_dictionary = {
                       # '2024032400': (nominal, 'all', '20240109', fts_efp, 4,
                       #                "toroidal M12, EFP fields 1-9, full LM-bands"),
                       # '2024032401': (extended, 'all', '20240109', {1: 13, 2: 13, 3: 13}, 1,
                       #                "extended coverage, toroidal M12"),
                       # '2024043000': (nominal, 'all', '20240305', fts_dfp, 0,
                       #                "torM12, defocus .0,.05,.1 mm, wave 2.7 um"),
                       # '2024050700': (nominal, 'all', '20240305', fts_dfp, 0,
                       #                "torM12, DFP fields 10-12, defocus .0,.05,.1 mm, wave 4.57 um"),
                       # '2024050701': (nominal, 'all', '20240305', fts_dfp, 0,
                       #                "torM12, DFP fields 10-12, defocus .0,.05,.1 mm, wave 5.00 um"),
                       # '2024050702': (nominal, 'all', '20240305', fts_dfp, 0,
                       #                "torM12, DFP fields 10-12, defocus .0,.05,.1 mm, wave 5.24 um"),
                       #
                       # '2024060700': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus   0 um'),
                       # '2024060705': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus  50 um'),
                       # '2024060710': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus 100 um'),
                       # '2024060720': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus 200 um'),
                       '2024073000': (nominal, 'all', '20240109', fts_efp, 4,
                                      "spherical M12, EFP fields 1-9, full LM-bands"),
                       '2024061403': (extended, 'all', '20231009', fts_spifu, 0,
                                      "METIS_M19manufacture_errors (updated)"),
                       }

    iq_date_stamp = '2024073000'
    opticon, mc_bounds, dist_date_stamp, field_tgt_slice, slice_radius, data_label = data_dictionary[iq_date_stamp]
    data_identifier = {'optical_path': opticon,
                       'iq_date_stamp': iq_date_stamp,
                       'mc_bounds': mc_bounds,                    # Set to None, [mlo, mhi] or 'all'
                       'dist_date_stamp': dist_date_stamp,
                       'field_tgt_slice': field_tgt_slice,
                       'slice_radius': slice_radius,
                       'data_label': data_label
                       }
    _ = Globals()

    fmt = "Analysing dataset 'data/iq/{:s}/{:s}"
    print(fmt.format(opticon, iq_date_stamp))

    base_debug_level = 'off'
    Globals.set_debug_level(base_debug_level)

    # Initialise static classes
    analyse = Analyse()
    plot = Plot()
    detector = Detector()
    util = Util()
    fitsio = FitsIo(opticon)

    # model_configuration = analysis_type, opticon, iq_date_stamp, None, None, None
    iq_filer = Filer()
    iq_filer.set_configuration('iq', opticon, is_ait_data=False)

    image_manager = ImageManager()
    image_manager.make_dictionary(data_identifier, iq_filer)

    summariser = Summariser()
    ipc = Ipc()

    run_test = False  # Generate test case with 50 % IPC and an image with one bright pixel.
    print("Running test case = {:s}".format(str(run_test)))
    if run_test:
        test = Test()
        test.run(iq_filer)
        print('Test - done')

    inter_pixels = [True]                        # True = include diffusion kernel convolution
    mc_bounds = image_manager.model_dict['mc_bounds']
    req_mc_bounds = data_identifier['mc_bounds']
    if req_mc_bounds is not None:
        if req_mc_bounds != 'all':
            mc_bounds = mc_bounds if req_mc_bounds == 'all' else req_mc_bounds
            print("\nSetting mc_start={:d}, mc_end={:d}".format(mc_bounds[0], mc_bounds[1]))
    process_control = mc_bounds, inter_pixels

    # Analyse Zemax data
    process_phase_data = False
    if process_phase_data:
        # Calculate the impact of sub-pixel shifts on photometry and line centroiding.
        print("Processing centroid shift impact for dataset {:s}".format(iq_date_stamp))
        phase = Phase()                                 # Module for phase (wavelength) shift analysis
        phase.process(data_identifier, process_control, iq_filer, image_manager,
                      config_nos=[0], field_nos=[2], plot_level=2)

    build_cubes = True
    if build_cubes:
        print()
        print('Reconstructing cubes and analysing slice profile data')
        print('-----------------------------------------------------')
        cuber = Cuber()
        # dist_model_configuration = 'distortion', opticon, dist_date_stamp, None, None, None
        dist_filer = Filer()
        dist_filer.set_configuration('distortion', opticon, is_ait_data=False)
        cuber.build(data_identifier, process_control, image_manager, iq_filer, dist_filer,
                    debug=False, plot_to_png=False)

    plot_cubes = True
    if plot_cubes:
        title = '\nPlot cube series data (wavelength/field) and calculate key statistics'
        print(title)
        print('-'*len(title))
        cuber = Cuber()
        cube_packages = cuber.read_pkl_cubes(iq_filer)
        if opticon == Globals.nominal:           # Filter out weird nominal configuration...!
            cube_packages = Cuber.remove_configs(cube_packages, [21])
        cuber.plot(opticon, cube_packages, iq_filer, is_defocus=False)

    print('LMS Repeatability (lmsiq.py) - done')


if __name__ == '__main__':
    main()
