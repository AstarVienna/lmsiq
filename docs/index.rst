lmsiq: METIS LMS image quality, distortion and AIT analysis tools
=====================================================================

``lmsiq`` is Alistair Glasse's (UK ATC) collection of analysis programs for the
METIS **LMS**, the L/M-band integral-field spectrograph of the ELT mid-infrared
instrument METIS. It contains four command-line programs sharing one flat
``src/`` namespace:

``lmsiq``
    Image-quality analysis of Zemax PSF stacks (perfect, as-designed and
    Monte-Carlo perturbed): line-spread functions, resolving power, enslitted
    energy, along- and across-slice FWHM.
``lmsdist``
    Fits the LMS distortion transforms described in E-TNT-ATC-MET-1058 from
    Zemax ray traces, models their dependence on prism and echelle angle, and
    writes the transform FITS products consumed by ScopeSim.
``lmssim``
    Generates simulated AIT performance-test frames, either through ScopeSim
    or through the bespoke transform-driven ``ToySim``.
``lmsaiv``
    Analyses the LMS AIT performance tests (LMS-RAD-01 dark current,
    LMS-RAD-04 pixel flat, LMS-OPT-01 field of view and slice positions,
    LMS-OPT-02 distortion; stubs for LMS-OPT-03, -04 and -09) on simulated or
    measured detector frames.

This documentation was written by the METIS pipeline team in September 2026
while assessing which parts of ``lmsiq`` can be extracted into the METIS
data-reduction pipeline (``pymetis``) as QC parameters and diagnostics. The
docstrings and the narrative pages record, alongside the description of what
each function does, the **open questions**, **inconsistencies**, **bugs** and
**clean-up suggestions** found while reading the code. They are collected on
one page in :doc:`open_questions`.

The docstrings and the program and data-layout pages describe upstream commit
``9b6143a`` ("General tidy up for PIP team walkthrough", merged November 2026).
The pipeline extraction analysis is dated September 2026 and refers to
``1547ea4``; its addendum lists what the later commit changes.

.. toctree::
   :maxdepth: 2
   :caption: Using lmsiq

   programs
   data_layout

.. toctree::
   :maxdepth: 2
   :caption: Pipeline extraction analysis

   analysis

.. toctree::
   :maxdepth: 2
   :caption: Code notes

   open_questions

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
