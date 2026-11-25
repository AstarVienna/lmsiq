# lmsiq
LMS image quality analysis

### Summary

The package includes several free-standing programs, as follows, 

lmsiq - Ingest Zemax images representing the ideal, as-designed and Monte-Carlo perturbed instances of the anticipated 
aberrated system. Output includes en-slitted energy and line profiles along both the spatial and spectral directions,
along with the wavelength dependence of key image quality performance parameters.

lmsdist - Generate LMS distortion transforms as described in 'E-TNT-ATC-MET-1058' from ray coordinates in key focal 
planes.  These coordinates can be derived from the Zemax model or performance test data.  

lmssim - Generate smiulated performance test data using either 'ScopeSim' or a bespoke 'ToySim' program which models 
specific performace test features (WCU and Leiden sky spectra, plus transforms generated using 'lmsdist'.

lmsaiv - LMS performance analysis tools tailored for the tests described in 'E-PRO-NOVA-MET-2018' (Polarion) and 
modelled using lmssim.  

### User Guide
The code is intended to be run by execution of the parent program (eg file 'lmsiq.py' for lmsiq).  

### Tasks
- Generate simulated data and analysis tools for performance testing (currently planned for summer 2026).
### Documentation
Sphinx documentation lives in `docs/` (Read the Docs configuration in `.readthedocs.yaml`).
It contains a description of the four programs, the expected data layout and header
contracts, the METIS pipeline team's analysis of which parts can be extracted into the
pipeline's LMS recipes, a consolidated list of open questions / inconsistencies / bugs /
clean-up suggestions harvested from the docstrings, and the API reference for every module.

To build locally:

    pip install -r docs/requirements.txt
    cd docs
    python tools/collect_notes.py     # refresh open_questions.md from the docstrings
    sphinx-build -b html . _build/html
