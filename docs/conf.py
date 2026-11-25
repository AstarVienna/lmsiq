"""Sphinx configuration for the lmsiq documentation.

The lmsiq modules live in a flat ``src/`` namespace and import each other by bare
name (``from lms_globals import Globals``), so ``src/`` is put on ``sys.path`` here.
Heavy or network-touching dependencies (ScopeSim, synphot) are mocked so the API
pages build without an instrument package or an internet connection.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

project = 'lmsiq'
author = 'Alistair Glasse (UK ATC); documentation by the METIS pipeline team (A* Vienna)'
copyright = f'{date.today().year}, {author}'
release = '1.0'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
templates_path = ['_templates']
html_static_path = ['_static']

# -- autodoc --------------------------------------------------------------
# Modules that are not needed to *document* the code. Mocking them lets
# autodoc import lmssim_scope / lmssim_model without a ScopeSim install.
autodoc_mock_imports = [
    'scopesim',
    'scopesim_templates',
    'synphot',
]
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autosummary_generate = False

# -- napoleon (numpydoc-style docstrings) ---------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True   # class attributes as :ivar: fields, avoids duplicate autodoc entries
napoleon_custom_sections = [
    ('Open questions', 'notes_style'),
    ('Inconsistencies', 'notes_style'),
    ('Clean-up', 'notes_style'),
    ('Bugs', 'notes_style'),
]

# -- myst -----------------------------------------------------------------
myst_enable_extensions = ['colon_fence', 'deflist', 'tasklist']
myst_heading_anchors = 3

# -- intersphinx ----------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}
intersphinx_timeout = 5

# -- HTML -----------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
}
html_title = 'lmsiq documentation'
