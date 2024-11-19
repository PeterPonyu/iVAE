import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import iVAE

project = 'iVAE'
copyright = '2024, Zeyu Fu'
author = 'Zeyu Fu'
release = iVAE.__version__

extensions = ['sphinx.ext.autodoc',
            'sphinx.ext.mathjax',
            'sphinx.ext.napoleon',
            'sphinx.ext.intersphinx',
            'sphinx.ext.viewcode',
            'sphinx.ext.todo',
            'sphinx.ext.autosummary',
             'nbsphinx']

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_css_files = ['custom.css']
