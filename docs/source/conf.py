import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

print(os.getcwd())


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyEulerCurves"
copyright = "2023, Davide Gurnari"
author = "Davide Gurnari"
release = "0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_nb",
    # "myst_parser",
]

nb_execution_mode = "off"  # disables notebook execution

# Enable MathJax for rendering LaTeX
# mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"

source_suffix = {
    ".rst": "restructuredtext",
    # ".md": "markdown",
    ".ipynb": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = []

autoclass_content = "both"

autodoc_mock_imports = [
    "pyEulerCurves._compute_local_EC_VR",  # mock the C++ extension
    "numpy",
    "pandas",
    "numba",
    "matplotlib",
    "sklearn",
    "scipy",
    "tqdm",
    "ipywidgets",
    "IPython",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = 'classic'

html_static_path = ["_static"]

debug = True
