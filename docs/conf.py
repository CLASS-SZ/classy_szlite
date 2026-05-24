"""Sphinx configuration for classy_szlite."""
from __future__ import annotations
import os
import sys

# So autodoc can find the package when building on RTD
sys.path.insert(0, os.path.abspath(".."))

# -- Project ----------------------------------------------------------------
project   = "classy_szlite"
copyright = "2026, Boris Bolliet and the CLASS-SZ contributors"
author    = "Boris Bolliet"

try:
    import classy_szlite
    release = classy_szlite.__version__
except Exception:
    release = "0.2.0"
version = ".".join(release.split(".")[:2])

# -- General ----------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "jax":    ("https://docs.jax.dev/en/latest", None),
}

# -- HTML output ------------------------------------------------------------
# pydata-sphinx-theme — same scientific look as numpy / scipy / matplotlib /
# jax / astropy / pandas. Clean, dense, two-column layout with a real navbar.
html_theme   = "pydata_sphinx_theme"
html_title   = "classy_szlite"
html_static_path = ["_static"]
html_show_sourcelink = False

html_theme_options = {
    "github_url":      "https://github.com/CLASS-SZ/classy_szlite",
    "navbar_align":    "left",
    "show_prev_next":  True,
    "use_edit_page_button": False,
    "logo":            {"text": "classy_szlite"},
    "icon_links": [
        {"name": "PyPI",   "url": "https://pypi.org/project/classy-szlite/",
         "icon": "fa-brands fa-python"},
        {"name": "GitHub", "url": "https://github.com/CLASS-SZ/classy_szlite",
         "icon": "fa-brands fa-github"},
    ],
    "header_links_before_dropdown": 7,
}
html_context = {
    "github_user":   "CLASS-SZ",
    "github_repo":   "classy_szlite",
    "github_version":"main",
    "doc_path":      "docs",
}

# Force the LEFT sidebar to show the full toctree on EVERY page.
# pydata-sphinx-theme's default per-page "Section Navigation" goes empty
# when a page is a toctree leaf (no children of its own). Use "sidebar-nav-bs"
# everywhere so users always see the full table of contents.
html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

# autodoc: include all members by default
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
