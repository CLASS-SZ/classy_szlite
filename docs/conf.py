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
html_theme   = "furo"
html_title   = "classy_szlite"
html_static_path = ["_static"]
html_show_sourcelink = False

html_theme_options = {
    "source_repository": "https://github.com/CLASS-SZ/classy_szlite",
    "source_branch":     "main",
    "source_directory":  "docs/",
}

# autodoc: include all members by default
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
