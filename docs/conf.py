"""Sphinx config."""
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from importlib.metadata import version

import numpy as np

# sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = "ptsa"
copyright = "2023, Dominik Beutel"
author = "Dominik Beutel"
release = version("ptsa")
version = ".".join(release.split(".")[:3])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "matplotlib.sphinxext.plot_directive",
]

autosectionlabel_prefix_document = True
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
todo_include_todos = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pyramid"

html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


def preprocess_ufunc(app, what, name, obj, options, lines):
    """Remove the first four docstring lines from numpy ufuncs.

    The docstring of ufuncs have a line containing the function name and signature and a
    blank line automatically prepended. The signature contains all arguments and keyword
    arguments but the positional arguments only get dummy variable names ('x1', 'x2',
    ...) and is therefore often confusing.

    By convention all ufuncs contain in their first manually added lines the function
    name with a simplified signature followed by a blank line. We explicitly add the
    signature with a separate process to autodoc.

    The four lines containing function names and signatures or being blank are therefore
    removed for a clean documentation produced by sphinx.
    """
    if isinstance(obj, np.ufunc):
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)


def ufunc_signature(app, what, name, obj, options, signature, return_annotation):
    """Extract the signature from the docstring for numpy ufuncs.

    As described in :func:`preprocess_ufunc` the third line in a ufunc docstring
    contains the manually added function name and signature intended to be displayed in
    the documentation (the preceeding two lines are automatically added). We extract the
    signature and add it manually.
    """
    if isinstance(obj, np.ufunc):
        signature = "(" + obj.__doc__.split("\n")[2].split("(")[1]
    return signature, return_annotation


def setup(app):
    """Add processes to autodoc."""
    app.connect("autodoc-process-docstring", preprocess_ufunc)
    app.connect("autodoc-process-signature", ufunc_signature)
