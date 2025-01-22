# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import logging as pylogging

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

import requests
from sphinx.util import logging

sys.path.insert(0, os.path.abspath(".."))
from odc.stac._version import __version__ as _odc_stac_version
from scripts import notebook_hash

# isort: off
# extra imports to check env
import odc.stac.bench
import odc.stac.eo3


# Workaround for https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
# When this https://github.com/agronholm/sphinx-autodoc-typehints/pull/153
# gets merged, we can remove this
class FilterForIssue123(pylogging.Filter):
    def filter(self, record: pylogging.LogRecord) -> bool:
        # You probably should make this check more specific by checking
        # that dataclass name is in the message, so that you don't filter out
        # other meaningful warnings
        return not record.getMessage().startswith("Cannot treat a function")


logging.getLogger("sphinx_autodoc_typehints").logger.addFilter(FilterForIssue123())
# End of a workaround


def ensure_notebooks(dst_folder):
    """
    Download pre-rendered notebooks from a tar archive
    """
    dst_folder = Path(dst_folder)
    if dst_folder.exists():
        print(f"Found pre-rendered notebooks in {dst_folder}")
        return True

    dst_folder.mkdir()
    nb_hash, nb_paths = notebook_hash.compute("../notebooks")
    nb_names = [p.rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".ipynb" for p in nb_paths]

    for nb in nb_names:
        url = f"https://{nb_hash[:16]}--odc-stac-docs.netlify.app/notebooks/{nb}"
        print(f"{url} -> notebooks/{nb}")
        rr = requests.get(url, timeout=5)
        if not rr:
            return False
        with open(dst_folder / nb, "wt", encoding="utf") as dst:
            dst.write(rr.text)

    return True


# working directory is docs/
# download pre-rendered notebooks unless folder is already populated
if not ensure_notebooks("notebooks"):
    notebooks_directory = os.path.abspath("../notebooks")
    raise RuntimeException(
        "There is no cached version of these notebooks. "
        "Build the notebooks before building the documentation. "
        f"Notebooks are located in {notebooks_directory}."
    )

# -- Project information -----------------------------------------------------

project = "odc-stac"
copyright = "2021, ODC"
author = "ODC"

version = ".".join(_odc_stac_version.split(".", 2)[:2])
# The full version, including alpha/beta/rc tags
release = _odc_stac_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

autosummary_generate = True

extlinks = {
    "issue": ("https://github.com/opendatacube/odc-stac/issues/%s", "issue %s"),
    "pull": ("https://github.com/opendatacube/odc-stac/pulls/%s", "PR %s"),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "datacube": ("https://datacube-core.readthedocs.io/en/latest/", None),
    "odc-geo": ("https://odc-geo.readthedocs.io/en/latest/", None),
    "pystac": ("https://pystac.readthedocs.io/en/latest/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True,
}

# html_logo = '_static/logo.svg'
html_last_updated_fmt = "%b %d, %Y"
html_show_sphinx = False


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["xr-fixes.css"]
