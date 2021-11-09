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
import os
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, os.path.abspath(".."))
from odc.stac._version import __version__ as _odc_stac_version


def compute_nb_hash(folder):
    nb_hash = (
        subprocess.check_output(
            [
                "/bin/bash",
                "-c",
                f"find {folder} -maxdepth 1 -name '*.py' -type f | sort -f -d | xargs cat | sha256sum | cut -b -16",
            ]
        )
        .decode("ascii")
        .split()[0]
    )
    return nb_hash


def ensure_notebooks(https_url, dst_folder):
    """
    Download pre-rendered notebooks from a tar archive
    """

    dst_folder = Path(dst_folder)
    if dst_folder.exists():
        print(f"Found pre-rendered notebooks in {dst_folder}")
        return

    dst_folder.mkdir()
    print(f"Fetching: {https_url} to {dst_folder}")
    log = subprocess.check_output(
        ["/bin/bash", "-c", f"curl -s {https_url} | tar xz -C {dst_folder}"]
    ).decode("utf-8")
    print(log)


# working directory is docs/
# download pre-rendered notebooks unless folder is already populated
nb_hash = compute_nb_hash("../notebooks")
https_url = (
    f"https://packages.dea.ga.gov.au/odc-stac/nb/odc-stac-notebooks-{nb_hash}.tar.gz"
)
ensure_notebooks(https_url, "notebooks")

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
    "issue": ("https://github.com/opendatacube/odc-stac/issues/%s", "issue "),
    "pull": ("https://github.com/opendatacube/odc-stac/pulls/%s", "PR "),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "np": ("https://docs.scipy.org/doc/numpy/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "xr": ("https://xarray.pydata.org/en/stable/", None),
    "datacube": ("https://datacube-core.readthedocs.io/en/latest/", None),
    "pystac": ("https://pystac.readthedocs.io/en/latest/", None),
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
