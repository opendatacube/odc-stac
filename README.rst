odc.stac
########

|Documentation Status| |Test Status| |Test Coverage| |Binder|

Tooling for converting STAC metadata to ODC data model.

Usage
#####


odc.stac.load
~~~~~~~~~~~~~

.. code-block:: python

   catalog = pystac_client.Client.open(...)
   query = catalog.search(...)
   xx = odc.stac.load(
       query.get_items(),
       bands=["red", "green", "blue"],
       crs="EPSG:32606",
       resolution=(-100, 100),
   )
   xx.red.plot.imshow(col="time")



Installation
############

Using pip
~~~~~~~~~

.. code-block:: bash

   pip install odc-stac


Using Conda
~~~~~~~~~~~

This package is be available on ``conda-forge`` channel:

.. code-block:: bash

   conda install -c conda-forge odc-stac

To use development version of ``odc-stac`` install dependencies from conda, then
install ``odc-stac`` with ``pip``.

Sample ``environment.yml`` is provided below.


.. code-block:: yaml

   channels:
     - conda-forge
   dependencies:
     - datacube >=1.8.5
     - xarray ~= 0.20.1
     - numpy
     - pandas
     - affine
     - rasterio
     - toolz
     - jinja2
     - pystac
     - pystac-client
     - pip =20
     - pip:
       - odc-stac

Developing
##########

To develop ``odc-stac`` locally using pip (assuming you have virtualenvwrapper_ installed):

.. code-block:: bash

   git clone https://github.com/opendatacube/odc-stac
   cd odc-stac
   mkvirtualenv odc-stac
   pip install -e .
   pip install -r requirements-dev.txt

Run tests with pytest_:

.. code-block:: bash

   pytest

Linting is provided by mypy_, pylint_, and black_:

.. code-block:: bash

   black --check .
   pylint -v odc
   mypy odc


.. |Documentation Status| image:: https://readthedocs.org/projects/odc-stac/badge/?version=latest
   :target: https://odc-stac.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |Test Status| image:: https://github.com/opendatacube/odc-stac/actions/workflows/main.yml/badge.svg
   :target: https://github.com/opendatacube/odc-stac/actions/workflows/main.yml
   :alt: Test Status

.. |Test Coverage| image:: https://codecov.io/gh/opendatacube/odc-stac/branch/develop/graph/badge.svg?token=HQ8nTuZHH5
   :target: https://codecov.io/gh/opendatacube/odc-stac
   :alt: Test Coverage

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/opendatacube/odc-stac/develop?urlpath=lab/workspaces/demo
   :alt: Run Examples in Binder

.. _virtualenvwrapper: https://virtualenvwrapper.readthedocs.io

.. _pytest: https://docs.pytest.org

.. _mypy: http://mypy-lang.org/

.. _pylint: https://pylint.org/

.. _black: https://github.com/psf/black
