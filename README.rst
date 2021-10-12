odc.stac
########

|Documentation Status| |Test Status| |Test Coverage|

Tooling for converting STAC metadata to ODC data model.

Usage
#####


odc.stac.load
~~~~~~~~~~~~~

.. code-block:: python

   catalog = pystac.Client.open(...)
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
     - xarray
     - numpy
     - pandas
     - affine
     - rasterio
     - toolz
     - jinja2
     - pystac
     - pip =20
     - pip:
       - odc-stac



.. |Documentation Status| image:: https://readthedocs.org/projects/odc-stac/badge/?version=latest
   :target: https://odc-stac.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |Test Status| image:: https://github.com/opendatacube/odc-stac/actions/workflows/main.yml/badge.svg
   :target: https://github.com/opendatacube/odc-stac/actions/workflows/main.yml
   :alt: Test Status

.. |Test Coverage| image:: https://codecov.io/gh/opendatacube/odc-stac/branch/develop/graph/badge.svg?token=HQ8nTuZHH5
   :target: https://codecov.io/gh/opendatacube/odc-stac
   :alt: Test Coverage
