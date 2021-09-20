.. highlight:: python

Overview
########

Load STAC :class:`pystac.Item`\s into :class:`xarray.Dataset`.

.. code-block:: python

   catalog = pystac.Client.open(...)
   query = catalog.search(...)
   xx = odc.stac.load(
       query.get_items(),
       bands=["red", "green", "blue"],
       output_crs="EPSG:32606",
       resolution=(100, -100),
   )
   xx.red.plot.imshow(col="time")


See :func:`odc.stac.load`.


Installation
############

Using pip
*********

.. code-block:: bash

   pip install odc-stac

Using Conda
***********

Currently conda package is not done yet. It's best to install dependencies
using conda then install ``odc-stac`` with pip. Sample ``environment.yml`` is
provided below.


.. code-block:: yaml

   channels:
     - conda-forge
   dependencies:
     - datacube>=1.8.5
     - xarray
     - numpy
     - pandas
     - affine
     - rasterio
     - toolz
     - jinja2
     - pystac
     - pip=20
     - pip:
       - odc-stac
