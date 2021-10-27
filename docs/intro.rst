.. highlight:: python

Overview
########

Load STAC :class:`pystac.Item`\s into :class:`xarray.Dataset`.

.. code-block:: python

   catalog = pystac_client.Client.open(...)
   query = catalog.search(...)
   xx = odc.stac.load(
       query.get_items(),
       bands=["red", "green", "blue"],
       resolution=100,
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

.. code-block:: bash

   conda install -c conda-forge odc-stac


Using latest unreleased code in ``conda`` is also possible. It's best to install
dependencies using conda then install ``odc-stac`` with pip. Sample
``environment.yml`` is provided below.


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
       - git+https://github.com/opendatacube/odc-stac/
