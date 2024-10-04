.. highlight:: python

Overview
########

Load STAC :py:class:`pystac.Item`\s into :py:class:`xarray.Dataset`.

.. code-block:: python

   catalog = pystac_client.Client.open(...)
   query = catalog.search(...)
   xx = odc.stac.load(
       query.items(),
       bands=["red", "green", "blue"],
       resolution=100,
   )
   xx.red.plot.imshow(col="time")


See :py:func:`odc.stac.load`.


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


From unreleased source
**********************

Using latest unreleased code in ``conda`` is also possible. It's best to install
dependencies using conda then install ``odc-stac`` with pip. Sample
``environment.yml`` is provided below.


.. code-block:: yaml

   channels:
     - conda-forge
   dependencies:
     - odc-geo >=0.1.3
     - xarray >=0.20.1
     - numpy
     - dask
     - pandas
     - affine
     - rasterio
     - boto3
     - toolz
     - pystac
     - pystac-client
     - pip =20
     - pip:
       - git+https://github.com/opendatacube/odc-stac/
