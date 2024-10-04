odc.stac
########

|Documentation Status| |Test Status| |Test Coverage| |Binder| |Discord|

Load STAC items into ``xarray`` Datasets. Process locally or distribute data
loading and computation with Dask_.

Usage
#####


odc.stac.load
~~~~~~~~~~~~~

.. code-block:: python

   catalog = pystac_client.Client.open(...)
   query = catalog.search(...)
   xx = odc.stac.load(
       query.items(),
       bands=["red", "green", "blue"],
   )
   xx.red.plot.imshow(col="time")

For more details see `Documentation`_ and `Sample Notebooks`_, or try it out on Binder_.


Installation
############

Using pip
~~~~~~~~~

.. code-block:: bash

   pip install odc-stac

To install with ``botocore`` support (for working with AWS):

.. code-block:: bash

   pip install 'odc-stac[botocore]'


Using Conda
~~~~~~~~~~~

This package is be available on ``conda-forge`` channel:

.. code-block:: bash

   conda install -c conda-forge odc-stac


From unreleased source
~~~~~~~~~~~~~~~~~~~~~~

To use development version of ``odc-stac`` install dependencies from ``conda``, then
install ``odc-stac`` with ``pip``.

Sample ``environment.yml`` is provided below.

.. code-block:: yaml

   channels:
     - conda-forge
   dependencies:
     - odc-geo
     - xarray
     - numpy
     - dask
     - pandas
     - affine
     - rasterio
     - toolz
     - pystac
     - pystac-client
     - pip
     - pip:
       - "git+https://github.com/opendatacube/odc-stac/"

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

.. |Discord| image:: https://img.shields.io/discord/1212501566326571070?label=Discord&logo=discord&logoColor=white&color=7289DA
   :target: https://discord.gg/4hhBQVas5U
   :alt: Join Discord for support

.. _Binder: https://mybinder.org/v2/gh/opendatacube/odc-stac/develop?urlpath=lab/workspaces/demo

.. _virtualenvwrapper: https://virtualenvwrapper.readthedocs.io

.. _pytest: https://docs.pytest.org

.. _mypy: http://mypy-lang.org/

.. _pylint: https://pylint.org/

.. _black: https://github.com/psf/black

.. _`Documentation`: https://odc-stac.readthedocs.io/

.. _`Sample Notebooks`: https://odc-stac.readthedocs.io/en/latest/examples.html

.. _Dask: https://dask.org/
