Best Practices
##############

:mod:`odc.stac` can operate on STAC items with only minimal information present,
however user experience is best when following information is included:
``data_type`` and ``nodata`` from `Raster Extension`_, ``proj:{shape,transform,epsg}``
from `Projection Extension`_.

For a full list of understood extension elements see table below.

.. list-table::

   * - `Raster Extension`_
     -
   * - ``data_type``
     - used to determine output pixel type
   * - ``nodata``
     - used when combining multiple items into one raster plane
   * - ``unit``
     - passed on as an attribute
       (can be useful for further processing)
   * - *[planned]* ``scale``, ``offset``
     - currently ignored, but will be supported in the future

   * - `Projection Extension`_
     -
   * - ``proj:shape``
     - contains image size per asset
   * - ``proj:transform``
     - contains geo-registration per asset
   * - ``proj:epsg``
     - contains native CRS
   * - ``proj:wkt2``, ``proj:projjson``
     - can be used instead of ``proj:epsg`` for CRS without EPSG code
   * - `Electro Optical Extension`_
     -
   * - ``eo:bands.common_name``
     - used to assign an alias for a band
       (use ``red`` instead of ``B04``).


Assumptions
===========

Items from the same collection are assumed to have the same number and names of
bands, and bands are assumed to use the same ``data_type`` across the
collection.

It is assumed that Assets within a single Item share common native projection.

.. _`Raster Extension`: https://github.com/stac-extensions/eo
.. _`Projection Extension`: https://github.com/stac-extensions/eo
.. _`Electro Optical Extension`: https://github.com/stac-extensions/eo
