.. _api-reference:

API Reference
#############

.. highlight:: python
.. py:module:: odc.stac
.. py:module:: odc.stac.bench
.. py:module:: odc.stac.eo3


odc.stac
********

.. currentmodule:: odc.stac
.. autosummary::
   :toctree: _api/

   load
   configure_rio
   configure_s3_access
   parse_item
   parse_items
   extract_collection_metadata
   output_geobox

odc.stac.ParsedItem
*******************

.. currentmodule:: odc.stac
.. autosummary::
   :toctree: _api/

   ParsedItem
   ParsedItem.assets
   ParsedItem.crs
   ParsedItem.geoboxes
   ParsedItem.image_geometry
   ParsedItem.resolve_bands
   ParsedItem.safe_geometry
   ParsedItem.solar_date_at
   ParsedItem.strip

   RasterBandMetadata
   RasterCollectionMetadata
   RasterLoadParams
   RasterSource

odc.stac.bench
**************

.. currentmodule:: odc.stac.bench
.. autosummary::
   :toctree: _api/

   BenchmarkContext
   BenchLoadParams

   dump_site
   load_from_json
   run_bench
   load_results

odc.stac.eo3
************

.. currentmodule:: odc.stac.eo3
.. autosummary::
   :toctree: _api/

   stac2ds
