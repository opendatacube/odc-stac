STAC vs Open Datacube
#####################

The `Open Datacube`_ (ODC) project, on which this library is based, started before `STAC`_
spec existed. As a result ODC uses different terminology for otherwise very
similar concepts.


.. list-table::
   :header-rows: 1

   * - STAC
     - ODC
     - Description
   * - :class:`~pystac.Collection`
     - Product or :class:`~datacube.model.DatasetType`
     - Collection of observations across space and time
   * - :class:`~pystac.Item`
     - :class:`~datacube.model.Dataset`
     - Single observation (specific time and place), multi-channel
   * - :class:`~pystac.Asset`
     - :class:`~datacube.model.Measurement`
     - Component of a single observation
   * - Band_
     - :class:`~datacube.model.Measurement`
     - Pixel plane within a multi-plane asset
   * - Common Name
     - Alias
     - Refer to the same band by different names

Similarly to STAC, ODC uses several levels of hierarchy to model metadata. At
the highest level there is *Product* which is a collection of *Datasets*. Each
*Dataset* contains a set of *Measurements* and related metadata. Finally
*Measurement* describes a single plane of pixels captured at roughly the same
time. Metadata includes location of the "file" and possibly location within a
file.

Multiple Bands per File
=======================

Multiple bands in a single file are supported by both ODC and STAC, but
representation differs. In STAC another level of hierarchy is added below an
*Asset* via the [bands attribute of the EO
extension](https://github.com/stac-extensions/eo#band-object). Resources pointed
to by an *Asset* may contain more than one band of pixels, and an *Asset*
contains descriptions of those bands. In ODC, *Asset* is not modelled
explicitly, instead resource path and potential location within this resource
are properties of a *Measurement* object. It is common in STAC to have one to
one mapping between band and asset, and in that scenario ODC *Measurement* and
STAC *Asset* can be seen as equivalent.

Geo Referencing Metadata
========================

Precise geo referencing metadata is stored within a file pointed to by
*Asset*/*Measurement*, but it can also be recorded within a STAC *Item*/ODC
*Dataset* document. Having geo-referencing information at this level can enable
more efficient data access by providing spatial information without needing to
access the source (data file) itself.

In STAC, the `Projection Extension`_ is used to bring this metadata from file to
*Item* document. In STAC each band might have different projection, but in ODC
projection is a *Dataset* level property and has to be shared across all
*Measurements*. In ODC individual bands can be of different resolution and have
different footprints (usually with a lot of overlap), but **must** be in the
same projection.

Consistency Assumptions
=======================

In STAC, *Collection* is a very loose term, in theory it can point to very
heterogeneous set of *Items*. In practice *Items* are typically very similar in
structure, most contain the same set of *Assets* and bands. ODC is more strict
in that regard. ODC *Product* contains expected set of *Measurements* per
*Dataset* as well as some basic common metadata per *Measurement*, specifically
pixel data type, which is assumed to stay the same across all *Datasets* for a
given *Measurement*.

STAC equivalent would be `Item Assets`_ extension with `Raster Extension`_
inside. It describes at the *Collection* level, expected structure of *Items*
contained within.


.. _`Open Datacube`: https://www.opendatacube.org/
.. _`STAC`: https://stacspec.org/
.. _`Projection Extension`: https://github.com/stac-extensions/projection
.. _`Raster Extension`: https://github.com/stac-extensions/eo
.. _`Item Assets`: https://github.com/stac-extensions/item-assets
.. _Band: https://github.com/stac-extensions/eo#band-object
