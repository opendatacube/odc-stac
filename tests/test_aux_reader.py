# pylint: disable=redefined-outer-name,missing-module-docstring,missing-function-docstring,missing-class-docstring
# pylint: disable=use-implicit-booleaness-not-comparison,protected-access
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.loader.types import AuxBandMetadata, AuxDataSource, AuxLoadParams

from odc.stac._mdtools import StacAuxReader
from odc.stac.model import PropertyLoadRequest, RasterBandMetadata
from odc.stac.testing.stac import b_, mk_parsed_item


class TestStacAuxReader:
    """Test cases for StacAuxReader class."""

    def test_init(self):
        """Test StacAuxReader initialization."""
        reader = StacAuxReader()
        assert reader is not None

    def test_read_basic(self):
        """Test basic reading functionality with simple data."""
        reader = StacAuxReader()

        # Create test data sources
        prop_req = PropertyLoadRequest(key="test_prop", dtype="float32", units="1")
        aux_source = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=42.0,
        )

        srcs = [[aux_source]]  # Single time step, single source
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time",)
        assert result.shape == (1,)
        assert result.values[0] == 42.0
        assert result.attrs["units"] == "1"

    def test_read_empty_sources(self):
        """Test reading with empty sources (should use fill value)."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(
            key="test_prop", dtype="float32", units="1", nodata=-999
        )

        srcs = [[]]  # Empty sources
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=-999,
            meta=AuxBandMetadata("float32", -999, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time",)
        assert result.shape == (1,)
        assert result.values[0] == -999
        assert result.attrs["units"] == "1"
        assert result.attrs["nodata"] == -999

    def test_read_multiple_sources(self):
        """Test reading with multiple sources in a single time step."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(key="test_prop", dtype="float32", units="1")

        # Multiple sources for same time step
        aux_source1 = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=10.0,
        )
        aux_source2 = AuxDataSource(
            uri="virtual://test/2",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=20.0,
        )

        srcs = [[aux_source1, aux_source2]]  # Single time step, multiple sources
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time",)
        assert result.shape == (1,)
        # Should use the fuser function to combine values
        assert result.values[0] == 15.0  # Default fuser averages values (10 + 20) / 2

    def test_read_multiple_time_steps(self):
        """Test reading with multiple time steps."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(key="test_prop", dtype="float32", units="1")

        aux_source1 = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=10.0,
        )
        aux_source2 = AuxDataSource(
            uri="virtual://test/2",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=20.0,
        )

        srcs = [[aux_source1], [aux_source2]]  # Two time steps
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")],
            dims=["time"],
            name="time",
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time",)
        assert result.shape == (2,)
        assert result.values[0] == 10.0
        assert result.values[1] == 20.0

    def test_read_with_none_values(self):
        """Test reading with None values in driver_data."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(key="test_prop", dtype="float32", units="1")

        aux_source1 = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=None,
        )
        aux_source2 = AuxDataSource(
            uri="virtual://test/2",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=20.0,
        )

        srcs = [[aux_source1, aux_source2]]
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=-999,
            meta=AuxBandMetadata("float32", -999, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time",)
        assert result.shape == (1,)
        # Should filter out None values and use only 20.0
        assert result.values[0] == 20.0

    def test_read_different_dtypes(self):
        """Test reading with different data types."""
        reader = StacAuxReader()

        # Test int32
        prop_req_int = PropertyLoadRequest(key="test_prop", dtype="int32", units="1")
        aux_source_int = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("int32", None, "1", driver_data=prop_req_int),
            driver_data=42,
        )

        srcs = [[aux_source_int]]
        cfg = AuxLoadParams(
            dtype="int32",
            fill_value=None,
            meta=AuxBandMetadata("int32", None, "1", driver_data=prop_req_int),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert result.dtype == np.int32
        assert result.values[0] == 42

    def test_read_with_custom_fuser(self):
        """Test reading with custom fuser function."""
        reader = StacAuxReader()

        def custom_fuser(values):
            return sum(values)  # Sum all values

        prop_req = PropertyLoadRequest(
            key="test_prop", dtype="float32", units="1", fuser=custom_fuser
        )

        aux_source1 = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=10.0,
        )
        aux_source2 = AuxDataSource(
            uri="virtual://test/2",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=20.0,
        )

        srcs = [[aux_source1, aux_source2]]
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert result.values[0] == 30.0  # 10 + 20

    def test_read_with_nodata_fill_value(self):
        """Test reading with nodata fill value."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(
            key="test_prop", dtype="float32", units="1", nodata=-999
        )

        aux_source = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", -999, "1", driver_data=prop_req),
            driver_data=42.0,
        )

        srcs = [[aux_source]]
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=-999,
            meta=AuxBandMetadata("float32", -999, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert result.attrs["nodata"] == -999

    def test_read_with_string_data(self):
        """Test reading with string data."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(key="test_prop", dtype="object", units="1")

        aux_source = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("object", None, "1", driver_data=prop_req),
            driver_data="test_string",
        )

        srcs = [[aux_source]]
        cfg = AuxLoadParams(
            dtype="object",
            fill_value=None,
            meta=AuxBandMetadata("object", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert result.values[0] == "test_string"

    def test_read_with_mixed_data_types(self):
        """Test reading with mixed data types in sources."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(key="test_prop", dtype="float32", units="1")

        aux_source1 = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=10.0,
        )
        aux_source2 = AuxDataSource(
            uri="virtual://test/2",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data="20",  # String instead of float
        )

        srcs = [[aux_source1, aux_source2]]
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        # Should handle mixed types gracefully
        assert isinstance(result, xr.DataArray)

    def test_read_with_empty_time_coord(self):
        """Test reading with empty time coordinate."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(key="test_prop", dtype="float32", units="1")

        srcs = []  # No time steps for empty time
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=-999,
            meta=AuxBandMetadata("float32", -999, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray([], dims=["time"], name="time")
        available_coords = {"time": time_coord}
        ctx = None

        result = reader.read(srcs, cfg, used_names, available_coords, ctx)

        assert result.shape == (0,)

    def test_read_with_missing_time_coord(self):
        """Test reading with missing time coordinate."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(key="test_prop", dtype="float32", units="1")

        aux_source = AuxDataSource(
            uri="virtual://test/1",
            subdataset=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
            driver_data=42.0,
        )

        srcs = [[aux_source]]
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        available_coords = {}  # Missing time coordinate
        ctx = None

        with pytest.raises(KeyError):
            reader.read(srcs, cfg, used_names, available_coords, ctx)

    def test_read_with_invalid_cfg(self):
        """Test reading with invalid configuration."""
        reader = StacAuxReader()

        srcs = [[]]
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=None,  # Invalid: meta should not be None
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        with pytest.raises(AssertionError):
            reader.read(srcs, cfg, used_names, available_coords, ctx)

    def test_read_with_invalid_driver_data(self):
        """Test reading with invalid driver_data in cfg.meta."""
        reader = StacAuxReader()

        srcs = [[]]
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data="invalid"),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        with pytest.raises(AssertionError):
            reader.read(srcs, cfg, used_names, available_coords, ctx)

    def test_read_with_none_fill_value(self):
        """Test reading with None fill value."""
        reader = StacAuxReader()

        prop_req = PropertyLoadRequest(
            key="test_prop", dtype="float32", units="1", nodata=None
        )

        srcs = [[]]  # Empty sources
        cfg = AuxLoadParams(
            dtype="float32",
            fill_value=None,
            meta=AuxBandMetadata("float32", None, "1", driver_data=prop_req),
        )
        used_names = {"time"}
        time_coord = xr.DataArray(
            [np.datetime64("2020-01-01")], dims=["time"], name="time"
        )
        available_coords = {"time": time_coord}
        ctx = None

        # The PropertyLoadRequest.fill_value should handle None nodata
        result = reader.read(srcs, cfg, used_names, available_coords, ctx)
        assert np.isnan(result.values[0])  # Default for float32 when nodata is None


class TestMkParsedItemWithProps:
    """Test cases for mk_parsed_item with props argument."""

    def test_mk_parsed_item_with_simple_values(self):
        """Test mk_parsed_item with simple values."""

        # Test with simple values
        props = {"eo:cloud_cover": 30, "some-other": 42}
        item = mk_parsed_item(bands=[b_("B01")], props=props)

        # Check that auxiliary bands were created
        aux_bands = [(k, v) for k, v in item.bands.items() if k[0] == "_stac_metadata"]
        assert len(aux_bands) == 2

        # Check that driver_data contains the actual values
        for _, source in aux_bands:
            assert isinstance(source, AuxDataSource)
            assert source.meta.driver_data is not None
            assert isinstance(source.meta.driver_data, PropertyLoadRequest)
            assert source.driver_data in [30, 42]  # Actual values

        # Check that aliases were created
        assert "eo_cloud_cover" in item.collection.meta.aliases
        assert "some_other" in item.collection.meta.aliases

    def test_mk_parsed_item_with_tuple_config(self):
        """Test mk_parsed_item with tuple format: (value, config_dict)."""

        # Test with tuple format: (value, config_dict)
        props = {
            "eo:cloud_cover": (
                30,
                {
                    "key": "eo:cloud_cover",
                    "dtype": "float32",
                    "units": "percent",
                    "nodata": -999,
                },
            ),
            "some-other": (
                42,
                {"key": "some-other", "dtype": "int32", "name": "custom_alias"},
            ),
        }
        item = mk_parsed_item(bands=[b_("B01")], props=props)

        # Check that auxiliary bands were created
        aux_bands = [(k, v) for k, v in item.bands.items() if k[0] == "_stac_metadata"]
        assert len(aux_bands) == 2

        # Check that driver_data contains the actual values
        for _, source in aux_bands:
            assert isinstance(source, AuxDataSource)
            assert source.meta.driver_data is not None
            assert isinstance(source.meta.driver_data, PropertyLoadRequest)
            assert source.driver_data in [30, 42]  # Actual values

        # Check that metadata was properly set
        for _, source in aux_bands:
            prop_req = source.meta.driver_data
            if prop_req.key == "eo:cloud_cover":
                assert prop_req.dtype == "float32"
                assert prop_req.units == "percent"
                assert prop_req.nodata == -999
                assert source.driver_data == 30
            elif prop_req.key == "some-other":
                assert prop_req.dtype == "int32"
                assert prop_req.output_name == "custom_alias"
                assert source.driver_data == 42

        # Check aliases
        assert "eo_cloud_cover" in item.collection.meta.aliases
        assert "custom_alias" in item.collection.meta.aliases

    def test_mk_parsed_item_with_mixed_formats(self):
        """Test mk_parsed_item with mixed simple values and tuple configs."""

        # Test with mixed formats
        props = {
            "simple_prop": 25,
            "tuple_prop": (
                0.8,
                {"key": "tuple_prop", "dtype": "float64", "nodata": -999},
            ),
            "another_simple": 100,
        }
        item = mk_parsed_item(bands=[b_("B01")], props=props)

        # Check that auxiliary bands were created
        aux_bands = [(k, v) for k, v in item.bands.items() if k[0] == "_stac_metadata"]
        assert len(aux_bands) == 3

        # Check that driver_data contains the actual values
        for _, source in aux_bands:
            assert isinstance(source, AuxDataSource)
            assert source.meta.driver_data is not None
            assert isinstance(source.meta.driver_data, PropertyLoadRequest)
            assert source.driver_data in [25, 0.8, 100]  # Actual values

        # Check that metadata was properly set
        for _, source in aux_bands:
            prop_req = source.meta.driver_data
            if prop_req.key == "simple_prop":
                assert source.driver_data == 25
            elif prop_req.key == "tuple_prop":
                assert prop_req.dtype == "float64"
                assert prop_req.nodata == -999
                assert source.driver_data == 0.8
            elif prop_req.key == "another_simple":
                assert source.driver_data == 100

        # Check aliases
        assert "simple_prop" in item.collection.meta.aliases
        assert "tuple_prop" in item.collection.meta.aliases
        assert "another_simple" in item.collection.meta.aliases

    def test_mk_parsed_item_without_props(self):
        """Test mk_parsed_item without props (should work as before)."""

        item = mk_parsed_item(bands=[b_("B01")])

        # Should not have any auxiliary bands
        aux_bands = [(k, v) for k, v in item.bands.items() if k[0] == "_stac_metadata"]
        assert len(aux_bands) == 0

        # Should not have any aliases for auxiliary bands
        aux_aliases = [
            k
            for k, v in item.collection.meta.aliases.items()
            if v and v[0][0] == "_stac_metadata"
        ]
        assert len(aux_aliases) == 0

    def test_mk_parsed_item_props_output_names(self):
        """Test that output names are correctly generated from props."""

        props = {
            "cloud.cover": 25,  # Should become "cloud_cover"
            "vegetation:index": 0.8,  # Should become "vegetation_index"
            "temp-data": 15,  # Should become "temp_data"
        }
        item = mk_parsed_item(bands=[b_("B01")], props=props)

        # Check aliases
        assert "cloud_cover" in item.collection.meta.aliases
        assert "vegetation_index" in item.collection.meta.aliases
        assert "temp_data" in item.collection.meta.aliases

    def test_mk_parsed_item_props_metadata(self):
        """Test that auxiliary band metadata is correctly set."""

        props = {"cloud_cover": 25}
        item = mk_parsed_item(bands=[b_("B01")], props=props)

        # Find the auxiliary band
        aux_band = None
        for k, v in item.bands.items():
            if k[0] == "_stac_metadata":
                aux_band = v
                break

        assert aux_band is not None
        assert isinstance(aux_band, AuxDataSource)
        assert aux_band.meta.driver_data is not None
        assert aux_band.meta.driver_data.key == "cloud_cover"
        assert aux_band.driver_data == 25

    def test_mk_parsed_item_raster_group_metadata_includes_aux_bands(self):
        """Test that RasterGroupMetadata includes auxiliary band metadata."""

        props = {
            "cloud_cover": 25,
            "vegetation_index": 0.8,
        }
        item = mk_parsed_item(bands=[b_("B01")], props=props)

        # Check that auxiliary bands are included in the metadata
        aux_bands_in_metadata = {
            k: v
            for k, v in item.collection.meta.bands.items()
            if k[0] == "_stac_metadata"
        }
        assert len(aux_bands_in_metadata) == 2

        # Check that they are AuxBandMetadata objects
        for k, v in aux_bands_in_metadata.items():
            assert isinstance(v, AuxBandMetadata)
            assert v.driver_data is not None
            assert isinstance(v.driver_data, PropertyLoadRequest)

        # Check that raster bands are also included
        raster_bands_in_metadata = {
            k: v
            for k, v in item.collection.meta.bands.items()
            if k[0] != "_stac_metadata"
        }
        assert len(raster_bands_in_metadata) == 1  # B01
        for k, v in raster_bands_in_metadata.items():
            assert isinstance(v, RasterBandMetadata)

        # Check that aux_bands property works
        aux_bands = item.collection.meta.aux_bands
        assert len(aux_bands) == 2
        for k, v in aux_bands.items():
            assert isinstance(v, AuxBandMetadata)

        # Check that raster_bands property works
        raster_bands = item.collection.meta.raster_bands
        assert len(raster_bands) == 1
        for k, v in raster_bands.items():
            assert isinstance(v, RasterBandMetadata)
