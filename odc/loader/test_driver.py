# pylint: disable=missing-function-docstring,missing-module-docstring,too-many-statements,too-many-locals
import pytest

from ._driver import reader_driver, register_driver, unregister_driver
from ._rio import RioDriver
from .testing.fixtures import FakeReaderDriver
from .types import RasterGroupMetadata


def test_driver_load():
    drv = RioDriver()
    drv_fake = FakeReaderDriver(RasterGroupMetadata({}))

    register_driver("fake", lambda: FakeReaderDriver({}))
    register_driver("fake2", drv_fake)

    assert isinstance(reader_driver(), RioDriver)
    assert isinstance(reader_driver("rio"), RioDriver)
    assert reader_driver("odc.loader._rio.RioDriver") is not None
    assert reader_driver(drv) is drv
    assert reader_driver(drv_fake) is drv_fake
    assert reader_driver("fake2") is drv_fake
    assert reader_driver("fake") is not drv_fake
    assert isinstance(reader_driver("fake"), FakeReaderDriver)

    unregister_driver("fake2")
    with pytest.raises(ValueError):
        reader_driver("fake2")

    for bad_spec in [
        "nosuchthing",
        "odc.loader.NoSuchDriver",
        "odc.nosuchmodule.NoSuchDriver",
    ]:
        with pytest.raises(ValueError):
            reader_driver(bad_spec)
