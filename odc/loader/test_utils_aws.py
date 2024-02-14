# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=unused-argument,unused-variable,missing-module-docstring,wrong-import-position,import-error
# pylint: disable=redefined-outer-name
import json
import os
from unittest import mock

import pytest

_ = pytest.importorskip("botocore")

from ._aws import (
    _fetch_text,
    auto_find_region,
    ec2_current_region,
    get_aws_settings,
    get_creds_with_retry,
    mk_boto_session,
)
from .testing.fixtures import write_files


@pytest.fixture()
def without_aws_env(monkeypatch):
    for e in os.environ:
        if e.startswith("AWS_"):
            monkeypatch.delenv(e, raising=False)
    yield


def patch_aws(func, *args, **kw):
    return mock.patch(__package__ + "._aws." + func, *args, **kw)


def _json(**kw):
    return json.dumps(kw)


def mock_urlopen(text, code=200):
    m = mock.MagicMock()
    m.getcode.return_value = code
    m.read.return_value = text.encode("utf8")
    m.__enter__.return_value = m
    return m


def test_ec2_current_region():
    tests = [
        (None, None),
        (_json(region="TT"), "TT"),
        (_json(x=3), None),
        ("not valid json", None),
    ]

    for rv, expect in tests:
        with patch_aws("_fetch_text", return_value=rv):
            assert ec2_current_region() == expect


@patch_aws("botocore_default_region", return_value=None)
def test_auto_find_region(*mocks):
    with patch_aws("_fetch_text", return_value=None):
        with pytest.raises(ValueError):
            auto_find_region()

    with patch_aws("_fetch_text", return_value=_json(region="TT")):
        assert auto_find_region() == "TT"


@patch_aws("botocore_default_region", return_value="tt-from-botocore")
def test_auto_find_region_2(*mocks):
    assert auto_find_region() == "tt-from-botocore"


def test_fetch_text():
    with patch_aws("urlopen", return_value=mock_urlopen("", 505)):
        assert _fetch_text("http://localhost:8817") is None

    with patch_aws("urlopen", return_value=mock_urlopen("text", 200)):
        assert _fetch_text("http://localhost:8817") == "text"

    def fake_urlopen(*args, **kw):
        raise IOError("Always broken")

    with patch_aws("urlopen", fake_urlopen):
        assert _fetch_text("http://localhost:8817") is None


def test_get_aws_settings(monkeypatch, without_aws_env):
    pp = write_files(
        {
            "config": """
[default]
region = us-west-2

[profile east]
region = us-east-1
[profile no_region]
""",
            "credentials": """
[default]
aws_access_key_id = AKIAWYXYXYXYXYXYXYXY
aws_secret_access_key = fake-fake-fake
[east]
aws_access_key_id = AKIAEYXYXYXYXYXYXYXY
aws_secret_access_key = fake-fake-fake
""",
        }
    )

    assert (pp / "credentials").exists()
    assert (pp / "config").exists()

    monkeypatch.setenv("AWS_CONFIG_FILE", str(pp / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(pp / "credentials"))

    aws, creds = get_aws_settings()
    assert aws["region_name"] == "us-west-2"
    assert aws["aws_access_key_id"] == "AKIAWYXYXYXYXYXYXYXY"
    assert aws["aws_secret_access_key"] == "fake-fake-fake"

    sess = mk_boto_session(
        profile="no_region", creds=creds.get_frozen_credentials(), region_name="mordor"
    )

    assert (
        sess.get_credentials().get_frozen_credentials()
        == creds.get_frozen_credentials()
    )

    aws, creds = get_aws_settings(profile="east")
    assert aws["region_name"] == "us-east-1"
    assert aws["aws_access_key_id"] == "AKIAEYXYXYXYXYXYXYXY"
    assert aws["aws_secret_access_key"] == "fake-fake-fake"

    aws, creds = get_aws_settings(aws_unsigned=True)
    assert creds is None
    assert aws["region_name"] == "us-west-2"
    assert aws["aws_unsigned"] is True

    aws, creds = get_aws_settings(
        profile="no_region", region_name="us-west-1", aws_unsigned=True
    )

    assert creds is None
    assert aws["region_name"] == "us-west-1"
    assert aws["aws_unsigned"] is True

    with patch_aws("_fetch_text", return_value=_json(region="mordor")):
        aws, creds = get_aws_settings(profile="no_region", aws_unsigned=True)

        assert aws["region_name"] == "mordor"
        assert aws["aws_unsigned"] is True


@patch_aws("get_creds_with_retry", return_value=None)
def test_get_aws_settings_no_credentials(without_aws_env):
    # get_aws_settings should fail when credentials are not available
    with pytest.raises(ValueError, match="Couldn't get credentials"):
        aws, creds = get_aws_settings(region_name="fake")


def test_creds_with_retry():
    session = mock.MagicMock()
    session.get_credentials = mock.MagicMock(return_value=None)

    assert get_creds_with_retry(session, 2, 0.01) is None
    assert session.get_credentials.call_count == 2
