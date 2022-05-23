# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
rasterio helpers 
"""
import threading
from typing import Any, Dict, Optional

import rasterio
import rasterio.env
from rasterio.session import Session

SECRET_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_ACCESS_TOKEN",
    "AZURE_STORAGE_ACCESS_KEY",
    "AZURE_STORAGE_SAS_TOKEN",
    "AZURE_SAS",
    "GS_ACCESS_KEY_ID",
    "GS_SECRET_ACCESS_KEY",
    "OSS_ACCESS_KEY_ID",
    "OSS_SECRET_ACCESS_KEY",
    "SWIFT_AUTH_TOKEN",
)

SESSION_KEYS = (
    *SECRET_KEYS,
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_S3_ENDPOINT",
    "AWS_NO_SIGN_REQUEST",
    "AZURE_STORAGE_ACCOUNT",
    "AZURE_NO_SIGN_REQUEST",
    "OSS_ENDPOINT",
    "SWIFT_STORAGE_URL",
)


class ThreadSession(threading.local):
    """
    Caches Session between rio_env calls.
    """

    def __init__(self) -> None:
        self._session: Optional[Session] = None

    def session(self, session: Optional[Session] = None) -> Session:
        if self._session is None:
            # first call in this thread
            # 1. Start GDAL environment
            rasterio.env.defenv()

            if session is None:
                # Figure out session from environment variables
                with rasterio.env.Env() as env:
                    self._session = env.session
            else:
                self._session = session

        if session is not None:
            return session

        assert self._session is not None
        return self._session


_local = ThreadSession()


def _sanitize(opts, keys):
    return {k: (v if k not in keys else "xx..xx") for k, v in opts.items()}


def get_rio_env(sanitize: bool = True, no_session_keys: bool = False) -> Dict[str, Any]:
    """
    Get GDAL params configured by rasterio for the current thread.

    :param sanitize: If True replace sensitive Values with 'x'
    :param no_session_keys: Remove keys that need to be supplied via Session classes.
    """

    if not rasterio.env.hasenv():
        return {}

    opts = rasterio.env.getenv()
    if no_session_keys:
        opts = {k: v for k, v in opts.items() if k not in SESSION_KEYS}
    if sanitize:
        opts = _sanitize(opts, SECRET_KEYS)

    return opts


def rio_env(session=None, *args, **kw):
    """
    Wraps rasterio.env.Env.

    re-uses GDAL environment and session between calls.
    """
    return rasterio.env.Env(_local.session(session), *args, **kw)
