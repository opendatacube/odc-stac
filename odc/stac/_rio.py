# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
rasterio helpers
"""
import threading
from typing import Any, Dict, Optional, Union

import rasterio
import rasterio.env
from rasterio.session import AWSSession, Session

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
GDAL_CLOUD_DEFAULTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
}


class _GlobalRioConfig:
    def __init__(self) -> None:
        self._configured = False
        self._aws: Optional[Dict[str, Any]] = None
        self._gdal_opts: Dict[str, Any] = {}

    def set(
        self,
        *,
        aws: Optional[Dict[str, Any]],
        gdal_opts: Dict[str, Any],
    ):
        self._aws = {**aws} if aws is not None else None
        self._gdal_opts = {**gdal_opts}
        self._configured = True

    @property
    def configured(self) -> bool:
        return self._configured

    def env(self) -> rasterio.env.Env:
        if self._configured is False:
            return rasterio.env.Env(_local.session())

        session: Optional[Session] = None
        if self._aws is not None:
            session = AWSSession(**self._aws)
        return rasterio.env.Env(_local.session(session), **self._gdal_opts)


_CFG = _GlobalRioConfig()


class ThreadSession(threading.local):
    """
    Caches Session between rio_env calls.
    """

    def __init__(self) -> None:
        super().__init__()
        self._session: Optional[Session] = None
        self._aws: Optional[Dict[str, Any]] = None

    @property
    def configured(self) -> bool:
        return self._session is not None

    def reset(self):
        self._session = None
        self._aws = None
        if rasterio.env.hasenv():
            rasterio.env.delenv()

    def session(self, session: Union[Dict[str, Any], Session] = None) -> Session:
        if self._session is None:
            # first call in this thread
            # 1. Start GDAL environment
            rasterio.env.defenv()

            if session is None:
                # Figure out session from environment variables
                with rasterio.env.Env() as env:
                    self._session = env.session
            else:
                if isinstance(session, dict):
                    self._aws = session
                    session = AWSSession(**session)
                self._session = session

            assert self._session is not None
            return self._session

        if session is not None:
            if isinstance(session, Session):
                return session
            # TODO: cache more than one session?
            if session == self._aws:
                return self._session
            return AWSSession(**session)

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


def rio_env(session=None, **kw):
    """
    Wraps rasterio.env.Env.

    re-uses GDAL environment and session between calls.
    """
    return rasterio.env.Env(_local.session(session), **kw)


def _set_default_rio_config(
    aws: Optional[Dict[str, Any]] = None,
    cloud_defaults: bool = False,
    **kwargs,
):
    opts = {**GDAL_CLOUD_DEFAULTS, **kwargs} if cloud_defaults else {**kwargs}
    _CFG.set(aws=aws, gdal_opts=opts)


def configure_rio(
    *,
    cloud_defaults: bool = False,
    verbose: bool = False,
    aws: Optional[Dict[str, Any]] = None,
    **params,
):
    """
    Change GDAL/rasterio configuration.

    Change can be applied locally or on a Dask cluster using ``WorkerPlugin`` mechanism.

    :param cloud_defaults: When ``True`` enable common cloud settings in GDAL
    :param verbose: Dump GDAL environment settings to stdout
    :param aws: Arguments for :py:class:`rasterio.session.AWSSession`
    """
    # backward compatible flags that don't make a difference anymore
    activate = params.pop("activate", None)
    client = params.pop("client", None)
    if client is not None:
        pass
    if activate is not None:
        pass

    _set_default_rio_config(cloud_defaults=cloud_defaults, aws=aws, **params)
    if verbose:
        with _CFG.env():
            _dump_rio_config()


def _dump_rio_config():
    cfg = get_rio_env()
    nw = max(len(k) for k in cfg)
    for k, v in cfg.items():
        print(f"{k:<{nw}} = {v}")


def configure_s3_access(
    profile: Optional[str] = None,
    region_name: str = "auto",
    aws_unsigned: Optional[bool] = None,
    requester_pays: bool = False,
    cloud_defaults: bool = True,
    **gdal_opts,
):
    """
    Credentialize for S3 bucket access or configure public access.

    This function obtains credentials for S3 access and passes them on to
    processing threads, either local or on dask cluster.


    .. note::

       if credentials are STS based they will eventually expire, currently
       this case is not handled very well, reads will just start failing
       eventually and will never recover.

    :param profile:        AWS profile name to use
    :param region_name:    Default region_name to use if not configured for a given/default AWS profile
    :param aws_unsigned:   If ``True`` don't bother with credentials when reading from S3
    :param requester_pays: Needed when accessing requester pays buckets

    :param cloud_defaults: Assume files are in the cloud native format, i.e. no side-car files, disables
                           looking for side-car files, makes things faster but won't work for files
                           that do have side-car files with extra metadata.

    :param gdal_opts:      Any other option to pass to GDAL environment setup

    :returns: credentials object or ``None`` if ``aws_unsigned=True``
    """
    # pylint: disable=import-outside-toplevel
    from ._aws import get_aws_settings

    aws, creds = get_aws_settings(
        profile=profile,
        region_name=region_name,
        aws_unsigned=aws_unsigned,
        requester_pays=requester_pays,
    )

    _set_default_rio_config(aws=aws, cloud_defaults=cloud_defaults, **gdal_opts)
    return creds
