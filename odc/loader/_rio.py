# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
rasterio helpers
"""
import logging
import threading
from typing import Any, ContextManager, Dict, Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.enums
import rasterio.env
import rasterio.warp
from odc.geo.converters import rio_geobox
from odc.geo.geobox import GeoBox
from odc.geo.overlap import ReprojectInfo, compute_reproject_roi
from odc.geo.roi import NormalizedROI, roi_is_empty, roi_shape, w_
from odc.geo.warp import resampling_s2rio
from rasterio.session import AWSSession, Session

from ._reader import (
    nodata_mask,
    pick_overview,
    resolve_dst_dtype,
    resolve_dst_nodata,
    resolve_src_nodata,
    same_nodata,
)
from .types import MDParser, RasterLoadParams, RasterSource

log = logging.getLogger(__name__)

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
    "AWS_REQUEST_PAYER",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
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


class RioReader:
    """
    Protocol for readers.
    """

    def capture_env(self) -> Dict[str, Any]:
        return capture_rio_env()

    def restore_env(self, env: Dict[str, Any]) -> ContextManager[Any]:
        return rio_env(**env)

    def read(
        self,
        src: RasterSource,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        dst: Optional[np.ndarray] = None,
    ) -> Tuple[NormalizedROI, np.ndarray]:
        return rio_read(src, cfg, dst_geobox, dst=dst)

    @property
    def md_parser(self) -> Optional[MDParser]:
        return None


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
    if session is None:
        session = kw.pop("_aws", None)
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
    if not cfg:
        return
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
    try:
        from ._aws import get_aws_settings
    except ImportError as e:
        raise ImportError(
            "botocore is required to configure s3 access. "
            "Install botocore directly or via `pip install 'odc-stac[botocore]'"
        ) from e

    aws, creds = get_aws_settings(
        profile=profile,
        region_name=region_name,
        aws_unsigned=aws_unsigned,
        requester_pays=requester_pays,
    )

    _set_default_rio_config(aws=aws, cloud_defaults=cloud_defaults, **gdal_opts)
    return creds


def _reproject_info_from_rio(
    rdr: rasterio.DatasetReader, dst_geobox: GeoBox, ttol: float
) -> ReprojectInfo:
    return compute_reproject_roi(rio_geobox(rdr), dst_geobox, ttol=ttol)


def _do_read(
    src: rasterio.Band,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    rr: ReprojectInfo,
    dst: Optional[np.ndarray] = None,
) -> Tuple[NormalizedROI, np.ndarray]:
    resampling = resampling_s2rio(cfg.resampling)
    rdr = src.ds

    if dst is not None:
        _dst = dst[rr.roi_dst]  # type: ignore
    else:
        _dst = np.ndarray(
            roi_shape(rr.roi_dst), dtype=resolve_dst_dtype(src.dtype, cfg)
        )

    src_nodata0 = rdr.nodatavals[src.bidx - 1]
    src_nodata = resolve_src_nodata(src_nodata0, cfg)
    dst_nodata = resolve_dst_nodata(_dst.dtype, cfg, src_nodata)

    if roi_is_empty(rr.roi_dst):
        return (rr.roi_dst, _dst)

    if roi_is_empty(rr.roi_src):
        # no overlap case
        if dst_nodata is not None:
            np.copyto(_dst, dst_nodata)
        return (rr.roi_dst, _dst)

    if rr.paste_ok and rr.read_shrink == 1:
        rdr.read(src.bidx, out=_dst, window=w_[rr.roi_src])

        if dst_nodata is not None and not same_nodata(src_nodata, dst_nodata):
            # remap nodata from source to output
            np.copyto(_dst, dst_nodata, where=nodata_mask(_dst, src_nodata))
    else:
        # some form of reproject
        # TODO: support read with integer shrink then reproject more
        # TODO: deal with int8 inputs

        rasterio.warp.reproject(
            src,
            _dst,
            src_nodata=src_nodata,
            dst_crs=str(dst_geobox.crs),
            dst_transform=dst_geobox[rr.roi_dst].transform,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    return (rr.roi_dst, _dst)


def rio_read(
    src: RasterSource,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    dst: Optional[np.ndarray] = None,
) -> Tuple[NormalizedROI, np.ndarray]:
    """
    Internal read method.

    Returns part of the destination image that overlaps with the given source.

    .. code-block: python

       cfg, geobox, sources = ...
       mosaic = np.full(geobox.shape, cfg.fill_value, dtype=cfg.dtype)

       for src in sources:
           roi, pix = rio_read(src, cfg, geobox)
           assert mosaic[roi].shape == pix.shape
           assert pix.dtype == mosaic.dtype

           # paste over destination pixels that are empty
           np.copyto(mosaic[roi], pix, where=(mosaic[roi] == nodata))
           # OR
           mosaic[roi] = pix  # if sources are true tiles (no overlaps)

    """

    try:
        return _rio_read(src, cfg, dst_geobox, dst)
    except (
        rasterio.errors.RasterioIOError,
        rasterio.errors.RasterBlockError,
        rasterio.errors.WarpOperationError,
        rasterio.errors.WindowEvaluationError,
    ) as e:
        if cfg.fail_on_error:
            log.error(
                "Aborting load due to failure while reading: %s:%d",
                src.uri,
                src.band,
            )
            raise e
    except rasterio.errors.RasterioError as e:
        if cfg.fail_on_error:
            log.error(
                "Aborting load due to some rasterio error: %s:%d",
                src.uri,
                src.band,
            )
            raise e

    # Failed to read, but asked to continue
    log.warning("Ignoring read failure while reading: %s:%d", src.uri, src.band)

    # TODO: capture errors somehow

    if dst is not None:
        out = dst[0:0, 0:0]
    else:
        out = np.ndarray((0, 0), dtype=cfg.dtype)

    return np.s_[0:0, 0:0], out


def _rio_read(
    src: RasterSource,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    dst: Optional[np.ndarray] = None,
) -> Tuple[NormalizedROI, np.ndarray]:
    # if resampling is `nearest` then ignore sub-pixel translation when deciding
    # whether we can just paste source into destination
    ttol = 0.9 if cfg.nearest else 0.05

    with rasterio.open(src.uri, "r", sharing=False) as rdr:
        assert isinstance(rdr, rasterio.DatasetReader)
        ovr_idx: Optional[int] = None

        if src.band > rdr.count:
            raise ValueError(f"No band {src.band} in '{src.uri}'")

        rr = _reproject_info_from_rio(rdr, dst_geobox, ttol=ttol)

        if cfg.use_overviews and rr.read_shrink > 1:
            ovr_idx = pick_overview(rr.read_shrink, rdr.overviews(src.band))

        if ovr_idx is None:
            with rio_env(VSI_CACHE=False):
                return _do_read(
                    rasterio.band(rdr, src.band), cfg, dst_geobox, rr, dst=dst
                )

        # read from overview
        with rasterio.open(
            src.uri, "r", sharing=False, overview_level=ovr_idx
        ) as rdr_ovr:
            rr = _reproject_info_from_rio(rdr, dst_geobox, ttol=ttol)
            with rio_env(VSI_CACHE=False):
                return _do_read(
                    rasterio.band(rdr_ovr, src.band), cfg, dst_geobox, rr, dst=dst
                )


def capture_rio_env() -> Dict[str, Any]:
    # pylint: disable=protected-access
    if _CFG._configured:
        env = {**_CFG._gdal_opts, "_aws": _CFG._aws}
    else:
        env = {}

    env.update(get_rio_env(sanitize=False, no_session_keys=True))
    # don't want that copied across to workers who might be on different machine
    env.pop("GDAL_DATA", None)

    if len(env) == 0:
        # not customized, supply defaults
        return {**GDAL_CLOUD_DEFAULTS}

    return env
