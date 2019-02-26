""" reader
"""
from typing import (
    List, Optional, Union, Any,
    Iterator, Tuple,
)
import numpy as np
from affine import Affine
from concurrent.futures import ThreadPoolExecutor
import rasterio
from rasterio.io import DatasetReader
from rasterio.crs import CRS as RioCRS
from types import SimpleNamespace
from datetime import datetime

from datacube.storage import BandInfo
from datacube.utils.geometry import CRS
from datacube.utils import (
    uri_to_local_path,
    get_part_from_uri,
    datetime_to_seconds_since_1970,
)
from datacube.drivers._types import (
    ReaderDriverEntry,
    ReaderDriver,
    GeoRasterReader,
    FutureGeoRasterReader,
    FutureNdarray,
    RasterShape,
    RasterWindow,
)


def _is_netcdf(fmt: str) -> bool:
    return fmt == 'NetCDF'


def _dc_crs(crs: Optional[RioCRS]) -> Optional[CRS]:
    """ Convert RIO version of CRS to datacube
    """
    if crs is None:
        return None
    if crs.is_epsg_code:
        return CRS('epsg:{}'.format(crs.to_epsg()))
    return CRS(crs.wkt)


def _read(src: DatasetReader,
          window: Optional[RasterWindow],
          out_shape: Optional[RasterShape]) -> np.ndarray:
    raise NotImplementedError("TODO: read")


def _rio_uri(band: BandInfo) -> str:
    """
    - file uris are converted to file names
       - if also netcdf wrap in NETCDF:"${filename}":${layer}
    - All other protocols go through unmodified
    """
    if band.uri_scheme == 'file':
        fname = str(uri_to_local_path(band.uri))

        if _is_netcdf(band.format):
            fname = 'NETCDF:"{}":{}'.format(fname, band.layer)

        return fname

    return band.uri


def _find_netcdf_band_by_time(src: DatasetReader, ts: datetime) -> int:
    """ backwards compatibility code.

        finds band that is nearest to a given timestamp
    """
    time_tag = 'NETCDF_DIM_time'
    ts0 = datetime_to_seconds_since_1970(ts)

    def get_ts() -> Iterator[Tuple[int, float]]:
        for bidx in range(1, src.count+1):
            tag_value = src.get_tag_item(time_tag, bidx=bidx)
            if tag_value is not None:
                yield (bidx, float(tag_value))

    all_ts = list((bidx, abs(ts - ts0))
                  for bidx, ts in get_ts())

    if len(all_ts) == 0:
        return 1  # TODO: copying previous behaviour, should at least log something

    all_ts.sort(key=lambda xx: xx[1])
    bidx, _ = all_ts[0]
    return bidx


def _rio_band_idx(band: BandInfo, src: DatasetReader) -> int:
    if band.band is not None:
        return band.band

    if not _is_netcdf(band.format):
        return 1

    bidx = get_part_from_uri(band.uri)
    if bidx is not None:
        return bidx

    if src.count == 1:  # Single-slice netcdf file
        return 1

    return _find_netcdf_band_by_time(src, band.center_time)


class RIOReader(GeoRasterReader):
    def __init__(self, src: DatasetReader,
                 band_idx: int,
                 pool: ThreadPoolExecutor):
        # TODO: add overrides for CRS/Transform/nodata
        self._src = src
        self._crs = _dc_crs(src.crs)
        self._band_idx = band_idx
        self._pool = pool
        self._meta = SimpleNamespace(**src.meta)

    @property
    def crs(self) -> Optional[CRS]:
        return self._crs

    @property
    def transform(self) -> Optional[Affine]:
        return self._meta.transform

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._meta.dtype)

    @property
    def shape(self) -> RasterShape:
        return self._src.shape

    @property
    def nodata(self) -> Optional[Union[int, float]]:
        return self._meta.nodata

    def read(self,
             window: Optional[RasterWindow] = None,
             out_shape: Optional[RasterShape] = None) -> FutureNdarray:
        return self._pool.submit(_read, self._src, window, out_shape)


def _rdr_open(band: BandInfo, ctx: Any, pool: ThreadPoolExecutor) -> RIOReader:
    """ Open file pointed by BandInfo and return RIOReader instance.

        raises Exception on failure

        TODO: start using ctx for handle cache
    """
    normalised_uri = _rio_uri(band)
    src = rasterio.open(normalised_uri, 'r')
    bidx = _rio_band_idx(band, src)

    return RIOReader(src, bidx, pool)


class RIORdrDriver(ReaderDriver):
    def __init__(self, pool: ThreadPoolExecutor, cfg: dict):
        self._pool = pool
        self._cfg = cfg

    def new_load_context(self, old_ctx: Optional[Any]) -> Any:
        return None  # TODO: implement file handle cache with this

    def open(self, band: BandInfo, ctx: Any) -> FutureGeoRasterReader:
        return self._pool.submit(_rdr_open, band, ctx, self._pool)


class RDEntry(ReaderDriverEntry):
    PROTOCOLS = ['file', 'http', 'https', 's3', 'ftp', 'zip']
    FORMATS = ['GeoTIFF', 'NetCDF', 'JPEG2000']

    @property
    def protocols(self) -> List[str]:
        return RDEntry.PROTOCOLS

    @property
    def formats(self) -> List[str]:
        return RDEntry.FORMATS

    def supports(self, protocol: str, fmt: str) -> bool:
        # TODO: might need better support matrix structures

        if fmt == 'NetCDF':
            return protocol == 'file'

        return True

    def new_instance(self, cfg: dict) -> ReaderDriver:
        cfg = cfg.copy()
        pool = cfg.pop('pool', None)
        if pool is None:
            max_workers = cfg.pop('max_workers', 1)
            pool = ThreadPoolExecutor(max_workers=max_workers)
        elif not isinstance(pool, ThreadPoolExecutor):
            if not cfg.pop('allow_custom_pool', False):
                raise ValueError("External `pool` should be a `ThreadPoolExecutor`")

        return RIORdrDriver(pool, cfg)
