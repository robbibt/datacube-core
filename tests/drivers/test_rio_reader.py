""" Tests for new RIO reader driver
"""
import pytest
import numpy as np
import rasterio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future

from datacube.testutils import mk_sample_dataset
from datacube.drivers.rio._reader import (
    RDEntry,
    _dc_crs,
    _rio_uri,
    _rio_band_idx,
)
from datacube.storage import BandInfo
from datacube.utils import datetime_to_seconds_since_1970
from datacube.testutils.geom import SAMPLE_WKT_WITHOUT_AUTHORITY

NetCDF = 'NetCDF'    # pylint: disable=invalid-name
GeoTIFF = 'GeoTIFF'  # pylint: disable=invalid-name


def mk_band(name: str,
            base_uri: str,
            path: str = '',
            format: str = GeoTIFF,
            **extras) -> BandInfo:
    band_opts = {k: extras.pop(k)
                 for k in 'path layer band'.split() if k in extras}

    band = dict(name=name, path=path, **band_opts)
    ds = mk_sample_dataset([band], base_uri, format=format, **extras)
    return BandInfo(ds, name)


def test_rio_rd_entry():
    rde = RDEntry()

    assert 'file' in rde.protocols
    assert 's3' in rde.protocols

    assert GeoTIFF in rde.formats
    assert NetCDF in rde.formats

    assert rde.supports('file', NetCDF) is True
    assert rde.supports('s3', NetCDF) is False

    assert rde.supports('file', GeoTIFF) is True
    assert rde.supports('s3', GeoTIFF) is True

    assert rde.new_instance({}) is not None
    assert rde.new_instance({'max_workers': 2}) is not None

    with pytest.raises(ValueError):
        rde.new_instance({'pool': []})

    # check pool re-use
    pool = ThreadPoolExecutor(max_workers=1)
    rdr = rde.new_instance({'pool': pool})
    assert rdr._pool is pool


def test_rd_internals_crs():
    from rasterio.crs import CRS as RioCRS

    assert _dc_crs(None) is None
    assert _dc_crs(RioCRS.from_epsg(3857)).epsg == 3857
    assert _dc_crs(RioCRS.from_wkt(SAMPLE_WKT_WITHOUT_AUTHORITY)).epsg is None


def test_rd_internals_bidx(data_folder):
    base = "file://" + str(data_folder) + "/metadata.yml"
    bi = mk_band('a',
                 base,
                 path="multi_doc.nc",
                 format=NetCDF,
                 timestamp=datetime.utcfromtimestamp(1),
                 layer='a')
    assert bi.uri.endswith('multi_doc.nc')
    assert datetime_to_seconds_since_1970(bi.center_time) == 1

    rio_fname = _rio_uri(bi)
    assert rio_fname.startswith('NETCDF:')

    with rasterio.open(rio_fname) as src:
        # timestamp search
        bidx = _rio_band_idx(bi, src)
        assert bidx == 2

        # extract from .uri
        bi.uri = bi.uri + "#part=5"
        assert _rio_band_idx(bi, src) == 5

        # extract from .band
        bi.band = 33
        assert _rio_band_idx(bi, src) == 33

    bi = mk_band('a',
                 base,
                 path="test.tif",
                 format=GeoTIFF)

    with rasterio.open(_rio_uri(bi), 'r') as src:
        # should default to 1
        assert _rio_band_idx(bi, src) == 1

        # layer containing int should become index
        bi = mk_band('a', base, path="test.tif", format=GeoTIFF, layer=2)
        assert _rio_band_idx(bi, src) == 2

        # band is the keyword
        bi = mk_band('a', base, path="test.tif", format=GeoTIFF, band=3)
        assert _rio_band_idx(bi, src) == 3


def test_rd_internals_uri():
    base = "file:///some/path/"

    bi = mk_band('green', base, path="f.tiff", format=GeoTIFF)
    assert _rio_uri(bi) == '/some/path/f.tiff'

    bi = mk_band('x', base, path="x.nc", layer='x', format=NetCDF)
    assert _rio_uri(bi) == 'NETCDF:"/some/path/x.nc":x'

    bi = mk_band('jj', 's3://some/path/config.yml', "jj.tiff")
    assert _rio_uri(bi) == 's3://some/path/jj.tiff'
    assert _rio_uri(bi) is bi.uri


def test_rio_driver_fail_to_open():
    nosuch_uri = 'file:///this-file-hopefully/doesnot/exist-4718193.tiff'
    rde = RDEntry()
    rdr = rde.new_instance({})

    assert rdr is not None

    load_ctx = rdr.new_load_context(None)
    load_ctx = rdr.new_load_context(load_ctx)

    bi = mk_band('green', nosuch_uri)
    assert bi.uri == nosuch_uri
    fut = rdr.open(bi, load_ctx)

    assert isinstance(fut, Future)

    with pytest.raises(IOError):
        fut.result()


def test_rio_driver_open(data_folder):
    base = "file://" + str(data_folder) + "/metadata.yml"

    pool = ThreadPoolExecutor(max_workers=1)
    rde = RDEntry()
    rdr = rde.new_instance({'pool': pool})

    assert rdr is not None

    load_ctx = rdr.new_load_context(None)
    load_ctx = rdr.new_load_context(load_ctx)

    bi = mk_band('b1', base, path="test.tif", format=GeoTIFF)
    fut = rdr.open(bi, load_ctx)
    assert isinstance(fut, Future)

    rdr = fut.result()
    assert rdr.crs is not None
    assert rdr.transform is not None
    assert rdr.crs.epsg == 4326
    assert rdr.shape == (2000, 4000)
    assert rdr.nodata == -999
    assert rdr.dtype == np.dtype(np.int16)

    pool.shutdown()
