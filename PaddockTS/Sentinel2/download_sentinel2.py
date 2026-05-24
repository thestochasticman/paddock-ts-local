"""Download a Sentinel-2 Analysis-Ready Data (ARD) cube from a STAC catalog.

The default STAC source is Geoscience Australia's
`Digital Earth Australia <https://explorer.dea.ga.gov.au/>`_ ARD
collection (``ga_s2am_ard_3`` / ``ga_s2bm_ard_3``). The fetched cube
(raw bands including fmask, no cloud masking applied) is written to
``query.sentinel2_path`` as a Zarr v2 store. The path is keyed by
``(bbox, time)`` — same query re-runs reuse the cached file; different
bbox or different time range gets its own folder.
"""

import os
import odc.stac
import rioxarray
import numpy as np
import xarray as xr
import pystac_client
from os import makedirs
from os.path import exists
from datetime import datetime
from xarray import Dataset
from PaddockTS.query import Query
from dask.distributed import Client as DaskClient
from PaddockTS.Sentinel2.sentinel2 import Sentinel2
from PaddockTS.Sentinel2.sentinel2 import defaultsentinel2
from PaddockTS.Sentinel2.check_if_valid_zarr_exists import check_if_valid_zarr_exists


odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})


def download_sentinel2(
    query: Query,
    num_workers: int = 1,
    threads_per_worker: int = 8,
    chunk_x: int = 256,
    chunk_y: int = 256,
    chunk_time: int = 1,
    sentinel2: Sentinel2 = defaultsentinel2
) -> Dataset:
    """Fetch and persist a Sentinel-2 ARD cube for ``query``.

    If ``query.sentinel2_path`` already exists (same bbox + time range
    has been downloaded before), opens and returns it. Otherwise searches
    the STAC catalog, materialises the requested bands via odc.stac on a
    local Dask cluster, and writes the result as a Zarr v2 store at
    ``query.sentinel2_path``. Raw bands — including the fmask band —
    are written as-is; cloud masking is a downstream concern.

    The function temporarily restores ``OMP_NUM_THREADS`` after Dask sets
    it to ``1``, since downstream stages (PyTorch in segmentation,
    TFLite in fractional cover) need multi-threaded BLAS.

    Args:
        query: The :class:`PaddockTS.query.Query` describing the region
            and time range. ``query.query_dir`` is created if missing.
        num_workers: Number of Dask processes. Each holds its own copy
            of GDAL/PROJ state, so memory grows roughly linearly. Tune
            down on small machines.
        threads_per_worker: Threads per Dask worker for I/O concurrency.
        chunk_x: Chunk size along the ``x`` dimension (pixels).
        chunk_y: Chunk size along the ``y`` dimension (pixels).
        chunk_time: Chunk size along the ``time`` dimension (scenes).
            Keep at ``1`` unless you know what you're doing — most
            downstream operations are per-timestep.
        sentinel2: A :class:`PaddockTS.Sentinel2.sentinel2.Sentinel2`
            config object specifying the STAC URL, collections, bands,
            CRS, resolution, and fmask values. Defaults to the bundled
            DEA Sentinel-2 config.

    Returns:
        xarray.Dataset: The raw cube with dims ``(time, y, x)`` and the
        requested bands as data variables. Also persisted to
        ``query.sentinel2_path``.

    Raises:
        RuntimeError: STAC returned no scenes for the query.
        Exception: Re-raises any error from the underlying ``odc.stac``
            load or Dask compute, after printing a diagnostic.
    """
    success_marker = f'{query.sentinel2_path}/_SUCCESS'
    if check_if_valid_zarr_exists(query.sentinel2_path):
        try:
            return xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords='all')
        except Exception as e:
            print(f'Cache at {query.sentinel2_path} unreadable ({e}); refetching')
        # Marker present but zarr unreadable → fall through and refetch.

    # Defensive cleanup: a previous run may have written a partial zarr
    # without the _SUCCESS marker (kill-9, OOM, network drop mid-write).
    # Wipe it before refetching so we never start a fresh download on top
    # of stale chunks — that's the failure mode where odc.stac/rasterio
    # later raises confusing errors on what looks like a clean download.
    if exists(query.sentinel2_path):
        from shutil import rmtree
        rmtree(query.sentinel2_path)

    bands = sentinel2.bands
    # DEA STAC's first request after a cold cache often hits a 504; the next
    # one (~150ms later) succeeds. See diagnostics.md at the repo root.
    from urllib3 import Retry
    from pystac_client.stac_api_io import StacApiIO
    stac_io = StacApiIO(max_retries=Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[408, 429, 502, 503, 504],
        allowed_methods=['GET', 'POST'],
    ))
    catalog = pystac_client.Client.open(sentinel2.stac_url, stac_io=stac_io)
    result = catalog.search(
        bbox=query.bbox,
        collections=sentinel2.collections,
        datetime=f'{query.start}/{query.end}',
        filter=sentinel2.cloud_cover_filter
    )
    items = list(result.items())

    if not items:
        raise RuntimeError(
            f'No Sentinel-2 scenes found for bbox {query.bbox} between '
            f'{query.start} and {query.end}.'
        )

    # Dask sets OMP_NUM_THREADS=1 (and the MKL / OpenBLAS equivalents) on the
    # parent process when the cluster spins up, which would cripple PyTorch
    # in the SAM segmentation stage that runs later in the same process.
    # Snapshot the originals so we can restore them on the way out — and put
    # the restore in a `finally` so a failed download doesn't poison the env
    # for the rest of the run.
    _THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
    thread_env_before = {v: os.environ.get(v) for v in _THREAD_VARS}
    makedirs(os.path.dirname(query.sentinel2_path), exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + 'Z'
    try:
        with DaskClient(n_workers=num_workers, threads_per_worker=threads_per_worker) as client:
            odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
            try:
                ds: Dataset = odc.stac.load(
                    items,
                    bands=bands,
                    crs=sentinel2.crs,
                    resolution=sentinel2.resolution,
                    groupby=sentinel2.groupby,
                    bbox=query.bbox,
                    chunks={'time': chunk_time, 'x': chunk_x, 'y': chunk_y},
                )
                # Pin the CRS in a form rioxarray reliably picks up on re-open;
                # without this the spatial_ref coord round-trips but ds.rio.crs
                # comes back as None, which breaks downstream stages (preseg,
                # SAM, paddocks).
                ds = ds.rio.write_crs(sentinel2.crs, inplace=False)
                ds = ds.assign_attrs(downloaded_at=timestamp)
                # Stream chunks directly to disk via the active Dask client —
                # avoids materializing the full cube in the driver process. On
                # large AOIs the in-driver path (client.compute + .result)
                # would peak at the whole-cube memory footprint; this path
                # peaks at ~workers × chunk_size.
                ds.to_zarr(query.sentinel2_path, mode='w', zarr_format=2)
            except Exception as e:
                print(f'Creating dataset using dask failed due to: {e}')
                raise
    finally:
        for var, prev in thread_env_before.items():
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev

    # Touch the _SUCCESS marker *after* the zarr write completes; its presence
    # is what the next call uses as the cache-validity check, so a kill-9
    # mid-write leaves the cache invalidated.
    with open(success_marker, 'w') as f:
        f.write(timestamp)
    # Re-open from disk so the returned dataset is a stable, eager reference
    # to the persisted store — not a lazy view over the now-shut Dask cluster
    # that would re-fetch from STAC on any access.
    return xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords='all')


from PaddockTS.utils import test_internet

# Shared across the test suite so all tests with the same (bbox, time) reuse
# one downloaded zarr — cuts test runtime from ~4 downloads to ~2 (the second
# is the forced refetch in test_missing_marker_triggers_refetch).
_TEST_BBOX = [148.36265, -33.52606, 148.38265, -33.50606]
from datetime import date as _date
_TEST_START, _TEST_END = _date(2024, 1, 1), _date(2024, 1, 21)
_test_cfg = None


def _shared_test_cfg():
    global _test_cfg
    if _test_cfg is None:
        import tempfile
        from PaddockTS.config import Config
        tmpdir = tempfile.mkdtemp(prefix='paddockts_s2_test_')
        _test_cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    return _test_cfg


def test_download_writes_zarr():
    """First call for a (bbox, time) writes the zarr at query.sentinel2_path."""
    q = Query(
        bbox=_TEST_BBOX, start=_TEST_START, end=_TEST_END,
        stub='s2_write', config=_shared_test_cfg(),
    )
    ds = download_sentinel2(q)
    return exists(q.sentinel2_path) and ds.time.size > 0


def test_repeated_query_uses_cache():
    """Second call with the same query reuses the zarr (no rewrite)."""
    q = Query(
        bbox=_TEST_BBOX, start=_TEST_START, end=_TEST_END,
        stub='s2_reuse', config=_shared_test_cfg(),
    )
    download_sentinel2(q)
    mtime_before = os.path.getmtime(q.sentinel2_path)

    # Same query → cache hit, no write
    download_sentinel2(q)
    mtime_after = os.path.getmtime(q.sentinel2_path)
    return mtime_before == mtime_after


def test_different_stub_same_attrs_reuses_cache():
    """Different stub but same (bbox, time) → same sentinel2_path → cache hit."""
    cfg = _shared_test_cfg()
    qa = Query(bbox=_TEST_BBOX, start=_TEST_START, end=_TEST_END, stub='stub_aaa', config=cfg)
    download_sentinel2(qa)
    mtime_before = os.path.getmtime(qa.sentinel2_path)

    qb = Query(bbox=_TEST_BBOX, start=_TEST_START, end=_TEST_END, stub='stub_bbb', config=cfg)
    if qa.sentinel2_path != qb.sentinel2_path:
        return False

    download_sentinel2(qb)
    mtime_after = os.path.getmtime(qb.sentinel2_path)
    return mtime_before == mtime_after


def test_different_time_different_path():
    """Same bbox + different time range → different sentinel2_path."""
    cfg = _shared_test_cfg()
    qa = Query(bbox=_TEST_BBOX, start=_TEST_START, end=_TEST_END,
               stub='s2_t1', config=cfg)
    qb = Query(bbox=_TEST_BBOX, start=_date(2024, 2, 1), end=_date(2024, 2, 21),
               stub='s2_t2', config=cfg)
    # Same bbox → shared aoi_dir
    if qa.aoi_dir != qb.aoi_dir:
        return False
    # Different time → different query_dir and sentinel2_path
    return qa.sentinel2_path != qb.sentinel2_path


def test_success_marker_written():
    """A successful download leaves a _SUCCESS file inside the zarr and a
    ``downloaded_at`` attribute on the dataset."""
    q = Query(
        bbox=_TEST_BBOX, start=_TEST_START, end=_TEST_END,
        stub='s2_marker', config=_shared_test_cfg(),
    )
    download_sentinel2(q)
    marker = f'{q.sentinel2_path}/_SUCCESS'
    persisted = xr.open_zarr(q.sentinel2_path, chunks=None)
    return exists(marker) and 'downloaded_at' in persisted.attrs


def test_missing_marker_triggers_refetch():
    """A cache with the zarr present but no _SUCCESS file is re-downloaded."""
    q = Query(
        bbox=_TEST_BBOX, start=_TEST_START, end=_TEST_END,
        stub='s2_refetch', config=_shared_test_cfg(),
    )
    download_sentinel2(q)

    # Simulate a crash mid-write: zarr exists but _SUCCESS was never touched
    marker = f'{q.sentinel2_path}/_SUCCESS'
    os.remove(marker)
    mtime_before = os.path.getmtime(q.sentinel2_path)

    # Next call should treat the cache as invalid and re-download
    download_sentinel2(q)
    mtime_after = os.path.getmtime(q.sentinel2_path)
    return exists(marker) and mtime_after > mtime_before


def test():
    return all([
        test_internet(None),
        test_different_time_different_path(),
        test_download_writes_zarr(),
        test_repeated_query_uses_cache(),
        test_different_stub_same_attrs_reuses_cache(),
        test_success_marker_written(),
        test_missing_marker_triggers_refetch(),
    ])


if __name__ == '__main__':
    print(test())
