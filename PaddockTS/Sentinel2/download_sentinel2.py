"""Download a Sentinel-2 Analysis-Ready Data (ARD) cube from a STAC catalog.

The default STAC source is Geoscience Australia's
`Digital Earth Australia <https://explorer.dea.ga.gov.au/>`_ ARD
collection (``ga_s2am_ard_3`` / ``ga_s2bm_ard_3``); both raw S3 and HTTPS
endpoints are handled. The downloaded cube is cloud-masked using the DEA
fmask band, sparsely-clear scenes are dropped via a NaN-fraction
threshold, and the result is written to ``query.sentinel2_path`` as a
Zarr v2 store.
"""

import os
import odc.stac
import rioxarray
import numpy as np
import pystac_client
from os import makedirs
from xarray import Dataset
from .sentinel2 import Sentinel2
from PaddockTS.query import Query
from .sentinel2 import defaultsentinel2
from dask.distributed import Client as DaskClient

def _s3_to_https(url: str) -> str:
    """Convert S3 URLs to HTTPS to avoid AWS auth issues."""
    if url.startswith('s3://dea-public-data/'):
        return url.replace('s3://dea-public-data/', 'https://data.dea.ga.gov.au/')
    return url

odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

def download_sentinel2(
    query: Query,
    num_workers: int = 4,
    threads_per_worker: int = 2,
    chunk_x: int = 1024,
    chunk_y: int = 1024,
    chunk_time: int = 1,
    max_nan_fraction: float = 0.20,
    sentinel2: Sentinel2 = defaultsentinel2
) -> Dataset:
    """Fetch, cloud-mask, and persist a Sentinel-2 ARD cube for ``query``.

    Searches the configured STAC catalog for items intersecting
    ``query.bbox`` over ``[query.start, query.end]``, lazily loads the
    requested bands with `odc.stac <https://odc-stac.readthedocs.io/>`_,
    materialises the cube on a local Dask cluster, then:

    1. Builds a clear-pixel mask from the fmask band (drops cloud and
       cloud-shadow pixels).
    2. Drops scenes whose remaining NaN fraction (across all bands and
       pixels) exceeds ``max_nan_fraction``.
    3. Writes the result to ``query.sentinel2_path`` as Zarr v2.

    The function temporarily restores ``OMP_NUM_THREADS`` after Dask sets
    it to ``1``, since downstream stages (PyTorch in segmentation,
    TFLite in fractional cover) need multi-threaded BLAS.

    Args:
        query: The :class:`PaddockTS.query.Query` describing the region
            and time range. ``query.tmp_dir`` is created if missing.
        num_workers: Number of Dask processes. Each holds its own copy
            of GDAL/PROJ state, so memory grows roughly linearly. Tune
            down on small machines.
        threads_per_worker: Threads per Dask worker for I/O concurrency.
        chunk_x: Chunk size along the ``x`` dimension (pixels).
        chunk_y: Chunk size along the ``y`` dimension (pixels).
        chunk_time: Chunk size along the ``time`` dimension (scenes).
            Keep at ``1`` unless you know what you're doing — most
            downstream operations are per-timestep.
        max_nan_fraction: Drop any scene whose fraction of NaN pixels
            (after cloud masking, across all bands and the AOI) exceeds
            this value. ``0.20`` keeps mostly-clear scenes only.
        sentinel2: A :class:`PaddockTS.Sentinel2.sentinel2.Sentinel2`
            config object specifying the STAC URL, collections, bands,
            CRS, resolution, and fmask values. Defaults to the bundled
            DEA Sentinel-2 config.

    Returns:
        xarray.Dataset: The cloud-masked, scene-filtered cube with dims
        ``(time, y, x)`` and the requested bands as data variables.
        Also persisted to ``query.sentinel2_path``.

    Raises:
        Exception: Re-raises any error from the underlying ``odc.stac``
            load or Dask compute, after printing a diagnostic.

    Example:
        >>> from datetime import date
        >>> from PaddockTS.query import Query
        >>> from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
        >>> q = Query(bbox=[148.46, -34.39, 148.50, -34.36],
        ...           start=date(2023, 1, 1), end=date(2023, 12, 31),
        ...           stub='milgadara')
        >>> ds = download_sentinel2(q)
    """
    bands = sentinel2.bands
    catalog = pystac_client.Client.open(sentinel2.stac_url)
    result = catalog.search(
        bbox=query.bbox,
        collections=sentinel2.collections,
        datetime=f'{query.start}/{query.end}',
        filter=sentinel2.cloud_cover_filter
    )
    omp_before = os.environ.get('OMP_NUM_THREADS')
    with DaskClient(n_workers=num_workers, threads_per_worker=threads_per_worker) as client:
        odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
        try:
            ds: Dataset = odc.stac.load(
                list(result.items()),
                bands=bands,
                crs=sentinel2.crs,
                resolution=sentinel2.resolution,
                groupby=sentinel2.groupby,
                bbox=query.bbox,
                chunks={'time': chunk_time, 'x': chunk_x, 'y': chunk_y},
                patch_url=_s3_to_https,
            )
            ds = client.compute(ds).result()
        except Exception as e:
            print(f'Creating dataset using dask failed due to: {e}')
            raise
        finally:
            client.close()
    # Dask sets OMP_NUM_THREADS=1 which cripples PyTorch in later pipeline steps
    if omp_before is None:
        os.environ.pop('OMP_NUM_THREADS', None)
    else:
        os.environ['OMP_NUM_THREADS'] = omp_before
    
    #creating cloud mask
    fmask = ds[sentinel2.cloud_mask_band]
    clear_mask = (fmask != sentinel2.fmask_cloud) & (fmask != sentinel2.fmask_shadow)
    ds = ds.drop_vars(sentinel2.cloud_mask_band).where(clear_mask)
    #dropping frames using a nan percentage threshold
    nan_frac = ds.to_array().isnull().mean(dim=['variable', 'x', 'y'])
    ds = ds.sel(time=nan_frac < max_nan_fraction)
    makedirs(query.tmp_dir, exist_ok=True)
    ds.to_zarr(query.sentinel2_path, mode='w', zarr_format=2)
    return ds


def test():
    from PaddockTS.utils import get_example_query
    download_sentinel2(get_example_query())

if __name__ == '__main__':
    test()