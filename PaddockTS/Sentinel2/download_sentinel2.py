"""
Sentinel-2 data download and processing module.

This module provides functionality to query, download, and process Sentinel-2
imagery from the DEA STAC catalog, including cloud masking and quality filtering.
"""

from __future__ import annotations

import pickle
from contextlib import contextmanager
from os import makedirs
from typing import TYPE_CHECKING, Generator

import numpy as np
import odc.stac
import pystac_client
import rioxarray  # noqa: F401 - required for rio accessor
from dask.distributed import Client as DaskClient
from xarray import Dataset

if TYPE_CHECKING:
    from PaddockTS.query import Query

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEA_STAC_URL = "https://explorer.dea.ga.gov.au/stac"
CLOUD_MASK_BAND = "oa_fmask"

# fmask values
FMASK_NODATA = 0
FMASK_VALID = 1
FMASK_CLOUD = 2
FMASK_SHADOW = 3
FMASK_SNOW = 4
FMASK_WATER = 5

# Default STAC collections (Sentinel-2A and 2B)
DEFAULT_COLLECTIONS = [
    "ga_s2am_ard_3",
    "ga_s2bm_ard_3",
]

# Default Sentinel-2 bands
DEFAULT_BANDS = [
    "nbart_blue",
    "nbart_green",
    "nbart_red",
    "nbart_red_edge_1",
    "nbart_red_edge_2",
    "nbart_red_edge_3",
    "nbart_nir_1",
    "nbart_nir_2",
    "nbart_swir_2",
    "nbart_swir_3",
]

# Default cloud cover filter (< 10%)
DEFAULT_FILTER = {"op": "<", "args": [{"property": "eo:cloud_cover"}, 0.10]}


# -----------------------------------------------------------------------------
# Dask Client Management
# -----------------------------------------------------------------------------

@contextmanager
def dask_client(
    num_workers: int = 4,
    threads_per_worker: int = 2,
) -> Generator[DaskClient, None, None]:
    """
    Context manager for Dask distributed client.

    Args:
        num_workers: Number of Dask worker processes.
        threads_per_worker: Threads per worker.

    Yields:
        Configured DaskClient instance.
    """
    client = DaskClient(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker,
    )
    try:
        yield client
    finally:
        client.close()


# -----------------------------------------------------------------------------
# STAC Query Functions
# -----------------------------------------------------------------------------

def configure_stac() -> pystac_client.Client:
    """
    Configure and return a STAC client for DEA.

    Returns:
        Configured pystac Client.
    """
    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={"aws_unsigned": True},
    )
    return pystac_client.Client.open(DEA_STAC_URL)


def search_stac(
    catalog: pystac_client.Client,
    query: Query,
    collections: list[str],
) -> list:
    """
    Search STAC catalog for items matching query parameters.

    Args:
        catalog: STAC client instance.
        query: Query parameters.
        collections: STAC collection IDs to search.

    Returns:
        List of STAC items.
    """
    results = catalog.search(
        bbox=query.bbox,
        collections=collections,
        datetime=query.datetime,
        filter=DEFAULT_FILTER,
    )
    return list(results.items())


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_stac_data(
    items: list,
    query: Query,
    bands: list[str],
    chunks: dict[str, int],
) -> Dataset:
    """
    Load STAC items into an xarray Dataset.

    Args:
        items: List of STAC items to load.
        query: Query parameters for CRS, resolution, etc.
        bands: List of band names to load.
        chunks: Dask chunk sizes {'time': n, 'x': n, 'y': n}.

    Returns:
        Lazy xarray Dataset.
    """
    return odc.stac.load(
        items,
        bands=bands,
        crs=query.crs,
        resolution=query.resolution,
        groupby=query.groupby,
        bbox=query.bbox,
        chunks=chunks,
    )


def compute_dataset(ds: Dataset, client: DaskClient) -> Dataset:
    """
    Compute a lazy Dataset using Dask.

    Args:
        ds: Lazy xarray Dataset.
        client: Dask client for computation.

    Returns:
        Computed (in-memory) Dataset.
    """
    future = client.compute(ds)
    return future.result()


# -----------------------------------------------------------------------------
# Cloud Masking
# -----------------------------------------------------------------------------

def apply_cloud_mask(
    ds: Dataset,
    cloud_band: str = CLOUD_MASK_BAND,
) -> Dataset:
    """
    Apply cloud masking using the fmask band.

    Masks out cloud (2) and shadow (3) pixels by setting them to NaN.

    Args:
        ds: Dataset with cloud mask band.
        cloud_band: Name of the cloud mask band.

    Returns:
        Dataset with cloudy pixels set to NaN, cloud band removed.
    """
    if cloud_band not in ds.data_vars:
        print(f"Warning: {cloud_band} band not found, skipping cloud masking")
        return ds

    fmask = ds[cloud_band]
    clear_mask = (fmask != FMASK_CLOUD) & (fmask != FMASK_SHADOW)

    masked_ds = ds.drop_vars(cloud_band)
    for var in masked_ds.data_vars:
        masked_ds[var] = masked_ds[var].where(clear_mask)

    return masked_ds


def drop_bad_frames(
    ds: Dataset,
    max_nan_fraction: float = 0.20,
) -> Dataset:
    """
    Drop time steps where NaN fraction exceeds threshold.

    Args:
        ds: Dataset with time dimension.
        max_nan_fraction: Maximum allowed NaN fraction (0-1).

    Returns:
        Dataset with bad frames removed.
    """
    first_var = list(ds.data_vars)[0]
    data = ds[first_var].values

    good_times = []
    for t in range(len(ds.time)):
        frame = data[t]
        nan_fraction = np.isnan(frame).sum() / frame.size
        if nan_fraction < max_nan_fraction:
            good_times.append(t)

    n_dropped = len(ds.time) - len(good_times)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} frames with >{max_nan_fraction * 100:.0f}% NaN")

    return ds.isel(time=good_times)


# -----------------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------------

def save_dataset(ds: Dataset, path: str) -> None:
    """
    Save Dataset to pickle file.

    Args:
        ds: Dataset to save.
        path: Output file path.
    """
    with open(path, "wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(path: str) -> Dataset:
    """
    Load Dataset from pickle file.

    Args:
        path: Path to pickle file.

    Returns:
        Loaded Dataset.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Main Download Function
# -----------------------------------------------------------------------------

def download_sentinel2(
    query: Query,
    num_workers: int = 4,
    threads_per_worker: int = 2,
    chunk_x: int = 1024,
    chunk_y: int = 1024,
    chunk_time: int = 1,
    apply_cloud_masking: bool = True,
    max_nan_fraction: float = 0.20,
) -> Dataset:
    """
    Download Sentinel-2 data from DEA STAC catalog.

    Queries the DEA STAC catalog, loads data using Dask for parallel
    processing, optionally applies cloud masking, and saves to disk.

    Args:
        query: Query parameters (bbox, datetime, bands, etc.).
        num_workers: Dask worker process count.
        threads_per_worker: Threads per Dask worker.
        chunk_x: Chunk width in pixels.
        chunk_y: Chunk height in pixels.
        chunk_time: Time chunk length.
        apply_cloud_masking: Whether to mask clouds and shadows.
        max_nan_fraction: Max NaN fraction for frame filtering.

    Returns:
        Processed xarray Dataset (also saved to query.path_ds2).
    """
    makedirs(query.stub_tmp_dir, exist_ok=True)

    # Prepare bands list
    bands = list(DEFAULT_BANDS)
    if apply_cloud_masking and CLOUD_MASK_BAND not in bands:
        bands.append(CLOUD_MASK_BAND)

    # Query STAC catalog
    catalog = configure_stac()
    items = search_stac(catalog, query, DEFAULT_COLLECTIONS)

    # Load and compute data
    chunks = {"time": chunk_time, "x": chunk_x, "y": chunk_y}

    with dask_client(num_workers, threads_per_worker) as client:
        ds_lazy = load_stac_data(items, query, bands, chunks)
        ds = compute_dataset(ds_lazy, client)

    # Set CRS
    ds = ds.rio.write_crs(query.crs)

    # Apply cloud masking
    if apply_cloud_masking:
        ds = apply_cloud_mask(ds)
        ds = drop_bad_frames(ds, max_nan_fraction)

    # Save and return
    save_dataset(ds, query.path_ds2)
    return ds
