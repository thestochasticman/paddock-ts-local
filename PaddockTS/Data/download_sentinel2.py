from dask.distributed import Client as DaskClient
from xarray.core.dataset import Dataset
from PaddockTS.query import Query
from os.path import exists
from os import makedirs
import pystac_client
import odc.stac
import pickle
import rioxarray
import numpy as np

def apply_cloud_mask(ds: Dataset, cloud_band: str = 'oa_fmask') -> Dataset:
    """
    Apply cloud masking to a dataset using the fmask band.

    fmask values:
        0 = nodata
        1 = valid (clear)
        2 = cloud
        3 = shadow
        4 = snow
        5 = water

    This function masks out clouds (2) and shadows (3), setting those pixels to NaN.

    Args:
        ds: xarray Dataset with cloud mask band
        cloud_band: Name of the cloud mask band

    Returns:
        Dataset with cloudy pixels set to NaN
    """
    if cloud_band not in ds.data_vars:
        print(f"Warning: {cloud_band} band not found, skipping cloud masking")
        return ds

    # Create mask: True where pixels are clear (not cloud=2 or shadow=3)
    fmask = ds[cloud_band]
    clear_mask = (fmask != 2) & (fmask != 3)

    # Apply mask to all other bands
    masked_ds = ds.drop_vars(cloud_band)
    for var in masked_ds.data_vars:
        masked_ds[var] = masked_ds[var].where(clear_mask)

    return masked_ds


def drop_bad_frames(ds: Dataset, max_nan_fraction: float = 0.20) -> Dataset:
    """
    Drop time steps where NaN fraction exceeds threshold.

    Args:
        ds: xarray Dataset with time dimension
        max_nan_fraction: Maximum allowed NaN fraction (0-1). Default 0.20 = 20%.
            Frames with more NaNs than this are dropped.

    Returns:
        Dataset with bad frames removed
    """
    # Use first data variable to check NaN fraction
    first_var = list(ds.data_vars)[0]
    data = ds[first_var].values  # (time, y, x)

    good_times = []
    for t in range(len(ds.time)):
        frame = data[t]
        nan_fraction = np.isnan(frame).sum() / frame.size
        if nan_fraction < max_nan_fraction:
            good_times.append(t)

    n_dropped = len(ds.time) - len(good_times)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} frames with >{max_nan_fraction*100:.0f}% NaN")

    return ds.isel(time=good_times)


def download_sentinel2(
    query: Query,
    num_workers: int = 4,
    threads_per_worker: int = 2,
    tile_width: int = 1024,
    tile_height: int = 1024,
    tile_time_series_length: int = 1,
    apply_cloud_masking: bool = True,
) -> Dataset:
    """
    Perform a STAC query, load the resulting data into an xarray.Dataset,
    and save it to disk as a pickle.

    Args:
        query (Query): The query parameters.
        stub (str or None): Optional filename stub; if None, uses query.get_stub().
        num_workers (int): Dask worker process count.
        threads_per_worker (int): Dask threads per worker.
        tile_width (int): Chunk width in pixels.
        tile_height (int): Chunk height in pixels.
        tile_time_series_length (int): Time-chunk length.

    Returns:
        Dataset: The loaded xarray Dataset (also saved to `{DS2_DIR}/{stub}.pkl`).
    """

    makedirs(query.stub_tmp_dir, exist_ok=True)
    # makedirs(query.stub_out_dir, exist_ok=True)
    
    dask_client = DaskClient(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker
    )

    catalog = pystac_client.Client.open('https://explorer.dea.ga.gov.au/stac')
    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={'aws_unsigned': True},
    )
    query_results = catalog.search(
        bbox=query.bbox,
        collections=query.collections,
        datetime=query.datetime,
        filter=query.filter.to_dict()
    )
    items = list(query_results.items())

    # Add fmask band if cloud masking is enabled and not already included
    bands_to_load = list(query.bands)
    if apply_cloud_masking and 'oa_fmask' not in bands_to_load:
        bands_to_load.append('oa_fmask')

    ds2_pipeline = odc.stac.load(
        items,
        bands=bands_to_load,
        crs=query.crs,
        resolution=query.resolution,
        groupby=query.groupby,
        bbox=query.bbox,
        chunks={
            'time': tile_time_series_length,
            'x': tile_width,
            'y': tile_height
        }
    )
    future = dask_client.compute(ds2_pipeline)
    ds2: Dataset = future.result()
    dask_client.close()
    ds2 = ds2.rio.write_crs(query.crs)

    # Apply cloud masking if requested
    if apply_cloud_masking:
        ds2 = apply_cloud_mask(ds2)
        ds2 = drop_bad_frames(ds2, max_nan_fraction=0.20)

    with open(query.path_ds2, 'wb') as handle:
        pickle.dump(ds2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ds2

def test_returned_dataset_values(query: Query) -> bool:
    """
    Load a saved dataset pickle and verify that:
      - its time values fall within the query's date range
      - it contains exactly the bands requested in the query

    Args:
        query (Query): Provides `start_time`, `end_time`, and `bands`.

    Returns:
        bool: True if both date-range and band checks pass, False otherwise.
    """
    from pandas import to_datetime
    load_pickle = lambda p: pickle.load(open(p, 'rb'))
    dataset: Dataset = load_pickle(query.path_ds2)
    returned_dates = [to_datetime(ts).date() for ts in dataset.time.values]
    if min(returned_dates) < query.start_time or max(returned_dates) > query.end_time:
        return False
    if not all(band in dataset.data_vars for band in query.bands):
        return False
    print(query.path_ds2)
    return True

def test() -> bool:
    """
    Run an end-to-end test:
      1. Remove any existing pickle for the example query.
      2. Download & save a new dataset.
      3. Verify the file exists and its contents match expectations.

    Returns:
        bool: True if both path-existence and data-value tests pass.
    """
    from PaddockTS.query import get_example_query

    query = get_example_query()
    download_sentinel2(query)
    return test_returned_dataset_values(query)

if __name__ == '__main__':
    print(test())
