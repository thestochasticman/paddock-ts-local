from dask.distributed import Client as DaskClient
from PaddockTSLocal.legend import DS2_DIR
from xarray.core.dataset import Dataset
from PaddockTSLocal.query import Query
from typing_extensions import Union
from os import makedirs
import pystac_client
import odc.stac
import pickle
import rioxarray
import sys
import platform

def query_to_ds2(
    query: Query,
    num_workers: int = 4,
    threads_per_worker: int = 2,
    tile_width: int = 1024,
    tile_height: int = 1024,
    tile_time_series_length: int = 1
) -> Dataset:
    """
    Load data from a STAC catalog into an xarray.Dataset using parallel Dask execution.

    Args:
        query (Query)                   : Contains bbox, collections, datetime, and bands for the STAC search.
        num_workers (int)               : Number of Dask worker processes to spawn.
        threads_per_worker (int)        : Number of threads per Dask worker.
        tile_width (int)                : Chunk width (x dimension) in pixels.
        tile_height (int)               : Chunk height (y dimension) in pixels.
        tile_time_series_length (int)   : Chunk length along the time axis.

    Returns:
        xarray.core.dataset.Dataset:
            An xarray Dataset reprojected to EPSG:6933 and chunked according to parameters.
    """
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

    ds2_pipeline = odc.stac.load(
        items,
        bands=query.bands,
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
    return ds2

def save_ds2_to_file(ds2: Dataset, path: str) -> None:
    """
    Serialize and save an xarray.Dataset to disk using Python pickle.

    Args:
        ds2 (xarray.core.dataset.Dataset): The dataset to serialize.
        path (str): Filesystem path for the output pickle file.

    Returns:
        None
    """
    with open(path, 'wb') as handle:
        pickle.dump(ds2, handle, protocol=pickle.HIGHEST_PROTOCOL)


def download_ds2_from_query(
    stub: Union[str, None],
    query: Query,
    num_workers: int = 4,
    threads_per_worker: int = 2,
    tile_width: int = 1024,
    tile_height: int = 1024,
    tile_time_series_length: int = 1,
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
    ds2 = query_to_ds2(
        query=query,
        num_workers=num_workers,
        threads_per_worker=threads_per_worker,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_time_series_length=tile_time_series_length
    )
    path = f"{DS2_DIR}/{stub}.pkl"
    save_ds2_to_file(ds2=ds2, path=path)
    return ds2


def test_path_existence(query: Query) -> bool:
    """
    Check whether a previously saved dataset pickle exists on disk.

    Args:
        query (Query): Used to derive the filename via `query.get_stub()`.

    Returns:
        bool: True if the file `{DS2_DIR}/{stub}.pkl` exists, False otherwise.
    """
    from os.path import exists
    stub = query.get_stub()
    path = f"{DS2_DIR}/{stub}.pkl"
    return exists(path)


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
    stub = query.get_stub()
    path = f"{DS2_DIR}/{stub}.pkl"
    load_pickle = lambda p: pickle.load(open(p, 'rb'))
    dataset: Dataset = load_pickle(path)

    returned_dates = [to_datetime(ts).date() for ts in dataset.time.values]
    if min(returned_dates) < query.start_time or max(returned_dates) > query.end_time:
        return False

    if not all(band in dataset.data_vars for band in query.bands):
        return False

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
    from PaddockTSLocal.query import get_example_query
    from os.path import exists
    from os import remove

    query = get_example_query()
    stub = query.get_stub()
    path = f"{DS2_DIR}/{stub}.pkl"
    if exists(path):
        remove(path)

    download_ds2_from_query(stub, query)
    return all([
        test_path_existence(query),
        test_returned_dataset_values(query)
    ])

if __name__ == '__main__':
    print(test())
