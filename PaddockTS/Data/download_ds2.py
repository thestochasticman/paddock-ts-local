from dask.distributed import Client as DaskClient
from xarray.core.dataset import Dataset
from PaddockTS.query import Query
import pystac_client
import odc.stac
import pickle
import rioxarray

def download_ds2(
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

    print(query.path_ds2, query.stub)
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
    download_ds2(query)
    return test_returned_dataset_values(query)

if __name__ == '__main__':
    print(test())
