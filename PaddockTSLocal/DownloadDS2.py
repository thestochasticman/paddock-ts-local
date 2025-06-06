from dask.distributed import Client as DaskClient
from PaddockTSLocal.Legend import DS2_DIR
from xarray.core.dataset import Dataset
from PaddockTSLocal.Query import Query
from typing_extensions import Union
from os import makedirs
import pystac_client
import odc.stac
import pickle
import rioxarray

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

    Parameters
    ----------
    query : Query
        A Query object containing bbox, collections, datetime, and bands attributes for the STAC search.
    num_workers : int, optional
        Number of Dask worker processes to spawn (default is 4).
    threads_per_worker : int, optional
        Number of threads per Dask worker (default is 2).
    tile_width : int, optional
        Width of each tile (in pixels) for chunking in the x dimension when loading (default is 1024).
    tile_height : int, optional
        Height of each tile (in pixels) for chunking in the y dimension when loading (default is 1024).
    tile_time_series_length : int, optional
        Length of the time dimension chunk (in number of time slices) per task (default is 1).

    Returns
    -------
    Dataset
        An xarray.Dataset containing the loaded data, reprojected to EPSG:6933 and chunked as specified.

    """
    # Start a Dask client for parallel execution
    dask_client = DaskClient(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker
    )

    # Configure STAC access
    catalog = pystac_client.Client.open('https://explorer.dea.ga.gov.au/stac')
    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={'aws_unsigned': True},
    )

    # Filter for low cloud cover
    filter_expression = {
        "op": "<",
        "args": [{"property": "eo:cloud_cover"}, 10]
    }
    query_results = catalog.search(
        bbox=query.bbox,
        collections=query.collections,
        datetime=query.datetime,
        filter=filter_expression
    )
    items = list(query_results.items())

    # Load and chunk data via ODC STAC loader
    ds2_pipeline = odc.stac.load(
        items,
        bands=query.bands,
        crs='EPSG:6933',
        resolution=10,
        groupby='solar_day',
        bbox=query.bbox,
        chunks={
            'time': tile_time_series_length,
            'x': tile_width,
            'y': tile_height
        }
    )

    # Execute the pipeline in parallel and retrieve the result
    future = dask_client.compute(ds2_pipeline)
    ds2: Dataset = future.result()

    # Close the Dask client
    dask_client.close()
    return ds2

def save_ds2_to_file(ds2: Dataset, path: str)->None:
    """
    Serialize and save an xarray.Dataset to disk using Python pickle.

    Parameters
    ----------
    ds2 : Dataset
        The xarray.Dataset to serialize.
    path : str
        Filesystem path where the pickled dataset will be saved.

    Returns
    -------
    None
        Writes the dataset to the specified file path.
    """
    with open(path, 'wb') as handle:
        pickle.dump(ds2, handle, protocol=pickle.HIGHEST_PROTOCOL)

def download_ds2_from_query(
    query: Query,
    stub: Union[str, None] = None,
    num_workers: int = 4, 
    threads_per_worker: int = 2,
    tile_width: int = 1024,
    tile_height: int = 1024,
    tile_time_series_length: int = 1,
)->Dataset:
    
    ds2 = query_to_ds2(
        query=query,
        num_workers=num_workers,
        threads_per_worker=threads_per_worker,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_time_series_length=tile_time_series_length 
    )
    stub = stub if stub is not None else query.get_stub()
    path = f"{DS2_DIR}/{stub}.pkl"
    save_ds2_to_file(ds2=ds2, path=path)
    return ds2

def test_path_existence(query: Query)->bool:
    from os.path import exists
    stub = query.get_stub()
    path = f"{DS2_DIR}/{stub}.pkl"
    return exists(path)

def test_returned_dataset_values(query: Query)->bool:
    from pandas import to_datetime
    stub = query.get_stub()
    path = f"{DS2_DIR}/{stub}.pkl"
    load_pickle = lambda path: pickle.load(open(path, 'rb'))
    dataset: Dataset = load_pickle(path)
    returned_dates = [to_datetime(timestamp).date() for timestamp in dataset.time.values]
    earliest_date = min(returned_dates)
    latest_date = max(returned_dates)
    if (earliest_date < query.start_time) or (latest_date > query.end_time):
        return False
    
    if not all([band in query.bands for band in dataset.data_vars.keys()]):
        return False
    return True

def test():
    from PaddockTSLocal.Query import get_example_query
    from os.path import exists
    from os import remove
    
    query = get_example_query()
    stub = query.get_stub()
    path = f"{DS2_DIR}/{stub}.pkl"
    if exists(path): remove(path)
    download_ds2_from_query(query)

    return all(
        [
            test_path_existence(query),
            test_returned_dataset_values(query)
        ]
    )

if __name__ == '__main__':
    print(test())