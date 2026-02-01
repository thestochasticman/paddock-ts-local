import odc.stac
import rioxarray
import numpy as np
import pystac_client
from xarray import Dataset
from .sentinel2 import Sentinel2
from PaddockTS.query import Query
from .sentinel2 import defaultsentinel2
from dask.distributed import Client as DaskClient

odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True},)

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

    bands = sentinel2.bands
    catalog = pystac_client.Client.open(sentinel2.stac_url)
    result = catalog.search(
        bbox=query.bbox,
        collections=sentinel2.collections,
        datetime=f'{query.start}/{query.end}',
        filter=sentinel2.cloud_cover_filter
    )
    with DaskClient(n_workers=num_workers, n_threads=threads_per_worker) as client:
        try:
            ds: Dataset = odc.stac.load(
                list(result.items()),
                bands=bands,
                crs=sentinel2.crs,
                resolution=sentinel2.resolution,
                groupby=sentinel2.groupby,
                bbox=query.bbox,
                chunks={'time': chunk_time, 'x': chunk_x, 'y': chunk_y},
            )

            ds = client.compute(ds).result()
        except Exception as e:
            print(f'Creating dataset using dask failed due to: {e}')
            raise
        finally:
            client.close()
    
    #creating cloud mask
    fmask = ds[sentinel2.cloud_mask_band]
    clear_mask = (fmask != sentinel2.fmask_cloud) & (fmask != sentinel2.fmask_shadow)
    ds = ds.drop_vars(sentinel2.cloud_mask_band).where(clear_mask)
    #dropping frames using a nan percentage threshold
    nan_frac = ds.to_array().isnull().mean(dim=['variable', 'x', 'y'])
    ds = ds.sel(time=nan_frac < max_nan_fraction)
    return ds

