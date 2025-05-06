from PaddockTSLocal.Presegment.compute_ndwi_fourier import f as compute_ndwi_fourier
from PaddockTSLocal.Presegment.rescale_image import f as rescale_image
from PaddockTSLocal.Presegment.export import f as export
from PaddockTSLocal.Download.query_to_ds import f as ds_from_stac
from dea_tools.bandindices import calculate_indices
from datetime import date
from PaddockTSLocal.Logger import Logger
from PaddockTSLocal.Query import Query
from os.path import exists
import rioxarray
import xarray as xr
import numpy as np
import pickle
import hdstats
from os.path import abspath
from matplotlib import pyplot as plt

load_pickle = lambda path: pickle.load(open(path, 'rb'))

def f(query: Query | None, stub: str | None, logger: Logger = None):
    path_ds = logger.get_path_dataset(stub, query)
    print(path_ds)
    if not exists(path_ds): ds_from_stac(query, logger)
    ds = load_pickle(logger.get_path_dataset(stub, query))
    print(ds)
    print(type(ds))
    ds = calculate_indices(ds, ['NDVI', 'NDWI', 'SAVI'], collection='ga_s2_3')
    img_fourier = compute_ndwi_fourier(ds)
    img = rescale_image(img_fourier)
    export(ds, img, logger.get_path_query_presegment_tiff(stub, query))

def t():
    query = Query(
        lat=-33.5040,
        lon=148.4,
        buffer=0.01,
        start_time=date(2020, 1, 1),
        end_time=date(2020, 6, 1),
        collections=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
        bands=[
            'nbart_blue',
            'nbart_green',
            'nbart_red', 
            'nbart_red_edge_1',
            'nbart_red_edge_2',
            'nbart_red_edge_3',
            'nbart_nir_1',
            'nbart_nir_2',
            'nbart_swir_2',
            'nbart_swir_3'
        ]
    )
    logger = Logger()
    stub = '4'
    f(query, stub, logger)

if __name__ == '__main__':
    t()


