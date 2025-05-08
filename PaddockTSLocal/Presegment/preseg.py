from PaddockTSLocal.Presegment.compute_ndwi_fourier import f as compute_ndwi_fourier
from PaddockTSLocal.Presegment.rescale_image import f as rescale_image
from PaddockTSLocal.Presegment.export import f as export
from datetime import date
from PaddockTSLocal.Query import Query
from os.path import join
from os import getcwd
from os.path import exists
import rioxarray
import pickle


load_pickle = lambda path: pickle.load(open(path, 'rb'))

def f(path_ds: str, path_out: str):
    ds = load_pickle(path_ds)
    # ds = calculate_indices(ds, ['NDVI', 'CFI', 'NIRv'], collection='ga_s2_3')
    img_fourier = compute_ndwi_fourier(ds)
    img = rescale_image(img_fourier)
    print(img)
    export(ds, img, path_out)

def t():
    from os.path import basename
    
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
    path_ds = join(getcwd(), 'Data', 'ds2', f"{query.get_stub()}.pkl")
    if not exists(path_ds):
        from PaddockTSLocal.Download.query_to_ds import f as query_to_ds
        query_to_ds(query=query)
    out_dir: str=join(getcwd(), 'Data', 'preseg')
    stub = basename(path_ds).split('.')[0]
    path_out = join(out_dir, f"{stub}.tif")
    f(path_ds, path_out)

if __name__ == '__main__':
    t()


