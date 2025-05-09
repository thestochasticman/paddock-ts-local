from PaddockTSLocal.VegFracAndIndices.calculate_fractional_cover import f as calculate_fractional_cover
from PaddockTSLocal.VegFracAndIndices.calculate_indices import f as calculate_indices
from PaddockTSLocal.VegFracAndIndices.add_fractional_cover_to_ds import f as add_fractional_cover_to_ds
from typing_extensions import Callable
import pickle


def f(ds, band_names: list[str], indices: dict[str, Callable], path: str):
    fractions = calculate_fractional_cover(ds, band_names, i=4, correction=False)
    ds = add_fractional_cover_to_ds(ds, fractions)
    ds = calculate_indices(ds, indices)
    with open(path, 'wb') as _file:
        print(path)
        pickle.dump(ds, _file, protocol=pickle.HIGHEST_PROTOCOL)

def t():
    from PaddockTSLocal.Query import Query
    from datetime import date
    from os.path import join
    from os import getcwd
    from os.path import exists
    import pickle
    from os.path import dirname
    from os import makedirs

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
        query_to_ds(query=query, path_ds=path_ds)
    load_pickle = lambda path: pickle.load(open(path, 'rb'))
    ds = load_pickle(path_ds)
    band_names = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3']
    from PaddockTSLocal.VegFracAndIndices.calculate_indices import calculate_ndvi
    from PaddockTSLocal.VegFracAndIndices.calculate_indices import calculate_cfi
    from PaddockTSLocal.VegFracAndIndices.calculate_indices import calculate_nirv

    path_ds2i = join(dirname(dirname(path_ds)), 'ds2i', f"{query.get_stub()}.pkl")
    makedirs(join(dirname(dirname(path_ds)), 'ds2i'), exist_ok=True)
    f(
        ds,
        band_names,
        indices = {
            'NDVI': calculate_ndvi,
            'CFI': calculate_cfi,
            'NIRv': calculate_nirv
        },
        path=path_ds2i
    )
    
if __name__ == '__main__':
    t()
