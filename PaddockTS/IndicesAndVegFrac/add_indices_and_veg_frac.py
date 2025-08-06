from PaddockTS.IndicesAndVegFrac.veg_frac import calculate_fractional_cover
from PaddockTS.IndicesAndVegFrac.veg_frac import add_fractional_cover_to_ds
from PaddockTS.IndicesAndVegFrac.indices import *
import pickle
from typing_extensions import Callable
from PaddockTS.query import Query
import pickle

def add_indices_and_veg_frac(
    query: Query,
    band_names: list[str] = [
        'nbart_blue',
        'nbart_green',
        'nbart_red',
        'nbart_nir_1',
        'nbart_swir_2',
        'nbart_swir_3'
    ],
    indices: dict[str, Callable] = {
        'NDVI': calculate_ndvi,
        'CFI': calculate_cfi,
        'NIRv': calculate_nirv
    }
):
    stub = query.stub
    path_ds2 = query.path_ds2
    path_ds2i = f"{query.stub_tmp_dir}/DS2I.pkl"
    ds = pickle.load(open(path_ds2, 'rb'))
    fractions = calculate_fractional_cover(ds, band_names, i=4, correction=False)
    ds = add_fractional_cover_to_ds(ds, fractions)
    ds = calculate_indices(ds, indices)
    with open(path_ds2i, 'wb') as _file:
        pickle.dump(ds, _file, protocol=pickle.HIGHEST_PROTOCOL)

def test():
    from PaddockTS.query import get_example_query
    from os.path import exists
    query = get_example_query()
    add_indices_and_veg_frac(query)
    return exists(f"{query.stub_tmp_dir}/DS2I.pkl")
    
if __name__ == '__main__':
    print(test())