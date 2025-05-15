from PaddockTSLocal.Indices import calculate_indices
from PaddockTSLocal.Indices import *
from PaddockTSLocal.VegFrac import calculate_fractional_cover
from PaddockTSLocal.VegFrac import add_fractional_cover_to_ds
from PaddockTSLocal.utils import load_pickle
from typing_extensions import Callable
from typing_extensions import Union
from xarray import Dataset
import pickle

def add_fractional_cover_and_calculate_indices(
        ds: Union[str, Dataset],
        path_out: str,
        band_names: list[str] = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3'],
        indices: dict[str, Callable] = {
            'NDVI': calculate_ndvi,
            'CFI': calculate_cfi,
            'NIRv': calculate_nirv
        }
    ):
    ds = load_pickle(ds) if isinstance(ds, str) else ds
    fractions = calculate_fractional_cover(ds, band_names, i=4, correction=False)
    ds = add_fractional_cover_to_ds(ds, fractions)
    ds = calculate_indices(ds, indices)
    with open(path_out, 'wb') as _file:
        print(path_out)
        pickle.dump(ds, _file, protocol=pickle.HIGHEST_PROTOCOL)
def test():

    from PaddockTSLocal.Query import get_example_query
    from datetime import date
    from os.path import join
    from os import getcwd
    from os.path import exists
    import pickle
    from os.path import dirname
    from os import makedirs

    query = get_example_query()
    path_ds = join(getcwd(), 'Data', 'ds2', f"{query.get_stub()}.pkl")
    
    load_pickle = lambda path: pickle.load(open(path, 'rb'))
    ds = load_pickle(path_ds)
    band_names = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3']
    path_ds2i = join(dirname(dirname(path_ds)), 'ds2i', f"{query.get_stub()}.pkl")
    makedirs(join(dirname(dirname(path_ds)), 'ds2i'), exist_ok=True)
    add_fractional_cover_and_calculate_indices(
        path_ds,
        path_ds2i
    )
    
if __name__ == '__main__':
    test()