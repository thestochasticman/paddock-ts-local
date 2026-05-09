import numpy as np
import xarray as xr
from xarray import Dataset
from numpy.typing import NDArray
from PaddockTS.query import Query

def _band(ds: Dataset, name: str) -> NDArray[np.float32]:
    b = ds[name].transpose('y', 'x', 'time').values.astype(np.float32)
    b[b == 0] = np.nan
    b /= 10000.0
    return b

def _normalised_diff(a, b):
    nd = (a - b) / (a + b)
    nd[~np.isfinite(nd)] = np.nan
    return nd

def compute_ndvi(ds: Dataset) -> NDArray[np.float32]:
    return _normalised_diff(_band(ds, 'nbart_nir_1'), _band(ds, 'nbart_red'))

def compute_cfi(ds: Dataset) -> NDArray[np.float32]:
    ndvi = compute_ndvi(ds)
    red = _band(ds, 'nbart_red')
    green = _band(ds, 'nbart_green')
    blue = _band(ds, 'nbart_blue')
    return ndvi * (red + green + green - blue)

def compute_nirv(ds: Dataset) -> NDArray[np.float32]:
    return compute_ndvi(ds) * _band(ds, 'nbart_nir_1')

def compute_ndti(ds: Dataset) -> NDArray[np.float32]:
    return _normalised_diff(_band(ds, 'nbart_swir_2'), _band(ds, 'nbart_swir_3'))

def compute_cai(ds: Dataset) -> NDArray[np.float32]:
    return 0.5 * (_band(ds, 'nbart_swir_2') + _band(ds, 'nbart_swir_3')) - _band(ds, 'nbart_nir_1')

def compute_indices(query: Query, ds_sentinel2=None, indices=None):
    from os.path import exists

    if ds_sentinel2 is None:
        if not exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds = xr.open_zarr(query.sentinel2_path, chunks=None)
    else:
        ds = ds_sentinel2

    if indices is None:
        indices = {'NDVI': compute_ndvi, 'CFI': compute_cfi, 'NIRv': compute_nirv, 'NDTI': compute_ndti, 'CAI': compute_cai}

    for name, func in indices.items():
        data = func(ds).transpose(2, 0, 1)  # (y, x, time) -> (time, y, x)
        ds[name] = xr.DataArray(data, dims=['time', 'y', 'x'], coords={'time': ds.time, 'y': ds.y, 'x': ds.x})
        print(f'{name}: {data.shape}')

    return ds


def test():
    from PaddockTS.utils import get_example_query
    ds = compute_indices(get_example_query())
    for name in ['NDVI', 'CFI', 'NIRv', 'NDTI', 'CAI']:
        print(f'{name} range: {float(ds[name].min()):.3f} to {float(ds[name].max()):.3f}')

if __name__ == '__main__':
    test()
