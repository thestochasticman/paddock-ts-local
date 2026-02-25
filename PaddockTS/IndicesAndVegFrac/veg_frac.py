import numpy as np
import xarray as xr
from xarray import Dataset
from PaddockTS.query import Query


BANDS = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3']


def compute_fractional_cover(query: Query, ds_sentinel2=None, model_n: int = 4, correction: bool = False):
    from os.path import exists
    from fractionalcover3.unmixcover import unmix_fractional_cover
    from fractionalcover3 import data

    if ds_sentinel2 is None:
        if not exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds = xr.open_zarr(query.sentinel2_path, chunks=None)
    else:
        ds = ds_sentinel2
    inref = np.stack([ds[b].values for b in BANDS], axis=1).astype(np.float32)

    if correction:
        factors = np.array([0.9551, 1.0582, 0.9871, 1.0187, 0.9528, 0.9688]) + \
                  np.array([-0.0022, 0.0031, 0.0064, 0.012, 0.0079, -0.0042])
        inref *= factors[:, np.newaxis, np.newaxis]
    else:
        inref *= 0.0001

    fractions = np.empty((inref.shape[0], 3, inref.shape[2], inref.shape[3]))
    for t in range(inref.shape[0]):
        fractions[t] = unmix_fractional_cover(inref[t], fc_model=data.get_model(n=model_n))

    coords = {'time': ds.time, 'y': ds.y, 'x': ds.x}
    frac_ds = xr.Dataset({
        'bg': xr.DataArray(fractions[:, 0], dims=['time', 'y', 'x'], coords=coords),
        'pv': xr.DataArray(fractions[:, 1], dims=['time', 'y', 'x'], coords=coords),
        'npv': xr.DataArray(fractions[:, 2], dims=['time', 'y', 'x'], coords=coords),
    })

    from os import makedirs
    makedirs(query.tmp_dir, exist_ok=True)
    frac_ds.to_zarr(query.vegfrac_path, mode='w')
    print(f'Saved to {query.vegfrac_path}')
    print(f'  bg={fractions[:, 0].mean():.3f}, pv={fractions[:, 1].mean():.3f}, npv={fractions[:, 2].mean():.3f}')
    return frac_ds


def test():
    from PaddockTS.utils import get_example_query
    ds = compute_fractional_cover(get_example_query())
    for name in ['bg', 'pv', 'npv']:
        print(f'{name} range: {float(ds[name].min()):.3f} to {float(ds[name].max()):.3f}')

if __name__ == '__main__':
    test()
