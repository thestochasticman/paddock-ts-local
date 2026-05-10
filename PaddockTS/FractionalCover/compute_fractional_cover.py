"""Spectral unmixing of Sentinel-2 reflectance into bg / pv / npv fractions.

The model is a small TFLite MLP adapted from
`fractionalcover3 <https://github.com/jrsrp/fractionalcover3>`_ by
Robert Denham (MIT-licensed; see
``PaddockTS/LICENSES/fractionalcover3.LICENSE``). Four model variants
ship with the package, indexed ``n=1..4`` from least to most complex;
``n=4`` is the default and most accurate.

Output bands:

- ``bg`` — bare ground fraction
- ``pv`` — green (photosynthetic) vegetation fraction
- ``npv`` — non-green (non-photosynthetic) vegetation fraction

Fractions are produced per-pixel per-timestep and persisted to
``query.fractional_cover_path`` as Zarr v2.
"""

import os
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='keras')

import numpy as np
import xarray as xr
from xarray import Dataset
from PaddockTS.query import Query


BANDS = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3']


def compute_fractional_cover(query: Query, ds_sentinel2=None, model_n: int = 4, correction: bool = False):
    """Run the TFLite unmixing model over every Sentinel-2 timestep.

    Stacks the six Sentinel-2 SR bands into a ``(time, band, y, x)``
    tensor, scales reflectance, and invokes the chosen model variant
    once per timestep. The result is written to
    ``query.fractional_cover_path`` and returned as an xarray Dataset
    with ``bg``, ``pv``, ``npv`` data variables on dims
    ``(time, y, x)``.

    Args:
        query: The :class:`PaddockTS.query.Query`. ``query.tmp_dir`` is
            created if missing and the output Zarr is written to
            ``query.fractional_cover_path``.
        ds_sentinel2: Optional in-memory Sentinel-2 dataset. If ``None``,
            opens (or downloads, then opens) ``query.sentinel2_path``.
            Must contain the six bands in :data:`BANDS`.
        model_n: Which bundled model variant to use (``1..4``). Higher
            ``n`` is more accurate but slower; ``n=4`` is the default.
        correction: If ``True``, apply per-band sensor calibration
            factors (gains and offsets fitted in the upstream
            fractionalcover3 work) instead of the simple ``* 0.0001``
            DN-to-reflectance scaling. Use only when your inputs match
            the calibration assumptions of the original model.

    Returns:
        xarray.Dataset: Dataset with variables ``bg``, ``pv``, ``npv``
        on dims ``(time, y, x)``. Also persisted to
        ``query.fractional_cover_path``.
    """
    from os.path import exists
    from PaddockTS.FractionalCover._unmix import unmix_fractional_cover, get_model

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
        fractions[t] = unmix_fractional_cover(inref[t], fc_model=get_model(n=model_n))

    coords = {'time': ds.time, 'y': ds.y, 'x': ds.x}
    frac_ds = xr.Dataset({
        'bg': xr.DataArray(fractions[:, 0], dims=['time', 'y', 'x'], coords=coords),
        'pv': xr.DataArray(fractions[:, 1], dims=['time', 'y', 'x'], coords=coords),
        'npv': xr.DataArray(fractions[:, 2], dims=['time', 'y', 'x'], coords=coords),
    })

    from os import makedirs
    makedirs(query.tmp_dir, exist_ok=True)
    frac_ds.to_zarr(query.fractional_cover_path, mode='w', zarr_format=2)
    return frac_ds


def test():
    from PaddockTS.utils import get_example_query
    ds = compute_fractional_cover(get_example_query())
    for name in ['bg', 'pv', 'npv']:
        print(f'{name} range: {float(ds[name].min()):.3f} to {float(ds[name].max()):.3f}')

if __name__ == '__main__':
    test()
