"""Vegetation and tillage indices computed from Sentinel-2 reflectance.

Each index is returned as a 32-bit float array with shape ``(y, x, time)``.
Zero pixels are treated as missing (Sentinel-2 ARD nodata) and DN values
are scaled by ``1/10000`` to get reflectance in ``[0, 1]``.

The high-level entry point :func:`compute_indices` adds every index as a
new ``(time, y, x)`` data variable to the input dataset, resolving its
inputs from ``query`` if no in-memory dataset is supplied.
"""

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
    """Normalised Difference Vegetation Index ``(NIR - Red) / (NIR + Red)``.

    Healthy vegetation strongly reflects NIR and absorbs red. NDVI ranges
    in ``[-1, 1]``; bare soil sits near 0, dense canopy near 0.8–0.9.

    Args:
        ds: Sentinel-2 dataset with ``nbart_nir_1`` and ``nbart_red`` bands.

    Returns:
        numpy.ndarray: Float32 array shaped ``(y, x, time)``.
    """
    return _normalised_diff(_band(ds, 'nbart_nir_1'), _band(ds, 'nbart_red'))

def compute_cfi(ds: Dataset) -> NDArray[np.float32]:
    """Crop Foliage Index — NDVI weighted by visible-band brightness.

    ``CFI = NDVI * (Red + 2 * Green - Blue)``. Amplifies the contrast
    between green crops and other land cover by combining NDVI with a
    visible "greenness" term.

    Args:
        ds: Sentinel-2 dataset with ``nbart_red``, ``nbart_green``,
            ``nbart_blue``, and ``nbart_nir_1`` bands.

    Returns:
        numpy.ndarray: Float32 array shaped ``(y, x, time)``.
    """
    ndvi = compute_ndvi(ds)
    red = _band(ds, 'nbart_red')
    green = _band(ds, 'nbart_green')
    blue = _band(ds, 'nbart_blue')
    return ndvi * (red + green + green - blue)

def compute_nirv(ds: Dataset) -> NDArray[np.float32]:
    """Near-Infrared Reflectance of Vegetation: ``NDVI * NIR``.

    A proxy for the fraction of NIR radiation reflected by vegetation
    (rather than the soil background); often a stronger correlate of
    GPP than NDVI.

    Args:
        ds: Sentinel-2 dataset with ``nbart_nir_1`` and ``nbart_red`` bands.

    Returns:
        numpy.ndarray: Float32 array shaped ``(y, x, time)``.
    """
    return compute_ndvi(ds) * _band(ds, 'nbart_nir_1')

def compute_ndti(ds: Dataset) -> NDArray[np.float32]:
    """Normalised Difference Tillage Index ``(SWIR2 - SWIR3) / (SWIR2 + SWIR3)``.

    Sensitive to crop residue cover, used as a proxy for tillage and
    stubble retention. Higher values indicate more lignin/cellulose
    relative to bare soil.

    Args:
        ds: Sentinel-2 dataset with ``nbart_swir_2`` and ``nbart_swir_3`` bands.

    Returns:
        numpy.ndarray: Float32 array shaped ``(y, x, time)``.
    """
    return _normalised_diff(_band(ds, 'nbart_swir_2'), _band(ds, 'nbart_swir_3'))

def compute_cai(ds: Dataset) -> NDArray[np.float32]:
    """Cellulose Absorption Index ``0.5 * (SWIR2 + SWIR3) - NIR``.

    Targets the cellulose/lignin absorption feature near 2100 nm; useful
    for distinguishing dry plant matter from bare soil.

    Args:
        ds: Sentinel-2 dataset with ``nbart_nir_1``, ``nbart_swir_2``,
            and ``nbart_swir_3`` bands.

    Returns:
        numpy.ndarray: Float32 array shaped ``(y, x, time)``.
    """
    return 0.5 * (_band(ds, 'nbart_swir_2') + _band(ds, 'nbart_swir_3')) - _band(ds, 'nbart_nir_1')

def compute_indices(query: Query, ds_sentinel2=None, indices=None):
    """Add NDVI, CFI, NIRv, NDTI, CAI as data variables to a Sentinel-2 cube.

    Each index is computed in float32 and inserted as a ``(time, y, x)``
    data variable on the Sentinel-2 dataset. The dataset is returned
    in-place — nothing is written to disk; persistence is handled by
    downstream consumers (e.g. ``make_paddockTS``).

    Args:
        query: The :class:`PaddockTS.query.Query` for resolving the
            Sentinel-2 Zarr if ``ds_sentinel2`` is not supplied.
        ds_sentinel2: Optional in-memory Sentinel-2 dataset. If ``None``,
            opens (or downloads, then opens) ``query.sentinel2_path``.
        indices: Optional ``{name: callable(ds) -> ndarray}`` mapping
            for custom or subset index sets. Defaults to the five built-in
            indices ``{'NDVI', 'CFI', 'NIRv', 'NDTI', 'CAI'}``.

    Returns:
        xarray.Dataset: The input dataset with one new ``(time, y, x)``
        data variable per requested index.
    """
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

    return ds


def test():
    from PaddockTS.utils import get_example_query
    ds = compute_indices(get_example_query())
    for name in ['NDVI', 'CFI', 'NIRv', 'NDTI', 'CAI']:
        print(f'{name} range: {float(ds[name].min()):.3f} to {float(ds[name].max()):.3f}')

if __name__ == '__main__':
    test()
