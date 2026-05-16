"""Vegetation and tillage indices computed from Sentinel-2 reflectance.

Each index is returned as a 32-bit float array with shape ``(y, x, time)``.
Zero pixels are treated as missing (Sentinel-2 ARD nodata) and DN values
are scaled by ``1/10000`` to get reflectance in ``[0, 1]``.

The high-level entry point :func:`compute_indices` adds every index as a
new ``(time, y, x)`` data variable to the input dataset, resolving its
inputs from ``query`` if no in-memory dataset is supplied.
"""

import os
import numpy as np
import xarray as xr
from os import makedirs
from os.path import exists
from datetime import datetime
from xarray import Dataset
from numpy.typing import NDArray
from PaddockTS.query import Query
from PaddockTS.SpectralIndices.check_if_valid_ds_indices_exists import check_if_valid_ds_indices_exists

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

    If a valid cache exists at ``query.indices_path``, it is opened and
    returned without recomputing. Otherwise each index is computed in
    float32 and inserted as a ``(time, y, x)`` data variable on the
    Sentinel-2 dataset; the result (original bands + indices) is then
    persisted as Zarr v2 at ``query.indices_path`` and a ``_SUCCESS``
    marker is written inside the zarr.

    Args:
        query: The :class:`PaddockTS.query.Query` for resolving the
            Sentinel-2 Zarr if ``ds_sentinel2`` is not supplied.
        ds_sentinel2: Optional in-memory Sentinel-2 dataset. If ``None``,
            opens (or downloads, then opens) ``query.sentinel2_path``.
        indices: Optional ``{name: callable(ds) -> ndarray}`` mapping
            for custom or subset index sets. Defaults to the five built-in
            indices ``{'NDVI', 'CFI', 'NIRv', 'NDTI', 'CAI'}``.

    Returns:
        xarray.Dataset: The Sentinel-2 dataset with one new ``(time, y, x)``
        data variable per requested index. Also persisted to
        ``query.indices_path``.
    """
    if check_if_valid_ds_indices_exists(query.indices_path):
        try:
            return xr.open_zarr(query.indices_path, chunks=None, decode_coords='all')
        except Exception as e:
            print(f'Indices cache at {query.indices_path} unreadable ({e}); recomputing')

    if ds_sentinel2 is None:
        if not exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds = xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords='all')
    else:
        ds = ds_sentinel2

    if indices is None:
        indices = {'NDVI': compute_ndvi, 'CFI': compute_cfi, 'NIRv': compute_nirv, 'NDTI': compute_ndti, 'CAI': compute_cai}

    for name, func in indices.items():
        data = func(ds).transpose(2, 0, 1)  # (y, x, time) -> (time, y, x)
        ds[name] = xr.DataArray(data, dims=['time', 'y', 'x'], coords={'time': ds.time, 'y': ds.y, 'x': ds.x})

    makedirs(os.path.dirname(query.indices_path), exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + 'Z'
    ds = ds.assign_attrs(indices_computed_at=timestamp)
    ds.to_zarr(query.indices_path, mode='w', zarr_format=2)
    # Touch _SUCCESS *after* the zarr write completes; its presence is what
    # the next call uses as the cache-validity check.
    with open(f'{query.indices_path}/_SUCCESS', 'w') as f:
        f.write(timestamp)
    return ds


def _temp_query():
    import tempfile
    from datetime import date
    from PaddockTS.config import Config
    tmpdir = tempfile.mkdtemp(prefix='paddockts_idx_test_')
    cfg = Config(out_dir=tmpdir, tmp_dir=tmpdir)
    return Query(
        bbox=[148.36265, -33.52606, 148.38265, -33.50606],
        start=date(2024, 1, 1), end=date(2024, 1, 21),
        stub=f'idx_{os.path.basename(tmpdir)}', config=cfg,
    )


def test_compute_writes_zarr_and_marker():
    """First call computes, writes indices.zarr, and touches the _SUCCESS marker."""
    q = _temp_query()
    ds = compute_indices(q)
    if not exists(q.indices_path):
        return False
    if not exists(f'{q.indices_path}/_SUCCESS'):
        return False
    return all(name in ds.data_vars for name in ('NDVI', 'CFI', 'NIRv', 'NDTI', 'CAI'))


def test_repeated_call_uses_cache():
    """Second call with same query reuses the zarr (no rewrite)."""
    q = _temp_query()
    compute_indices(q)
    mtime_before = os.path.getmtime(q.indices_path)
    compute_indices(q)
    mtime_after = os.path.getmtime(q.indices_path)
    return mtime_before == mtime_after


def test_missing_marker_triggers_recompute():
    """A cache with the zarr present but no _SUCCESS file is recomputed."""
    q = _temp_query()
    compute_indices(q)
    marker = f'{q.indices_path}/_SUCCESS'
    os.remove(marker)
    mtime_before = os.path.getmtime(q.indices_path)
    compute_indices(q)
    mtime_after = os.path.getmtime(q.indices_path)
    return exists(marker) and mtime_after > mtime_before


def test():
    return all([
        test_compute_writes_zarr_and_marker(),
        test_repeated_call_uses_cache(),
        test_missing_marker_triggers_recompute(),
    ])


if __name__ == '__main__':
    print(test())
