"""Aggregate per-pixel Sentinel-2 data into per-paddock medians.

Rasterises the paddock polygons onto the Sentinel-2 grid, then for every
band in the dataset computes the per-paddock median across pixels at
each timestep. The result is the central time-series dataset that
downstream stages (yearly split, smoothing, phenology, plotting) consume.
"""

import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from PaddockTS.query import Query
import os

def _band_medians(band_array, mask_flat, paddock_ids):
    """
    band_array: np.ndarray (time, y, x)
    mask_flat:  1D np.ndarray of length y*x, containing integer paddock IDs
    paddock_ids: list/array of integer paddock IDs (same ones used in rasterize)
    """
    T, Y, X = band_array.shape
    flat   = band_array.reshape(T, -1)
    out    = np.empty((len(paddock_ids), T), dtype=np.float64)

    for t in range(T):
        row = flat[t]
        for i, pid in enumerate(paddock_ids):
            sel = row[mask_flat == pid]
            if sel.size and not np.all(np.isnan(sel)):  # suppress warning if all NaN
                out[i, t] = np.nanmedian(sel)
            else:
                out[i, t] = np.nan

    return out

def make_paddock_time_series(query: Query, ds_sentinel2=None, paddocks_filepath=None, crs="epsg:6933"):
    """Compute per-paddock medians for every band at every timestep.

    Steps:

    1. Compute the five spectral indices (NDVI, CFI, NIRv, NDTI, CAI)
       and add them to the input dataset.
    2. Rasterise paddock polygons to integer IDs aligned with the
       Sentinel-2 grid.
    3. For each band, in parallel across processes, compute the
       per-paddock NaN-aware median across pixels at every timestep.
    4. Stitch results back into an xarray Dataset on dims
       ``(paddock, time)`` and persist as Zarr v2 to
       ``{paddocks_filepath stem}_timeseries.zarr``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        ds_sentinel2: Optional in-memory Sentinel-2 dataset. If ``None``,
            ``query.sentinel2_path`` is opened (or downloaded first).
        paddocks_filepath: Path to a GeoPackage (.gpkg) containing paddock
            polygons (must include a ``paddock`` column for IDs). If
            ``None``, defaults to ``{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg``
            (loaded or generated via :func:`PaddockTS.PaddockSegmentation.get_paddocks`).
        crs: Equal-area CRS to write onto the dataset for
            georeferencing the rasterised mask. Defaults to EPSG:6933
            (WGS84 / NSIDC EASE-Grid 2.0 Global).

    Returns:
        xarray.Dataset: Per-paddock medians on dims ``(paddock, time)``
        with one data variable per Sentinel-2 band and per spectral
        index. Also persisted to ``{paddocks_filepath stem}_timeseries.zarr``.
    """
    import rasterio.features
    from affine import Affine
    import pandas as pd
    from os.path import exists
    from pathlib import Path
    import geopandas as gpd

    if ds_sentinel2 is None:
        if not exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)

    # Compute vegetation indices (NDVI, CFI, NIRv, NDTI, CAI)
    from PaddockTS.SpectralIndices.indices import compute_indices
    ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)

    if paddocks_filepath is None:
        paddocks_filepath = f'{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg'
        if not exists(paddocks_filepath):
            from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
            get_paddocks(query)

    # Use load_user_paddocks to ensure 'paddock' column exists
    from PaddockTS.utils import load_user_paddocks
    paddocks = load_user_paddocks(paddocks_filepath)

    ds = ds_sentinel2

    # 1) Ensure CRS is written
    ds = ds.rio.write_crs(crs, inplace=False)

    # Reproject paddocks to match the dataset CRS
    if paddocks.crs != ds.rio.crs:
        paddocks = paddocks.to_crs(ds.rio.crs)

    pol = paddocks
    transform = ds.rio.transform()
    H, W = ds.rio.height, ds.rio.width

    # 2) Build a mapping from paddock label (string) -> integer ID for rasterization
    #    This works whether pol.paddock is int, string, or mixed.
    paddock_labels = pol["paddock"].astype(str)
    # unique labels in a stable order
    unique_labels = pd.Index(paddock_labels.unique().tolist())
    int_ids = np.arange(1, len(unique_labels) + 1, dtype=np.int32)  # start at 1, 0 = background
    label_to_int = dict(zip(unique_labels, int_ids))

    # integer paddock IDs for each polygon row
    poly_ids = paddock_labels.map(label_to_int).to_numpy(dtype=np.int32)

    # 3) Rasterize once using integer IDs
    shapes = [(geom, int(pid)) for geom, pid in zip(pol.geometry, poly_ids)]
    mask = rasterio.features.rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,              # background
        dtype=np.int32,
    )
    mask_flat   = mask.ravel()

    # These are the IDs we’ll compute medians for, and their corresponding labels
    paddock_ids  = int_ids.tolist()             # integer IDs in order
    paddock_strs = unique_labels.astype(str).tolist()  # original labels as strings

    # 4) Grab all band arrays in memory
    bands = {var: ds[var].values for var in ds.data_vars}

    # 5) Parallel median for each band
    results = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = {
            var: exe.submit(_band_medians, arr, mask_flat, paddock_ids)
            for var, arr in bands.items()
        }
        for var, fut in futures.items():
            results[var] = fut.result()

    # 6) Stitch back into xarray
    coords = {
        "paddock": paddock_strs,              # original paddock labels (strings)
        "time": ds.coords["time"],
        "spatial_ref": np.int32(ds.rio.crs.to_epsg()),
    }
    data_vars = {
        var: (("paddock", "time"), results[var])
        for var in results
    }
    result = xr.Dataset(data_vars, coords=coords)

    paddocks_path = Path(paddocks_filepath)
    zarr_path = f'{query.tmp_dir}/{paddocks_path.stem}_timeseries.zarr'
    result.to_zarr(zarr_path, mode='w', zarr_format=2)
    print(f'Saved to {zarr_path}')
    return result


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    result = make_paddock_time_series(query)
    print(result)
    for var in result.data_vars:
        print(f'{var}: {float(result[var].mean()):.3f}')


if __name__ == '__main__':
    test()
