### updated verison of make_paddockTS3() that can take integer or string for pol[paddock]

import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
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

def make_paddockTS(query, ds_sentinel2=None, paddocks=None, crs="epsg:6933"):
    import rasterio.features
    from affine import Affine
    import pandas as pd
    from os.path import exists

    if ds_sentinel2 is None:
        if not exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds_sentinel2 = xr.open_zarr(query.sentinel2_path, chunks=None)

    # Compute vegetation indices (NDVI, CFI, NIRv, NDTI, CAI)
    from PaddockTS.IndicesAndVegFrac.indices import compute_indices
    ds_sentinel2 = compute_indices(query, ds_sentinel2=ds_sentinel2)

    if paddocks is None:
        import geopandas as gpd
        gpkg_path = f'{query.tmp_dir}/{query.stub}_paddocks.gpkg'
        if exists(gpkg_path):
            paddocks = gpd.read_file(gpkg_path)
        else:
            from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
            paddocks = get_paddocks(query)

    ds = ds_sentinel2
    pol = paddocks

    # 1) Ensure CRS is written
    ds = ds.rio.write_crs(crs, inplace=False)
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

    zarr_path = f'{query.tmp_dir}/{query.stub}_paddockTS.zarr'
    result.to_zarr(zarr_path, mode='w')
    print(f'Saved to {zarr_path}')
    return result


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    result = make_paddockTS(query)
    print(result)
    for var in result.data_vars:
        print(f'{var}: {float(result[var].mean()):.3f}')


if __name__ == '__main__':
    test()
