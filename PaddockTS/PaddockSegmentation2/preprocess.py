import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray import Dataset
from scipy.ndimage import gaussian_filter
from os.path import exists
from PaddockTS.query import Query
from .utils import compute_ndvi, compute_ndwi, timeseries_kmeans, compute_cluster_edges, find_optimal_clusters


def compute_preseg_features(
    ds: Dataset,
    n_clusters: int | str = 8,
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    k_range: range = range(4, 16),
    scoring: str = 'coverage',
    epsilon_factor: float = 0.005,
    method: str = 'contours',
) -> NDArray[np.float32]:
    ndvi = compute_ndvi(ds)
    ndwi = compute_ndwi(ds)

    # smooth spatially per timestep
    sigma = 1.5
    for t in range(ndvi.shape[2]):
        ndvi[:, :, t] = gaussian_filter(np.nan_to_num(ndvi[:, :, t], nan=0.0), sigma=sigma)
        ndwi[:, :, t] = gaussian_filter(np.nan_to_num(ndwi[:, :, t], nan=0.0), sigma=sigma)

    if n_clusters == 'auto':
        transform = ds.rio.transform()
        crs = ds.rio.crs
        opt = find_optimal_clusters(
            ndvi, ndwi,
            transform=transform, crs=crs,
            k_range=k_range,
            min_area_ha=min_area_ha,
            max_area_ha=max_area_ha,
            min_compactness=min_compactness,
            epsilon_factor=epsilon_factor,
            method=method,
            scoring=scoring,
        )
        n_clusters = opt['optimal_k']

    labels = timeseries_kmeans(ndvi, ndwi, n_clusters=n_clusters)
    edges = compute_cluster_edges(labels)
    ndvi_median = np.nanmedian(ndvi, axis=2)
    ndwi_median = np.nanmedian(ndwi, axis=2)

    def norm(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)

    return np.stack([norm(labels.astype(np.float32)), edges, norm(ndvi_median), norm(ndwi_median)], axis=-1).astype(np.float32)


def rescale_uint8(im: NDArray[np.float32]) -> NDArray[np.uint8]:
    if im.ndim == 2:
        im = im[:, :, None]
    out = np.empty_like(im, dtype=np.uint8)
    for i in range(im.shape[2]):
        band = im[:, :, i]
        finite = np.isfinite(band)
        vmin, vmax = np.nanmin(band), np.nanmax(band)
        if not np.any(finite) or vmax <= vmin:
            out[:, :, i] = 0
            continue
        scaled = np.clip((band - vmin) / (vmax - vmin), 0, 1)
        scaled[~finite] = 0.0
        out[:, :, i] = (scaled * 255).astype(np.uint8)
    return out


def convert_to_geotiff(ds: Dataset, inp_uint8: NDArray[np.uint8]) -> xr.DataArray:
    if inp_uint8.ndim == 2:
        inp_uint8 = inp_uint8[:, :, None]
    da = xr.DataArray(
        inp_uint8,
        coords={'y': ds.y.values, 'x': ds.x.values, 'band': np.arange(1, inp_uint8.shape[2] + 1)},
        dims=('y', 'x', 'band'),
    )
    da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    if ds.rio.crs is not None:
        da.rio.write_crs(ds.rio.crs, inplace=True)
    try:
        da.rio.write_transform(ds.rio.transform(), inplace=True)
    except Exception:
        pass
    return da


def preprocess(
    query: Query,
    n_clusters: int | str = 'auto',
    min_area_ha: float = 5,
    max_area_ha: float = 1500,
    min_compactness: float = 0.1,
    k_range: range = range(4, 16),
    scoring: str = 'coverage',
    epsilon_factor: float = 0.005,
    method: str = 'contours',
) -> xr.DataArray:
    
    if not exists(query.sentinel2_path):
        raise FileNotFoundError(f'Sentinel-2 data not found at {query.sentinel2_path}. Run download_sentinel2(query) first.')

    ds = xr.open_zarr(query.sentinel2_path, chunks=None)

    features = compute_preseg_features(
        ds,
        n_clusters=n_clusters,
        min_area_ha=min_area_ha,
        max_area_ha=max_area_ha,
        min_compactness=min_compactness,
        k_range=k_range,
        scoring=scoring,
        epsilon_factor=epsilon_factor,
        method=method,
    )
    u8 = rescale_uint8(features)
    geotiff = convert_to_geotiff(ds, u8)
    tif_path = f'{query.tmp_dir}/{query.stub}_preseg.tif'
    geotiff.transpose('band', 'y', 'x').rio.to_raster(tif_path)
    print(f'Saved preseg to {tif_path}')
    return geotiff

