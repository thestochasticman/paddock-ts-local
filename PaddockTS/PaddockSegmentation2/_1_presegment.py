"""
Stage 1: Presegmentation using temporal spectral features.

This module computes temporal features from Sentinel-2 time series
using only numpy/scipy operations (no network access or ML models required).

Features computed (4 bands):
- Median NDVI: overall vegetation level
- Std NDVI: temporal variation (crop cycles, management)
- Edge magnitude: gradient of median NDVI (highlights boundaries)
- Median NDWI: water/moisture content (irrigation, wet areas)

Key differences from PaddockSegmentation (SAM-based):
- Uses both NDVI and NDWI
- Uses temporal statistics + edge detection
- Lighter weight, works offline
"""

import pickle
from os.path import exists

import numpy as np
import xarray as xr
import rioxarray
from numpy.typing import NDArray
from xarray.core.dataset import Dataset

from PaddockTS.query import Query
from PaddockTS.PaddockSegmentation2.utils import completion, compute_temporal_features


def compute_ndvi(ds: Dataset) -> NDArray[np.float32]:
    """
    Compute NDVI = (nir - red) / (nir + red) from Sentinel-2 dataset.

    Args:
        ds: xarray Dataset with 'nbart_red' and 'nbart_nir_1' bands

    Returns:
        NDVI time series with shape (H, W, T), values in [-1, 1]
    """
    red = ds["nbart_red"].transpose("y", "x", "time").values.astype(np.float32)
    nir = ds["nbart_nir_1"].transpose("y", "x", "time").values.astype(np.float32)

    # Mask invalid values
    red[red == 0] = np.nan
    nir[nir == 0] = np.nan

    # Scale to reflectance [0, 1]
    red /= 10000.0
    nir /= 10000.0

    # Compute NDVI
    den = nir + red
    ndvi = (nir - red) / den
    ndvi[~np.isfinite(ndvi)] = np.nan

    return ndvi


def compute_ndwi(ds: Dataset) -> NDArray[np.float32]:
    """
    Compute NDWI = (green - nir) / (green + nir) from Sentinel-2 dataset.

    NDWI highlights water bodies and wet/irrigated areas, which often
    correspond to paddock boundaries (rivers, dams, irrigation channels).

    Args:
        ds: xarray Dataset with 'nbart_green' and 'nbart_nir_1' bands

    Returns:
        NDWI time series with shape (H, W, T), values in [-1, 1]
    """
    green = ds["nbart_green"].transpose("y", "x", "time").values.astype(np.float32)
    nir = ds["nbart_nir_1"].transpose("y", "x", "time").values.astype(np.float32)

    # Mask invalid values
    green[green == 0] = np.nan
    nir[nir == 0] = np.nan

    # Scale to reflectance [0, 1]
    green /= 10000.0
    nir /= 10000.0

    # Compute NDWI
    den = green + nir
    ndwi = (green - nir) / den
    ndwi[~np.isfinite(ndwi)] = np.nan

    return ndwi


def compute_spectral_temporal_features(ds: Dataset) -> NDArray[np.float32]:
    """
    Compute NDVI and NDWI time series, gap-fill, then extract temporal statistics.

    Args:
        ds: xarray Dataset with Sentinel-2 bands

    Returns:
        Feature array with shape (H, W, 4) containing:
        - Median NDVI (vegetation level)
        - Std NDVI (temporal variation)
        - Edge magnitude (boundary detection)
        - Median NDWI (water/moisture)
    """
    # Compute and gap-fill NDVI
    ndvi = compute_ndvi(ds)
    ndvi_filled = completion(ndvi)

    # Compute and gap-fill NDWI
    ndwi = compute_ndwi(ds)
    ndwi_filled = completion(ndwi)

    # Extract temporal features
    features = compute_temporal_features(ndvi_filled, ndwi_filled)
    return features


def rescale_uint8(im: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Scale each band independently to [0, 255] as uint8.
    Handles NaNs and constant bands safely.

    Args:
        im: Input array with shape (H, W, B) or (H, W)

    Returns:
        uint8 array scaled to [0, 255]
    """
    if im.ndim == 2:
        im = im[:, :, None]

    h, w, b = im.shape
    out = np.empty((h, w, b), dtype=np.uint8)

    for i in range(b):
        band = im[:, :, i]
        finite = np.isfinite(band)

        if not np.any(finite):
            out[:, :, i] = 0
            continue

        vmin = np.nanmin(band)
        vmax = np.nanmax(band)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            out[:, :, i] = 0
            continue

        scaled = (band - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~finite] = 0.0
        out[:, :, i] = (scaled * 255.0).astype(np.uint8)

    return out


def convert_to_geotiff(ds2: Dataset, inp_uint8: NDArray[np.uint8]) -> xr.DataArray:
    """
    Wrap uint8 H x W x B array into a georeferenced DataArray.

    Args:
        ds2: Original dataset for CRS and coordinate reference
        inp_uint8: uint8 feature array (H, W, B)

    Returns:
        Georeferenced xarray DataArray
    """
    if inp_uint8.ndim == 2:
        inp_uint8 = inp_uint8[:, :, None]

    image = inp_uint8

    lat = ds2.y.values
    lon = ds2.x.values
    bands = np.arange(1, image.shape[2] + 1)

    data_xr = xr.DataArray(
        image,
        coords={"y": lat, "x": lon, "band": bands},
        dims=("y", "x", "band"),
    )

    # Keep georeferencing identical to ds2
    data_xr.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    if ds2.rio.crs is not None:
        data_xr.rio.write_crs(ds2.rio.crs, inplace=True)
    try:
        data_xr.rio.write_transform(ds2.rio.transform(), inplace=True)
    except Exception:
        pass

    return data_xr


def ds2_to_preseg_geotiff(ds2: Dataset) -> xr.DataArray:
    """
    Convert Sentinel-2 dataset to spectral temporal features GeoTIFF.

    Args:
        ds2: xarray Dataset with Sentinel-2 bands

    Returns:
        Georeferenced DataArray with temporal features (4 bands)
    """
    features = compute_spectral_temporal_features(ds2)
    u8 = rescale_uint8(features)
    return convert_to_geotiff(ds2, u8)


def save_ndvi_geotiff(data_xr: xr.DataArray, path: str) -> None:
    """Save DataArray as GeoTIFF."""
    data_xr.transpose("band", "y", "x").rio.to_raster(path)


def presegment(query: Query) -> xr.DataArray:
    """
    Main entry point for Stage 1 presegmentation.

    Loads Sentinel-2 data, computes NDVI temporal features,
    and saves as GeoTIFF.

    Args:
        query: Query object with paths and parameters

    Returns:
        Georeferenced DataArray with temporal features
    """
    if not exists(query.path_ds2):
        raise FileNotFoundError(
            f"Sentinel-2 data not found at {query.path_ds2}. "
            "Run download_sentinel2(query) first."
        )

    ds2 = pickle.load(open(query.path_ds2, 'rb'))
    preseg_geotiff = ds2_to_preseg_geotiff(ds2)
    save_ndvi_geotiff(preseg_geotiff, query.path_preseg_tif)
    return preseg_geotiff


def test():
    from PaddockTS.query import get_example_query
    from os import remove

    query = get_example_query()
    if exists(query.path_preseg_tif):
        remove(query.path_preseg_tif)
    presegment(query)
    print(f"Output: {query.path_preseg_tif}")
    return exists(query.path_preseg_tif)


if __name__ == '__main__':
    print(test())
