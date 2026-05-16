import numpy as np
import xarray as xr
from numpy.typing import NDArray
from datetime import datetime
from os import makedirs
from os.path import exists, dirname
from PaddockTS.query import Query
from PaddockTS.Sentinel2.check_if_valid_zarr_exists import check_if_valid_zarr_exists
from PaddockTS.PaddockSegmentation.check_if_valid_preseg_exists import check_if_valid_preseg_exists


def completion(arr):
    """Gap-fill NaN values in a (H, W, T) time series by forward-filling then backfilling."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[-1]), 0)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    i, j = np.meshgrid(
        np.arange(idx.shape[0]), np.arange(idx.shape[1]), indexing="ij"
    )
    dat = arr[i[:, :, np.newaxis], j[:, :, np.newaxis], idx]
    if np.isnan(np.sum(dat[:, :, 0])):
        fill = np.nanmean(dat, axis=-1)
        for t in range(dat.shape[-1]):
            m = np.isnan(dat[:, :, t])
            if m.any():
                dat[m, t] = fill[m]
            else:
                break
    return dat


def fourier_mean(x, n=3, step=5):
    """Extract n Fourier band means from a (H, W, T) time series -> (H, W, n)."""
    spectra = np.abs(np.fft.fft(x, axis=2))
    result = np.empty((x.shape[0], x.shape[1], n), dtype=np.float32)
    for k in range(n):
        result[:, :, k] = np.mean(spectra[:, :, 1 + k * step:(k + 1) * step + 1], axis=2)
    return result


def compute_ndwi_fourier(ds: xr.Dataset) -> NDArray[np.float32]:
    """Compute NDWI from sentinel-2 bands, gap-fill, then Fourier summary -> (H, W, 3)."""
    green = ds["nbart_green"].transpose("y", "x", "time").values.astype(np.float32)
    nir = ds["nbart_nir_1"].transpose("y", "x", "time").values.astype(np.float32)

    green[green == 0] = np.nan
    nir[nir == 0] = np.nan
    green /= 10000.0
    nir /= 10000.0

    ndwi_obs = (green - nir) / (green + nir)
    ndwi_obs[~np.isfinite(ndwi_obs)] = np.nan

    ndwi = completion(ndwi_obs)
    return fourier_mean(ndwi).astype(np.float32)


def rescale_uint8(im: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Per-band min-max normalization to [0, 255] uint8."""
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


def save_geotiff(ds: xr.Dataset, uint8_image: NDArray[np.uint8], path: str) -> None:
    """Save a 3-band uint8 image as a georeferenced GeoTIFF using CRS/transform from ds."""
    import rioxarray  # noqa: F401

    if uint8_image.ndim == 2:
        uint8_image = np.repeat(uint8_image[:, :, None], 3, axis=2)
    elif uint8_image.shape[2] == 1:
        uint8_image = np.repeat(uint8_image, 3, axis=2)
    elif uint8_image.shape[2] == 2:
        uint8_image = np.concatenate([uint8_image, uint8_image[:, :, :1]], axis=2)
    elif uint8_image.shape[2] > 3:
        uint8_image = uint8_image[:, :, :3]

    da = xr.DataArray(
        uint8_image,
        coords={"y": ds.y.values, "x": ds.x.values, "band": np.arange(1, 4)},
        dims=("y", "x", "band"),
    )
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    if ds.rio.crs is not None:
        da.rio.write_crs(ds.rio.crs, inplace=True)
    try:
        da.rio.write_transform(ds.rio.transform(), inplace=True)
    except Exception:
        pass
    da.transpose("band", "y", "x").rio.to_raster(path)


def presegment(query: Query, ds_sentinel2=None) -> str:
    """
    Create a 3-band uint8 NDWI Fourier GeoTIFF from Sentinel-2 data.
    Returns the path to the saved preseg GeoTIFF.
    """
    if check_if_valid_preseg_exists(query.preseg_path):
        return query.preseg_path

    if ds_sentinel2 is None:
        if not check_if_valid_zarr_exists(query.sentinel2_path):
            from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
            download_sentinel2(query)
        ds = xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords='all')
    else:
        ds = ds_sentinel2
    features = compute_ndwi_fourier(ds)
    uint8_image = rescale_uint8(features)

    makedirs(dirname(query.preseg_path), exist_ok=True)
    save_geotiff(ds, uint8_image, query.preseg_path)
    # Touch ``<tif>._SUCCESS`` *after* the TIFF write completes; its presence
    # is what the next call uses as the cache-validity check.
    with open(f'{query.preseg_path}._SUCCESS', 'w') as f:
        f.write(datetime.utcnow().isoformat() + 'Z')
    print(f"Saved preseg to {query.preseg_path}")
    return query.preseg_path
