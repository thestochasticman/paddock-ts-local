import numpy as np
import xarray as xr
from numpy.typing import NDArray
from os.path import exists
from PaddockTS.query import Query


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


def presegment(query: Query) -> str:
    """
    Create a 3-band uint8 NDWI Fourier GeoTIFF from Sentinel-2 data.
    Returns the path to the saved preseg GeoTIFF.
    """
    preseg_path = f"{query.tmp_dir}/{query.stub}_preseg.tif"
    if exists(preseg_path):
        return preseg_path

    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2

    if not exists(query.sentinel2_path):
        download_sentinel2(query)

    ds = xr.open_zarr(query.sentinel2_path, chunks=None)
    features = compute_ndwi_fourier(ds)
    uint8_image = rescale_uint8(features)

    save_geotiff(ds, uint8_image, preseg_path)
    print(f"Saved preseg to {preseg_path}")
    return preseg_path
