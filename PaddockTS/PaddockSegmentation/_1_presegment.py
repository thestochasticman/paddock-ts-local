from PaddockTS.query import Query
from os.path import exists
from PaddockTS.Data.download_sentinel2 import download_sentinel2
from PaddockTS.PaddockSegmentation.utils import fourier_mean
from PaddockTS.PaddockSegmentation.utils import completion
import pickle
from xarray.core.dataset import Dataset
from typing_extensions import Union
from numpy.typing import NDArray
import xarray as xr
import numpy as np
import rioxarray

# """
# _1_presegemnt.py stands for the precossesing needed to perform paddock segmentation.
# The '_1' in it entails that it is the first part of paddock segmentation
# """

# def compute_ndwi_fourier(ds: Dataset)->NDArray[np.float64]:
#     """
#       From an xarray Dataset of DS2 bands, compute the NDWI time series,
#       fill missing data, then extract Fourier summary features.
      
#       Steps:
#         1. Stack the 4 bands (red, green, blue, NIR) into an array.
#         2. Mask out zeros, scale reflectance to [0,1].
#         3. Compute NDWI_obs = (green − nir) / (green + nir).
#         4. Fill gaps with `completion`.
#         5. Compute Fourier-band means with `fourier_mean`.
#     """
#     keep_vars = ['nbart_red','nbart_green','nbart_blue','nbart_nir_1']
#     data = ds[keep_vars].to_array().transpose('y', 'x','variable', 'time').values.astype(np.float32)
#     data[data == 0] = np.nan
#     data /= 10000.
#     ndwi_obs = (data[:,:,1,:]-data[:,:,3,:])/(data[:,:,1,:]+data[:,:,3,:]) # w = water. (g-nir)/(g+nir)
#     ndwi = completion(ndwi_obs)
#     f2 = fourier_mean(ndwi)
#     return f2

# def rescale(im: NDArray[np.float64])->NDArray[np.float64]:
#     '''
#     Rescale raster (im) to between 0 and 255.
#     This makes it suitable for a 8-bit GeoTIFF export

#     Attempts to rescale each band separately, then join them back together to achieve exact same shape as input.
#     Note. Assumes multiple bands, otherwise breaks'''
#     n_bands = im.shape[2]
#     _im = np.empty(im.shape)
#     for n in range(0,n_bands):
#         matrix = im[:,:,n]
#         scaled_matrix = (255*(matrix - np.min(matrix))/np.ptp(matrix)).astype(int)
#         _im[:,:,n] = scaled_matrix
#     return(_im)

# def convert_to_geotiff(ds2: Dataset, inp: NDArray[np.float64])->xr.DataArray:
#     inp = inp.astype(np.float32)
#     '''
#     Take a 3-band H×W×3 array, wrap it in xarray with original geolocation,
#     and attach the CRS for saving as GeoTIFF.
#     '''
#     if inp.shape[2] == 3:
#         image = rescale(inp) # 3d array 
#         lat = list(ds2.y.values) # latitude is the same size as the first axis
#         lon = list(ds2.x.values) # longitude is the same size as second axis
#         bands = list(range(1,image.shape[2]+1)) # band is the 3rd axis
#         crs = ds2.rio.crs
#         # create xarray object
#         data_xr = xr.DataArray(
#                 image, 
#                 coords={'y': lat,'x': lon,'band': bands}, 
#                 dims=["y", "x", "band"]
#         )
#         data_xr.rio.write_crs(crs, inplace=True)
#         return data_xr
#         # save as geotif:
#         # data_xr.transpose('band', 'y', 'x').rio.to_raster(path)
#     else:
#         print("Input image is wrong shape! No action taken")

# def ds2_to_ndwi_geotiff(ds2: Dataset)->xr.DataArray:
#     return convert_to_geotiff(ds2, rescale(compute_ndwi_fourier(ds2)))

# def save_ndwi_geotiff(data_xr: xr.DataArray, path)->None:
#     print(path, '----------')
#     data_xr.transpose('band', 'y', 'x').rio.to_raster(path)


import numpy as np
import xarray as xr
import rioxarray

from numpy.typing import NDArray
from xarray.core.dataset import Dataset

from PaddockTS.PaddockSegmentation.utils import fourier_mean, completion


def compute_ndwi_fourier(ds: Dataset) -> NDArray[np.float32]:
    """
    Compute NDWI = (green - nir) / (green + nir), gap-fill, then Fourier summary.
    Uses only green + nir to reduce memory vs stacking 4 bands.
    """
    green = ds["nbart_green"].transpose("y", "x", "time").values.astype(np.float32)
    nir   = ds["nbart_nir_1"].transpose("y", "x", "time").values.astype(np.float32)

    green[green == 0] = np.nan
    nir[nir == 0] = np.nan

    green /= 10000.0
    nir   /= 10000.0

    den = green + nir
    ndwi_obs = (green - nir) / den
    ndwi_obs[~np.isfinite(ndwi_obs)] = np.nan

    ndwi = completion(ndwi_obs)
    f = fourier_mean(ndwi)  # expected shape (H, W, B) or (H, W) depending on your impl
    return f.astype(np.float32)


def rescale_uint8(im: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Scale each band independently to [0, 255] as uint8.
    Handles NaNs and constant bands safely.
    Expects im shape (H, W, B) or (H, W).
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
    Wrap uint8 H×W×B into a georeferenced DataArray and copy CRS/transform from ds2.
    Ensures output stays uint8 (so FastSAM can open it).
    """
    if inp_uint8.ndim == 2:
        inp_uint8 = inp_uint8[:, :, None]

    # FastSAM likes 3-channel images. If you have !=3 bands, coerce here.
    if inp_uint8.shape[2] == 1:
        image = np.repeat(inp_uint8, 3, axis=2)
    elif inp_uint8.shape[2] >= 3:
        image = inp_uint8[:, :, :3]
    else:  # 2 bands
        image = np.concatenate([inp_uint8, inp_uint8[:, :, :1]], axis=2)

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
        # If ds2 lacks a transform, rioxarray may infer it from coords; leave as-is.
        pass

    return data_xr


def ds2_to_ndwi_geotiff(ds2: Dataset) -> xr.DataArray:
    f = compute_ndwi_fourier(ds2)              # float32 features
    u8 = rescale_uint8(f)                      # uint8 features
    return convert_to_geotiff(ds2, u8)         # uint8 DataArray


def save_ndwi_geotiff(data_xr: xr.DataArray, path: str) -> None:
    # IMPORTANT: write uint8, band-first
    data_xr.transpose("band", "y", "x").rio.to_raster(path)


def presegment(query: Query)->xr.DataArray:
    if not exists(query.path_ds2):
        raise FileNotFoundError(f"You have not downloaded ds2 data for the given stub yet.")
    ds2 = pickle.load(open(query.path_ds2, 'rb'))
    ndwi_geotiff = ds2_to_ndwi_geotiff(ds2)
    save_ndwi_geotiff(ndwi_geotiff, query.path_preseg_tif)
    return ndwi_geotiff

def test():
    from PaddockTS.query import get_example_query
    from os.path import exists
    from os import remove


    query = get_example_query()
    if exists(query.path_preseg_tif): remove(query.path_preseg_tif)
    presegment(query)
    print(query.path_preseg_tif)
    return exists(query.path_preseg_tif)

if __name__ == '__main__':
    print(test())
