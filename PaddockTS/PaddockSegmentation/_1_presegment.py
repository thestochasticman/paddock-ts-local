from PaddockTS.legend import NDWI_FOURIER_GEOTIFF_DIR
from PaddockTS.query import Query
from os.path import exists
from PaddockTS.Data.download_ds2 import download_ds2
from PaddockTS.PaddockSegmentation.utils import fourier_mean
from PaddockTS.PaddockSegmentation.utils import completion
import pickle
from PaddockTS.legend import DS2_DIR
from xarray.core.dataset import Dataset
from typing_extensions import Union
from numpy.typing import NDArray
import xarray as xr
import numpy as np
import rioxarray

"""
_1_presegemnt.py stands for the precossesing needed to perform paddock segmentation.
The '_1' in it entails that it is the first part of paddock segmentation
"""

def compute_ndwi_fourier(ds: Dataset)->NDArray[np.float64]:
    """
      From an xarray Dataset of DS2 bands, compute the NDWI time series,
      fill missing data, then extract Fourier summary features.
      
      Steps:
        1. Stack the 4 bands (red, green, blue, NIR) into an array.
        2. Mask out zeros, scale reflectance to [0,1].
        3. Compute NDWI_obs = (green − nir) / (green + nir).
        4. Fill gaps with `completion`.
        5. Compute Fourier-band means with `fourier_mean`.
    """
    keep_vars = ['nbart_red','nbart_green','nbart_blue','nbart_nir_1']
    data = ds[keep_vars].to_array().transpose('y', 'x','variable', 'time').values.astype(np.float32)
    data[data == 0] = np.nan
    data /= 10000.
    ndwi_obs = (data[:,:,1,:]-data[:,:,3,:])/(data[:,:,1,:]+data[:,:,3,:]) # w = water. (g-nir)/(g+nir)
    ndwi = completion(ndwi_obs)
    f2 = fourier_mean(ndwi)
    return f2

def rescale(im: NDArray[np.float64])->NDArray[np.float64]:
    '''
    Rescale raster (im) to between 0 and 255.
    This makes it suitable for a 8-bit GeoTIFF export

    Attempts to rescale each band separately, then join them back together to achieve exact same shape as input.
    Note. Assumes multiple bands, otherwise breaks'''
    n_bands = im.shape[2]
    _im = np.empty(im.shape)
    for n in range(0,n_bands):
        matrix = im[:,:,n]
        scaled_matrix = (255*(matrix - np.min(matrix))/np.ptp(matrix)).astype(int)
        _im[:,:,n] = scaled_matrix
    return(_im)

def convert_to_geotiff(ds2: Dataset, inp: NDArray[np.float64])->xr.DataArray:
    inp = inp.astype(np.float32)
    '''
    Take a 3-band H×W×3 array, wrap it in xarray with original geolocation,
    and attach the CRS for saving as GeoTIFF.
    '''
    if inp.shape[2] == 3:
        image = rescale(inp) # 3d array 
        lat = list(ds2.y.values) # latitude is the same size as the first axis
        lon = list(ds2.x.values) # longitude is the same size as second axis
        bands = list(range(1,image.shape[2]+1)) # band is the 3rd axis
        crs = ds2.rio.crs
        # create xarray object
        data_xr = xr.DataArray(
                image, 
                coords={'y': lat,'x': lon,'band': bands}, 
                dims=["y", "x", "band"]
        )
        data_xr.rio.write_crs(crs, inplace=True)
        return data_xr
        # save as geotif:
        # data_xr.transpose('band', 'y', 'x').rio.to_raster(path)
    else:
        print("Input image is wrong shape! No action taken")

def ds2_to_ndwi_geotiff(ds2: Dataset)->xr.DataArray:
    return convert_to_geotiff(ds2, rescale(compute_ndwi_fourier(ds2)))

def save_ndwi_geotiff(data_xr: xr.DataArray, path)->None:
    print(path, '----------')
    data_xr.transpose('band', 'y', 'x').rio.to_raster(path)

def presegment(query: Query)->xr.DataArray:
    stub = query.stub
    if not exists(query.path_ds2):
        raise FileNotFoundError(f"You have not downloaded ds2 data for the given stub yet.")
    
    ds2 = pickle.load(open(query.path_ds2, 'rb'))
    ndwi_geotiff = ds2_to_ndwi_geotiff(ds2)
    path = f"{NDWI_FOURIER_GEOTIFF_DIR}/{stub}.tif"
    save_ndwi_geotiff(ndwi_geotiff, path)
    return ndwi_geotiff
    # save_ndwi_geotiff(ds2_to_ndwi_geotiff(load_pickle(ds2) if isinstance(ds2, str) else ds2), path)

def test():
    from PaddockTS.query import get_example_query
    from os.path import exists
    from os import remove

    stub = 'test_example_query'
    query = get_example_query()
    path = f"{NDWI_FOURIER_GEOTIFF_DIR}/{stub}.tif"
    if exists(path): remove(path)
    presegment(stub)
    print(path)
    return exists(path)

if __name__ == '__main__':
    print(test())