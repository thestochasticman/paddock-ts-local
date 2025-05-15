from PaddockTSLocal.utils import load_pickle
from xarray.core.dataset import Dataset
from typing_extensions import Union
from numpy.typing import NDArray
import xarray as xr
import numpy as np
import rioxarray

def fourier_mean(x, n=3, step=5):
    result = np.empty((x.shape[0], x.shape[1], n), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y = np.fft.fft(x[i,j,:])
            for k in range(n):
                result[i,j,k] = np.mean(np.abs(y[1+k*step:((k+1)*step+1) or None]))

    return result

def completion(arr):
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
            mask = np.isnan(dat[:, :, t])
            if mask.any():
                dat[mask, t] = fill[mask]
            else:
                break
    return dat

def compute_ndwi_fourier(ds: Dataset)->NDArray[np.float64]:
    keep_vars = ['nbart_red','nbart_green','nbart_blue','nbart_nir_1']
    data = ds[keep_vars].to_array().transpose('y', 'x','variable', 'time').values.astype(np.float32)
    data[data == 0] = np.nan
    data /= 10000.
    ndwi_obs = (data[:,:,1,:]-data[:,:,3,:])/(data[:,:,1,:]+data[:,:,3,:]) # w = water. (g-nir)/(g+nir)
    ndwi = completion(ndwi_obs)
    f2 = fourier_mean(ndwi)
    print(f2.shape)
    return f2

def rescale(im: NDArray[np.float64])->NDArray[np.float64]:
    '''rescale raster (im) to between 0 and 255.
    Attempts to rescale each band separately, then join them back together to achieve exact same shape as input.
    Note. Assumes multiple bands, otherwise breaks'''
    n_bands = im.shape[2]
    _im = np.empty(im.shape)
    for n in range(0,n_bands):
        matrix = im[:,:,n]
        scaled_matrix = (255*(matrix - np.min(matrix))/np.ptp(matrix)).astype(int)
        _im[:,:,n] = scaled_matrix
    return(_im)

def convert_to_geotif(ds2: Dataset, inp: NDArray[np.float64])->xr.DataArray:
    inp = inp.astype(np.float32)
    '''prepares a 3-band image for SAMgeo. 
    First rescale bands in the image. Then convert to xarray with original geo info. Then save geotif'''
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
    return convert_to_geotif(ds2, rescale(compute_ndwi_fourier(ds2)))

def save_ndwi_geotiff(data_xr: xr.DataArray, path)->None:
    data_xr.transpose('band', 'y', 'x').rio.to_raster(path)

def presegment(ds2: Union[str, Dataset], path: str)->xr.DataArray:
    ds2 = load_pickle(ds2) if isinstance(ds2, str) else ds2
    ndwi_geotiff = ds2_to_ndwi_geotiff(ds2)
    save_ndwi_geotiff(ndwi_geotiff, path)
    print(ndwi_geotiff)
    return ndwi_geotiff
    # save_ndwi_geotiff(ds2_to_ndwi_geotiff(load_pickle(ds2) if isinstance(ds2, str) else ds2), path)

def test():
    from PaddockTSLocal.Query import get_example_query
    from os.path import join
    from os import makedirs
    from os import getcwd
    
    query = get_example_query()
    ds2_dir: str=join(getcwd(), 'Data', 'ds2')
    makedirs(ds2_dir, exist_ok=True)
    path_ds2 = join(ds2_dir, f"{query.get_stub()}.pkl")
    ndwi_tiff_dir:  str=join(getcwd(), 'Data', 'ndwi_tif')
    makedirs(ndwi_tiff_dir, exist_ok=True)
    path_ndwi_tiff = join(ndwi_tiff_dir, f"{query.get_stub()}.tif")
    presegment(path_ds2, path_ndwi_tiff)

if __name__ == '__main__':
    test()
