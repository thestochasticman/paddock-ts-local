from dea_tools.bandindices import calculate_indices
from ds_from_stac import f as ds_from_stac
from os.path import exists
import rioxarray 
import xarray as xr
from Args import Args
import numpy as np
import pickle

load_pickle = lambda path: pickle.load(open(path, 'rb'))

import numpy as np
import pandas as pd

def completion(data):
    """
    data: numpy array of shape (Y, X, T) with np.nan for missing.
    Returns same shape, with nan gaps filled by linear interpolation along T,
    then any remaining nans replaced by the mean of that pixel’s time series.
    """
    Y, X, T = data.shape
    out = np.empty_like(data, dtype=np.float32)

    for i in range(Y):
        for j in range(X):
            ts = pd.Series(data[i, j, :])
            filled = ts.interpolate(limit_direction="both")
            # if still nan (all-nan series), fill with 0
            if filled.isna().all():
                out[i, j, :] = 0.0
            else:
                out[i, j, :] = filled.fillna(filled.mean()).values
    return out


def fourier_mean(x, n=3, step=5):
    result = np.empty((x.shape[0], x.shape[1], n), dtype=np.float32)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y = np.fft.fft(x[i,j,:])
            for k in range(n):
                result[i,j,k] = np.mean(np.abs(y[1+k*step:((k+1)*step+1) or None]))

    return result

def rescale(im):
    '''rescale raster (im) to between 0 and 255.
    Attempts to rescale each band separately, then join them back together to achieve exact same shape as input.
    Note. Assumes multiple bands, otherwise breaks'''
    n_bands = im.shape[2]
    _im = np.empty(im.shape)
    for n in range(0,n_bands):
        matrix = im[:,:,n]
        scaled_matrix = (255*(matrix - np.min(matrix))/np.ptp(matrix)).astype(int)
        _im[:,:,n] = scaled_matrix
    print('output shape equals input:', im.shape == im.shape)
    return(_im)

def transform(ds):
	keep_vars = ['nbart_red','nbart_green','nbart_blue','nbart_nir_1']
	data = ds[keep_vars].to_array().transpose('y', 'x','variable', 'time').values.astype(np.float32)
	data[data == 0] = np.nan
	data /= 10000.
	ndwi_obs = (data[:,:,1,:]-data[:,:,3,:])/(data[:,:,1,:]+data[:,:,3,:]) # w = water. (g-nir)/(g+nir)
	ndwi = completion(ndwi_obs)
	f2 = fourier_mean(ndwi)
	return f2

def export_for_segmentation(ds, inp, out_stub):
    '''prepares a 3-band image for SAMgeo. 
    First rescale bands in the image. Then convert to xarray with original geo info. Then save geotif'''
    if inp.shape[2] == 3:
        image = rescale(inp) # 3d array 
        lat = list(ds.y.values) # latitude is the same size as the first axis
        lon = list(ds.x.values) # longitude is the same size as second axis
        bands = list(range(1,image.shape[2]+1)) # band is the 3rd axis
        crs = ds.rio.crs
        # create xarray object
        data_xr = xr.DataArray(image, 
                       coords={'y': lat,'x': lon,'band': bands}, 
                       dims=["y", "x", "band"])
        data_xr.rio.write_crs(crs, inplace=True)
        # save as geotif:
        data_xr.transpose('band', 'y', 'x').rio.to_raster(out_stub + '.tif')
    else:
        print("Input image is wrong shape! No action taken")

def f(args: Args = Args.from_cli()):
    ds = load_pickle(args.path_out)
    ds = calculate_indices(
        ds,
        ['NDVI', 'NDWI', 'SAVI'],
        collection='ga_s2_3'
    )
    f2 = transform(ds)
    im = rescale(f2)
    export_for_segmentation(ds, im, 'something')

def t(): return f(Args.from_cli())

if __name__ == '__main__': print('passed' if t() else 'failed')