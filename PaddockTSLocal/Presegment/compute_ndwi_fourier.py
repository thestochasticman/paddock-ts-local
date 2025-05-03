import numpy as np
import hdstats
import numpy as np
from numpy.typing import NDArray
from xarray.core.dataset import Dataset

def f(ds: Dataset)->NDArray[np.float_]:
    keep_vars = ['nbart_red','nbart_green','nbart_blue','nbart_nir_1']
    data = ds[keep_vars].to_array().transpose('y', 'x','variable', 'time').values.astype(np.float32)
    data[data == 0] = np.nan
    data /= 10000.
    ndwi_obs = (data[:,:,1,:]-data[:,:,3,:])/(data[:,:,1,:]+data[:,:,3,:]) # w = water. (g-nir)/(g+nir)
    ndwi = hdstats.completion(ndwi_obs)
    f2 = hdstats.fourier_mean(ndwi)
    return f2
