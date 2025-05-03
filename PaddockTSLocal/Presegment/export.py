from PaddockTSLocal.Presegment.rescale_image import f as rescale_image
import xarray as xr
from xarray.core.dataset import Dataset
import numpy as np
from numpy.typing import NDArray

def f(ds: Dataset, inp: NDArray[np.int_], path: str)->None:
      '''prepares a 3-band image for SAMgeo. 
      First rescale bands in the image. Then convert to xarray with original geo info. Then save geotif'''
      if inp.shape[2] == 3:
            image = rescale_image(inp) # 3d array 
            lat = list(ds.y.values) # latitude is the same size as the first axis
            lon = list(ds.x.values) # longitude is the same size as second axis
            bands = list(range(1,image.shape[2]+1)) # band is the 3rd axis
            crs = ds.rio.crs
            # create xarray object
            data_xr = xr.DataArray(
                  image, 
                  coords={'y': lat,'x': lon,'band': bands}, 
                  dims=["y", "x", "band"]
            )
            print(data_xr)
            data_xr.rio.write_crs(crs, inplace=True)
            # save as geotif:
            data_xr.transpose('band', 'y', 'x').rio.to_raster(path)
      else:
            print("Input image is wrong shape! No action taken")
