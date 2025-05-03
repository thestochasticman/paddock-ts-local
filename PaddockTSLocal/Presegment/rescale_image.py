import numpy as np
from numpy.typing import NDArray

def f(image: NDArray[np.float_])->NDArray[np.int_]:
    '''
        rescale raster (img) to between 0 and 255
    '''

    min_vals = image.min(axis=(0, 1), keepdims=True)
    ptp_vals = image.ptp(axis=(0, 1), keepdims=True)
    scaled = 255 * (image - min_vals) / ptp_vals
    return scaled.astype(np.float32)
