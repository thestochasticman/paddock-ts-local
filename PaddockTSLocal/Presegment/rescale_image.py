import numpy as np
from numpy.typing import NDArray

# def f(image: NDArray[np.float_])->NDArray[np.int_]:
#     '''
#         rescale raster (img) to between 0 and 255
#     '''

#     min_vals = image.min(axis=(0, 1), keepdims=True)
#     ptp_vals = image.ptp(axis=(0, 1), keepdims=True)
#     scaled = 255 * (image - min_vals) / ptp_vals
#     return scaled.astype(np.float32)

def f(im):
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