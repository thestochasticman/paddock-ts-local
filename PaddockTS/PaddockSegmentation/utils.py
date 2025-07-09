import numpy as np

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
