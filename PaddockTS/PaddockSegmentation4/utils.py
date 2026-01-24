"""
Utility functions for PaddockSegmentation3 (time series clustering approach).
"""

import numpy as np
from numpy.typing import NDArray


def normalize(arr: NDArray) -> NDArray[np.float32]:
    """Normalize array to 0-1 range."""
    arr = arr.astype(np.float32)
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max - arr_min > 0:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr)


def completion(arr: NDArray) -> NDArray:
    """
    Forward-fill NaN values using last valid observation (vectorized).
    For initial NaNs with no forward history, uses back-fill.
    """
    import pandas as pd

    arr = arr.copy()
    h, w, t = arr.shape

    flat = arr.reshape(-1, t)
    df = pd.DataFrame(flat.T)
    df = df.ffill().bfill().fillna(0)
    arr = df.values.T.reshape(h, w, t)

    return arr


def compute_cluster_edges(labels: NDArray[np.int32]) -> NDArray[np.float32]:
    """
    Compute edge magnitude at cluster boundaries.

    Creates a gradient map where cluster transitions have high values.
    This serves as the elevation map for watershed.

    Args:
        labels: Cluster label array (H, W)

    Returns:
        Edge magnitude array (H, W) in [0, 1]
    """
    from scipy import ndimage

    h, w = labels.shape

    # Compute gradient of cluster labels
    # Where labels change, gradient is high
    sobel_x = ndimage.sobel(labels.astype(np.float32), axis=1)
    sobel_y = ndimage.sobel(labels.astype(np.float32), axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to [0, 1]
    if edges.max() > 0:
        edges = edges / edges.max()

    # Apply gaussian smoothing to spread edges slightly
    edges = ndimage.gaussian_filter(edges, sigma=2)

    return edges.astype(np.float32)
