"""
Utility functions for PaddockSegmentation2 (scikit-image approach).

This module provides helper functions for temporal feature extraction
and array normalization without requiring network access or heavy ML models.
"""

import numpy as np
from numpy.typing import NDArray


def normalize(arr: NDArray) -> NDArray[np.float32]:
    """
    Normalize array to 0-1 range.

    Args:
        arr: Input array of any shape

    Returns:
        Normalized array with values in [0, 1] range
    """
    arr = arr.astype(np.float32)
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max - arr_min > 0:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr)


def completion(arr: NDArray) -> NDArray:
    """
    Forward-fill NaN values using last valid observation (vectorized).
    For initial NaNs with no forward history, uses median of all valid values.

    Args:
        arr: Input array with shape (H, W, T) where T is time dimension

    Returns:
        Array with NaN values filled
    """
    import pandas as pd

    arr = arr.copy()
    h, w, t = arr.shape

    # Reshape to (pixels, time) for vectorized pandas operations
    flat = arr.reshape(-1, t)

    # Convert to DataFrame for efficient ffill
    df = pd.DataFrame(flat.T)  # (time, pixels)
    df = df.ffill()  # Forward fill
    df = df.bfill()  # Back fill any remaining leading NaNs

    # Fill any fully-NaN pixels with 0
    df = df.fillna(0)

    # Reshape back
    arr = df.values.T.reshape(h, w, t)

    return arr


def compute_edge_magnitude(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute gradient magnitude using Canny-like approach.

    Uses Gaussian smoothing + Sobel for cleaner edges that better
    highlight paddock boundaries.

    Args:
        arr: 2D array (H, W)

    Returns:
        Gradient magnitude array (H, W)
    """
    from scipy import ndimage

    # Smooth first to reduce noise (sigma=1.5 for 10m resolution)
    smoothed = ndimage.gaussian_filter(arr, sigma=1.5)

    # Sobel filters for x and y gradients
    sobel_x = ndimage.sobel(smoothed, axis=1)
    sobel_y = ndimage.sobel(smoothed, axis=0)

    # Gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Enhance edges: apply non-linear scaling to boost weak edges
    magnitude = np.power(magnitude, 0.7)  # Gamma correction to boost

    return magnitude


def compute_temporal_features(
    ndvi: NDArray[np.float32],
    ndwi: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """
    Compute temporal statistics from spectral index time series.

    Args:
        ndvi: NDVI time series with shape (H, W, T)
        ndwi: Optional NDWI time series with shape (H, W, T)

    Returns:
        Feature array with shape (H, W, C) containing:
        - Band 0: Median NDVI (normalized) - overall vegetation level
        - Band 1: Std NDVI (normalized) - highlights temporal variation
        - Band 2: Edge magnitude of median NDVI - highlights boundaries
        If ndwi provided:
        - Band 3: Median NDWI (normalized) - water/moisture content
    """
    ndvi_median = np.nanmedian(ndvi, axis=2)
    ndvi_std = np.nanstd(ndvi, axis=2)

    # Compute edge magnitude on median NDVI
    ndvi_edges = compute_edge_magnitude(ndvi_median)

    # Normalize each feature independently
    ndvi_median_norm = normalize(ndvi_median)
    ndvi_std_norm = normalize(ndvi_std)
    ndvi_edges_norm = normalize(ndvi_edges)

    feature_list = [
        ndvi_median_norm,
        ndvi_std_norm,
        ndvi_edges_norm,
    ]

    # Add NDWI if provided
    if ndwi is not None:
        ndwi_median = np.nanmedian(ndwi, axis=2)
        ndwi_median_norm = normalize(ndwi_median)
        feature_list.append(ndwi_median_norm)

    # Stack into multi-channel feature array
    features = np.stack(feature_list, axis=-1)

    return features.astype(np.float32)
