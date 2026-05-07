"""Utility functions for SMIPS downscaling."""

from __future__ import annotations

import numpy as np
import rasterio
import xarray as xr
from numpy.typing import NDArray
from os.path import exists
from scipy.sparse import lil_matrix, csr_matrix, diags
from rasterio.enums import Resampling


def compute_terrain_features(
    query,
    s2: xr.Dataset,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute TWI and HLI terrain features reprojected to S2 grid.

    Parameters
    ----------
    query : Query
        PaddockTS query with bbox.
    s2 : xr.Dataset
        Sentinel-2 dataset for grid reference.

    Returns
    -------
    twi : ndarray
        TWI reprojected to S2 grid, shape (ny, nx).
    hli : ndarray
        HLI reprojected to S2 grid, shape (ny, nx).
    """
    from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import (
        download_terrain,
        get_filename as terrain_get_filename,
    )
    from PaddockTS.Environmental.TerrainTiles.utils import (
        calculate_slope,
        calculate_aspect,
        calculate_twi,
        calculate_hli,
        pysheds_accumulation,
    )

    # Get terrain file
    terrain_tif = terrain_get_filename(query)
    if not exists(terrain_tif):
        print('    downloading terrain data...')
        download_terrain(query)

    # Calculate terrain derivatives
    slope = calculate_slope(terrain_tif)
    aspect = calculate_aspect(terrain_tif)
    _, _, _, acc = pysheds_accumulation(terrain_tif)
    twi = calculate_twi(acc, slope)

    # Get latitude for HLI
    lat_center = (query.bbox[1] + query.bbox[3]) / 2
    hli = calculate_hli(slope, aspect, lat_center)

    # Read terrain raster metadata for reprojection
    with rasterio.open(terrain_tif) as src:
        terrain_transform = src.transform
        terrain_crs = src.crs

    # Create xarray DataArrays for reprojection
    ny_t, nx_t = twi.shape
    y_coords = [terrain_transform.f + terrain_transform.e * i for i in range(ny_t)]
    x_coords = [terrain_transform.c + terrain_transform.a * i for i in range(nx_t)]

    twi_da = xr.DataArray(
        twi.astype(np.float32),
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
    )
    twi_da = twi_da.rio.write_crs(terrain_crs)

    hli_da = xr.DataArray(
        hli.astype(np.float32),
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
    )
    hli_da = hli_da.rio.write_crs(terrain_crs)

    # Reproject to S2 grid
    import rioxarray  # noqa: F401

    s2_crs = s2.rio.crs if hasattr(s2, 'rio') and s2.rio.crs else 'EPSG:6933'

    twi_reproj = twi_da.rio.reproject_match(
        s2.isel(time=0) if 'time' in s2.dims else s2,
        resampling=Resampling.bilinear,
    )
    hli_reproj = hli_da.rio.reproject_match(
        s2.isel(time=0) if 'time' in s2.dims else s2,
        resampling=Resampling.bilinear,
    )

    return twi_reproj.values, hli_reproj.values


def reproject_smips_to_s2_grid(
    smips: xr.DataArray,
    s2: xr.Dataset,
    coarse_resolution: float = 1000.0,
) -> xr.DataArray:
    """Reproject SMIPS from EPSG:4326 to EPSG:6933 covering S2 extent.

    Keeps coarse resolution (~1km) but aligns to S2 coordinate system.

    Parameters
    ----------
    smips : xr.DataArray
        SMIPS data in EPSG:4326 (lat/lon), dims (y, x).
    s2 : xr.Dataset
        Sentinel-2 dataset in EPSG:6933, used for bounds reference.
    coarse_resolution : float
        Output resolution in meters. Default 1000m.

    Returns
    -------
    xr.DataArray
        SMIPS reprojected to EPSG:6933 at coarse resolution.
    """
    import rioxarray  # noqa: F401

    # Ensure CRS is set
    if smips.rio.crs is None:
        smips = smips.rio.write_crs('EPSG:4326')

    # Get S2 bounds
    s2_bounds = (
        float(s2.x.min()),
        float(s2.y.min()),
        float(s2.x.max()),
        float(s2.y.max()),
    )

    # Reproject to EPSG:6933 at coarse resolution
    smips_reproj = smips.rio.reproject(
        'EPSG:6933',
        resolution=coarse_resolution,
        resampling=Resampling.nearest,
    )

    # Clip to S2 bounds with small buffer
    buffer = coarse_resolution * 2
    smips_reproj = smips_reproj.rio.clip_box(
        minx=s2_bounds[0] - buffer,
        miny=s2_bounds[1] - buffer,
        maxx=s2_bounds[2] + buffer,
        maxy=s2_bounds[3] + buffer,
    )

    return smips_reproj


def build_aggregation_matrix(
    smips_aligned: xr.DataArray,
    s2: xr.Dataset,
) -> csr_matrix:
    """Build sparse matrix mapping fine S2 pixels to coarse SMIPS pixels.

    Each row k corresponds to a coarse SMIPS pixel. The row contains
    weights 1/n_k for each fine pixel falling within that coarse pixel,
    where n_k is the count of fine pixels in coarse pixel k.

    Parameters
    ----------
    smips_aligned : xr.DataArray
        SMIPS data reprojected to EPSG:6933, dims (y, x).
    s2 : xr.Dataset
        Sentinel-2 dataset, dims include (y, x).

    Returns
    -------
    csr_matrix
        Shape (n_coarse, n_fine). A @ theta_fine gives aggregated values.
    """
    # Get coordinate arrays
    y_fine = s2.y.values
    x_fine = s2.x.values
    y_coarse = smips_aligned.y.values
    x_coarse = smips_aligned.x.values

    n_fine = len(y_fine) * len(x_fine)
    n_coarse = len(y_coarse) * len(x_coarse)

    # Compute coarse pixel spacing
    if len(y_coarse) > 1:
        dy_coarse = abs(float(y_coarse[1] - y_coarse[0]))
    else:
        dy_coarse = 1000.0

    if len(x_coarse) > 1:
        dx_coarse = abs(float(x_coarse[1] - x_coarse[0]))
    else:
        dx_coarse = 1000.0

    # Build sparse matrix
    A = lil_matrix((n_coarse, n_fine), dtype=np.float64)

    for k_y, yc in enumerate(y_coarse):
        # Find fine y indices within this coarse row
        in_y = (y_fine <= yc + dy_coarse / 2) & (y_fine > yc - dy_coarse / 2)
        y_indices = np.where(in_y)[0]

        if len(y_indices) == 0:
            continue

        for k_x, xc in enumerate(x_coarse):
            # Find fine x indices within this coarse column
            in_x = (x_fine >= xc - dx_coarse / 2) & (x_fine < xc + dx_coarse / 2)
            x_indices = np.where(in_x)[0]

            if len(x_indices) == 0:
                continue

            # Coarse pixel linear index
            coarse_idx = k_y * len(x_coarse) + k_x

            # Fine pixel linear indices (row-major: y * nx + x)
            fine_indices = []
            for yi in y_indices:
                for xi in x_indices:
                    fine_indices.append(yi * len(x_fine) + xi)

            n_in_cell = len(fine_indices)
            if n_in_cell > 0:
                A[coarse_idx, fine_indices] = 1.0 / n_in_cell

    return A.tocsr()


def build_laplacian_matrix(ny: int, nx: int) -> csr_matrix:
    """Build 2D discrete Laplacian for spatial smoothness.

    Uses 4-connectivity (up, down, left, right neighbors).
    L @ theta computes discrete second derivatives.

    Parameters
    ----------
    ny : int
        Number of rows in the grid.
    nx : int
        Number of columns in the grid.

    Returns
    -------
    csr_matrix
        Shape (n, n) where n = ny * nx.
    """
    n = ny * nx

    # Diagonal: each pixel has up to 4 neighbors
    main_diag = np.ones(n) * 4

    # Off-diagonals for left/right neighbors
    lr_diag = -np.ones(n - 1)
    # Zero out connections across row boundaries
    for i in range(1, ny):
        lr_diag[i * nx - 1] = 0

    # Off-diagonals for up/down neighbors
    ud_diag = -np.ones(n - nx)

    # Construct sparse matrix
    L = diags(
        [main_diag, lr_diag, lr_diag, ud_diag, ud_diag],
        [0, -1, 1, -nx, nx],
        shape=(n, n),
        format='csr',
    )

    return L


def compute_features(
    s2: xr.Dataset,
    twi: NDArray[np.float32] | None = None,
    hli: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Extract NDVI, SWIR, and optionally terrain features from Sentinel-2.

    Parameters
    ----------
    s2 : xr.Dataset
        Sentinel-2 dataset for one timestep with bands:
        nbart_nir_1, nbart_red, nbart_swir_2, nbart_swir_3.
    twi : ndarray, optional
        Topographic Wetness Index, shape (ny, nx). If provided, included in features.
    hli : ndarray, optional
        Heat Load Index, shape (ny, nx). If provided, included in features.

    Returns
    -------
    features : ndarray
        Shape (n_features, ny*nx) with [NDVI, SWIR2, SWIR3, TWI?, HLI?].
        n_features = 3 if no terrain, 5 if terrain provided.
    valid_mask : ndarray
        Shape (ny*nx,) boolean mask where all features are valid.
    """

    def _get_band(ds: xr.Dataset, name: str) -> NDArray[np.float32]:
        b = ds[name].values.astype(np.float32)
        b[b == 0] = np.nan
        b /= 10000.0
        return b

    nir = _get_band(s2, 'nbart_nir_1')
    red = _get_band(s2, 'nbart_red')
    swir2 = _get_band(s2, 'nbart_swir_2')
    swir3 = _get_band(s2, 'nbart_swir_3')

    # NDVI
    ndvi = (nir - red) / (nir + red)
    ndvi[~np.isfinite(ndvi)] = np.nan

    # Stack S2 features
    feature_list = [ndvi.ravel(), swir2.ravel(), swir3.ravel()]

    # Add terrain features if provided
    if twi is not None:
        twi_flat = twi.ravel().astype(np.float32)
        # Clip extreme TWI values
        twi_flat = np.clip(twi_flat, 0, 20)
        feature_list.append(twi_flat)

    if hli is not None:
        hli_flat = hli.ravel().astype(np.float32)
        feature_list.append(hli_flat)

    features = np.stack(feature_list, axis=0)

    # Valid where all features present
    valid_mask = ~np.isnan(features).any(axis=0)

    return features, valid_mask


def fit_linear_prior(
    smips_flat: NDArray[np.float64],
    features: NDArray[np.float32],
    A: csr_matrix,
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Fit linear model SM ~ a + b*f1 + c*f2 + ... and apply to fine scale.

    Uses ordinary least squares on coarse-scale aggregated features.
    Supports variable number of features (3 for S2-only, 5 with terrain).

    Parameters
    ----------
    smips_flat : ndarray
        Flattened coarse SMIPS values, shape (n_coarse,).
    features : ndarray
        Shape (n_features, n_fine). Features can be [NDVI, SWIR2, SWIR3]
        or [NDVI, SWIR2, SWIR3, TWI, HLI].
    A : csr_matrix
        Aggregation matrix (n_coarse, n_fine).
    valid_mask : ndarray
        Shape (n_fine,) boolean mask for valid features.

    Returns
    -------
    prior : ndarray
        Shape (n_fine,) predicted SM at fine scale. NaN where features invalid.
    """
    n_features = features.shape[0]
    n_fine = features.shape[1]

    # Aggregate features to coarse scale
    features_coarse = np.zeros((n_features, A.shape[0]), dtype=np.float64)
    for i in range(n_features):
        f = features[i].copy()
        f[~valid_mask] = 0
        features_coarse[i] = A @ f

    # Also compute valid fraction per coarse pixel
    valid_frac = A @ valid_mask.astype(np.float64)
    # Normalize by valid fraction (suppress divide-by-zero warning)
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(n_features):
            features_coarse[i] = np.where(
                valid_frac > 0.1,
                features_coarse[i] / valid_frac,
                np.nan,
            )

    # Find coarse pixels where both SMIPS and features valid
    coarse_valid = ~np.isnan(smips_flat) & ~np.isnan(features_coarse).any(axis=0)

    if coarse_valid.sum() < n_features + 2:
        # Not enough points for regression, return uniform prior
        mean_sm = np.nanmean(smips_flat)
        prior = np.full(n_fine, mean_sm, dtype=np.float64)
        prior[~valid_mask] = np.nan
        return prior

    # Design matrix [1, f1, f2, f3, ...]
    X_columns = [np.ones(coarse_valid.sum())]
    for i in range(n_features):
        X_columns.append(features_coarse[i, coarse_valid])
    X = np.column_stack(X_columns)
    y = smips_flat[coarse_valid]

    # Solve least squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        mean_sm = np.nanmean(smips_flat)
        prior = np.full(n_fine, mean_sm, dtype=np.float64)
        prior[~valid_mask] = np.nan
        return prior

    # Apply to fine scale
    prior = np.full(n_fine, np.nan, dtype=np.float64)
    # Start with intercept
    prior[valid_mask] = coeffs[0]
    # Add each feature contribution
    for i in range(n_features):
        prior[valid_mask] += coeffs[i + 1] * features[i, valid_mask]

    return prior


def select_nearest_smips(
    smips_cube: xr.DataArray,
    target_time: np.datetime64,
    max_gap_days: int = 1,
) -> xr.DataArray | None:
    """Select SMIPS observation nearest to target time.

    Parameters
    ----------
    smips_cube : xr.DataArray
        SMIPS data cube with time dimension.
    target_time : np.datetime64
        Target time to match.
    max_gap_days : int
        Maximum allowed gap in days. Return None if exceeded.

    Returns
    -------
    xr.DataArray or None
        2D spatial slice, or None if no valid match.
    """
    # Find the time dimension name
    time_dim = None
    for dim in smips_cube.dims:
        if dim == 'time' or dim.lower().startswith('time'):
            time_dim = dim
            break

    if time_dim is None:
        # Assume first dimension is time if it has coordinates
        dims = list(smips_cube.dims)
        if len(dims) >= 1:
            time_dim = dims[0]
        else:
            raise ValueError(f'Cannot find time dimension in SMIPS. Dims: {smips_cube.dims}')

    smips_times = smips_cube.coords[time_dim].values
    target = np.datetime64(target_time, 'D')
    smips_times_d = np.array(smips_times).astype('datetime64[D]')

    diffs = np.abs(smips_times_d - target)
    nearest_idx = int(np.argmin(diffs))

    if diffs[nearest_idx] > np.timedelta64(max_gap_days, 'D'):
        return None

    return smips_cube.isel({time_dim: nearest_idx})
