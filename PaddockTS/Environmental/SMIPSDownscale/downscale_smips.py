"""Downscale SMIPS soil moisture to Sentinel-2 resolution using convex optimization.

Uses mass conservation constraints to ensure aggregated fine-scale values match
coarse SMIPS observations, while leveraging Sentinel-2 SWIR and NDVI to inform
sub-pixel spatial patterns.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import makedirs
from os.path import exists

import cvxpy as cp
import numpy as np
import xarray as xr

from PaddockTS.query import Query
from PaddockTS.Environmental.SMIPS import download_smips

from .smips_downscale_config import SMIPSDownscaleConfig
from .utils import (
    reproject_smips_to_s2_grid,
    build_aggregation_matrix,
    build_laplacian_matrix,
    compute_features,
    compute_terrain_features,
    fit_linear_prior,
    select_nearest_smips,
)


get_filename = lambda q: f'{q.tmp_dir}/Environmental/{q.stub}_smips_downscaled.nc'


def downscale_timestep(
    smips_t: xr.DataArray,
    s2_t: xr.Dataset,
    config: SMIPSDownscaleConfig,
    twi: np.ndarray | None = None,
    hli: np.ndarray | None = None,
) -> xr.DataArray | None:
    """Solve convex optimization for one timestep.

    Formulation:
        minimize ||θ - prior||² + λ||L @ θ||²
        subject to: A @ θ = smips_coarse (mass conservation)
                    θ >= 0 (non-negativity)

    Parameters
    ----------
    smips_t : xr.DataArray
        SMIPS for this timestep, already reprojected to EPSG:6933.
    s2_t : xr.Dataset
        Sentinel-2 bands for this timestep.
    config : SMIPSDownscaleConfig
        Optimization parameters.
    twi : ndarray, optional
        Topographic Wetness Index at S2 resolution.
    hli : ndarray, optional
        Heat Load Index at S2 resolution.

    Returns
    -------
    xr.DataArray or None
        Downscaled soil moisture at S2 resolution, or None if failed.
    """
    ny = s2_t.y.size
    nx = s2_t.x.size
    n_fine = ny * nx

    # Extract features from S2 (and terrain if provided)
    features, valid_mask = compute_features(s2_t, twi=twi, hli=hli)

    # Build aggregation matrix
    A = build_aggregation_matrix(smips_t, s2_t)
    n_coarse = A.shape[0]

    # Flatten SMIPS values
    smips_flat = smips_t.values.ravel().astype(np.float64)

    # Find valid coarse pixels (have SMIPS data)
    coarse_valid = ~np.isnan(smips_flat)
    if coarse_valid.sum() == 0:
        return None

    # Compute prior from linear model
    prior = fit_linear_prior(smips_flat, features, A, valid_mask)

    # Replace NaN in prior with mean SMIPS value
    mean_sm = np.nanmean(smips_flat)
    prior_filled = np.where(np.isnan(prior), mean_sm, prior)

    # Build Laplacian for smoothness
    L = build_laplacian_matrix(ny, nx)

    # CVX Problem
    theta = cp.Variable(n_fine)

    # Constraints: mass conservation for valid coarse pixels
    A_valid = A[coarse_valid, :]
    smips_valid = smips_flat[coarse_valid]
    constraints = [
        A_valid @ theta == smips_valid,
        theta >= 0,
    ]

    # Objective: data fidelity + smoothness
    data_term = cp.sum_squares(theta - prior_filled)
    smooth_term = cp.sum_squares(L @ theta)

    objective = cp.Minimize(data_term + config.lambda_smoothness * smooth_term)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(
            solver=getattr(cp, config.solver),
            max_iters=config.max_iters,
            verbose=config.verbose,
        )
    except cp.SolverError as e:
        print(f'    Solver error: {e}')
        return None

    if problem.status not in ['optimal', 'optimal_inaccurate']:
        print(f'    Optimization status: {problem.status}')
        return None

    # Reshape to grid
    sm_fine = theta.value.reshape(ny, nx)

    return xr.DataArray(
        sm_fine,
        dims=['y', 'x'],
        coords={'y': s2_t.y, 'x': s2_t.x},
        attrs={'units': 'mm', 'long_name': 'Downscaled soil moisture'},
    )


def downscale_smips(
    query: Query,
    config: SMIPSDownscaleConfig = SMIPSDownscaleConfig(),
    workers: int = 4,
) -> xr.DataArray:
    """Downscale SMIPS soil moisture to Sentinel-2 resolution.

    Uses convex optimization with mass conservation constraints.
    Output is only generated for dates where Sentinel-2 observations exist.

    Parameters
    ----------
    query : Query
        PaddockTS query with bbox, start, end dates.
    config : SMIPSDownscaleConfig
        Optimization parameters.
    workers : int
        Parallel workers for timestep processing.

    Returns
    -------
    xr.DataArray
        Dims: (time, y, x) at S2 resolution (10m, EPSG:6933).
        Time: subset of S2 observation times.
    """
    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    filename = get_filename(query)

    # Check cache
    if exists(filename):
        print(f'  cached: {filename}')
        ds = xr.open_dataset(filename)
        data_vars = [v for v in ds.data_vars if v != 'spatial_ref']
        da = ds[data_vars[0]].load()
        ds.close()
        return da

    # Load dependencies
    print('  loading SMIPS data...')
    # Load SMIPS directly to avoid cache bug in download_smips
    from PaddockTS.Environmental.SMIPS.download_smips import get_filename as smips_get_filename
    smips_path = smips_get_filename(query)
    if exists(smips_path):
        print(f'    loading from: {smips_path}')
        smips_ds = xr.open_dataset(smips_path)
        # Find the actual data variable (not spatial_ref or other metadata)
        data_vars = [v for v in smips_ds.data_vars if v != 'spatial_ref']
        if not data_vars:
            raise ValueError(f'No data variables found in SMIPS file: {list(smips_ds.data_vars)}')
        var_name = data_vars[0]
        smips_cube = smips_ds[var_name].load()
    else:
        smips_cube = download_smips(query)
    print(f'    SMIPS dims: {smips_cube.dims}, shape: {smips_cube.shape}')

    print('  loading Sentinel-2 data...')
    if not exists(query.sentinel2_path):
        raise FileNotFoundError(
            f'Sentinel-2 data not found at {query.sentinel2_path}. '
            'Run download_sentinel2() first.'
        )
    s2_ds = xr.open_zarr(query.sentinel2_path, chunks=None, decode_coords="all")

    s2_times = s2_ds.time.values
    print(f'  found {len(s2_times)} S2 timesteps')

    # Load terrain features if enabled
    twi = None
    hli = None
    if config.use_terrain:
        print('  loading terrain features...')
        twi, hli = compute_terrain_features(query, s2_ds)
        print(f'    TWI range: {np.nanmin(twi):.1f} - {np.nanmax(twi):.1f}')
        print(f'    HLI range: {np.nanmin(hli):.3f} - {np.nanmax(hli):.3f}')

    # Process each S2 timestep
    results: dict[np.datetime64, xr.DataArray] = {}

    def process_timestep(t):
        # Find matching SMIPS
        smips_t = select_nearest_smips(smips_cube, t, config.max_gap_days)
        if smips_t is None:
            return t, None, 'no_smips'

        # Reproject SMIPS to S2 grid
        s2_t = s2_ds.sel(time=t)
        smips_reproj = reproject_smips_to_s2_grid(smips_t, s2_t)

        # Downscale
        result = downscale_timestep(smips_reproj, s2_t, config, twi=twi, hli=hli)
        if result is None:
            return t, None, 'solver_failed'
        return t, result, 'ok'

    print(f'  downscaling with {workers} workers...')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_timestep, t): t for t in s2_times}

        for fut in as_completed(futures):
            t, result, status = fut.result()
            if status == 'ok':
                results[t] = result
                print(f'    completed {str(t)[:10]}')
            elif status == 'no_smips':
                print(f'    skipped {str(t)[:10]} (no matching SMIPS within {config.max_gap_days} days)')
            else:
                print(f'    failed {str(t)[:10]} (solver failed)')

    if not results:
        raise RuntimeError('No timesteps successfully downscaled.')

    # Stack into cube
    ordered = sorted(results.items(), key=lambda x: x[0])
    times = [t for t, _ in ordered]
    slices = [da for _, da in ordered]

    cube = xr.concat(slices, dim='time')
    cube = cube.assign_coords(time=('time', times))
    cube.name = 'soil_moisture_downscaled'
    cube.attrs.update(
        source='SMIPS downscaled via convex optimization',
        smips_doi='10.25901/b020-nm39',
        lambda_smoothness=config.lambda_smoothness,
        solver=config.solver,
        units='mm',
        long_name='Downscaled soil moisture',
    )

    # Save
    cube.to_dataset(name='soil_moisture_downscaled').to_netcdf(filename)
    print(f'  saved: {filename} ({len(times)} timesteps)')

    return cube


def test():
    """Test downscaling on hilly terrain with seasonal variation."""
    from datetime import date
    import matplotlib.pyplot as plt

    # Murrumbateman area, NSW - hilly grazing/wine country
    # Good terrain variation + seasonal soil moisture changes
    # test_q = Query(
    #     bbox=[149.00, -34.98, 149.02, -34.96],  # ~2km x 2km
    #     start=date(2022, 1, 1),   # Summer (dry)
    #     end=date(2022, 12, 31),   # Full year for seasonal variation
    #     stub='SMIPS_DOWNSCALE_HILLY',
    # )

    from PaddockTS.utils import get_example_query2

    test_q = get_example_query2()

    # Ensure S2 data exists
    if not exists(test_q.sentinel2_path):
        print('Downloading Sentinel-2 data...')
        from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
        download_sentinel2(test_q)

    # Run downscaling
    print('Running downscaling...')
    result = downscale_smips(test_q)

    print(f'\nOutput shape: {result.shape}')
    print(f'Time steps: {len(result.time)}')
    print(f'Spatial dims: {result.y.size} x {result.x.size}')
    print(f'Value range: {float(result.min()):.2f} - {float(result.max()):.2f} mm')

    # Validation: load original SMIPS directly from NetCDF to avoid cache bug
    from PaddockTS.Environmental.SMIPS.download_smips import get_filename as smips_filename
    smips_path = smips_filename(test_q)
    smips_ds = xr.open_dataset(smips_path)
    data_vars = [v for v in smips_ds.data_vars if v != 'spatial_ref']
    smips = smips_ds[data_vars[0]]
    print(f'\nOriginal SMIPS shape: {smips.shape}')
    print(f'Original SMIPS range: {float(smips.min()):.2f} - {float(smips.max()):.2f} mm')

    # Load S2 data for NDVI/NDWI computation
    s2_ds = xr.open_zarr(test_q.sentinel2_path, chunks=None, decode_coords="all")

    # Compute NDVI and NDWI for all timesteps
    def _band(ds, name):
        b = ds[name].values.astype(np.float32)
        b[b == 0] = np.nan
        b /= 10000.0
        return b

    nir = _band(s2_ds, 'nbart_nir_1')
    red = _band(s2_ds, 'nbart_red')
    swir2 = _band(s2_ds, 'nbart_swir_2')

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red)
    # NDWI = (Green - NIR) / (Green + NIR) for water bodies
    # Or (NIR - SWIR) / (NIR + SWIR) for vegetation water content
    ndwi = (nir - swir2) / (nir + swir2)

    # Median across time
    ndvi_median = np.nanmedian(ndvi, axis=0)
    ndwi_median = np.nanmedian(ndwi, axis=0)

    print(f'\nNDVI median range: {np.nanmin(ndvi_median):.3f} - {np.nanmax(ndvi_median):.3f}')
    print(f'NDWI median range: {np.nanmin(ndwi_median):.3f} - {np.nanmax(ndwi_median):.3f}')

    # Plot comparison: 2x2 grid
    if len(result.time) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        t0 = result.time[2].values

        # Original SMIPS (find nearest)
        smips_t = select_nearest_smips(smips, t0)
        if smips_t is not None:
            smips_t.plot(ax=axes[0, 0], cmap='Blues', cbar_kwargs={'label': 'mm'})
            axes[0, 0].set_title(f'SMIPS (~1km)\n{str(t0)[:10]}')

        # Downscaled
        result.isel(time=0).plot(ax=axes[0, 1], cmap='Blues', cbar_kwargs={'label': 'mm'})
        axes[0, 1].set_title(f'Downscaled (10m)\n{str(t0)[:10]}')

        # Median NDVI
        im_ndvi = axes[1, 0].imshow(
            ndvi_median, cmap='RdYlGn', vmin=-0.2, vmax=0.8,
            extent=[float(s2_ds.x.min()), float(s2_ds.x.max()),
                    float(s2_ds.y.min()), float(s2_ds.y.max())],
            origin='upper'
        )
        axes[1, 0].set_title('Median NDVI')
        plt.colorbar(im_ndvi, ax=axes[1, 0], label='NDVI')

        # Median NDWI
        im_ndwi = axes[1, 1].imshow(
            ndwi_median, cmap='RdYlBu', vmin=-0.3, vmax=0.3,
            extent=[float(s2_ds.x.min()), float(s2_ds.x.max()),
                    float(s2_ds.y.min()), float(s2_ds.y.max())],
            origin='upper'
        )
        axes[1, 1].set_title('Median NDWI (NIR-SWIR)')
        plt.colorbar(im_ndwi, ax=axes[1, 1], label='NDWI')

        plt.tight_layout()
        out_path = f'{test_q.tmp_dir}/smips_downscale_test.png'
        plt.savefig(out_path, dpi=150)
        print(f'\nPlot saved: {out_path}')

        smips_ds.close()
        s2_ds.close()


if __name__ == '__main__':
    test()
