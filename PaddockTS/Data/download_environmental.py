"""
Download all environmental data in parallel using multiprocessing.

Note: Uses ProcessPoolExecutor for task-level parallelism. This avoids
netCDF4/HDF5 thread-safety issues by using separate processes.
Year-level parallelism is disabled when called from here to avoid
nested parallelism issues.
"""
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import xarray as xr

from PaddockTS.query import Query


def _run_terrain_tiles(query: Query, verbose: bool):
    from PaddockTS.Data.Environmental.terrain_tiles import terrain_tiles
    terrain_tiles(query, verbose=verbose)
    return 'terrain'


def _run_slga_soils(query: Query, verbose: bool):
    from PaddockTS.Data.Environmental.slga_soils import slga_soils
    slga_soils(query, verbose=verbose)
    return 'soils'


def _run_ozwald_daily_pg(query: Query, verbose: bool):
    from PaddockTS.Data.Environmental.ozwald_daily import ozwald_daily
    # Disable year-level parallelism to avoid netCDF4 thread-safety issues
    ozwald_daily(query, variables=['Pg'], save_netcdf=True, save_json=True, verbose=verbose, parallel=False)
    return 'ozwald_daily_Pg'


def _run_ozwald_daily_tmax(query: Query, verbose: bool):
    from PaddockTS.Data.Environmental.ozwald_daily import ozwald_daily
    ozwald_daily(query, variables=['Tmax', 'Tmin'], save_netcdf=True, save_json=True, verbose=verbose, parallel=False)
    return 'ozwald_daily_Tmax'


def _run_ozwald_daily_uavg(query: Query, verbose: bool):
    from PaddockTS.Data.Environmental.ozwald_daily import ozwald_daily
    ozwald_daily(query, variables=['Uavg', 'VPeff'], save_netcdf=True, save_json=True, verbose=verbose, parallel=False)
    return 'ozwald_daily_Uavg'


def _run_ozwald_8day(query: Query, verbose: bool):
    from PaddockTS.Data.Environmental.ozwald_8day import ozwald_8day
    ozwald_8day(query, variables=['Ssoil', 'Qtot', 'LAI', 'GPP'], save_netcdf=True, save_json=True, verbose=verbose, parallel=False)
    return 'ozwald_8day'


def _run_silo_daily(query: Query, verbose: bool):
    from PaddockTS.Data.Environmental.silo_daily import silo_daily
    silo_daily(query, save_netcdf=True, save_json=True, verbose=verbose)
    return 'silo_daily'


def download_environmental(query: Query, verbose=True):
    """Download all environmental data for a query, running downloads in parallel.

    Uses multiprocessing to run downloads concurrently since the underlying
    libraries (GDAL, HDF5, netCDF4) are not thread-safe.

    Parameters
    ----------
        query: Query object specifying location and time range
        verbose: Print progress messages

    Returns
    -------
        dict with all downloaded datasets and dataframes
    """
    from PaddockTS.Data.Environmental.daesim_forcing import daesim_forcing, daesim_soils

    # Ensure output directories exist before spawning processes
    os.makedirs(query.stub_out_dir, exist_ok=True)
    os.makedirs(query.stub_tmp_dir, exist_ok=True)

    # All download tasks - these are independent and can run in parallel
    download_tasks = [
        (_run_terrain_tiles, 'terrain_tiles'),
        (_run_slga_soils, 'slga_soils'),
        (_run_ozwald_daily_pg, 'ozwald_daily_Pg'),
        (_run_ozwald_daily_tmax, 'ozwald_daily_Tmax'),
        (_run_ozwald_daily_uavg, 'ozwald_daily_Uavg'),
        (_run_ozwald_8day, 'ozwald_8day'),
        (_run_silo_daily, 'silo_daily'),
    ]

    if verbose:
        print(f"Starting {len(download_tasks)} parallel downloads...")

    completed = set()
    # Use 'spawn' context to avoid issues with fork() and zarr/HDF5/netCDF4 libraries
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=len(download_tasks), mp_context=ctx) as executor:
        futures = {executor.submit(func, query, verbose): name for func, name in download_tasks}

        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
                completed.add(result)
                if verbose:
                    print(f"[done] Completed {task_name}")
            except Exception as e:
                if verbose:
                    print(f"[failed] Failed {task_name}: {e}")
                raise

    # Load results from saved files
    if verbose:
        print("\nLoading datasets from files...")

    outdir = query.stub_out_dir
    stub = query.stub

    results = {
        'terrain': os.path.join(outdir, f'{stub}_terrain.tif'),  # Path to terrain GeoTiff
        'soils': None,  # Soils are saved as individual tiffs in tmpdir
        'silo_daily': xr.open_dataset(os.path.join(outdir, f'{stub}_silo_daily.nc'), engine='netcdf4'),
        'ozwald_8day': xr.open_dataset(os.path.join(outdir, f'{stub}_ozwald_8day.nc'), engine='netcdf4'),
        'ozwald_daily_Pg': xr.open_dataset(os.path.join(outdir, f'{stub}_ozwald_daily_Pg.nc'), engine='netcdf4'),
        'ozwald_daily_Tmax': xr.open_dataset(os.path.join(outdir, f'{stub}_ozwald_daily_Tmax.nc'), engine='netcdf4'),
        'ozwald_daily_Uavg': xr.open_dataset(os.path.join(outdir, f'{stub}_ozwald_daily_Uavg.nc'), engine='netcdf4'),
    }

    # Run aggregation functions that depend on the downloads
    if verbose:
        print("Running aggregation functions...")

    results['daesim_forcing'] = daesim_forcing(
        query,
        ds_silo_daily=results['silo_daily'],
        ds_ozwald_8day=results['ozwald_8day'],
        ds_ozwald_daily_Pg=results['ozwald_daily_Pg'],
        ds_ozwald_daily_Tmax=results['ozwald_daily_Tmax'],
        ds_ozwald_daily_Uavg=results['ozwald_daily_Uavg'],
        verbose=verbose
    )

    results['daesim_soils'] = daesim_soils(query, verbose=verbose)

    if verbose:
        print("\n[done] All environmental downloads complete")

    return results
