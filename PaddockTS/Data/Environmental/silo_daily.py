"""
Download daily climate variables from SILO at 5km resolution.

Documentation: https://www.longpaddock.qld.gov.au/silo/gridded-data
"""
import os
import shutil
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

import requests
import xarray as xr
import numpy as np

from PaddockTS.query import Query

# Global lock for netCDF4 operations (library has global state that's not thread-safe)
_NETCDF_LOCK = threading.Lock()

# Cache staleness threshold for current year (in hours)
CURRENT_YEAR_CACHE_MAX_AGE_HOURS = 24


def _is_cache_stale(zarr_path: str, year: str) -> bool:
    """Check if cache needs refreshing.

    Past years: never stale (data is static)
    Current year: stale if cache is older than CURRENT_YEAR_CACHE_MAX_AGE_HOURS
    """
    current_year = str(datetime.now().year)

    if year != current_year:
        return False  # Past years are static, never stale

    if not os.path.exists(zarr_path):
        return True  # No cache = stale

    # Check cache age for current year
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(zarr_path))
    age = datetime.now() - cache_mtime
    return age > timedelta(hours=CURRENT_YEAR_CACHE_MAX_AGE_HOURS)


SILO_VARIABLES = {
    "daily_rain": "Daily rainfall, mm",
    "monthly_rain": "Monthly rainfall, mm",
    "max_temp": "Maximum temperature, degrees Celsius",
    "min_temp": "Minimum temperature, degrees Celsius",
    "vp": "Vapour pressure, hPa",
    "vp_deficit": "Vapour pressure deficit, hPa",
    "evap_pan": "Class A pan evaporation, mm",
    "evap_syn": "Synthetic estimate, mm",
    "evap_morton_lake": "Morton's shallow lake evaporation, mm",
    "radiation": "Solar radiation: Solar exposure, MJ/m2",
    "rh_tmax": "Relative humidity at time of maximum temperature, %",
    "rh_tmin": "Relative humidity at time of minimum temperature, %",
    "et_short_crop": "Evapotranspiration FAO564 short crop, mm",
    "et_tall_crop": "ASCE5 tall crop6, mm",
    "et_morton_actual": "Morton's areal actual evapotranspiration, mm",
    "et_morton_potential": "Morton's point potential evapotranspiration, mm",
    "et_morton_wet": "Morton's wet-environment areal potential evapotranspiration, mm",
    "mslp": "Mean sea level pressure, hPa",
}


def update_silo_cache(silo_folder: str, variables: list[str] = None, years: list[str] = None, verbose=True):
    """Update SILO cache for specified variables and years.

    This function is designed to be called by a background updater service.
    It downloads/refreshes the cache for the specified variables and years.

    Parameters
    ----------
    silo_folder : str
        Directory for SILO cache files
    variables : list[str], optional
        Variables to update. Defaults to common variables.
    years : list[str], optional
        Years to update. Defaults to current year only.
    verbose : bool
        Print progress messages
    """
    if variables is None:
        variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain",
                     "et_morton_actual", "et_morton_potential"]

    if years is None:
        years = [str(datetime.now().year)]

    os.makedirs(silo_folder, exist_ok=True)

    for year in years:
        for var in variables:
            zarr_path = os.path.join(silo_folder, f"{year}.{var}.zarr")
            nc_filename = os.path.join(silo_folder, f"{year}.{var}.nc")

            # Check if update needed
            if os.path.exists(zarr_path) and not _is_cache_stale(zarr_path, year):
                if verbose:
                    print(f"Cache fresh: {year}.{var}")
                continue

            if verbose:
                print(f"Updating cache: {year}.{var}")

            with _NETCDF_LOCK:
                # Remove stale cache
                if os.path.exists(zarr_path):
                    shutil.rmtree(zarr_path)

                # Download if needed
                if not os.path.exists(nc_filename):
                    _download_from_silo(var, year, silo_folder, verbose=verbose)

                # Convert to Zarr v2 format (compatible with zarr 2.x and 3.x)
                if os.path.exists(nc_filename):
                    ds = xr.open_dataset(nc_filename, engine='netcdf4')
                    ds.to_zarr(zarr_path, mode='w', zarr_format=2)
                    ds.close()
                    os.remove(nc_filename)
                    if verbose:
                        print(f"Converted to Zarr: {zarr_path}")


def _download_from_silo(var, year, silo_folder, verbose=True):
    """Download a NetCDF for the whole of Australia for a given year and variable."""
    silo_baseurl = "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/"
    url = silo_baseurl + var + "/" + str(year) + "." + var + ".nc"
    filename = os.path.join(silo_folder, f"{year}.{var}.nc")

    response = requests.head(url)
    if response.status_code == 200:
        if verbose:
            print(f"Downloading from SILO: {var} {year} ~400MB")
        with requests.get(url, stream=True) as stream:
            with open(filename, "wb") as file:
                shutil.copyfileobj(stream.raw, file)
        if verbose:
            print(f"Downloaded {filename}")


def _singleyear(var, latitude, longitude, buffer, year, silo_folder, verbose=True):
    """Select the region of interest from the Australia wide NetCDF file."""
    nc_filename = os.path.join(silo_folder, f"{year}.{var}.nc")
    zarr_path = os.path.join(silo_folder, f"{year}.{var}.zarr")

    # Use Zarr cache if available and fresh (thread-safe reads)
    if os.path.exists(zarr_path) and not _is_cache_stale(zarr_path, year):
        ds = xr.open_zarr(zarr_path)
    else:
        # netCDF4 has global state that's not thread-safe, so serialize all conversions
        with _NETCDF_LOCK:
            # Check again after acquiring lock (another thread may have converted)
            if os.path.exists(zarr_path) and not _is_cache_stale(zarr_path, year):
                ds = xr.open_zarr(zarr_path)
            else:
                # Remove stale cache if it exists
                if os.path.exists(zarr_path):
                    shutil.rmtree(zarr_path)
                    if verbose:
                        print(f"Removing stale cache: {zarr_path}")

                if not os.path.exists(nc_filename):
                    _download_from_silo(var, year, silo_folder, verbose=verbose)
                # Convert to Zarr v2 format (compatible with zarr 2.x and 3.x)
                ds = xr.open_dataset(nc_filename, engine='netcdf4')
                ds.to_zarr(zarr_path, mode='w', zarr_format=2)
                ds.close()
                # Remove netCDF to save space (Zarr is the cache now)
                if os.path.exists(nc_filename):
                    os.remove(nc_filename)
                if verbose:
                    print(f"Converted to Zarr: {zarr_path}")
                ds = xr.open_zarr(zarr_path)

    if 'crs' in list(ds.data_vars):
        ds = ds.drop_vars('crs')

    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))

    min_buffer_size = 0.03
    if buffer < min_buffer_size:
        ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')

    return ds_region


def _multiyear(var, latitude, longitude, buffer, years, silo_folder, verbose=True, parallel=True):
    """Fetch multiple years of SILO data, optionally in parallel.

    With Zarr caching, parallel reads are thread-safe. The conversion step
    (download + convert to Zarr) uses file locking to prevent race conditions.
    """
    if parallel and len(years) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_year(year):
            return year, _singleyear(var, latitude, longitude, buffer, year, silo_folder, verbose=verbose)

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(years), 4)) as executor:
            futures = {executor.submit(fetch_year, year): year for year in years}
            for future in as_completed(futures):
                year, ds_year = future.result()
                if ds_year is not None:
                    results[year] = ds_year

        # Concatenate in chronological order
        dss = [results[year] for year in years if year in results]
    else:
        # Sequential fallback
        dss = []
        for year in years:
            ds_year = _singleyear(var, latitude, longitude, buffer, year, silo_folder, verbose=verbose)
            if ds_year is not None:
                dss.append(ds_year)

    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def _save_json(ds, outdir, stub, start_year, end_year, buffer, reducer='median', verbose=True):
    """Save SILO daily data as JSON for frontend consumption."""
    if reducer == 'median':
        ds_point = ds.median(dim=['lat', 'lon'])
    elif reducer == 'mean':
        ds_point = ds.mean(dim=['lat', 'lon'])
    elif reducer == 'min':
        ds_point = ds.min(dim=['lat', 'lon'])
    elif reducer == 'max':
        ds_point = ds.max(dim=['lat', 'lon'])
    else:
        ds_point = ds.median(dim=['lat', 'lon'])

    data = []
    for i, time_val in enumerate(ds_point.time.values):
        row = {"time": str(time_val)[:10]}
        for var in ds_point.data_vars:
            val = float(ds_point[var].isel(time=i).values)
            row[var] = None if np.isnan(val) else round(val, 2)
        data.append(row)

    payload = {
        "meta": {
            "start_year": start_year,
            "end_year": end_year,
            "buffer": buffer,
            "reducer": reducer,
            "variables": list(ds.data_vars.keys())
        },
        "data": data
    }

    json_path = Path(outdir) / f"{stub}_silo_daily.json"
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    if verbose:
        print(f"Saved JSON with {len(data)} records and {len(payload['meta']['variables'])} variables: {json_path}")

    return json_path


def silo_daily(
    query: Query,
    variables: list[str] = None,
    silo_folder: str = None,
    save_netcdf=True,
    save_json=True,
    plot=False,
    reducer='median',
    verbose=True,
    parallel=True
):
    """Download daily variables from SILO at 5km resolution for the region/time of interest.

    Parameters
    ----------
        query: Query object with lat, lon, buffer, start_time, end_time, stub_out_dir, stub, stub_tmp_dir
        variables: List of variables to download (default: ["radiation", "vp", "max_temp", "min_temp", "daily_rain", "et_morton_actual", "et_morton_potential"])
        silo_folder: Directory for caching Australia-wide SILO files (~400MB each)
        save_netcdf: Save the data as NetCDF file
        save_json: Save the data as JSON for frontend consumption
        plot: Generate time series plots
        reducer: How to spatially aggregate data for JSON export
        verbose: Print progress messages

    Returns
    -------
        xarray.Dataset with the requested variables
    """
    from os import makedirs

    if variables is None:
        variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain", "et_morton_actual", "et_morton_potential"]

    lat, lon, buffer = query.lat, query.lon, query.buffer
    start_year = str(query.start_time.year)
    end_year = str(query.end_time.year)
    outdir, stub, tmpdir = query.stub_out_dir, query.stub, query.stub_tmp_dir
    makedirs(outdir, exist_ok=True)
    makedirs(tmpdir, exist_ok=True)

    # SILO files are large (~400MB each) and shared across all queries
    # Use config.silo_dir (defaults to {tmp_dir}/SILO if not set)
    if silo_folder is None:
        from PaddockTS.config import config
        silo_folder = config.silo_dir
    makedirs(silo_folder, exist_ok=True)

    if verbose:
        print(f"Starting silo_daily for stub {stub}")

    dss = []
    years = [str(year) for year in range(int(start_year), int(end_year) + 1)]
    for variable in variables:
        ds = _multiyear(variable, lat, lon, buffer, years, silo_folder, verbose=verbose, parallel=parallel)
        dss.append(ds)
    ds_concat = xr.merge(dss)

    if save_netcdf:
        filename = os.path.join(outdir, f'{stub}_silo_daily.nc')
        ds_concat.to_netcdf(filename, engine='netcdf4')
        if verbose:
            print("Saved:", filename)

    if save_json:
        _save_json(ds_concat, outdir, stub, start_year, end_year, buffer, reducer, verbose)

    if plot:
        import matplotlib.pyplot as plt
        vars_list = list(ds_concat.data_vars)
        figsize = (10, 2 * len(vars_list))
        ds_point = ds_concat.median(dim=['lat', 'lon'])
        fig, axes = plt.subplots(nrows=len(vars_list), figsize=figsize, sharex=True)
        if len(vars_list) == 1:
            axes = [axes]
        for ax, var in zip(axes, vars_list):
            ds_point[var].plot(ax=ax, add_legend=False)
            ax.set_xlabel("")
        filename = os.path.join(outdir, f'{stub}_silo_daily.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print("Saved:", filename)

    return ds_concat
