"""
Download daily climate variables from SILO at 5km resolution.

Documentation: https://www.longpaddock.qld.gov.au/silo/gridded-data
"""
import os
import shutil
import json
from pathlib import Path

import requests
import xarray as xr
import numpy as np

from PaddockTS.query import Query


import threading

# Lock to prevent concurrent downloads of the same SILO file
_download_locks = {}
_locks_lock = threading.Lock()


def _get_download_lock(filename):
    """Get or create a lock for a specific file to prevent concurrent downloads."""
    with _locks_lock:
        if filename not in _download_locks:
            _download_locks[filename] = threading.Lock()
        return _download_locks[filename]


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
    filename = os.path.join(silo_folder, f"{year}.{var}.nc")

    # Use lock to prevent concurrent downloads of the same file
    lock = _get_download_lock(filename)
    with lock:
        if not os.path.exists(filename):
            _download_from_silo(var, year, silo_folder, verbose=verbose)

    ds = xr.open_dataset(filename, engine='netcdf4')

    if 'crs' in list(ds.data_vars):
        ds = ds.drop_vars('crs')

    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))

    min_buffer_size = 0.03
    if buffer < min_buffer_size:
        ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')

    return ds_region


def _multiyear(var, latitude, longitude, buffer, years, silo_folder, verbose=True):
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
    verbose=True
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
    # Store in base tmp_dir, not stub-specific tmpdir
    if silo_folder is None:
        silo_folder = os.path.join(query.tmp_dir, "SILO")
    makedirs(silo_folder, exist_ok=True)

    if verbose:
        print(f"Starting silo_daily for stub {stub}")

    dss = []
    years = [str(year) for year in range(int(start_year), int(end_year) + 1)]
    for variable in variables:
        ds = _multiyear(variable, lat, lon, buffer, years, silo_folder, verbose=verbose)
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
