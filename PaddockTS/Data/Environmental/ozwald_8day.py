"""
Download 8-day climate variables from OzWald at 500m resolution.

Catalog: https://thredds.nci.org.au/thredds/catalog/ub8/au/OzWALD/8day/catalog.html
"""
import os
import json
from pathlib import Path

import requests
import xarray as xr
import numpy as np

from PaddockTS.query import Query


OZWALD_8DAY_VARIABLES = {
    "Alb": "Albedo",
    "BS": "Bare Surface",
    "EVI": "Enhanced Vegetation Index",
    "FMC": "Fuel Moisture Content",
    "GPP": "Gross Primary Productivity",
    "LAI": "Leaf Area Index",
    "NDVI": "Normalised Difference Vegetation Index",
    "NPV": "Non Photosynthetic Vegetation",
    "OW": "Open Water",
    "PV": "Photosynthetic Vegetation",
    "Qtot": "Streamflow",
    "SN": "Snow",
    "Ssoil": "Soil profile moisture change",
}


def _singleyear_thredds(var, latitude, longitude, buffer, year, stub, tmpdir, verbose=True):
    buffer = max(0.003, buffer)

    if var == 'Ssoil':
        north = latitude + buffer
        south = latitude - buffer
        west = longitude - buffer
        east = longitude + buffer

        time_start = f"{year}-01-01"
        time_end = f"{year}-12-31"

        base_url = "https://thredds.nci.org.au"
        url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}'

        head_response = requests.head(url)
        if head_response.status_code == 200:
            response = requests.get(url)
            filename = os.path.join(tmpdir, f"{stub}_{var}_{year}.nc")
            with open(filename, 'wb') as f:
                f.write(response.content)
            if verbose:
                print("Downloaded", filename)
            ds = xr.open_dataset(filename, engine='netcdf4')
        else:
            return None
    else:
        url = f"https://thredds.nci.org.au/thredds/dodsC/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc"
        ds = xr.open_dataset(url)

    if not ds:
        return None

    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))

    if buffer < 0.03:
        ds_region = ds.sel(latitude=[latitude], longitude=[longitude], method='nearest')

    return ds_region


def _singleyear_gdata(var, latitude, longitude, buffer, year):
    filename = f"/g/data/ub8/au/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc"

    if not os.path.exists(filename):
        return None

    ds = xr.open_dataset(filename)
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))

    if buffer < 0.03:
        ds_region = ds.sel(latitude=[latitude], longitude=[longitude], method='nearest')

    return ds_region


def _multiyear(var, latitude, longitude, buffer, years, stub, tmpdir, thredds=True, verbose=True):
    dss = []
    for year in years:
        if thredds:
            ds_year = _singleyear_thredds(var, latitude, longitude, buffer, year, stub, tmpdir, verbose=verbose)
        else:
            ds_year = _singleyear_gdata(var, latitude, longitude, buffer, year)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def _save_json(ds, outdir, stub, start_year, end_year, buffer, reducer='median', verbose=True):
    """Save ozwald 8day data as JSON for frontend consumption."""
    if reducer == 'median':
        ds_point = ds.median(dim=['latitude', 'longitude'])
    elif reducer == 'mean':
        ds_point = ds.mean(dim=['latitude', 'longitude'])
    elif reducer == 'min':
        ds_point = ds.min(dim=['latitude', 'longitude'])
    elif reducer == 'max':
        ds_point = ds.max(dim=['latitude', 'longitude'])
    else:
        ds_point = ds.median(dim=['latitude', 'longitude'])

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

    json_path = Path(outdir) / f"{stub}_ozwald_8day.json"
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    if verbose:
        print(f"Saved JSON with {len(data)} records and {len(payload['meta']['variables'])} variables: {json_path}")

    return json_path


def ozwald_8day(
    query: Query,
    variables: list[str] = None,
    thredds=True,
    save_netcdf=True,
    save_json=True,
    plot=False,
    reducer='median',
    verbose=True
):
    """Download 8day variables from OzWald at 500m resolution for the region/time of interest.

    Parameters
    ----------
        query: Query object with lat, lon, buffer, start_time, end_time, stub_out_dir, stub, stub_tmp_dir
        variables: List of variables to download (default: ["Ssoil", "Qtot", "LAI", "GPP"])
        thredds: Use public Thredds API (True) or NCI gdata (False)
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
        variables = ["Ssoil", "Qtot", "LAI", "GPP"]

    lat, lon, buffer = query.lat, query.lon, query.buffer
    start_year = str(query.start_time.year)
    end_year = str(query.end_time.year)
    outdir, stub, tmpdir = query.stub_out_dir, query.stub, query.stub_tmp_dir
    makedirs(outdir, exist_ok=True)
    makedirs(tmpdir, exist_ok=True)

    if verbose:
        print("Starting ozwald_8day")

    dss = []
    years = [str(year) for year in range(int(start_year), int(end_year) + 1)]
    for variable in variables:
        ds_variable = _multiyear(variable, lat, lon, buffer, years, stub, tmpdir, thredds, verbose=verbose)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)

    if save_netcdf:
        filename = os.path.join(outdir, f'{stub}_ozwald_8day.nc')
        ds_concat.to_netcdf(filename, engine='netcdf4')
        if verbose:
            print("Saved:", filename)

    if save_json:
        _save_json(ds_concat, outdir, stub, start_year, end_year, buffer, reducer, verbose)

    if plot:
        import matplotlib.pyplot as plt
        vars_list = list(ds_concat.data_vars)
        figsize = (10, 2 * len(vars_list))
        ds_point = ds_concat.median(dim=['latitude', 'longitude'])
        fig, axes = plt.subplots(nrows=len(vars_list), figsize=figsize, sharex=True)
        if len(vars_list) == 1:
            axes = [axes]
        for ax, var in zip(axes, vars_list):
            ds_point[var].plot(ax=ax, add_legend=False)
            ax.set_xlabel("")
        filename = os.path.join(outdir, f'{stub}_ozwald_8day.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print("Saved:", filename)

    return ds_concat
