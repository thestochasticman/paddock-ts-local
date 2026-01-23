"""
Merge SILO and OzWald data into CSV files for DAESim input.
"""
import os

import pandas as pd
import xarray as xr
import rioxarray as rxr

from PaddockTS.query import Query


VARIABLE_NAMES = {
    "daily_rain": "Precipitation (SILO)",
    "max_temp": "Maximum temperature (SILO)",
    "min_temp": "Minimum temperature (SILO)",
    "vp": "VPeff (SILO)",
    "radiation": "SRAD",
    "Pg": "Precipitation",
    "Tmax": "Maximum temperature",
    "Tmin": "Minimum temperature",
    "VPeff": "VPeff",
    "Uavg": "Uavg",
    "Ssoil": "Soil moisture",
    "Qtot": "Runoff",
    "LAI": "Vegetation leaf area",
    "GPP": "Vegetation growth",
}


def _aggregate_pixels(ds):
    """Aggregate pixels to a single value per timepoint."""
    return ds.median(dim=["latitude", "longitude"])


def _normalize_silo(ds):
    """Normalize SILO dataset to match OzWald coordinate names."""
    if 'crs' in ds.data_vars:
        ds = ds.drop_vars(['crs'])
    if 'lat' in ds.dims:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    return ds


def daesim_forcing(
    query: Query,
    ds_silo_daily: xr.Dataset,
    ds_ozwald_8day: xr.Dataset,
    ds_ozwald_daily_Pg: xr.Dataset,
    ds_ozwald_daily_Tmax: xr.Dataset,
    ds_ozwald_daily_Uavg: xr.Dataset,
    verbose=True
):
    """Merge the ozwald and silo datasets into a dataframe for input into DAESim.

    Parameters
    ----------
        query: Query object with stub_out_dir and stub
        ds_silo_daily: Dataset from silo_daily()
        ds_ozwald_8day: Dataset from ozwald_8day()
        ds_ozwald_daily_Pg: Dataset from ozwald_daily(variables=['Pg'])
        ds_ozwald_daily_Tmax: Dataset from ozwald_daily(variables=['Tmax', 'Tmin'])
        ds_ozwald_daily_Uavg: Dataset from ozwald_daily(variables=['Uavg', 'VPeff'])

    Returns
    -------
        DataFrame with all variables required for DAESim input
        Also saves to {stub}_DAESim_forcing.csv
    """
    outdir = query.stub_out_dir
    stub = query.stub

    ds_silo_daily = _normalize_silo(ds_silo_daily)

    ds_silo_daily = _aggregate_pixels(ds_silo_daily)
    ds_ozwald_8day = _aggregate_pixels(ds_ozwald_8day)
    ds_ozwald_daily_Pg = _aggregate_pixels(ds_ozwald_daily_Pg)
    ds_ozwald_daily_Tmax = _aggregate_pixels(ds_ozwald_daily_Tmax)
    ds_ozwald_daily_Uavg = _aggregate_pixels(ds_ozwald_daily_Uavg)

    ds_merged = xr.merge([ds_silo_daily, ds_ozwald_8day, ds_ozwald_daily_Pg, ds_ozwald_daily_Tmax, ds_ozwald_daily_Uavg])

    df = ds_merged.to_dataframe().reset_index()
    df = df.set_index('time')
    df.rename(columns=VARIABLE_NAMES, inplace=True)
    df.rename_axis("date", inplace=True)

    daesim_ordering = [
        "Precipitation", "Runoff", "Minimum temperature", "Maximum temperature",
        "Soil moisture", "Vegetation growth", "Vegetation leaf area", "VPeff", "Uavg", "SRAD"
    ]
    df_ordered = df[daesim_ordering]

    filepath = os.path.join(outdir, stub + "_DAESim_forcing.csv")
    df_ordered.to_csv(filepath)
    if verbose:
        print("Saved", filepath)

    return df_ordered


def daesim_soils(query: Query, verbose=True):
    """Merge the soil tiffs into a csv required for DAESim.

    Parameters
    ----------
        query: Query object with stub_out_dir, stub, stub_tmp_dir

    Requirements
    ------------
        This function expects 9 soil variables x 4 depths = 36 tiff files to be predownloaded

    Returns
    -------
        DataFrame with all variables required for DAESim input
        Also saves to {stub}_DAESim_Soils.csv
    """
    outdir = query.stub_out_dir
    stub = query.stub
    tmpdir = query.stub_tmp_dir

    variables = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density', 'Available_Water_Capacity',
                 'Effective_Cation_Exchange_Capacity', 'Total_Nitrogen', 'Total_Phosphorus']
    depths = ['5-15cm', '15-30cm', '30-60cm', '60-100cm']

    values = []
    for variable in variables:
        for depth in depths:
            filename = os.path.join(tmpdir, f"{stub}_{variable}_{depth}.tif")
            ds = rxr.open_rasterio(filename)
            value = float(ds.isel(band=0, x=0, y=0).values)
            values.append({
                "variable": variable,
                "depth": depth,
                "value": value
            })

    df = pd.DataFrame(values)
    pivot_df = df.pivot(index='depth', columns='variable', values='value')
    pivot_df = pivot_df.reset_index()

    depth_order = ['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    pivot_df['depth'] = pd.Categorical(pivot_df['depth'], categories=depth_order, ordered=True)
    sorted_df = pivot_df.sort_values(by='depth')

    filepath = os.path.join(outdir, stub + "_DAESim_Soils.csv")
    sorted_df.to_csv(filepath, index=False)
    if verbose:
        print("Saved", filepath)

    return sorted_df
