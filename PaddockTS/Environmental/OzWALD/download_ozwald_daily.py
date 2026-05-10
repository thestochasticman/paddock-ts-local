"""Download daily OzWALD climate variables for the AOI centre.

`OzWALD <https://www.wenfo.org/ozwald/>`_ is the Australian Water and
Landscape Dynamics dataset hosted by ANU, providing modelled daily
meteorology over Australia at ~5 km resolution. This module reads the
OPeNDAP-served NetCDFs, samples the time series at the centre of
``query.bbox``, concatenates yearly chunks, and persists a tidy CSV.

Variables include ``Tmax``, ``Tmin``, ``Pg`` (precipitation), wind
speed, and downwelling longwave radiation; the full default set is
read from :class:`PaddockTS.Environmental.OzWALD.ozwald.OzWALD`.
"""

from PaddockTS.query import Query
from .ozwald import OzWALD
from os import makedirs
from os.path import exists
import pandas as pd
import xarray as xr

ozwald = OzWALD()
get_filename = lambda q: f'{q.tmp_dir}/Environmental/{q.stub}_ozwald_daily.csv'


def _download_variable(variable, years, bbox):
    centre_lat = (bbox[1] + bbox[3]) / 2
    centre_lon = (bbox[0] + bbox[2]) / 2
    chunks = []

    for year in years:
        url = ozwald.get_url(variable, year)
        ds = xr.open_dataset(url)
        data = ds.sel(
            latitude=centre_lat, longitude=centre_lon, method='nearest',
        )[variable].load()
        ds.close()
        # Ensure time coordinate is datetime64 (some yearly files use int32)
        if not pd.api.types.is_datetime64_any_dtype(data.time.values):
            data['time'] = pd.to_datetime(data.time.values)
        chunks.append(data)

    return xr.concat(chunks, dim='time')


def download_ozwald_daily(query: Query, variables: list[str] = None):
    """Fetch daily OzWALD time series for ``query.bbox`` centre.

    For each variable, opens the per-year NetCDF over OPeNDAP, samples
    the nearest grid cell to the AOI centre, concatenates years, and
    writes a single tidy CSV (one row per day, one column per variable)
    to ``{query.tmp_dir}/Environmental/{query.stub}_ozwald_daily.csv``.

    Cached: if the output CSV already exists it is loaded and returned
    without contacting the server.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        variables: Optional list of OzWALD daily variable names to
            fetch. If ``None``, fetches every variable defined by the
            bundled :class:`OzWALD` config.

    Returns:
        pandas.DataFrame: One row per day in ``[query.start, query.end]``
        with a ``time`` column and one column per variable.
    """
    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    makedirs(f'{query.out_dir}/Environmental', exist_ok=True)
    filename = get_filename(query)

    if exists(filename):
        print(f'  cached: {filename}')
        return pd.read_csv(filename, parse_dates=['time'])

    variables = variables or list(ozwald.daily_meteo.keys())
    years = range(query.start.year, query.end.year + 1)
    records = {}

    for variable in variables:
        print(f'  {variable}...', flush=True)
        ts = _download_variable(variable, years, query.bbox)
        records[variable] = ts.values

    time = ts.time.values
    df = pd.DataFrame(records, index=pd.DatetimeIndex(time, name='time'))
    df = df.sort_index()
    df = df.loc[str(query.start):str(query.end)]    
    df.reset_index(inplace=True)
    df.to_csv(filename, index=False)
    print(f'  saved: {filename} ({len(df)} days, {len(variables)} variables)')
    return df


def test():
    from PaddockTS.utils import get_example_query
    df = download_ozwald_daily(get_example_query())
    print(df.head(10))
    print(f'\nShape: {df.shape}')
    print(f'Columns: {list(df.columns)}')


if __name__ == '__main__':
    test()
