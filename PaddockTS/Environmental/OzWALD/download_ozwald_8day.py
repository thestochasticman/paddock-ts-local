from PaddockTS.query import Query
from .ozwald import OzWALD
from os import makedirs
from os.path import exists
import pandas as pd
import xarray as xr

ozwald = OzWALD()
get_filename = lambda q: f'{q.tmp_dir}/Environmental/{q.stub}_ozwald_8day.csv'


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
        chunks.append(data)

    return xr.concat(chunks, dim='time')


def download_ozwald_8day(query: Query, variables: list[str] = None):
    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    filename = get_filename(query)

    if exists(filename):
        print(f'  cached: {filename}')
        return pd.read_csv(filename, parse_dates=['time'])

    variables = variables or list(ozwald.eight_day.keys())
    years = range(query.start.year, query.end.year + 1)
    records = {}

    for variable in variables:
        print(f'  {variable}...', flush=True)
        ts = _download_variable(variable, years, query.bbox)
        records[variable] = ts.values

    time = ts.time.values
    df = pd.DataFrame(records, index=pd.DatetimeIndex(time, name='time'))
    df = df.loc[str(query.start):str(query.end)]
    df.reset_index(inplace=True)
    df.to_csv(filename, index=False)
    print(f'  saved: {filename} ({len(df)} rows, {len(variables)} variables)')
    return df


def test():
    from PaddockTS.utils import get_example_query
    df = download_ozwald_8day(get_example_query())
    print(df.head(10))
    print(f'\nShape: {df.shape}')
    print(f'Columns: {list(df.columns)}')


if __name__ == '__main__':
    test()
