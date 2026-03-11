from PaddockTS.query import Query
from PaddockTS.Environmental.SILO.download_silo import download_silo, get_filename as silo_filename
from PaddockTS.Environmental.OzWALD.download_ozwald_daily import download_ozwald_daily, get_filename as ozwald_daily_filename
from PaddockTS.Environmental.OzWALD.download_ozwald_8day import download_ozwald_8day, get_filename as ozwald_8day_filename
from os.path import exists
from os import makedirs
import pandas as pd

RENAME = {
    'Pg': 'Precipitation',
    'Qtot': 'Runoff',
    'Tmin': 'Minimum temperature',
    'Tmax': 'Maximum temperature',
    'Ssoil': 'Soil moisture',
    'GPP': 'Vegetation growth',
    'LAI': 'Vegetation leaf area',
    'VPeff': 'VPeff',
    'Uavg': 'Uavg',
    'radiation': 'SRAD',
}

DAESIM_COLUMNS = [
    'Precipitation',
    'Runoff',
    'Minimum temperature',
    'Maximum temperature',
    'Soil moisture',
    'Vegetation growth',
    'Vegetation leaf area',
    'VPeff',
    'Uavg',
    'SRAD',
]

get_filename = lambda q: f'{q.out_dir}/{q.stub}_DAESim_forcing.csv'


def daesim_forcing(query: Query):
    makedirs(query.out_dir, exist_ok=True)
    filename = get_filename(query)

    if exists(filename):
        print(f'  cached: {filename}')
        return pd.read_csv(filename, parse_dates=['date'])

    # Download if not already cached
    df_silo = download_silo(query)
    df_daily = download_ozwald_daily(query, variables=['Pg', 'Tmax', 'Tmin', 'Uavg', 'VPeff'])
    df_8day = download_ozwald_8day(query, variables=['Ssoil', 'Qtot', 'LAI', 'GPP'])

    # SILO: just need radiation, indexed by date
    silo = df_silo[['YYYY-MM-DD', 'radiation']].copy()
    silo.rename(columns={'YYYY-MM-DD': 'date'}, inplace=True)
    silo['date'] = pd.to_datetime(silo['date'])
    silo.set_index('date', inplace=True)

    # OzWALD daily
    daily = df_daily.copy()
    daily['date'] = pd.to_datetime(daily['time'])
    daily.set_index('date', inplace=True)
    daily.drop(columns=['time'], inplace=True)

    # OzWALD 8-day: forward-fill to daily
    eightday = df_8day.copy()
    eightday['date'] = pd.to_datetime(eightday['time'])
    eightday.set_index('date', inplace=True)
    eightday.drop(columns=['time'], inplace=True)
    eightday = eightday.resample('D').ffill()

    # Merge on date
    df = silo.join(daily, how='inner').join(eightday, how='left')
    df.rename(columns=RENAME, inplace=True)
    df = df[DAESIM_COLUMNS]
    df.index.name = 'date'

    df.to_csv(filename)
    print(f'  saved: {filename} ({len(df)} days)')
    return df.reset_index()


def test():
    from PaddockTS.utils import get_example_query
    df = daesim_forcing(get_example_query())
    print(df.head(10))
    print(f'\nShape: {df.shape}')
    print(f'Columns: {list(df.columns)}')


if __name__ == '__main__':
    test()
