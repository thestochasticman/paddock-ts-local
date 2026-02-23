from matplotlib import pyplot as plt
from PaddockTS.query import Query
from PaddockTS.Environmental.OzWALD.download_ozwald_daily import get_filename as daily_filename
from PaddockTS.Environmental.OzWALD.download_ozwald_8day import get_filename as eightday_filename
from os import makedirs
import pandas as pd


DAILY_GROUPS = {
    'temperature': {
        'vars': ['Tmax', 'Tmin'],
        'ylabel': 'Temperature (°C)',
        'title': 'OzWALD Daily Temperature',
    },
    'precipitation': {
        'vars': ['Pg'],
        'ylabel': 'Precipitation (mm)',
        'title': 'OzWALD Monthly Precipitation',
        'kind': 'bar',
    },
    'wind': {
        'vars': ['Uavg', 'Ueff'],
        'ylabel': 'Wind Speed (m/s)',
        'title': 'OzWALD Wind Speed',
    },
    'radiation': {
        'vars': ['DWLReff'],
        'ylabel': 'Radiation (W/m²)',
        'title': 'OzWALD Downwelling Longwave Radiation',
    },
}

EIGHTDAY_GROUPS = {
    'vegetation_index': {
        'vars': ['NDVI', 'EVI'],
        'ylabel': 'Index',
        'title': 'OzWALD Vegetation Indices',
    },
    'vegetation_cover': {
        'vars': ['PV', 'NPV', 'BS'],
        'ylabel': 'Fraction',
        'title': 'OzWALD Fractional Cover',
    },
    'lai_gpp': {
        'vars': ['LAI', 'GPP'],
        'ylabel': 'LAI (m²/m²) / GPP (g m⁻² d⁻¹)',
        'title': 'OzWALD LAI & GPP',
    },
    'water': {
        'vars': ['Ssoil', 'Qtot'],
        'ylabel': 'mm',
        'title': 'OzWALD Soil Moisture & Runoff',
    },
}


def _plot_groups(df, time_col, groups, query, prefix):
    makedirs(query.out_dir, exist_ok=True)

    for name, cfg in groups.items():
        cols = [c for c in cfg['vars'] if c in df.columns]
        if not cols:
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        kind = cfg.get('kind', 'line')

        if kind == 'bar':
            monthly = df.set_index(time_col)[cols].resample('ME').sum()
            monthly.plot(kind='bar', ax=ax, width=0.8)
            ticks = range(0, len(monthly), max(1, len(monthly) // 12))
            ax.set_xticks(list(ticks))
            ax.set_xticklabels([monthly.index[i].strftime('%Y-%m') for i in ticks], rotation=45, ha='right')
        else:
            for col in cols:
                ax.plot(df[time_col], df[col], label=col, linewidth=0.5, alpha=0.8)
            ax.legend()

        ax.set_ylabel(cfg['ylabel'])
        ax.set_title(cfg['title'])
        plt.tight_layout()
        out_path = f'{query.out_dir}/{query.stub}_{prefix}_{name}.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'  saved: {out_path}')


def ozwald_daily_plot(query: Query, groups: dict = None):
    df = pd.read_csv(daily_filename(query), parse_dates=['time'])
    _plot_groups(df, 'time', groups or DAILY_GROUPS, query, 'ozwald_daily')


def ozwald_8day_plot(query: Query, groups: dict = None):
    df = pd.read_csv(eightday_filename(query), parse_dates=['time'])
    _plot_groups(df, 'time', groups or EIGHTDAY_GROUPS, query, 'ozwald_8day')


def test():
    from PaddockTS.utils import get_example_query
    q = get_example_query()
    ozwald_daily_plot(q)
    ozwald_8day_plot(q)


if __name__ == '__main__':
    test()
