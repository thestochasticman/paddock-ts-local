"""Diagnostic plots for OzWALD daily and 8-day climate / vegetation data.

Each plot file is a single panel covering the full date range of a
``Query``. Variables are grouped thematically (temperature, rainfall,
vegetation index, etc.); precipitation/rainfall groups use a monthly
bar plot and other groups use thin-line time-series.
"""

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
    """Plot OzWALD daily climate variables grouped by theme.

    Reads the cached daily CSV (downloaded by
    :func:`PaddockTS.Environmental.OzWALD.download_ozwald_daily.download_ozwald_daily`)
    and writes one PNG per group to
    ``{query.out_dir}/{query.stub}_ozwald_daily_{group}.png``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        groups: Optional override of the default grouping. Maps a group
            name to ``{'vars': [...], 'ylabel': str, 'title': str,
            'kind': 'line'|'bar'}``. If ``None``, uses
            :data:`DAILY_GROUPS` (temperature, precipitation, wind,
            radiation).
    """
    df = pd.read_csv(daily_filename(query), parse_dates=['time'])
    _plot_groups(df, 'time', groups or DAILY_GROUPS, query, 'ozwald_daily')


def ozwald_8day_plot(query: Query, groups: dict = None):
    """Plot OzWALD 8-day vegetation / water variables grouped by theme.

    Reads the cached 8-day CSV (downloaded by
    :func:`PaddockTS.Environmental.OzWALD.download_ozwald_8day.download_ozwald_8day`)
    and writes one PNG per group to
    ``{query.out_dir}/{query.stub}_ozwald_8day_{group}.png``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        groups: Optional override of the default grouping. If ``None``,
            uses :data:`EIGHTDAY_GROUPS` (vegetation index, fractional
            cover, LAI/GPP, soil water/runoff).
    """
    df = pd.read_csv(eightday_filename(query), parse_dates=['time'])
    _plot_groups(df, 'time', groups or EIGHTDAY_GROUPS, query, 'ozwald_8day')


def test():
    from PaddockTS.utils import get_example_query
    q = get_example_query()
    ozwald_daily_plot(q)
    ozwald_8day_plot(q)


if __name__ == '__main__':
    test()
