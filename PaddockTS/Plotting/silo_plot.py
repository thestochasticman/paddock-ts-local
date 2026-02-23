from matplotlib import pyplot as plt
from PaddockTS.query import Query
from PaddockTS.Environmental.SILO.download_silo import get_filename
from os import makedirs
import pandas as pd


PLOT_GROUPS = {
    'temperature': {
        'vars': ['max_temp', 'min_temp'],
        'ylabel': 'Temperature (°C)',
        'title': 'Daily Temperature',
    },
    'rainfall': {
        'vars': ['daily_rain'],
        'ylabel': 'Rainfall (mm)',
        'title': 'Monthly Rainfall',
        'kind': 'bar',
    },
    'radiation': {
        'vars': ['radiation'],
        'ylabel': 'Radiation (MJ/m²)',
        'title': 'Solar Radiation',
    },
    'evapotranspiration': {
        'vars': ['et_short_crop', 'evap_pan'],
        'ylabel': 'ET (mm)',
        'title': 'Evapotranspiration',
    },
    'humidity': {
        'vars': ['vp_deficit', 'vp'],
        'ylabel': 'hPa',
        'title': 'Vapour Pressure',
    },
}


def silo_plot(query: Query, groups: dict = None):
    filename = get_filename(query)
    df = pd.read_csv(filename, parse_dates=['YYYY-MM-DD'])
    groups = groups or PLOT_GROUPS
    makedirs(query.out_dir, exist_ok=True)

    for name, cfg in groups.items():
        cols = [c for c in cfg['vars'] if c in df.columns]
        if not cols:
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        kind = cfg.get('kind', 'line')

        if kind == 'bar':
            monthly = df.set_index('YYYY-MM-DD')[cols].resample('ME').sum()
            monthly.plot(kind='bar', ax=ax, width=0.8)
            ticks = range(0, len(monthly), max(1, len(monthly) // 12))
            ax.set_xticks(list(ticks))
            ax.set_xticklabels([monthly.index[i].strftime('%Y-%m') for i in ticks], rotation=45, ha='right')
        else:
            for col in cols:
                ax.plot(df['YYYY-MM-DD'], df[col], label=col, linewidth=0.5, alpha=0.8)
            ax.legend()

        ax.set_ylabel(cfg['ylabel'])
        ax.set_title(cfg['title'])
        plt.tight_layout()
        out_path = f'{query.out_dir}/{query.stub}_silo_{name}.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'  saved: {out_path}')


def test():
    from PaddockTS.utils import get_example_query
    silo_plot(get_example_query())


if __name__ == '__main__':
    test()
