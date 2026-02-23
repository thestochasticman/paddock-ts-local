from matplotlib import pyplot as plt
from os.path import dirname
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
        'title': 'Daily Rainfall',
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


def plot(filename: str, groups: dict = None):
    df = pd.read_csv(filename, parse_dates=['time'])
    groups = groups or PLOT_GROUPS
    dirpath = dirname(filename)

    for name, cfg in groups.items():
        cols = [c for c in cfg['vars'] if c in df.columns]
        if not cols:
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        kind = cfg.get('kind', 'line')

        if kind == 'bar':
            # Resample to monthly for bar charts
            monthly = df.set_index('time')[cols].resample('ME').sum()
            monthly.plot(kind='bar', ax=ax, width=0.8)
            ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly.index], rotation=45, ha='right')
        else:
            for col in cols:
                ax.plot(df['time'], df[col], label=col, linewidth=0.5, alpha=0.8)
            ax.legend()

        ax.set_ylabel(cfg['ylabel'])
        ax.set_title(cfg['title'])
        plt.tight_layout()
        plt.savefig(f'{dirpath}/silo_{name}.png', dpi=150)
        plt.close()
