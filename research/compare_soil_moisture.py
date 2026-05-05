"""
Compare SMIPS vs OzWALD Ssoil soil moisture products.

SMIPS: ~1 km resolution, daily
OzWALD Ssoil: ~5 km resolution, 8-day composites
"""
from datetime import date
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from PaddockTS.query import Query
from PaddockTS.Environmental.SMIPS.download_smips import download_smips
from PaddockTS.Environmental.OzWALD.ozwald import OzWALD


def download_ozwald_ssoil_cube(query: Query, retries: int = 3) -> xr.DataArray:
    """Download OzWALD Ssoil as a spatial cube (not just point data)."""
    ozwald = OzWALD()
    years = range(query.start.year, query.end.year + 1)
    bbox = query.bbox  # [west, south, east, north]

    chunks = []
    for year in years:
        url = ozwald.get_url('Ssoil', year)
        print(f'  fetching OzWALD Ssoil {year}...', flush=True)

        # Retry logic for NCI THREDDS server
        for attempt in range(retries):
            try:
                ds = xr.open_dataset(url)
                # Slice to bbox
                data = ds['Ssoil'].sel(
                    latitude=slice(bbox[3], bbox[1]),  # north to south (decreasing)
                    longitude=slice(bbox[0], bbox[2])  # west to east
                ).load()
                ds.close()
                chunks.append(data)
                break
            except OSError as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt  # exponential backoff
                    print(f'    Connection failed, retrying in {wait}s... ({e})')
                    time.sleep(wait)
                else:
                    raise RuntimeError(f'Failed to fetch OzWALD data after {retries} attempts: {e}')

    cube = xr.concat(chunks, dim='time')
    # Filter to query date range
    cube = cube.sel(time=slice(str(query.start), str(query.end)))
    return cube


def compare_soil_moisture():
    """Compare SMIPS and OzWALD soil moisture side by side."""
    # Create test query - use 2022 dates within SMIPS coverage
    test_q = Query(
        bbox=[148.36265, -33.52606, 148.38265, -33.50606],
        start=date(2022, 1, 1),
        end=date(2022, 1, 31),
        stub='SOIL_MOISTURE_COMPARISON'
    )

    print('Downloading SMIPS data...')
    smips_cube = download_smips(test_q)

    print('\nDownloading OzWALD Ssoil data...')
    ozwald_cube = download_ozwald_ssoil_cube(test_q)

    print(f'\nSMIPS shape: {smips_cube.shape} (time, y, x)')
    print(f'OzWALD shape: {ozwald_cube.shape} (time, lat, lon)')

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Row 1: SMIPS ---
    # Spatial map
    smips_cube.isel(time=0).plot(
        ax=axes[0, 0], cmap='Blues',
        cbar_kwargs={'label': 'Soil Water (mm)'}
    )
    axes[0, 0].set_title(f'SMIPS (~1 km)\n{str(smips_cube.time[0].values)[:10]}')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')

    # Time series
    smips_mean = smips_cube.mean(('x', 'y'))
    axes[0, 1].plot(smips_mean.time.values, smips_mean.values, marker='o', linewidth=2, label='SMIPS (daily)')
    axes[0, 1].set_title('SMIPS Mean Soil Moisture')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Total Bucket (mm)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()

    # --- Row 2: OzWALD ---
    # Spatial map
    if ozwald_cube.size > 0:
        ozwald_cube.isel(time=0).plot(
            ax=axes[1, 0], cmap='Blues',
            cbar_kwargs={'label': 'Soil Water (mm)'}
        )
        axes[1, 0].set_title(f'OzWALD Ssoil (~5 km)\n{str(ozwald_cube.time[0].values)[:10]}')
    else:
        axes[1, 0].text(0.5, 0.5, 'No OzWALD data\n(bbox too small for 5km grid)',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('OzWALD Ssoil (~5 km)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')

    # Time series
    if ozwald_cube.size > 0:
        ozwald_mean = ozwald_cube.mean(('latitude', 'longitude'))
        axes[1, 1].plot(ozwald_mean.time.values, ozwald_mean.values, marker='s', linewidth=2, color='orange', label='OzWALD (8-day)')
    axes[1, 1].set_title('OzWALD Mean Soil Moisture')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Ssoil (mm)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()

    plt.suptitle('Soil Moisture Comparison: SMIPS vs OzWALD', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = f'{test_q.tmp_dir}/soil_moisture_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f'\nPlot saved: {output_path}')
    plt.show()

    # Print statistics
    print('\n=== Statistics ===')
    print(f'SMIPS resolution: ~1 km (0.01°), daily')
    print(f'OzWALD resolution: ~5 km (0.05°), 8-day')
    print(f'\nSMIPS mean: {float(smips_mean.mean()):.1f} mm')
    print(f'SMIPS range: {float(smips_mean.min()):.1f} - {float(smips_mean.max()):.1f} mm')
    if ozwald_cube.size > 0:
        print(f'\nOzWALD mean: {float(ozwald_mean.mean()):.1f} mm')
        print(f'OzWALD range: {float(ozwald_mean.min()):.1f} - {float(ozwald_mean.max()):.1f} mm')


if __name__ == '__main__':
    compare_soil_moisture()
