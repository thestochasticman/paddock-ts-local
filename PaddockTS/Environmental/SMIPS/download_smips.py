"""
Download SMIPS soil moisture data for a Query.

Public CC-BY data from TERN (doi:10.25901/b020-nm39). Daily ~1 km volumetric
soil moisture across Australia. WMS temporal coverage: 2005-01-01 → ~2023-03-01.
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import requests
import rioxarray
import xarray as xr

from PaddockTS.query import Query
from .smips import SMIPS

smips = SMIPS()
get_filename = lambda q: f'{q.tmp_dir}/Environmental/{q.stub}_smips.nc'


def _fetch_tiff(
    d: date,
    bbox: tuple[float, float, float, float],
    layer: str = smips.layer,
) -> bytes:
    minx, miny, maxx, maxy = bbox
    width = min(smips.server_max_dim, max(1, round((maxx - minx) / smips.resolution_deg)))
    height = min(smips.server_max_dim, max(1, round((maxy - miny) / smips.resolution_deg)))

    params = {
        'service': 'WMS',
        'version': '1.1.1',
        'request': 'GetMap',
        'layers': layer,
        'styles': '',
        'srs': 'EPSG:4326',
        'bbox': f'{minx},{miny},{maxx},{maxy}',
        'width': width,
        'height': height,
        'format': 'image/tiff',
        'time': d.isoformat(),
    }
    r = requests.get(smips.wms_url, params=params, timeout=smips.timeout)
    r.raise_for_status()

    if not (r.content.startswith(b'II') or r.content.startswith(b'MM')):
        snippet = r.content[:400].decode('utf-8', errors='replace')
        raise RuntimeError(f'SMIPS WMS error for {d}: {snippet}')
    return r.content


def smips_day(
    d: date | str,
    bbox: tuple[float, float, float, float],
    layer: str = smips.layer,
) -> xr.DataArray:
    """Fetch SMIPS pixels over `bbox` for one day, as a 2D DataArray."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    tiff_bytes = _fetch_tiff(d, bbox, layer)

    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        tmp.write(tiff_bytes)
        tmp_path = tmp.name
    try:
        da = rioxarray.open_rasterio(tmp_path, masked=True).squeeze('band', drop=True).load()
    finally:
        os.unlink(tmp_path)
    return da


def smips_cube(
    start: date | str,
    end: date | str,
    bbox: tuple[float, float, float, float],
    layer: str = smips.layer,
    workers: int = 8,
    skip_missing: bool = True,
) -> xr.DataArray:
    """
    Fetch a (time, y, x) cube of SMIPS pixels over `bbox` between start and end
    (inclusive). Parallelised across days.
    """
    days = [d.date() for d in pd.date_range(start, end, freq='D')]
    slices: dict[date, xr.DataArray] = {}

    def fetch(d: date):
        return d, smips_day(d, bbox, layer=layer)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch, d): d for d in days}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                _, da = fut.result()
                slices[d] = da
            except (requests.RequestException, RuntimeError) as e:
                msg = f'[{d}] {type(e).__name__}: {e}'
                if skip_missing:
                    print(msg)
                else:
                    raise

    if not slices:
        raise RuntimeError('No SMIPS days returned data.')

    ordered = sorted(slices.items())
    times = pd.to_datetime([d for d, _ in ordered])
    cube = xr.concat([da for _, da in ordered], dim=pd.Index(times, name='time'))
    cube.name = layer.lower()
    cube.attrs.update(
        source='TERN SMIPS v1.0 via WMS',
        doi='10.25901/b020-nm39',
        license='CC-BY 4.0',
        endpoint=smips.wms_url,
        layer=layer,
    )
    return cube


def download_smips(query: Query, layer: str = smips.layer, workers: int = 8) -> xr.DataArray:
    """Download SMIPS data for a Query and cache to NetCDF."""
    from os import makedirs
    from os.path import exists

    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    filename = get_filename(query)

    if exists(filename):
        print(f'  cached: {filename}')
        ds = xr.open_dataset(filename)
        # Find the actual data variable (not spatial_ref or other metadata)
        data_vars = [v for v in ds.data_vars if v != 'spatial_ref']
        da = ds[data_vars[0]].load()
        ds.close()
        return da

    bbox = tuple(query.bbox)
    print(f'  fetching SMIPS data for bbox {bbox}...', flush=True)

    cube = smips_cube(query.start, query.end, bbox, layer=layer, workers=workers)
    # Compute to avoid dask scheduler issues, then save
    cube = cube.compute()
    cube.to_dataset(name=layer.lower()).to_netcdf(filename)
    print(f'  saved: {filename} ({len(cube.time)} days)')
    return cube


def test():
    import matplotlib.pyplot as plt
    from PaddockTS.utils import get_example_query
    q = get_example_query()
    # Use a shorter date range for testing
    from datetime import date
    from PaddockTS.query import Query
    test_q = Query(
        bbox=q.bbox,
        start=date(2022, 1, 1),
        end=date(2022, 1, 7),
        stub='SMIPS_TEST'
    )
    cube = download_smips(test_q)
    print(cube)
    print(f'\nShape: {cube.shape}  (time, y, x)')
    print(f'\nMean over bbox per day:\n{cube.mean(("x", "y")).to_pandas()}')

    # Plot spatial map for first timestep
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cube.isel(time=0).plot(ax=axes[0], cmap='Blues', cbar_kwargs={'label': 'Soil Water (mm)'})
    axes[0].set_title(f'SMIPS Soil Moisture\n{str(cube.time[0].values)[:10]}')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')

    # Plot time series of mean soil moisture
    mean_ts = cube.mean(('x', 'y'))
    axes[1].plot(mean_ts.time.values, mean_ts.values, marker='o', linewidth=2)
    axes[1].set_title('Mean Soil Moisture Over Bbox')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Total Bucket (mm)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{test_q.tmp_dir}/smips_test.png', dpi=150)
    print(f'\nPlot saved: {test_q.tmp_dir}/smips_test.png')
    # plt.show()


if __name__ == '__main__':
    test()
