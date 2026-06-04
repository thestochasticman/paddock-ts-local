"""Split a paddockTS dataset by calendar year and attach day-of-year coords.

Phenology metrics are inherently per-season, so downstream code expects
each year as its own dataset with a ``doy`` coordinate to align years
on a common axis. This module persists one Zarr per year alongside the
in-memory dict it returns.
"""

import numpy as np
import xarray as xr


def split_paddock_time_series_by_year(ds):
    """Group a paddock time-series dataset into one slice per calendar year.

    Adds a ``doy`` coordinate (day-of-year, 1–366) on the ``time``
    dimension of each per-year slice so seasonal curves can be aligned
    across years on a common DOY axis.

    Args:
        ds: An xarray.Dataset on dims ``(paddock, time)`` (typically
            the output of :func:`make_paddock_time_series`).

    Returns:
        dict[int, xarray.Dataset]: Mapping ``{year: ds_year}`` where each
        ``ds_year`` covers a single calendar year and carries a ``doy``
        coordinate alongside ``time``.
    """
    years = np.unique(ds.time.dt.year.values)
    datasets_by_year = {}

    for year in years:
        ds_year = ds.sel(time=ds.time.dt.year == year)
        ds_year.attrs['year'] = int(year)
        doy = ds_year.time.dt.dayofyear.data
        ds_year = ds_year.assign_coords(doy=('time', doy))
        datasets_by_year[int(year)] = ds_year
    return datasets_by_year

def make_yearly_paddock_time_series(query, ds_paddockTS=None, paddocks_filepath=None):
    """Persist one Zarr per year and return the same data as a dict.

    Loads the paddockTS Zarr if not provided, calls
    :func:`split_paddock_time_series_by_year`, and writes each per-year slice as
    Zarr v2 to ``{paddocks_filepath stem}_timeseries_{year}.zarr``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        ds_paddockTS: Optional in-memory paddockTS dataset. If ``None``,
            opens (or generates, then opens) the cached timeseries zarr.
        paddocks_filepath: Path to the paddocks GeoPackage. Used to derive
            the timeseries zarr path. If ``None``, defaults to
            ``{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg``.

    Returns:
        dict[int, xarray.Dataset]: Mapping ``{year: ds_year}``. Each
        per-year slice is also persisted to disk.
    """
    from datetime import datetime
    from os import makedirs
    from pathlib import Path
    from PaddockTS.Sentinel2.check_if_valid_zarr_exists import check_if_valid_zarr_exists

    if paddocks_filepath is None:
        paddocks_filepath = query.sam_paddocks_path

    paddocks_path = Path(paddocks_filepath)
    timeseries_zarr = f'{query.tmp_dir}/{paddocks_path.stem}_timeseries_smoothed.zarr'

    if ds_paddockTS is None:
        if not check_if_valid_zarr_exists(timeseries_zarr):
            from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
            make_paddock_time_series(query, paddocks_filepath=paddocks_filepath)
        ds_paddockTS = xr.open_zarr(timeseries_zarr, chunks=None, decode_coords='all')

    datasets_by_year = split_paddock_time_series_by_year(ds_paddockTS)

    makedirs(query.tmp_dir, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + 'Z'
    for year, ds_year in datasets_by_year.items():
        year_path = f'{query.tmp_dir}/{paddocks_path.stem}_timeseries_{year}.zarr'
        ds_year = ds_year.assign_attrs(yearly_split_computed_at=timestamp)
        ds_year.to_zarr(year_path, mode='w', zarr_format=2)
        with open(f'{year_path}/_SUCCESS', 'w') as f:
            f.write(timestamp)
        print(f'Saved {year}: {ds_year.sizes["time"]} time steps -> {year_path}')

    return datasets_by_year


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    yearly = make_yearly_paddock_time_series(query)
    for year, ds in yearly.items():
        print(f'{year}: {ds.sizes["time"]} time steps, doy range {int(ds.doy.min())}-{int(ds.doy.max())}')


if __name__ == '__main__':
    test()