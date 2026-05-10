"""Split a paddockTS dataset by calendar year and attach day-of-year coords.

Phenology metrics are inherently per-season, so downstream code expects
each year as its own dataset with a ``doy`` coordinate to align years
on a common axis. This module persists one Zarr per year alongside the
in-memory dict it returns.
"""

import numpy as np
import xarray as xr


def split_paddockTS_by_year(ds):
    """Group a paddock time-series dataset into one slice per calendar year.

    Adds a ``doy`` coordinate (day-of-year, 1–366) on the ``time``
    dimension of each per-year slice so seasonal curves can be aligned
    across years on a common DOY axis.

    Args:
        ds: An xarray.Dataset on dims ``(paddock, time)`` (typically
            the output of :func:`make_paddockTS`).

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


def make_yearly_paddockTS(query, ds_paddockTS=None):
    """Persist one Zarr per year and return the same data as a dict.

    Loads the paddockTS Zarr if not provided, calls
    :func:`split_paddockTS_by_year`, and writes each per-year slice as
    Zarr v2 to ``{query.tmp_dir}/{query.stub}_paddockTS_{year}.zarr``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        ds_paddockTS: Optional in-memory paddockTS dataset. If ``None``,
            opens (or generates, then opens) the cached
            ``{query.stub}_paddockTS.zarr``.

    Returns:
        dict[int, xarray.Dataset]: Mapping ``{year: ds_year}``. Each
        per-year slice is also persisted to disk.
    """
    from os.path import exists

    if ds_paddockTS is None:
        zarr_path = f'{query.tmp_dir}/{query.stub}_paddockTS.zarr'
        if not exists(zarr_path):
            from PaddockTS.PaddockTS.make_paddockTS import make_paddockTS
            make_paddockTS(query)
        ds_paddockTS = xr.open_zarr(zarr_path, chunks=None)

    datasets_by_year = split_paddockTS_by_year(ds_paddockTS)

    for year, ds_year in datasets_by_year.items():
        year_path = f'{query.tmp_dir}/{query.stub}_paddockTS_{year}.zarr'
        ds_year.to_zarr(year_path, mode='w', zarr_format=2)
        print(f'Saved {year}: {ds_year.sizes["time"]} time steps -> {year_path}')

    return datasets_by_year


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    yearly = make_yearly_paddockTS(query)
    for year, ds in yearly.items():
        print(f'{year}: {ds.sizes["time"]} time steps, doy range {int(ds.doy.min())}-{int(ds.doy.max())}')


if __name__ == '__main__':
    test()
