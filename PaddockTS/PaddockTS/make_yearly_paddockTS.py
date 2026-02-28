import numpy as np
import xarray as xr


def split_paddockTS_by_year(ds):
    """
    Split paddock time series data by year, add day of year (doy) coordinate.

    Args:
        ds (xarray.Dataset): The input dataset containing time series data for each paddock.

    Returns:
        dict: A dictionary where each key is a year (int), and the value is an xarray.Dataset
              for that year, with an added 'doy' coordinate.
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
        ds_year.to_zarr(year_path, mode='w')
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
