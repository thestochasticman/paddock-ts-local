import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


def make_smoothed_paddockTS(query, ds_paddockTS=None, days=10, window_length=7, polyorder=2):
    """
    Resample, conservatively interpolate, and smooth all time-dependent
    variables in a paddock-time xarray Dataset.

      1. Separate non-time-dependent variables.
      2. Resample time-dependent data every `days` days (median).
      3. Interpolate missing values with PCHIP (conservative).
      4. Smooth with Savitzky-Golay.
      5. Re-attach static variables and return the new dataset.

    Parameters
    ----------
    query : Query
        The query object.
    ds_paddockTS : xarray.Dataset, optional
        The paddock time series dataset. If None, loaded from query.
    days : int, optional
        The resampling frequency in days (default is 10).
    window_length : int, optional
        The window length for the Savitzky-Golay filter (default is 7). This value must be odd.
        This is how many resampled obs the polynomial is fit to.
    polyorder : int, optional
        The polynomial order for the Savitzky-Golay filter (default is 2).
        Should be smaller than window_length.
    """
    from os.path import exists

    if ds_paddockTS is None:
        zarr_path = f'{query.tmp_dir}/{query.stub}_paddockTS.zarr'
        if not exists(zarr_path):
            from PaddockTS.PaddockTS.make_paddockTS import make_paddockTS
            make_paddockTS(query)
        ds_paddockTS = xr.open_zarr(zarr_path, chunks=None)

    ds = ds_paddockTS

    # 1. split vars
    time_dependent_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
    non_time_dependent_vars = [v for v in ds.data_vars
                               if v not in time_dependent_vars]

    ds_non_time = ds[non_time_dependent_vars]
    ds_time_dep = ds[time_dependent_vars]

    # 2. resample on a fixed grid
    ds_resampled = ds_time_dep.resample(time=f"{days}D").median()
    ds_resampled = ds_resampled.transpose("paddock", "time")

    # 3. interpolate with PCHIP
    interp_dict = {}
    x = np.arange(ds_resampled.time.size)

    for var in time_dependent_vars:
        data = ds_resampled[var].values  # (paddock, time)
        data_interp = np.empty_like(data, dtype=np.float64)

        for i in range(data.shape[0]):
            y = data[i]
            valid = np.isfinite(y)
            if valid.sum() >= 2:
                try:
                    f = PchipInterpolator(x[valid], y[valid], extrapolate=True)
                    data_interp[i] = f(x)
                except ValueError:
                    data_interp[i] = np.nanmean(y) if valid.any() else np.nan
            else:
                data_interp[i] = np.nanmean(y) if valid.any() else np.nan

        # 4. Savitzky-Golay smoothing
        wl = window_length + (window_length + 1) % 2  # make odd
        wl = min(wl, data_interp.shape[1] | 1)  # <= n_time & odd
        data_smoothed = savgol_filter(data_interp,
                                      window_length=wl,
                                      polyorder=polyorder,
                                      axis=-1)
        interp_dict[var] = (("paddock", "time"), data_smoothed)

    # 5. rebuild dataset
    ds_new = ds_resampled.copy()
    for var, da in interp_dict.items():
        ds_new[var] = da
    for var in non_time_dependent_vars:
        ds_new[var] = ds_non_time[var]

    for c in ds.coords:
        if c not in ds_new.coords:
            ds_new = ds_new.assign_coords({c: ds[c]})

    smoothed_path = f'{query.tmp_dir}/{query.stub}_paddockTS_smoothed.zarr'
    ds_new.to_zarr(smoothed_path, mode='w')
    print(f'Saved to {smoothed_path}')
    return ds_new


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    smoothed = make_smoothed_paddockTS(query)
    print(smoothed)
    print(f'Time steps: {smoothed.sizes["time"]}')


if __name__ == '__main__':
    test()
