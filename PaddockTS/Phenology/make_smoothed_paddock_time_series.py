"""Resample, gap-fill, and smooth a per-paddock time series.

Sentinel-2 revisit gaps and cloud-mask drops leave irregular time
series. This module produces a uniform, smoothed version suitable for
phenology fitting and plotting:

1. resample to a fixed cadence (median per bin),
2. fill gaps with monotone PCHIP interpolation (no overshoot), and
3. smooth with a Savitzky-Golay filter (low-order polynomial fit).

Static (non-time) variables are passed through untouched.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


def make_smoothed_paddock_time_series(query, ds_paddockTS=None, paddocks_filepath=None, days=10, window_length=7, polyorder=2):
    """Resample-then-interpolate-then-smooth all time-dependent variables.

    Pipeline applied to each paddock × variable series:

    1. Split out non-time-dependent vars (static metadata, etc.).
    2. Resample time-dependent data with a ``days``-day median.
    3. Fill gaps with PCHIP (monotone cubic) interpolation;
       fall back to mean-fill for series with fewer than 2 valid points.
    4. Smooth with a Savitzky-Golay filter
       (``window_length`` and ``polyorder`` configurable).
    5. Re-attach static variables and persist as Zarr v2 to
       ``{paddocks_filepath stem}_timeseries_smoothed.zarr``.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        ds_paddockTS: Optional in-memory paddockTS dataset. If ``None``,
            opens (or generates, then opens) the cached timeseries zarr.
        paddocks_filepath: Path to the paddocks GeoPackage. Used to derive
            the timeseries zarr path. If ``None``, defaults to
            ``{query.tmp_dir}/{query.stub}_sam_paddocks.gpkg``.
        days: Resampling cadence in days. Default 10.
        window_length: Savitzky-Golay window size in number of resampled
            samples. Coerced to the next odd integer if even and clipped
            to fit short series. Default 7.
        polyorder: Savitzky-Golay polynomial order. Must be less than
            ``window_length``. Default 2.

    Returns:
        xarray.Dataset: Smoothed dataset on dims ``(paddock, time)``
        with the same data variables as the input, plus an ``observed``
        boolean variable marking which resampled bins contained at least
        one real observation (False = gap-filled). Also persisted to
        ``{paddocks_filepath stem}_timeseries_smoothed.zarr``.
    """
    from datetime import datetime
    from os import makedirs
    from pathlib import Path
    from PaddockTS.Sentinel2.check_if_valid_zarr_exists import check_if_valid_zarr_exists

    if paddocks_filepath is None:
        paddocks_filepath = query.sam_paddocks_path

    paddocks_path = Path(paddocks_filepath)
    timeseries_zarr = f'{query.tmp_dir}/{paddocks_path.stem}_timeseries.zarr'

    if ds_paddockTS is None:
        if not check_if_valid_zarr_exists(timeseries_zarr):
            from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
            make_paddock_time_series(query, paddocks_filepath=paddocks_filepath)
        ds_paddockTS = xr.open_zarr(timeseries_zarr, chunks=None, decode_coords='all')

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

    # Bins that contained at least one real observation, captured before
    # gap-fill. Persisted as an `observed` data variable so downstream
    # consumers (e.g. the web /phenology endpoint) can distinguish
    # observed from interpolated samples.
    observed = np.zeros((ds_resampled.sizes["paddock"], ds_resampled.sizes["time"]), dtype=bool)
    for var in time_dependent_vars:
        observed |= np.isfinite(ds_resampled[var].values)

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
    ds_new["observed"] = (("paddock", "time"), observed)
    for var in non_time_dependent_vars:
        ds_new[var] = ds_non_time[var]

    for c in ds.coords:
        if c not in ds_new.coords:
            ds_new = ds_new.assign_coords({c: ds[c]})

    smoothed_path = f'{query.tmp_dir}/{paddocks_path.stem}_timeseries_smoothed.zarr'
    makedirs(query.tmp_dir, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + 'Z'
    ds_new = ds_new.assign_attrs(smoothed_computed_at=timestamp)
    ds_new.to_zarr(smoothed_path, mode='w', zarr_format=2)
    with open(f'{smoothed_path}/_SUCCESS', 'w') as f:
        f.write(timestamp)
    print(f'Saved to {smoothed_path}')
    return ds_new


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    smoothed = make_smoothed_paddock_time_series(query)
    print(smoothed)
    print(f'Time steps: {smoothed.sizes["time"]}')


if __name__ == '__main__':
    test()
