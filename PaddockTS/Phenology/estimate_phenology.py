"""Per-paddock seasonal phenology metrics via the vendored phenolopy library.

For each year and each paddock, compute season-of-year metrics from a
single vegetation index time series:

- ``sos`` — start of season (DOY and value)
- ``pos`` — peak of season (DOY and value)
- ``eos`` — end of season (DOY and value)
- amplitudes, length-of-season, integrals, etc.

The implementation wraps :mod:`PaddockTS.Phenology._phenolopy` (a
vendored copy of `phenolopy <https://github.com/lewistrotter/phenolopy>`_
by Lewis Trotter, Apache 2.0; see
``PaddockTS/LICENSES/phenolopy.LICENSE``). A small monkey-patch is
applied to ``xr.merge`` during the call to silence a coordinate
mismatch upstream sees as a hard error.
"""

import xarray as xr
from contextlib import contextmanager
from PaddockTS.Phenology import _phenolopy as phenolopy

# Backup the original merge
_real_merge = xr.merge


@contextmanager
def _override_xr_merge():
    """Temporarily override xarray.merge to use compat='override' for phenolopy."""
    def _patched_merge(objs, *args, **kwargs):
        kwargs.pop("compat", None)
        return _real_merge(objs, compat="override", **kwargs)

    phenolopy.xr.merge = _patched_merge
    try:
        yield
    finally:
        phenolopy.xr.merge = _real_merge


def estimate_phenology(query, ds_yearly=None, variable='NDVI', min_observations=25):
    """Compute per-paddock phenology metrics for each year.

    For each year in ``ds_yearly``, this:

    1. Selects ``variable`` (e.g. ``'NDVI'``) and renames it to
       ``veg_index`` (phenolopy's expected variable name).
    2. Calls :func:`phenolopy.calc_num_seasons` to count peaks.
    3. Calls :func:`phenolopy.calc_phenometrics` with a seasonal-amplitude
       method (5% threshold, two-sided) to derive SoS / PoS / EoS and
       associated values.
    4. Flattens the result into a tidy :class:`pandas.DataFrame` and
       attaches the peak count.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        ds_yearly: Optional ``{year: xarray.Dataset}`` mapping (typically
            from :func:`PaddockTS.Phenology.make_yearly_paddock_time_series`). If
            ``None``, built on demand. Each dataset must have a ``doy``
            coordinate.
        variable: Name of the data variable to feed into phenolopy.
            Defaults to ``'NDVI'``; ``'NIRv'`` and ``'CFI'`` also work
            and are sometimes preferred for low-LAI canopies.
        min_observations: Minimum number of valid observations required
            per paddock. Paddocks with fewer are skipped. Default 25.

    Returns:
        dict[int, pandas.DataFrame]: One DataFrame per year. Columns
        include the phenolopy metrics (``sos_times``, ``sos_values``,
        ``pos_times``, ``eos_times``, etc.) plus ``num_peaks`` and a
        ``paddock`` identifier.
    """
    if ds_yearly is None:
        from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
        ds_yearly = make_yearly_paddock_time_series(query)

    results = {}
    for year, ds in ds_yearly.items():
        ds_veg = (
            ds[[variable]]
            .rename({variable: "veg_index"})
            .drop_vars("doy")
        )

        # Filter paddocks with insufficient observations
        valid_counts = ds_veg["veg_index"].count(dim="time")
        valid_mask = valid_counts >= min_observations

        if not valid_mask.any():
            print(f'  {year}: skipped (no paddocks with >= {min_observations} observations)')
            continue

        n_skipped = (~valid_mask).sum().item()
        if n_skipped > 0:
            print(f'  {year}: ignoring {n_skipped} paddocks with < {min_observations} observations')

        ds_veg = ds_veg.sel(paddock=valid_mask)

        da_num_seasons = phenolopy.calc_num_seasons(ds=ds_veg)

        with _override_xr_merge():
            ds_phenos = phenolopy.calc_phenometrics(
                da=ds_veg["veg_index"],
                peak_metric="pos",
                base_metric="bse",
                method="seasonal_amplitude",
                factor=0.05,
                thresh_sides="two_sided",
                abs_value=0,
            )

        phenos_df = (
            ds_phenos
            .drop_vars(["spatial_ref", "time"])
            .to_dataframe()
            .reset_index()
        )
        phenos_df["num_peaks"] = da_num_seasons.values

        results[year] = phenos_df
        print(f'  {year}: {len(phenos_df)} paddocks, {phenos_df["num_peaks"].mean():.1f} avg peaks')

    return results


def get_paddock_year_phenology(query, paddock_id, year: int, variable: str = 'NDVI') -> dict:
    """Return per-observation veg-index points + SoS/PoS/EoS metrics for one paddock × year.

    Same phenolopy configuration as :func:`estimate_phenology` (seasonal-amplitude method,
    5% factor, two-sided threshold), but computed for a single paddock so it can be
    served as a web payload.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        paddock_id: Paddock identifier matching the ``paddock`` coord of the cached
            timeseries zarrs (compared as strings).
        year: Calendar year to extract.
        variable: Vegetation index column name. Default ``'NDVI'``.

    Returns:
        dict with keys ``paddock_id`` (str), ``year`` (int), ``variable`` (str),
        ``observations`` (list[{doy, value}] for this paddock × year), and ``metrics``
        (dict with sos_time/value, pos_time/value, eos_time/value, num_peaks) or ``None``
        if phenometrics could not be computed.
    """
    import numpy as np
    import pandas as pd
    import traceback
    from os.path import exists

    paddock_str = str(paddock_id)
    yearly_zarr = f'{query.tmp_dir}/sam_paddocks_timeseries_{year}.zarr'
    if not exists(yearly_zarr):
        raise FileNotFoundError(f'yearly timeseries zarr not found: {yearly_zarr}')

    ds_year = xr.open_zarr(yearly_zarr, chunks=None, decode_coords='all')
    paddock_strs = [str(p) for p in ds_year.paddock.values]
    if paddock_str not in paddock_strs:
        raise ValueError(f'paddock_id {paddock_id!r} not in {yearly_zarr}')

    da = ds_year[variable].sel(paddock=paddock_str)
    doy = ds_year['doy'].values
    values = da.values
    observations = [
        {'doy': int(d), 'value': float(v)}
        for d, v in zip(doy, values)
        if np.isfinite(v)
    ]

    metrics = None
    try:
        ds_veg = (
            ds_year[[variable]]
            .rename({variable: 'veg_index'})
            .drop_vars('doy')
            .sel(paddock=[paddock_str])
        )
        da_num_seasons = phenolopy.calc_num_seasons(ds=ds_veg)
        with _override_xr_merge():
            ds_phenos = phenolopy.calc_phenometrics(
                da=ds_veg['veg_index'],
                peak_metric='pos',
                base_metric='bse',
                method='seasonal_amplitude',
                factor=0.05,
                thresh_sides='two_sided',
                abs_value=0,
            )
        phenos_df = (
            ds_phenos
            .drop_vars(['spatial_ref', 'time'], errors='ignore')
            .to_dataframe()
            .reset_index()
        )
        if not phenos_df.empty:
            r = phenos_df.iloc[0]
            def _safe(name):
                v = r.get(name)
                return float(v) if v is not None and pd.notna(v) else None
            metrics = {
                'sos_time': _safe('sos_times'),
                'sos_value': _safe('sos_values'),
                'pos_time': _safe('pos_times'),
                'pos_value': _safe('pos_values'),
                'eos_time': _safe('eos_times'),
                'eos_value': _safe('eos_values'),
                'num_peaks': int(da_num_seasons.values[0]) if da_num_seasons.size else None,
            }
    except Exception:
        traceback.print_exc()

    return {
        'paddock_id': paddock_str,
        'year': int(year),
        'variable': variable,
        'observations': observations,
        'metrics': metrics,
    }


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    results = estimate_phenology(query)
    for year, df in results.items():
        print(f'{year}: {df.columns.tolist()}')


if __name__ == '__main__':
    test()
