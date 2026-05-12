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


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    results = estimate_phenology(query)
    for year, df in results.items():
        print(f'{year}: {df.columns.tolist()}')


if __name__ == '__main__':
    test()
