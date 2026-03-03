import xarray as xr
from contextlib import contextmanager
from PaddockTS.Phenology.phenolopy.scripts import phenolopy

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


def estimate_phenology(query, ds_yearly=None, variable='NDVI'):
    """
    For each year in ds_yearly, compute phenology metrics using phenolopy.

    Parameters
    ----------
    query : Query
        The query object.
    ds_yearly : dict, optional
        Mapping from year (int) to xarray.Dataset with paddock time series and doy coordinate.
        If None, built from make_yearly_paddockTS.
    variable : str
        Name of the data variable to process.

    Returns
    -------
    dict of pd.DataFrame
        One DataFrame per year with phenology metrics.
    """
    if ds_yearly is None:
        from PaddockTS.PaddockTS.make_yearly_paddockTS import make_yearly_paddockTS
        ds_yearly = make_yearly_paddockTS(query)

    results = {}
    for year, ds in ds_yearly.items():
        ds_veg = (
            ds[[variable]]
            .rename({variable: "veg_index"})
            .drop_vars("doy")
        )

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
