"""Per-paddock × per-year phenology curves with SoS / PoS / EoS markers.

Generates a single multi-panel PNG: rows are paddocks, columns are
years, and each panel overlays the raw vegetation-index series (filled
blue dots) and the resampled-and-smoothed series (open blue dots) on a
DOY axis, with the start-of-season, peak-of-season, and end-of-season
DOYs drawn as vertical reference lines.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from PaddockTS.query import Query


def phenology_plot(query: Query, phenology_results: dict[int, pd.DataFrame] | None = None, ds_yearly: dict[int, xr.Dataset] | None = None, ds_paddockTS: xr.Dataset | None = None, variable: str = 'NDVI') -> str:
    """Plot per-paddock × per-year phenology curves with SoS / PoS / EoS markers.

    Args:
        query: The :class:`PaddockTS.query.Query`. Output is written to
            ``{query.out_dir}/{query.stub}_phenology.png``.
        phenology_results: Optional ``{year: DataFrame}`` from
            :func:`PaddockTS.Phenology.estimate_phenology`. If ``None``,
            recomputed on demand using ``variable``.
        ds_yearly: Optional ``{year: Dataset}`` from
            :func:`PaddockTS.PaddockTimeSeries.make_yearly_paddock_time_series`. If
            ``None``, built on demand.
        ds_paddockTS: Optional raw per-paddock time series Dataset. If
            ``None``, opened (or generated, then opened) from cache.
            Used to plot the un-resampled raw observations.
        variable: Vegetation index column to plot. Default ``'NDVI'``;
            ``'NIRv'`` and ``'CFI'`` also work.

    Returns:
        str: Filesystem path of the generated PNG.
    """
    import os
    os.makedirs(query.out_dir, exist_ok=True)

    if phenology_results is None:
        from PaddockTS.Phenology.estimate_phenology import estimate_phenology
        phenology_results = estimate_phenology(query, ds_yearly=ds_yearly, variable=variable)

    if ds_yearly is None:
        from PaddockTS.PaddockTimeSeries.make_yearly_paddock_time_series import make_yearly_paddock_time_series
        ds_yearly = make_yearly_paddock_time_series(query)

    if ds_paddockTS is None:
        zarr_path = f'{query.tmp_dir}/{query.stub}_paddockTS.zarr'
        if not os.path.exists(zarr_path):
            from PaddockTS.PaddockTimeSeries.make_paddock_time_series import make_paddock_time_series
            make_paddock_time_series(query)
        ds_paddockTS = xr.open_zarr(zarr_path, chunks=None)

    years = sorted(ds_yearly.keys())
    paddocks = list(ds_yearly[years[0]].paddock.values)
    n_rows, n_cols = len(paddocks), len(years)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 1.5 * n_rows),
                             squeeze=False)

    for i, paddock in enumerate(paddocks):
        for j, year in enumerate(years):
            ax = axes[i, j]
            df_year = phenology_results[year]
            ds_year = ds_yearly[year]

            # Interpolated series
            da_res = ds_year[variable].sel(paddock=str(paddock))
            res_doy = ds_year['doy'].values
            ax.scatter(res_doy, da_res.values,
                       facecolors='white', edgecolors='blue',
                       s=20, label='interpolated')

            # Raw series
            ds_raw_year = ds_paddockTS.sel(
                time=slice(f"{year}-01-01", f"{year}-12-31")
            )
            da_raw = ds_raw_year[variable].sel(paddock=str(paddock))
            raw_doy = da_raw['time'].dt.dayofyear.values
            ax.scatter(raw_doy, da_raw.values,
                       color='blue', s=20, label='raw')

            # Phenology lines
            row = df_year[df_year['paddock'].astype(str) == str(paddock)]
            if not row.empty:
                r = row.iloc[0]
                ax.axvline(r['sos_times'], color='green', linestyle='--', label='SoS')
                ax.axvline(r['pos_times'], color='blue', linestyle='-.', label='PoS')
                ax.axvline(r['eos_times'], color='red', linestyle=':', label='EoS')

            ax.set_ylim(0, 1)
            if j == 0:
                ax.tick_params(labelleft=True)
                ax.set_ylabel(variable)
            else:
                ax.tick_params(labelleft=False)

            if i == n_rows - 1:
                ax.set_xlabel("DOY")
            else:
                ax.tick_params(labelbottom=False)

            if i == 0:
                ax.set_title(f"{year}", pad=8)

            if j == n_cols - 1:
                ax.text(1.02, 0.5, f"Paddock {paddock}",
                        transform=ax.transAxes, va='center')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f'{query.out_dir}/{query.stub}_phenology.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved to {out_path}')
    return out_path


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    phenology_plot(query, variable='NDVI')


if __name__ == '__main__':
    test()
