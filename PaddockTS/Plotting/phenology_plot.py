"""Per-paddock × per-year phenology curves with SoS / PoS / EoS markers.

Generates a single multi-panel PNG: rows are paddocks, columns are
years, and each panel plots the sampled + interpolated vegetation-index
series (the smoothed yearly dataset: 10-day median resample -> PCHIP
gap-fill -> Savitzky-Golay) as markers on a DOY axis — filled circles
for sampled bins (held a real observation), hollow circles for
interpolated (gap-filled) bins — with the start-of-season,
peak-of-season, and end-of-season DOYs drawn as vertical reference
lines. This mirrors the web viewer's phenology panel, which scatters the
same series rather than connecting it into a line.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import xarray as xr

from PaddockTS.query import Query


def phenology_plot(query: Query, phenology_results: dict[int, pd.DataFrame] | None = None, ds_yearly: dict[int, xr.Dataset] | None = None, ds_paddockTS: xr.Dataset | None = None, variable: str = 'NDVI', paddocks_filepath: str | None = None, max_paddocks_per_page: int = 8, label_col: str | None = None) -> list[str]:
    """Plot per-paddock × per-year phenology curves with SoS / PoS / EoS markers.

    Args:
        query: The :class:`PaddockTS.query.Query`. Output is written to
            ``{query.out_dir}/{paddocks_stem}_phenology.png``.
        phenology_results: Optional ``{year: DataFrame}`` from
            :func:`PaddockTS.Phenology.estimate_phenology`. If ``None``,
            recomputed on demand using ``variable``.
        ds_yearly: Optional ``{year: Dataset}`` from
            :func:`PaddockTS.Phenology.make_yearly_paddock_time_series`. If
            ``None``, built on demand.
        ds_paddockTS: Deprecated and unused. Retained for backward
            compatibility with callers that still pass it (the raw
            overlay was removed in favour of plotting only the sampled +
            interpolated series, matching the web viewer).
        variable: Vegetation index column to plot. Default ``'NDVI'``;
            ``'NIRv'`` and ``'CFI'`` also work.
        paddocks_filepath: Path to the paddocks file. Used to derive
            the output filename stem. If ``None``, defaults to
            ``{query.stub}_sam_paddocks``.
        max_paddocks_per_page: Maximum number of paddocks per output image.
            Default 8. Prevents images from becoming too tall with many
            paddocks.
        label_col: Column name to use for paddock labels. If ``None``,
            uses the ``paddock`` column (numeric ID).

    Returns:
        list[str]: Filesystem paths of the generated PNGs.
    """
    import os
    from pathlib import Path
    os.makedirs(query.out_dir, exist_ok=True)

    # Derive output filename stem from paddocks_filepath
    if paddocks_filepath is not None:
        out_stem = Path(paddocks_filepath).stem
    else:
        out_stem = f'{query.stub}_sam_paddocks'

    # Build paddock label mapping
    if label_col is not None:
        from PaddockTS.utils import load_user_paddocks
        label_filepath = paddocks_filepath if paddocks_filepath else query.sam_paddocks_path
        gdf = load_user_paddocks(label_filepath)
        paddock_labels = dict(zip(gdf['paddock'].astype(str), gdf[label_col].astype(str)))
    else:
        paddock_labels = None

    if phenology_results is None:
        from PaddockTS.Phenology.estimate_phenology import estimate_phenology
        phenology_results = estimate_phenology(query, ds_yearly=ds_yearly, variable=variable)

    if ds_yearly is None:
        from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
        ds_yearly = make_yearly_paddock_time_series(query)

    # Only plot years that have phenology results (some may be skipped due to insufficient data)
    years = sorted(set(ds_yearly.keys()) & set(phenology_results.keys()))
    if not years:
        print('No years with phenology results to plot')
        return []
    paddocks = list(ds_yearly[years[0]].paddock.values)
    n_paddocks = len(paddocks)
    n_cols = len(years)

    # Clean up any existing phenology files for this stem
    import glob
    for old_file in glob.glob(f'{query.out_dir}/{out_stem}_phenology*.png'):
        os.remove(old_file)

    # Split paddocks into pages
    n_pages = (n_paddocks + max_paddocks_per_page - 1) // max_paddocks_per_page
    paddock_pages = [paddocks[i * max_paddocks_per_page:(i + 1) * max_paddocks_per_page] for i in range(n_pages)]

    out_paths = []

    # Fixed row height for consistent sizing across pages
    row_height = 2.5
    fig_height = row_height * max_paddocks_per_page

    for page_idx, page_paddocks in enumerate(paddock_pages):
        n_rows = len(page_paddocks)

        fig = plt.figure(figsize=(8 * n_cols, fig_height))

        # Use GridSpec to position axes at the top, leaving empty space at bottom if needed
        gs = GridSpec(max_paddocks_per_page, n_cols, figure=fig)

        axes = []
        for i in range(n_rows):
            row_axes = []
            for j in range(n_cols):
                ax = fig.add_subplot(gs[i, j])
                row_axes.append(ax)
            axes.append(row_axes)
        axes = np.array(axes)

        for i, paddock in enumerate(page_paddocks):
            for j, year in enumerate(years):
                ax = axes[i, j]
                df_year = phenology_results[year]
                ds_year = ds_yearly[year]

                # Sampled + interpolated series (10-day median resample ->
                # PCHIP gap-fill -> Savitzky-Golay) plotted as markers, to
                # match the web viewer's phenology panel. Sampled bins (held
                # a real observation) are filled circles; interpolated
                # (gap-filled) bins are hollow circles. The `observed` mask
                # comes from make_smoothed; if absent (older caches), every
                # point is treated as sampled.
                da_res = ds_year[variable].sel(paddock=str(paddock))
                res_doy = ds_year['doy'].values
                vals = da_res.values
                if 'observed' in ds_year:
                    obs = ds_year['observed'].sel(paddock=str(paddock)).values.astype(bool)
                else:
                    obs = np.ones(vals.shape, dtype=bool)
                ax.scatter(res_doy[obs], vals[obs],
                           color='#5aa8ff', s=18, label='sampled')
                ax.scatter(res_doy[~obs], vals[~obs],
                           facecolors='none', edgecolors='#5aa8ff', s=18,
                           label='interpolated')

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
                    label = paddock_labels[str(paddock)] if paddock_labels else f"Paddock {paddock}"
                    ax.text(1.02, 0.5, label, transform=ax.transAxes, va='center')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Output filename always includes page number
        out_path = f'{query.out_dir}/{out_stem}_phenology_p{page_idx + 1:02d}.png'
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f'Saved to {out_path}')
        out_paths.append(out_path)

    return out_paths


def test():
    from PaddockTS.utils import get_example_query

    query = get_example_query()
    phenology_plot(query, variable='NDVI')


if __name__ == '__main__':
    test()
