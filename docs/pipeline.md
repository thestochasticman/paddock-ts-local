# Pipeline

PaddockTS runs two pipelines in parallel from a single `Query`:

- **Sentinel-2 → PaddockTS** — up to 21 stages producing paddock
  segmentation, time series, phenology, plots, and a stitched PDF
  report.
- **Environmental** — 7 stages pulling terrain, climate, and soil data.

`PaddockTS.get_outputs.get_outputs(query)` orchestrates both on two
threads with a live `rich` dashboard.

```text
┌────────────────────────────────────────────────────────────────────┐
│                            get_outputs(query)                      │
└────────────────────────────────────────────────────────────────────┘
                │                                  │
                ▼                                  ▼
   ┌──────────────────────┐           ┌─────────────────────────┐
   │ Environmental thread │           │ Sentinel-2 → PaddockTS  │
   │  • terrain (DEM)     │           │  • S2 download + clean  │
   │  • OzWALD daily      │           │  • spectral indices     │
   │  • SILO climate      │           │  • fractional cover     │
   │  • SLGA soils        │           │  • SAM segmentation     │
   │  • diagnostic plots  │           │  • per-paddock TS       │
   │  • topography plot ──┼─ waits ─► │  • yearly split         │
   │      (needs S2 grid) │           │  • phenology metrics    │
   └──────────────────────┘           │  • calendar + phenology │
                                      │  • PDF report           │
                                      └─────────────────────────┘
```

## How a stage works

Each stage is a plain Python function that:

- takes the `Query` plus optionally a previous stage's in-memory
  output as a kwarg,
- if that kwarg is missing, loads the previous output from disk (and
  cascades — generating it first if needed),
- writes its own output to a deterministic path derived from `Query`,
- touches a `_SUCCESS` marker **after** the data write completes; the
  marker is the cache-validity check on the next call.

This means you can call any stage in isolation, in any order, and the
caching falls into place. `get_outputs` exists purely to orchestrate
the full run with progress output — none of the stage functions
require it.

## Sentinel-2 → PaddockTS pipeline

The driver runs each numbered stage in order. Stages marked **(SAM)**
operate on the auto-segmented SAM paddocks; stages marked **(user)**
operate on a user-provided paddocks file passed via the
`paddocks_filepath` argument to `get_outputs`. Stages are skipped if
their inputs aren't available (e.g. `skip_sam=True` skips the SAM
stages, or no `paddocks_filepath` skips the user stages).

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | Download Sentinel-2 + clean | `Sentinel2.download_sentinel2` + `Sentinel2.clean_sentinel2` | `sentinel2.zarr`, `sentinel2_clean.zarr` |
| 2 | Compute spectral indices | `SpectralIndices.indices.compute_indices` | `indices.zarr` |
| 3 | Compute fractional cover | `FractionalCover.compute_fractional_cover` | `fractional_cover.zarr` |
| 4 | Sentinel-2 video | `Plotting.sentinel2_video` | `{stub}_sentinel2.mp4` |
| 5 | Segment paddocks (SAM) | `PaddockSegmentation.get_paddocks` | `sam_paddocks.gpkg` |
| 6 | S2 + paddocks video (SAM) | `Plotting.sentinel2_paddocks_video` | `..._sentinel2_paddocks.mp4` |
| 7 | S2 + paddocks video (user) | `Plotting.sentinel2_paddocks_video` | `..._sentinel2_paddocks.mp4` |
| 8 | Fractional cover video | `Plotting.fractional_cover_video` | `{stub}_fractional_cover.mp4` |
| 9 | FC + paddocks video (SAM) | `Plotting.fractional_cover_paddocks_video` | `..._fractional_cover_paddocks.mp4` |
| 10 | FC + paddocks video (user) | `Plotting.fractional_cover_paddocks_video` | `..._fractional_cover_paddocks.mp4` |
| 11 | Make paddock TS (SAM) | `Phenology.make_paddock_time_series` | `..._timeseries.zarr` |
| 12 | Make paddock TS (user) | `Phenology.make_paddock_time_series` | `..._timeseries.zarr` |
| 13 | Make yearly paddock TS (SAM) | `Phenology.make_yearly_paddock_time_series` | `..._timeseries_<year>.zarr` |
| 14 | Make yearly paddock TS (user) | `Phenology.make_yearly_paddock_time_series` | `..._timeseries_<year>.zarr` |
| 15 | Estimate phenology (SAM) | `Phenology.estimate_phenology` | `{year: DataFrame}` in-memory |
| 16 | Estimate phenology (user) | `Phenology.estimate_phenology` | `{year: DataFrame}` in-memory |
| 17 | Calendar plot (SAM) | `Plotting.calendar_plot` | `..._calendar_<year>_p01.png` |
| 18 | Calendar plot (user) | `Plotting.calendar_plot` | `..._calendar_<year>_p01.png` |
| 19 | Phenology plot (SAM) | `Plotting.phenology_plot` | `..._phenology_p01.png` |
| 20 | Phenology plot (user) | `Plotting.phenology_plot` | `..._phenology_p01.png` |
| 21 | PDF report | `Plotting.make_pdf` | `{stub}.pdf` |

### Stage 1: Download Sentinel-2 + clean

Two sub-stages. `download_sentinel2` searches the DEA STAC catalog for
overlapping `ga_s2am_ard_3` / `ga_s2bm_ard_3` scenes, loads the
requested bands (including `oa_fmask`) via `odc.stac` on a Dask
cluster, and writes the raw cube to `query.sentinel2_path`.
`clean_sentinel2` then masks out fmask cloud/shadow pixels, drops
scenes whose NaN fraction exceeds `max_nan_fraction` (default 0.5), and
writes the result to `query.sentinel2_clean_path`.

Both writes are guarded by `_SUCCESS` markers — a kill mid-write
leaves the cache invalidated and the next call refetches cleanly.

Failure modes specific to DEA's STAC are documented in
[`diagnostics.md`](https://github.com/thestochasticman/paddock-ts-local/blob/main/diagnostics.md)
at the repo root.

### Stage 5: Paddock segmentation (SAM)

Three internal steps:

1. **Presegmentation** (`_presegment`) — derives a single grayscale
   image from the multi-temporal Sentinel-2 stack using NDWI Fourier
   features. This collapses time into a representation that emphasises
   stable field boundaries and suppresses transient noise (clouds,
   shadows, seasonal greenness). Written as a GeoTIFF at
   `query.preseg_path`.
2. **SAM mask generation** — feeds the presegmented image to
   [`segment-geospatial`](https://samgeo.gishub.org/) (default
   backbone: SAM ViT-H, ~2.4 GB checkpoint auto-downloaded to
   `{config.tmp_dir}/sam_weights` on first run) and writes a mask
   GeoTIFF plus raw polygons GeoPackage.
3. **Vectorisation and filtering** — explodes multipart geometries,
   reprojects to a local UTM zone for accurate area / perimeter,
   computes `area_ha` and isoperimetric `compactness = 4πA/L²`, drops
   polygons outside `[min_area_ha, max_area_ha]` or below
   `min_compactness`, sorts by area descending, and assigns 1-based
   `paddock` IDs.

The final filtered GeoPackage lives at `query.sam_paddocks_path`.

### Stages 11–14: Per-paddock time series

`make_paddock_time_series` is the pivot from pixel-space to
paddock-space. It:

1. Computes the five spectral indices and adds them to the Sentinel-2
   dataset (cached via `compute_indices`).
2. Rasterises paddock polygons onto the Sentinel-2 grid using integer
   IDs.
3. For every data variable, in parallel across processes, computes the
   per-paddock NaN-aware median across pixels at every timestep.
4. Stitches the results into an `xarray.Dataset` on dims
   `(paddock, time)` and persists as Zarr v2.

`make_yearly_paddock_time_series` then splits the cube by calendar year
and attaches a `doy` (day-of-year, 1–366) coordinate, so seasonal
curves from different years align on a common DOY axis.

### Stages 15–16: Phenology

`estimate_phenology` wraps the vendored
[`phenolopy`](https://github.com/lewistrotter/phenolopy) library. For
each year and each paddock, it computes:

- `sos_times` / `sos_values` — start of season
- `pos_times` / `pos_values` — peak of season
- `eos_times` / `eos_values` — end of season
- amplitudes, length-of-season, integrals over season
- `num_peaks` — independent count of identified seasons

Paddocks with fewer than `min_observations` (default 25) valid points
in a year are skipped for that year. The result is one tidy
`pandas.DataFrame` per year, returned as `{year: DataFrame}`.

## Environmental data pipeline

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | Download terrain (Copernicus DEM) | `Environmental.TerrainTiles.download_terrain_tiles` | AOI-keyed `terrain.tif` |
| 2 | Download OzWALD daily | `Environmental.OzWALD.download_ozwald_daily` | `{stub}_ozwald_daily.csv` |
| 3 | Download SILO climate | `Environmental.SILO.download_silo` | `{stub}_silo.csv` |
| 4 | Download SLGA soils | `Environmental.SLGASoils.download_slgasoils` | `{stub}_{var}_{depth}.tif` × N |
| 5 | OzWALD plot | `Plotting.ozwald_plot.ozwald_daily_plot` | `{stub}_ozwald_daily_*.png` |
| 6 | SILO plot | `Plotting.silo_plot` | `{stub}_silo_*.png` |
| 7 | Terrain plot | `Plotting.terrain_tiles_plot` | `{stub}_topography.png` |

Stage 7 (terrain plot) waits for the cleaned Sentinel-2 cube
(`sentinel2_clean.zarr`) to be produced because it overlays the
terrain rendering on the S2 grid extent.

Stages 3 and 6 are skipped if `config.email` is unset; stage 4 is
skipped if `config.tern_api_key` is unset. Terrain and OzWALD work
without any credentials.

## Skipping the dashboard

If you want one stage and don't need the live UI, call it directly —
nothing requires `get_outputs`:

```python
from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
gdf = get_paddocks(query)
```

The function loads (and if necessary downloads) its own inputs from
the cache.

## See also

- **[API reference](api/index.md)** — full function signatures with
  runnable examples
- [`PaddockTS.get_outputs`](api/get_outputs.md) — the orchestrator
- [`diagnostics.md`](https://github.com/thestochasticman/paddock-ts-local/blob/main/diagnostics.md)
  — known failure modes (DEA STAC cold-start, GDAL HTTP auth)
