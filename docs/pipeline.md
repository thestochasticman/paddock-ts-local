# Pipeline

PaddockTS runs two pipelines in parallel from a single `Query`:

- **Sentinel-2 → PaddockTS** — 13 stages producing paddock segmentation, time series, and phenology
- **Environmental** — 7 stages pulling terrain, climate, and soil data

`PaddockTS.get_outputs.get_outputs(query)` orchestrates both pipelines on
two threads and shows progress live.

## Sentinel-2 → PaddockTS

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | Download Sentinel-2 | `Sentinel2.download_sentinel2` | `{stub}_sentinel2.zarr` |
| 2 | Compute indices | `SpectralIndices.indices` | added to `_sentinel2.zarr` |
| 3 | Compute fractional cover | `FractionalCover.compute_fractional_cover` | `{stub}_fractional_cover.zarr` |
| 4 | Sentinel-2 video | `Plotting.sentinel2_video` | `{stub}_sentinel2.mp4` |
| 5 | Segment paddocks | `PaddockSegmentation.get_paddocks` | `{stub}_paddocks.gpkg` |
| 6 | Sentinel-2 + paddocks video | `Plotting.sentinel2_paddocks_video` | `{stub}_sentinel2_paddocks.mp4` |
| 7 | Fractional cover video | `Plotting.fractional_cover_video` | `{stub}_fractional_cover.mp4` |
| 8 | Fractional cover + paddocks video | `Plotting.fractional_cover_paddocks_video` | `{stub}_fractional_cover_paddocks.mp4` |
| 9 | Make paddockTS | `PaddockTS.make_paddockTS` | `{stub}_paddockTS.zarr` |
| 10 | Make yearly paddockTS | `PaddockTS.make_yearly_paddockTS` | `{stub}_paddockTS_<year>.zarr` |
| 11 | Estimate phenology | `Phenology.estimate_phenology` | `{stub}_phenology_<year>.csv` |
| 12 | Calendar plot | `Plotting.calendar_plot` | `{stub}_calendar.png` |
| 13 | Phenology plot | `Plotting.phenology_plot` | `{stub}_phenology.png` |

Each stage is a standalone function that:

- Takes the `Query` plus optionally a previous stage's output as a kwarg.
- Reads the previous stage's output from disk if not passed in.
- Writes its own output to disk.
- Is **cache-aware**: rerunning a stage when its output exists is a no-op
  (or a load) for most stages.

This means you can run any stage in isolation, or call them in sequence
yourself instead of going through `get_outputs`. For example:

```python
from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
from PaddockTS.SpectralIndices.indices import compute_indices
from PaddockTS.FractionalCover import compute_fractional_cover
from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks

ds = download_sentinel2(query)
ds = compute_indices(query, ds_sentinel2=ds)
fc = compute_fractional_cover(query, ds_sentinel2=ds)
paddocks = get_paddocks(query, ds_sentinel2=ds)
```

### Stage 5: Paddock segmentation in detail

The segmentation stage uses
[Segment Anything](https://segment-anything.com/) via
`segment-geospatial` (`samgeo`). It runs in two passes:

1. **Presegmentation** — derive a grayscale image from the Sentinel-2 stack
   using NDWI Fourier features. This emphasises field boundaries and
   suppresses temporal noise.
2. **SAM mask generation** — feed the presegmented image to SAM
   (default model: ViT-H) and convert the resulting masks to vector
   polygons.

Polygons are then filtered by area and compactness to drop very small
or strangely-shaped masks. The result is a `geopandas.GeoDataFrame` with
columns `paddock`, `area_ha`, `compactness`, `geometry`.

### Stage 9: Per-paddock time series

`make_paddockTS` rasterises the paddock polygons over the Sentinel-2 grid
and computes a per-paddock median for every band and index at every
timestamp. The output is a `(paddock, time)` xarray Dataset.

This is the central time-series object that downstream stages
(`make_yearly_paddockTS`, `estimate_phenology`, plots) consume.

## Environmental data pipeline

| # | Stage | Module |
|---|---|---|
| 1 | Download terrain (Mapbox tiles) | `Environmental.TerrainTiles.download_terrain_tiles` |
| 2 | Download OzWALD daily | `Environmental.OzWALD.download_ozwald_daily` |
| 3 | Download SILO climate | `Environmental.SILO.download_silo` |
| 4 | Download SLGA soils | `Environmental.SLGASoils.download_slgasoils` |
| 5 | OzWALD plot | `Plotting.ozwald_plot` |
| 6 | SILO plot | `Plotting.silo_plot` |
| 7 | Terrain plot | `Plotting.terrain_tiles_plot` |

Stage 7 (terrain plot) waits for the Sentinel-2 download (Sentinel-2
stage 1) to complete because it overlays the terrain rendering on the
S2 grid extent.

## Skipping the dashboard

If you just want a single stage and don't need the live UI, call it
directly. None of the stage functions require `get_outputs` — they're
plain functions. `get_outputs` exists purely to orchestrate the full run
with progress output.

## See also

- [API reference](api/query.md) for full function signatures
- [`PaddockTS.get_outputs`](api/get_outputs.md) for the orchestrator's options
