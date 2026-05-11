# PaddockTS

**Paddock segmentation and time-series analysis from Sentinel-2 imagery.**

PaddockTS turns a bounding box and a date range into a full set of
paddock-scale outputs: segmented paddock polygons, vegetation indices,
fractional cover (bare / green / non-green), per-paddock time series,
phenology metrics, and ready-to-use plots and videos.

## What it does

Given a `Query` (a bounding box + date range), the pipeline:

1. Downloads cloud-masked Sentinel-2 surface reflectance from a STAC catalog.
2. Computes vegetation indices (NDVI, CFI, NIRv, NDTI, CAI).
3. Estimates fractional cover via spectral unmixing.
4. Segments paddocks using SAM (Segment Anything) over a presegmentation
   image derived from NDWI Fourier features.
5. Aggregates the time series per paddock.
6. Computes phenology metrics per paddock per year.
7. Renders plots and videos for review.

Alongside this, an environmental-data pipeline pulls terrain (DEM-derived
flow accumulation, slope, TWI), daily climate (OzWALD, SILO), and
soil properties (SLGA) for the same area.

## Why PaddockTS

- **One bounding box → everything.** No glue scripts. The full chain runs
  end-to-end from a single `Query` object.
- **Built for Australian agronomy.** Defaults and integrations target Sentinel-2
  ARD, OzWALD, SILO, SLGA, and TERN datasets.
- **Composable.** Every stage (`download_sentinel2`, `compute_indices`,
  `compute_fractional_cover`, `get_paddocks`, `make_paddock_time_series`, `estimate_phenology`)
  is independently callable. Skip stages, swap in your own data.
- **Caches by content.** A query's `stub` is a SHA-256 of its inputs, so
  outputs are cached by what the query *means*, not by an arbitrary name.

## Quick example

```python
from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub="my_first_run",
)

get_outputs(query)
```

This kicks off the full pipeline with a live status dashboard. Outputs
land under `~/Documents/PaddockTS-Outputs/<stub>/` (configurable).

## Where to go next

- [Getting started](getting-started.md) — install, configure, run your first query
- [Pipeline](pipeline.md) — the 13 Sentinel-2 stages and 7 environmental stages, what they produce, and how to skip or replace any of them
- [API reference](api/query.md) — every public function and class

## License

PaddockTS is MIT-licensed. It vendors third-party code under permissive
licenses; see `PaddockTS/LICENSES/` for attribution to:

- `fractionalcover3` (Robert Denham, MIT)
- `phenolopy` (Lewis Trotter, Apache 2.0)
