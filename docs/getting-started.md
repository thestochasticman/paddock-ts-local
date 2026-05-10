# Getting started

## Install

PaddockTS depends on a substantial geospatial + ML stack
(`rasterio`, `geopandas`, `xarray`, `dask`, `tensorflow`, `torch`,
`segment-geospatial`, …) which is far easier to set up via conda.

### Conda (recommended)

```bash
git clone https://github.com/thestochasticman/paddock-ts-local.git
cd paddock-ts-local
conda env create -f paddock-ts-env.yml
conda activate paddockts-env
pip install -e .
```

### pip

If you have the native libraries already (GDAL, PROJ, GEOS, CUDA where
applicable), you can install via pip:

```bash
pip install PaddockTS
```

PaddockTS requires Python ≥ 3.11.

## Configure

Default output and tmp directories are:

| Setting    | Default                            |
|-----------:|:-----------------------------------|
| `out_dir`  | `~/Documents/PaddockTS-Outputs`    |
| `tmp_dir`  | `~/Downloads/PaddockTS-Tmp`        |
| `email`    | *(unset; required for SILO)*       |
| `tern_api_key` | *(unset; required for SLGA soils)* |

Override by writing `~/.config/PaddockTS.json`:

```json
{
  "out_dir": "/data/paddockts/outputs",
  "tmp_dir": "/data/paddockts/tmp",
  "email": "you@example.org",
  "tern_api_key": "<your-tern-key>"
}
```

The `email` and `tern_api_key` fields are only needed if you call the
SILO climate or SLGA soils stages. The Sentinel-2 → PaddockTS pipeline
itself works without them.

## Construct a query

Two ways to make a `Query`:

### From a bounding box

```python
from datetime import date
from PaddockTS.query import Query

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],  # [west, south, east, north]
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub="my_first_run",
)
```

### From a centre point + buffer in km

```python
query = Query.from_lat_lon(
    lat=-35.098087,
    lon=148.929983,
    buffer_km=2,
    start=date(2025, 6, 1),
    end=date(2025, 6, 30),
    stub="example_2",
)
```

If you don't pass `stub`, a content-addressed identifier is computed from
the bbox + dates (a SHA-256 hash). Two queries with the same inputs will
share outputs on disk.

## Run the pipeline

The simplest entry point is `get_outputs`, which runs the full
Sentinel-2 → PaddockTS pipeline alongside the environmental data
pipeline, with a live status dashboard:

```python
from PaddockTS.get_outputs import get_outputs

get_outputs(query)
```

To re-run from scratch (delete cached outputs first):

```python
get_outputs(query, reload=True)
```

To show the captured log panel under the status tables:

```python
get_outputs(query, show_log=True)
```

## Outputs

Per-query outputs land under `out_dir/<stub>/`:

- `<stub>_paddocks.gpkg` — segmented paddock polygons
- `<stub>_paddockTS.zarr` — per-paddock time series of all bands and indices
- `<stub>_paddockTS_<year>.zarr` — yearly slices of the time series
- `<stub>_phenology_<year>.csv` — per-paddock phenology metrics
- `<stub>_calendar.png`, `<stub>_phenology.png` — diagnostic plots
- `<stub>_sentinel2.mp4`, `<stub>_fractional_cover.mp4`, … — animation videos

Intermediate artefacts (raw Sentinel-2 zarr, presegmentation tif, SAM
masks) are kept in `tmp_dir/<stub>/` and can be safely deleted between
runs without losing your final outputs.

## Next steps

- [Pipeline](pipeline.md) — what each stage does and how to call them individually
- [API reference](api/query.md) — full public API
