# Getting started

This page takes you from a fresh checkout to a first paddock-scale
analysis. It covers installation, configuration, constructing a `Query`,
running the full pipeline, and the layout of the outputs on disk.

## Requirements

- **Python ≥ 3.11** (3.11 or 3.12 recommended).
- **Operating system:** Linux, macOS, or WSL. Native Windows is
  untested — the geospatial stack (GDAL, PROJ, GEOS) is much easier to
  set up under conda on a POSIX environment.
- **Disk:** ~3 GB for a single year over ~5 km² (Sentinel-2 zarr +
  intermediate masks). Add ~2.5 GB for the SAM ViT-H checkpoint on
  first segmentation run.
- **Memory:** 8 GB minimum, 16 GB+ recommended for AOIs above a few
  square km. SAM on CPU peaks at ~6 GB.
- **GPU (optional):** SAM segmentation auto-detects CUDA; everything
  else is CPU.
- **`ffmpeg`** with the `libopenh264` encoder for the MP4 outputs
  (bundled by the conda install).

## Install

### Conda (recommended)

A single environment file installs the geospatial native stack (GDAL,
PROJ, GEOS), the ML stack (PyTorch, TensorFlow), and PaddockTS itself.

```bash
git clone https://github.com/thestochasticman/paddock-ts-local.git
cd paddock-ts-local
conda env create -f paddock-ts-env.yml
conda activate paddockts
pip install -e .
```

### pip

If you already have GDAL, PROJ, GEOS, and (optionally) CUDA installed
system-wide:

```bash
pip install -e .
```

Confirm the install:

```bash
python -c "from PaddockTS.query import Query; print(Query.__module__)"
# -> PaddockTS.query
```

## Configure

PaddockTS reads optional configuration from `~/.config/PaddockTS.json`.
Defaults are sensible for a single-user laptop, so this step is
optional — only the SILO and SLGA stages require credentials.

| Setting | Default | Required for |
|---|---|---|
| `out_dir` | `~/Documents/PaddockTS-Outputs` | final outputs |
| `tmp_dir` | `~/Downloads/PaddockTS-Tmp` | intermediates + caches |
| `email` | unset | SILO climate stage |
| `tern_api_key` | unset | SLGA soils stage |

Example `~/.config/PaddockTS.json`:

```json
{
  "out_dir": "/data/paddockts/outputs",
  "tmp_dir": "/data/paddockts/tmp",
  "email": "you@example.org",
  "tern_api_key": "<your-tern-key>"
}
```

- **SILO email** is registered with the upstream service; any working
  address is fine.
- **TERN API key** is generated at <https://account.tern.org.au/>.

The Sentinel-2 → PaddockTS chain itself works without any credentials.

### Pass a custom config from code

If you'd rather not write to `~/.config`, build a `Config` and pass it
to your `Query`:

```python
from datetime import date
from PaddockTS.config import Config
from PaddockTS.query import Query

cfg = Config(
    out_dir="/data/paddockts/outputs",
    tmp_dir="/data/paddockts/tmp",
    email="you@example.org",
    tern_api_key="<your-tern-key>",
)

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub="my_run",
    config=cfg,
)
```

## Construct a `Query`

`Query` is the immutable, content-addressed object that flows through
every stage. There are three ways to build one.

### From a bounding box

```python
from datetime import date
from PaddockTS.query import Query

query = Query(
    bbox=[148.36265, -33.52606, 148.38265, -33.50606],  # [W, S, E, N]
    start=date(2020, 1, 1),
    end=date(2021, 12, 31),
    stub="my_first_run",
)
```

Bounding boxes are `[west, south, east, north]` in EPSG:4326 (decimal
degrees). Snapped to 3 dp internally (~100 m) so near-identical bboxes
share their downloaded Sentinel-2 cube.

### From a centre point + buffer in km

```python
query = Query.from_lat_lon(
    lat=-35.098087,
    lon=148.929983,
    buffer_km=2.0,            # ±2 km from centre on each axis (≈ 4×4 km AOI)
    start=date(2025, 1, 1),
    end=date(2025, 6, 30),
    stub="point_buffered",
)
```

### From an existing paddocks file

If you already have field boundaries (GeoPackage, Shapefile, or GeoJSON):

```python
query = Query.build_from_paddocks(
    paddocks_filepath="/path/to/paddocks.gpkg",
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    stub="my_farm",
    label_col="paddock_name",   # column with human-readable IDs
)
```

The bbox is the envelope of all geometries (reprojected to EPSG:4326).

### About the `stub`

If you omit `stub`, a SHA-256 hash of `(bbox, start, end)` is used —
two queries with identical inputs share outputs on disk. Pass an
explicit string for human-readable filenames. Stubs are registered in
`{config.hash_file}` and must uniquely identify a `Query`; reusing a
stub for a different `(bbox, start, end)` raises `ValueError`.

## Run the pipeline

### Full run

The simplest entry point is `get_outputs`, which spawns the Sentinel-2
and environmental pipelines on parallel threads and shows a live
dashboard:

```python
from PaddockTS.get_outputs import get_outputs

get_outputs(query)
```

Common options:

```python
get_outputs(query, reload=True)        # delete tmp_dir + out_dir, then rerun
get_outputs(query, show_log=True)      # render a tail-of-log panel
get_outputs(                           # skip SAM, use user-provided paddocks
    query,
    paddocks_filepath="/path/to/paddocks.gpkg",
    skip_sam=True,
    label_col="paddock_name",
)
```

### Single stage

Every stage is a plain function. Call it directly when you want one
output and don't need the dashboard:

```python
from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
from PaddockTS.SpectralIndices.indices import compute_indices
from PaddockTS.FractionalCover import compute_fractional_cover
from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks

ds = download_sentinel2(query)                      # raw S2 zarr
ds = compute_indices(query, ds_sentinel2=ds)        # adds NDVI/CFI/...
fc = compute_fractional_cover(query, ds_sentinel2=ds)  # bg / pv / npv
paddocks = get_paddocks(query, ds_sentinel2=ds)     # GeoDataFrame
```

Every stage either accepts its inputs as a kwarg or loads them from
the cache; you can call any subset, in any order.

## Outputs

Per-query outputs land under `out_dir/<stub>/` and intermediates under
`tmp_dir/<stub>/`. Final outputs include:

| File | What's in it |
|---|---|
| `{stub}_paddocks.gpkg` | Segmented paddock polygons + `area_ha` + `compactness` |
| `{stub}_paddockTS.zarr` | Per-paddock medians for every band + index on `(paddock, time)` |
| `{stub}_paddockTS_<year>.zarr` | Yearly slices with a `doy` coordinate |
| `{stub}_sentinel2.mp4` | True-colour Sentinel-2 timeline |
| `{stub}_fractional_cover.mp4` | Bare / green / non-green RGB timeline |
| `{stub}_calendar_<year>_p01.png` | Per-paddock thumbnail calendar |
| `{stub}_phenology_p01.png` | SoS / PoS / EoS curves per paddock per year |
| `{stub}_topography.png` | Elevation, slope, aspect, flow accumulation |
| `{stub}_silo_*.png`, `{stub}_ozwald_daily_*.png` | Climate diagnostic panels |
| `{stub}.pdf` | Stitched report combining all of the above |

Intermediate artefacts (raw S2 zarr, presegmentation tif, SAM masks)
in `tmp_dir/<stub>/` can be safely deleted between runs without losing
your final outputs.

## Caching contract

Every cached output is guarded by a `_SUCCESS` marker file written
**after** the data write completes:

- Zarr cubes: `path/to/foo.zarr/_SUCCESS`
- GeoTIFFs / GeoPackages: `path/to/foo.tif._SUCCESS`

On startup each stage checks for both the data file and the marker. A
data file without its marker means a previous run was killed mid-write
(OOM, kill -9, network drop) and the stage refetches/recomputes. Pass
`reload=True` to `get_outputs` to force a clean rebuild.

## Next steps

- **[Pipeline](pipeline.md)** — what each stage does, what it caches,
  and how to skip or replace it.
- **[API reference](api/index.md)** — full public API with runnable
  examples for every function.
