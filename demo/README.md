# PaddockTS demos

Runnable Jupyter notebooks showing how to use the system end-to-end.
Start with [`01_quickstart.ipynb`](01_quickstart.ipynb) and pick the
others based on what you need.

| Notebook | What it covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | The simplest end-to-end run: bbox + dates → `get_outputs(query)` → live dashboard → review the calendar / phenology / PDF. |
| [`02_pipeline_stages.ipynb`](02_pipeline_stages.ipynb) | Calling each stage individually (Sentinel-2 download, clean, indices, fractional cover, SAM segmentation, per-paddock TS, phenology) — useful for debugging, fine-tuning, or replacing a single stage with your own implementation. Also shows how to inspect the outputs (geopackage, zarr, dataframes). |
| [`03_custom_paddocks.ipynb`](03_custom_paddocks.ipynb) | Bringing your own paddock boundaries (GeoPackage / Shapefile / GeoJSON) and skipping SAM. Uses the bundled `artifacts/PaddockSet1.gpkg` as the worked example. |

## Running the notebooks

```bash
# From the repo root, with the paddockts conda env activated:
jupyter lab demo/
```

Each notebook is keyed to its own `stub` so the demos don't collide
with each other on disk. They use small AOIs (~2 km × 2 km) and short
date windows (a few months) so cold runs finish in a few minutes; the
content-addressed cache means rerunning is instant.

## Prerequisites

- PaddockTS installed (`pip install -e .` or `conda env create -f
  paddock-ts-env.yml && conda activate paddockts && pip install -e .`).
- Internet access — Sentinel-2 is downloaded on demand from Geoscience
  Australia's STAC.
- **No** credentials needed for the Sentinel-2 chain. SILO needs an
  email (any working address registered with the service) and SLGA
  needs a TERN API key — see the [Getting Started docs](https://thestochasticman.github.io/paddock-ts-local/getting-started/)
  if you plan to use those environmental layers.

## What gets written to disk

Each notebook prints the relevant paths as it goes. By default:

- **Outputs** (paddocks gpkg, time-series zarr, calendar PNGs, MP4
  videos, PDF report) → `~/Documents/PaddockTS-Outputs/<stub>/`
- **Caches** (raw + clean Sentinel-2 zarr, indices, fractional cover,
  presegmentation tif, SAM masks, terrain DEM, env CSVs) →
  `~/Downloads/PaddockTS-Tmp/...`

To redirect both, set `out_dir` / `tmp_dir` in `~/.config/PaddockTS.json`
before running. See [Getting Started → Configure](https://thestochasticman.github.io/paddock-ts-local/getting-started/#configure).
